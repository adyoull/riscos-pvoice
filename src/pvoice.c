/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
/* pvoice: Piper's VITS voices in plain C. The model is the one in Piper's
 * training code (piper/train/vits: models.py, modules.py, attentions.py,
 * transforms.py), run forward for inference only:
 *
 *   phoneme ids -> text encoder (embedding, transformer with relative
 *   attention) -> duration predictor (stochastic: flows with rational
 *   quadratic splines, run backwards from noise) -> each phoneme's frames ->
 *   flow (residual coupling layers, backwards) -> HiFi-GAN decoder -> audio.
 *
 * Signals are [channels][time] float arrays, as in PyTorch (batch of 1, so
 * the masks are all ones and left out). */
#include "pvoice.h"
#include "pv_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LAYERS 8
#define MAX_UPS 6
#define MAX_KERNELS 4
#define MAX_FLOWS 4
#define MAX_WN 8

enum { C_VERSION, C_RATE, C_SPEAKERS, C_VOCAB, C_HIDDEN, C_HEADS, C_LAYERS, C_WINDOW,
	C_INTER, C_SDP, C_UPS, C_KERNELS, C_RESBLOCK, C_FLOW_LAYERS, C_COUNT };

typedef struct {
	int n;
	pv_conv sep[3], pw[3];
	float *g1[3], *b1[3], *g2[3], *b2[3];
} dds;

typedef struct {
	pv_conv q, k, v, o, f1, f2;
	float *rel_k, *rel_v, *g1, *b1, *g2, *b2;
} enc_layer;

typedef struct {
	pv_conv pre, post, cond, in[MAX_WN], rs[MAX_WN];
	int has_cond;
} coupling;

typedef struct {
	pv_conv c[2][3];     /* ResBlock1: convs1, convs2; ResBlock2: convs in c[0] */
	int n;
} resblock;

struct pv_voice {
	unsigned char *file;
	int cfg[C_COUNT];
	float scales[3];
	int *pmap, npmap;          /* [codepoint, id] pairs */
	char espeak[64];
	float *emb, *emb_g;
	int gin;
	enc_layer enc[MAX_LAYERS];
	pv_conv enc_proj;
	/* Duration predictor (stochastic). */
	pv_conv dp_pre, dp_proj, dp_cond;
	dds dp_convs;
	struct { pv_conv pre, proj; dds convs; } dpf[MAX_FLOWS];   /* flows 7, 5, 3 */
	int n_dpf;
	float *dp_m, *dp_enl;
	/* Duration predictor (plain). */
	pv_conv d1, d2, dproj;
	float *dn1g, *dn1b, *dn2g, *dn2b;
	int has_dp_cond;
	coupling fl[MAX_FLOWS];
	int n_flows;
	pv_conv dec_pre, dec_cond, ups[MAX_UPS], dec_post;
	int has_dec_cond;
	resblock res[MAX_UPS * MAX_KERNELS];
	/* Every float array this voice allocated, to free. */
	void **allocs;
	int nallocs, capallocs;
	/* The last synthesis' stages, for the tests. */
	float *d_m, *d_logs, *d_logw, *d_z;
	int d_n, d_frames;
	int bad;
	volatile int *stop;        /* from the options of the synthesis under way */
	char *err;
	size_t err_size;
};

static double default_clock(void)
{
	return (double)clock() / CLOCKS_PER_SEC;
}
double (*pv_clock)(void) = default_clock;

/* ---- Loading --------------------------------------------------------------- */

typedef struct {
	const char *name;
	int nlen, type, ndim;
	unsigned dims[4];
	const unsigned char *data;
	size_t count;
} entry;

static entry *entries;
static int nentries;

static unsigned rd32(const unsigned char *p)
{
	return p[0] | p[1] << 8 | p[2] << 16 | (unsigned)p[3] << 24;
}

static float half_to_float(unsigned h)
{
	unsigned s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
	float f;
	if (e == 0)
		f = ldexpf((float)m, -24);
	else if (e == 31)
		f = m ? NAN : INFINITY;
	else
		f = ldexpf((float)(m | 1024), (int)e - 25);
	return s ? -f : f;
}

static void *track(pv_voice *v, void *p)
{
	if (!p) {
		v->bad = 1;
		return NULL;
	}
	if (v->nallocs == v->capallocs) {
		int cap = v->capallocs ? v->capallocs * 2 : 256;
		void **n = realloc(v->allocs, cap * sizeof *n);
		if (!n) {
			free(p);
			v->bad = 1;
			return NULL;
		}
		v->allocs = n;
		v->capallocs = cap;
	}
	v->allocs[v->nallocs++] = p;
	return p;
}

static const entry *find(const char *name)
{
	int i;
	size_t len = strlen(name);
	for (i = 0; i < nentries; i++)
		if ((size_t)entries[i].nlen == len && !memcmp(entries[i].name, name, len))
			return &entries[i];
	return NULL;
}

static void fail(pv_voice *v, const char *what, const char *name)
{
	if (!v->bad && v->err)
		snprintf(v->err, v->err_size, "%s: %s", what, name);
	v->bad = 1;
}

/* Element i of a float tensor (stored as float32 or float16). */
static float value(const entry *e, size_t i)
{
	if (e->type == 0) {
		unsigned u = rd32(e->data + 4 * i);
		float f;
		memcpy(&f, &u, 4);
		return f;
	}
	return half_to_float(e->data[2 * i] | e->data[2 * i + 1] << 8);
}

/* A float copy of a tensor. */
static float *get_f(pv_voice *v, const char *name, size_t *count, int required)
{
	const entry *e = find(name);
	float *f;
	size_t i;
	if (!e) {
		if (required)
			fail(v, "missing from the voice", name);
		return NULL;
	}
	if (e->type != 0 && e->type != 1) {
		fail(v, "not a float tensor", name);
		return NULL;
	}
	f = track(v, malloc((e->count ? e->count : 1) * sizeof *f));
	if (!f)
		return NULL;
	for (i = 0; i < e->count; i++)
		f[i] = value(e, i);
	if (count)
		*count = e->count;
	return f;
}

/* A convolution: weights packed for pv_conv_run. */
static int get_conv(pv_voice *v, pv_conv *c, const char *path, int required)
{
	char n[160];
	const entry *we, *ae;
	int o, i, k;
	memset(c, 0, sizeof *c);
	snprintf(n, sizeof n, "%s.weight", path);
	we = find(n);
	snprintf(n, sizeof n, "%s.conv", path);
	ae = find(n);
	if (!we || !ae || ae->type != 2 || we->ndim != 3) {
		if (required)
			fail(v, "convolution missing from the voice", path);
		return -1;
	}
	if (we->type != 0 && we->type != 1) {
		fail(v, "not a float tensor", path);
		return -1;
	}
	c->stride = (int)rd32(ae->data);
	c->pad_l = (int)rd32(ae->data + 4);
	c->pad_r = (int)rd32(ae->data + 8);
	c->dil = (int)rd32(ae->data + 12);
	c->groups = (int)rd32(ae->data + 16);
	c->transposed = (int)rd32(ae->data + 20);
	c->k = (int)we->dims[2];
	if (c->transposed) {
		c->in = (int)we->dims[0];
		c->out = (int)we->dims[1];
	} else {
		c->out = (int)we->dims[0];
		c->in = (int)we->dims[1] * c->groups;
	}
	if (!c->transposed && c->groups > 1) {
		if (c->groups != c->in || c->groups != c->out) {
			fail(v, "grouped convolution not supported", path);
			return -1;
		}
		snprintf(n, sizeof n, "%s.weight", path);
		c->w = get_f(v, n, NULL, 1);   /* depthwise: [out][1][k] is already [out][k] */
		if (!c->w)
			return -1;
	} else {
		/* Straight from the file into [k][out][in]. */
		float *p = track(v, malloc((size_t)c->out * c->in * c->k * sizeof *p));
		if (!p)
			return -1;
		for (o = 0; o < c->out; o++)
			for (i = 0; i < c->in; i++)
				for (k = 0; k < c->k; k++)
					p[((size_t)k * c->out + o) * c->in + i] = value(we, c->transposed
						? ((size_t)i * c->out + o) * c->k + k : ((size_t)o * c->in + i) * c->k + k);
		c->w = p;
	}
	snprintf(n, sizeof n, "%s.bias", path);
	c->b = find(n) ? get_f(v, n, NULL, 1) : NULL;
	return 0;
}

static void get_dds(pv_voice *v, dds *d, const char *path)
{
	char n[160];
	int i;
	d->n = 0;
	for (i = 0; i < 3; i++) {
		snprintf(n, sizeof n, "%s.convs_sep.%d", path, i);
		if (get_conv(v, &d->sep[i], n, 0))
			break;
		snprintf(n, sizeof n, "%s.convs_1x1.%d", path, i);
		get_conv(v, &d->pw[i], n, 1);
		snprintf(n, sizeof n, "%s.norms_1.%d.gamma", path, i);
		d->g1[i] = get_f(v, n, NULL, 1);
		snprintf(n, sizeof n, "%s.norms_1.%d.beta", path, i);
		d->b1[i] = get_f(v, n, NULL, 1);
		snprintf(n, sizeof n, "%s.norms_2.%d.gamma", path, i);
		d->g2[i] = get_f(v, n, NULL, 1);
		snprintf(n, sizeof n, "%s.norms_2.%d.beta", path, i);
		d->b2[i] = get_f(v, n, NULL, 1);
		d->n++;
	}
}

static int bind(pv_voice *v)
{
	char n[160];
	int i, j, l;
	const entry *e;
	if (!(e = find("config")) || e->type != 2 || e->count < C_COUNT) {
		fail(v, "not a voice file (no config)", "config");
		return -1;
	}
	for (i = 0; i < C_COUNT; i++)
		v->cfg[i] = (int)rd32(e->data + 4 * i);
	if (v->cfg[C_VERSION] != 1 || v->cfg[C_LAYERS] > MAX_LAYERS || v->cfg[C_UPS] > MAX_UPS ||
			v->cfg[C_KERNELS] > MAX_KERNELS || v->cfg[C_FLOW_LAYERS] > MAX_WN ||
			v->cfg[C_HIDDEN] % v->cfg[C_HEADS]) {
		fail(v, "unsupported voice", "config");
		return -1;
	}
	if ((e = find("scales")) && e->type == 0 && e->count >= 3)
		for (i = 0; i < 3; i++) {
			unsigned u = rd32(e->data + 4 * i);
			memcpy(&v->scales[i], &u, 4);
		}
	if ((e = find("phoneme_map")) && e->type == 2) {
		v->npmap = (int)(e->count / 2);
		v->pmap = track(v, malloc((e->count + 1) * sizeof(int)));
		if (v->pmap)
			for (i = 0; i < (int)e->count; i++)
				v->pmap[i] = (int)rd32(e->data + 4 * i);
	}
	if ((e = find("espeak_voice")) && e->type == 3) {
		size_t len = e->count < sizeof v->espeak - 1 ? e->count : sizeof v->espeak - 1;
		memcpy(v->espeak, e->data, len);
		v->espeak[len] = 0;
	}

	v->emb = get_f(v, "enc_p.emb.weight", NULL, 1);
	if (v->cfg[C_SPEAKERS] > 1) {
		size_t cnt;
		v->emb_g = get_f(v, "emb_g.weight", &cnt, 1);
		v->gin = v->emb_g ? (int)(cnt / v->cfg[C_SPEAKERS]) : 0;
	}
	for (l = 0; l < v->cfg[C_LAYERS]; l++) {
		enc_layer *L = &v->enc[l];
		snprintf(n, sizeof n, "enc_p.encoder.attn_layers.%d.conv_q", l); get_conv(v, &L->q, n, 1);
		snprintf(n, sizeof n, "enc_p.encoder.attn_layers.%d.conv_k", l); get_conv(v, &L->k, n, 1);
		snprintf(n, sizeof n, "enc_p.encoder.attn_layers.%d.conv_v", l); get_conv(v, &L->v, n, 1);
		snprintf(n, sizeof n, "enc_p.encoder.attn_layers.%d.conv_o", l); get_conv(v, &L->o, n, 1);
		snprintf(n, sizeof n, "enc_p.encoder.attn_layers.%d.emb_rel_k", l); L->rel_k = get_f(v, n, NULL, 1);
		snprintf(n, sizeof n, "enc_p.encoder.attn_layers.%d.emb_rel_v", l); L->rel_v = get_f(v, n, NULL, 1);
		snprintf(n, sizeof n, "enc_p.encoder.norm_layers_1.%d.gamma", l); L->g1 = get_f(v, n, NULL, 1);
		snprintf(n, sizeof n, "enc_p.encoder.norm_layers_1.%d.beta", l); L->b1 = get_f(v, n, NULL, 1);
		snprintf(n, sizeof n, "enc_p.encoder.norm_layers_2.%d.gamma", l); L->g2 = get_f(v, n, NULL, 1);
		snprintf(n, sizeof n, "enc_p.encoder.norm_layers_2.%d.beta", l); L->b2 = get_f(v, n, NULL, 1);
		snprintf(n, sizeof n, "enc_p.encoder.ffn_layers.%d.conv_1", l); get_conv(v, &L->f1, n, 1);
		snprintf(n, sizeof n, "enc_p.encoder.ffn_layers.%d.conv_2", l); get_conv(v, &L->f2, n, 1);
		/* The FFN pads with F.pad ("same"), not in its convolutions. */
		L->f1.pad_l = (L->f1.k - 1) / 2; L->f1.pad_r = L->f1.k / 2;
		L->f2.pad_l = (L->f2.k - 1) / 2; L->f2.pad_r = L->f2.k / 2;
	}
	get_conv(v, &v->enc_proj, "enc_p.proj", 1);

	v->has_dp_cond = get_conv(v, &v->dp_cond, "dp.cond", 0) == 0;
	if (v->cfg[C_SDP]) {
		/* Run backwards: Flip, flows.7, Flip, flows.5, Flip, flows.3, Flip,
		 * flows.0 (flows.1 is the "useless vflow" Piper drops). */
		static const int order[] = { 7, 5, 3 };
		get_conv(v, &v->dp_pre, "dp.pre", 1);
		get_conv(v, &v->dp_proj, "dp.proj", 1);
		get_dds(v, &v->dp_convs, "dp.convs");
		for (j = 0; j < 3; j++) {
			snprintf(n, sizeof n, "dp.flows.%d.pre", order[j]); get_conv(v, &v->dpf[j].pre, n, 1);
			snprintf(n, sizeof n, "dp.flows.%d.proj", order[j]); get_conv(v, &v->dpf[j].proj, n, 1);
			snprintf(n, sizeof n, "dp.flows.%d.convs", order[j]); get_dds(v, &v->dpf[j].convs, n);
		}
		v->n_dpf = 3;
		v->dp_m = get_f(v, "dp.flows.0.m", NULL, 1);
		v->dp_enl = get_f(v, "dp.flows.0.exp_neg_logs", NULL, 1);
	} else {
		get_conv(v, &v->d1, "dp.conv_1", 1);
		get_conv(v, &v->d2, "dp.conv_2", 1);
		get_conv(v, &v->dproj, "dp.proj", 1);
		v->dn1g = get_f(v, "dp.norm_1.gamma", NULL, 1);
		v->dn1b = get_f(v, "dp.norm_1.beta", NULL, 1);
		v->dn2g = get_f(v, "dp.norm_2.gamma", NULL, 1);
		v->dn2b = get_f(v, "dp.norm_2.beta", NULL, 1);
	}

	/* The flow: coupling layers 0, 2, 4, 6 (the others are Flips). */
	v->n_flows = 0;
	for (j = 0; j < MAX_FLOWS; j++) {
		coupling *c = &v->fl[j];
		snprintf(n, sizeof n, "flow.flows.%d.pre", 2 * j);
		if (get_conv(v, &c->pre, n, 0))
			break;
		snprintf(n, sizeof n, "flow.flows.%d.post", 2 * j); get_conv(v, &c->post, n, 1);
		snprintf(n, sizeof n, "flow.flows.%d.enc.cond_layer", 2 * j);
		c->has_cond = get_conv(v, &c->cond, n, 0) == 0;
		for (l = 0; l < v->cfg[C_FLOW_LAYERS]; l++) {
			snprintf(n, sizeof n, "flow.flows.%d.enc.in_layers.%d", 2 * j, l); get_conv(v, &c->in[l], n, 1);
			snprintf(n, sizeof n, "flow.flows.%d.enc.res_skip_layers.%d", 2 * j, l); get_conv(v, &c->rs[l], n, 1);
		}
		v->n_flows++;
	}

	get_conv(v, &v->dec_pre, "dec.conv_pre", 1);
	v->has_dec_cond = get_conv(v, &v->dec_cond, "dec.cond", 0) == 0;
	get_conv(v, &v->dec_post, "dec.conv_post", 1);
	for (i = 0; i < v->cfg[C_UPS]; i++) {
		snprintf(n, sizeof n, "dec.ups.%d", i);
		get_conv(v, &v->ups[i], n, 1);
	}
	for (i = 0; i < v->cfg[C_UPS] * v->cfg[C_KERNELS]; i++) {
		resblock *r = &v->res[i];
		r->n = 0;
		for (j = 0; j < 3; j++) {
			if (v->cfg[C_RESBLOCK] == 1) {
				snprintf(n, sizeof n, "dec.resblocks.%d.convs1.%d", i, j);
				if (get_conv(v, &r->c[0][j], n, 0))
					break;
				snprintf(n, sizeof n, "dec.resblocks.%d.convs2.%d", i, j);
				get_conv(v, &r->c[1][j], n, 1);
			} else {
				snprintf(n, sizeof n, "dec.resblocks.%d.convs.%d", i, j);
				if (get_conv(v, &r->c[0][j], n, 0))
					break;
			}
			r->n++;
		}
		if (!r->n)
			fail(v, "decoder block missing", "dec.resblocks");
	}
	return v->bad ? -1 : 0;
}

pv_voice *pv_load(const char *path, char *err, size_t err_size)
{
	FILE *f = fopen(path, "rb");
	pv_voice *v;
	long size;
	unsigned count, i;
	const unsigned char *p, *end;
	if (err && err_size)
		err[0] = 0;
	if (!f) {
		if (err)
			snprintf(err, err_size, "can't open %s", path);
		return NULL;
	}
	v = calloc(1, sizeof *v);
	if (!v) {
		fclose(f);
		return NULL;
	}
	v->err = err;
	v->err_size = err_size;
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	v->file = size > 12 ? malloc((size_t)size) : NULL;
	if (!v->file || fread(v->file, 1, (size_t)size, f) != (size_t)size || memcmp(v->file, "PVOI", 4)) {
		fclose(f);
		if (err)
			snprintf(err, err_size, "%s isn't a voice file", path);
		free(v->file);
		free(v);
		return NULL;
	}
	fclose(f);
	count = rd32(v->file + 8);
	entries = calloc(count ? count : 1, sizeof *entries);
	nentries = 0;
	p = v->file + 12;
	end = v->file + size;
	for (i = 0; entries && i < count && p + 4 <= end; i++) {
		entry *e = &entries[nentries];
		unsigned len = p[0] | p[1] << 8, d;
		size_t bytes, elem;
		e->nlen = (int)len;
		e->name = (const char *)p + 2;
		p += 2 + len;
		if (p + 2 > end)
			break;
		e->type = p[0];
		e->ndim = p[1];
		p += 2;
		if (e->ndim > 4 || p + 4 * e->ndim > end)
			break;
		e->count = 1;
		for (d = 0; d < (unsigned)e->ndim; d++) {
			e->dims[d] = rd32(p + 4 * d);
			e->count *= e->dims[d];
		}
		if (e->ndim == 0)
			e->count = 1;
		p += 4 * e->ndim;
		elem = e->type == 1 ? 2 : e->type == 3 ? 1 : 4;
		bytes = e->count * elem;
		e->data = p;
		p += (bytes + 3) & ~(size_t)3;
		if (p > end)
			break;
		nentries++;
	}
	if (!entries || nentries != (int)count || bind(v)) {
		if (err && !err[0])
			snprintf(err, err_size, "%s is damaged", path);
		free(entries);
		entries = NULL;
		pv_free(v);
		return NULL;
	}
	free(entries);
	entries = NULL;
	/* Everything is copied out; the file itself isn't needed now. */
	free(v->file);
	v->file = NULL;
	v->err = NULL;
	return v;
}

void pv_free(pv_voice *v)
{
	int i;
	if (!v)
		return;
	for (i = 0; i < v->nallocs; i++)
		free(v->allocs[i]);
	free(v->allocs);
	free(v->file);
	free(v->d_m);
	free(v->d_logs);
	free(v->d_logw);
	free(v->d_z);
	free(v);
}

int pv_sample_rate(const pv_voice *v) { return v->cfg[C_RATE]; }
const char *pv_espeak_voice(const pv_voice *v) { return v->espeak; }

void pv_default_options(const pv_voice *v, pv_options *o)
{
	o->noise_scale = v->scales[0];
	o->length_scale = v->scales[1] > 0 ? v->scales[1] : 1.0f;
	o->noise_w = v->scales[2];
	o->speaker = 0;
	o->seed = 0;
	o->stop = NULL;
}

static int stopped(const pv_voice *v)
{
	return v->stop && *v->stop;
}

static int id_of(const pv_voice *v, uint32_t cp)
{
	int i;
	for (i = 0; i < v->npmap; i++)
		if ((uint32_t)v->pmap[2 * i] == cp)
			return v->pmap[2 * i + 1];
	return -1;
}

int pv_phonemes_to_ids(const pv_voice *v, const uint32_t *ph, int n, int *ids, int max)
{
	int i, k = 0, pad = id_of(v, '_'), bos = id_of(v, '^'), eos = id_of(v, '$');
	if (pad < 0) pad = 0;
	if (bos < 0) bos = 1;
	if (eos < 0) eos = 2;
#define PUT(x) do { if (k < max) ids[k++] = (x); } while (0)
	PUT(bos);
	PUT(pad);
	for (i = 0; i < n; i++) {
		int j;
		for (j = 0; j < v->npmap; j++)
			if ((uint32_t)v->pmap[2 * j] == ph[i]) {
				PUT(v->pmap[2 * j + 1]);
				PUT(pad);
			}
	}
	PUT(eos);
#undef PUT
	return k;
}

void pv_last_debug(const pv_voice *v, pv_debug *d)
{
	d->enc_m = v->d_m;
	d->enc_logs = v->d_logs;
	d->logw = v->d_logw;
	d->z = v->d_z;
	d->n = v->d_n;
	d->frames = v->d_frames;
}

/* ---- Random numbers ------------------------------------------------------- */

static uint32_t rng_next(uint32_t *s)
{
	uint32_t x = *s;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return *s = x ? x : 0x9e3779b9u;
}

static float gaussian(uint32_t *s)
{
	float u1 = ((rng_next(s) >> 8) + 1) * (1.0f / 16777217.0f);
	float u2 = (rng_next(s) >> 8) * (1.0f / 16777216.0f);
	return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ---- Pieces of the model --------------------------------------------------- */

static float *alloc_f(size_t n)
{
	return malloc((n ? n : 1) * sizeof(float));
}

/* y = conv(x) into a new buffer; *tout = its length. */
static float *conv_new(const pv_conv *c, const float *x, int t, int *tout)
{
	int to = pv_conv_out_len(c, t);
	float *y = alloc_f((size_t)c->out * to);
	if (y)
		pv_conv_run(c, x, t, t, y, to);
	if (tout)
		*tout = to;
	return y;
}

/* Adds a [ch] vector (a 1x1 convolution of the speaker embedding) to every
 * time step. */
static void add_cond(const pv_conv *c, const float *g, float *x, int t)
{
	float *cv = conv_new(c, g, 1, NULL);
	int o, i;
	if (!cv)
		return;
	for (o = 0; o < c->out; o++)
		for (i = 0; i < t; i++)
			x[(size_t)o * t + i] += cv[o];
	free(cv);
}

/* Multi-head self-attention with relative positions (attentions.py). */
static float *attention(const pv_voice *v, const enc_layer *L, const float *x, int t)
{
	int h = v->cfg[C_HIDDEN], nh = v->cfg[C_HEADS], kc = h / nh, w = v->cfg[C_WINDOW];
	float scale = 1.0f / sqrtf((float)kc);
	float *q = conv_new(&L->q, x, t, NULL), *k = conv_new(&L->k, x, t, NULL), *vv = conv_new(&L->v, x, t, NULL);
	float *qt = alloc_f((size_t)t * kc), *vt = alloc_f((size_t)t * kc), *o = alloc_f((size_t)t * kc);
	float *sc = alloc_f((size_t)t * t), *out = alloc_f((size_t)h * t), *y = NULL;
	int hd, i, s, d, r;
	if (!q || !k || !vv || !qt || !vt || !o || !sc || !out)
		goto done;
	for (hd = 0; hd < nh; hd++) {
		for (d = 0; d < kc; d++)
			for (i = 0; i < t; i++) {
				qt[i * kc + d] = q[(size_t)(hd * kc + d) * t + i] * scale;
				vt[i * kc + d] = vv[(size_t)(hd * kc + d) * t + i];
			}
		memset(sc, 0, (size_t)t * t * sizeof *sc);
		pv_gemm(t, t, kc, qt, kc, k + (size_t)hd * kc * t, t, sc, t);
		/* Relative keys: position s - t within the window. */
		for (i = 0; i < t; i++)
			for (r = -w; r <= w; r++) {
				const float *e = L->rel_k + (size_t)(r + w) * kc;
				float acc = 0;
				s = i + r;
				if (s < 0 || s >= t)
					continue;
				for (d = 0; d < kc; d++)
					acc += qt[i * kc + d] * e[d];
				sc[(size_t)i * t + s] += acc;
			}
		for (i = 0; i < t; i++) {
			float *row = sc + (size_t)i * t, mx = row[0], sum = 0;
			for (s = 1; s < t; s++)
				if (row[s] > mx)
					mx = row[s];
			for (s = 0; s < t; s++)
				sum += row[s] = expf(row[s] - mx);
			for (s = 0; s < t; s++)
				row[s] /= sum;
		}
		memset(o, 0, (size_t)t * kc * sizeof *o);
		pv_gemm(t, kc, t, sc, t, vt, kc, o, kc);
		for (i = 0; i < t; i++)
			for (r = -w; r <= w; r++) {
				const float *e = L->rel_v + (size_t)(r + w) * kc;
				float p;
				s = i + r;
				if (s < 0 || s >= t)
					continue;
				p = sc[(size_t)i * t + s];
				for (d = 0; d < kc; d++)
					o[i * kc + d] += p * e[d];
			}
		for (d = 0; d < kc; d++)
			for (i = 0; i < t; i++)
				out[(size_t)(hd * kc + d) * t + i] = o[i * kc + d];
	}
	y = conv_new(&L->o, out, t, NULL);
done:
	free(q); free(k); free(vv); free(qt); free(vt); free(o); free(sc); free(out);
	return y;
}

/* Text encoder: x (hidden x t), m and logs (inter x t each). */
static int encode(pv_voice *v, const int *ids, int t, float **x_out, float **m, float **logs)
{
	int h = v->cfg[C_HIDDEN], l, c, i, inter = v->cfg[C_INTER];
	float sq = sqrtf((float)h);
	float *x = alloc_f((size_t)h * t), *stats;
	if (!x)
		return -1;
	for (i = 0; i < t; i++) {
		int id = ids[i] >= 0 && ids[i] < v->cfg[C_VOCAB] ? ids[i] : 0;
		for (c = 0; c < h; c++)
			x[(size_t)c * t + i] = v->emb[(size_t)id * h + c] * sq;
	}
	for (l = 0; l < v->cfg[C_LAYERS]; l++) {
		const enc_layer *L = &v->enc[l];
		float *y = stopped(v) ? NULL : attention(v, L, x, t), *f;
		size_t n = (size_t)h * t, j;
		if (!y) {
			free(x);
			return -1;
		}
		for (j = 0; j < n; j++)
			x[j] += y[j];
		free(y);
		pv_layer_norm(x, h, t, t, L->g1, L->b1);
		f = conv_new(&L->f1, x, t, NULL);
		if (!f) {
			free(x);
			return -1;
		}
		pv_relu(f, (size_t)L->f1.out * t);
		y = conv_new(&L->f2, f, t, NULL);
		free(f);
		if (!y) {
			free(x);
			return -1;
		}
		for (j = 0; j < n; j++)
			x[j] += y[j];
		free(y);
		pv_layer_norm(x, h, t, t, L->g2, L->b2);
	}
	stats = conv_new(&v->enc_proj, x, t, NULL);
	if (!stats) {
		free(x);
		return -1;
	}
	*m = alloc_f((size_t)inter * t);
	*logs = alloc_f((size_t)inter * t);
	if (!*m || !*logs) {
		free(*m); free(*logs); free(stats); free(x);
		return -1;
	}
	memcpy(*m, stats, (size_t)inter * t * sizeof(float));
	memcpy(*logs, stats + (size_t)inter * t, (size_t)inter * t * sizeof(float));
	free(stats);
	*x_out = x;
	return 0;
}

/* Dilated depth-separable convolutions (modules.DDSConv), in place. */
static void dds_run(const dds *d, float *x, int t)
{
	int i, ch = d->sep[0].out;
	size_t n = (size_t)ch * t, j;
	for (i = 0; i < d->n; i++) {
		float *y = conv_new(&d->sep[i], x, t, NULL), *z;
		if (!y)
			return;
		pv_layer_norm(y, ch, t, t, d->g1[i], d->b1[i]);
		pv_gelu(y, n);
		z = conv_new(&d->pw[i], y, t, NULL);
		free(y);
		if (!z)
			return;
		pv_layer_norm(z, ch, t, t, d->g2[i], d->b2[i]);
		pv_gelu(z, n);
		for (j = 0; j < n; j++)
			x[j] += z[j];
		free(z);
	}
}

/* Inverse rational quadratic spline with linear tails (transforms.py), for
 * one value: w, h (bins) and d (bins - 1) unnormalised. */
static float spline_inverse(float x, const float *uw, const float *uh, const float *ud, int bins,
		float bound, float norm)
{
	float wid[16], hei[16], cw[17], ch[17], der[17];
	float mx, sum, min_b = 1e-3f, min_d = 1e-3f;
	float input_cw, input_bw, input_ch, input_h, delta, d0, d1, a, b, c, disc, root;
	int i, bin;
	if (x < -bound || x > bound || bins > 16)
		return x;
	/* Widths and heights: softmax, then a minimum per bin. */
	mx = uw[0] / norm;
	for (i = 1; i < bins; i++)
		if (uw[i] / norm > mx)
			mx = uw[i] / norm;
	for (sum = 0, i = 0; i < bins; i++)
		sum += wid[i] = expf(uw[i] / norm - mx);
	cw[0] = 0;
	for (i = 0; i < bins; i++)
		cw[i + 1] = cw[i] + (min_b + (1 - min_b * bins) * wid[i] / sum);
	mx = uh[0] / norm;
	for (i = 1; i < bins; i++)
		if (uh[i] / norm > mx)
			mx = uh[i] / norm;
	for (sum = 0, i = 0; i < bins; i++)
		sum += hei[i] = expf(uh[i] / norm - mx);
	ch[0] = 0;
	for (i = 0; i < bins; i++)
		ch[i + 1] = ch[i] + (min_b + (1 - min_b * bins) * hei[i] / sum);
	for (i = 0; i <= bins; i++) {
		cw[i] = 2 * bound * cw[i] - bound;
		ch[i] = 2 * bound * ch[i] - bound;
	}
	cw[0] = ch[0] = -bound;
	cw[bins] = ch[bins] = bound;
	/* Derivatives: the ends fixed so the tails are linear (slope 1). */
	{
		float constant = logf(expf(1 - min_d) - 1);
		for (i = 0; i <= bins; i++) {
			float u = (i == 0 || i == bins) ? constant : ud[i - 1];
			der[i] = min_d + (u > 20 ? u : log1pf(expf(u)));
		}
	}
	/* Which bin (the last edge nudged up, as searchsorted does). */
	for (bin = -1, i = 0; i <= bins; i++)
		if (x >= ch[i] + (i == bins ? 1e-6f : 0))
			bin++;
	if (bin < 0) bin = 0;
	if (bin >= bins) bin = bins - 1;
	input_cw = cw[bin];
	input_bw = cw[bin + 1] - cw[bin];
	input_ch = ch[bin];
	input_h = ch[bin + 1] - ch[bin];
	delta = input_h / input_bw;
	d0 = der[bin];
	d1 = der[bin + 1];
	a = (x - input_ch) * (d0 + d1 - 2 * delta) + input_h * (delta - d0);
	b = input_h * d0 - (x - input_ch) * (d0 + d1 - 2 * delta);
	c = -delta * (x - input_ch);
	disc = b * b - 4 * a * c;
	if (disc < 0)
		disc = 0;
	root = (2 * c) / (-b - sqrtf(disc));
	return root * input_bw + input_cw;
}

/* Stochastic duration predictor, backwards from noise: log durations. */
static float *durations_sdp(pv_voice *v, const float *xenc, int t, const float *g, float noise_w,
		uint32_t *seed)
{
	int fch = v->dp_pre.out, i, j, bins;
	float *xd = conv_new(&v->dp_pre, xenc, t, NULL), *xp, *z0, *z1, *logw;
	float norm = sqrtf((float)fch);
	if (!xd)
		return NULL;
	if (g && v->has_dp_cond)
		add_cond(&v->dp_cond, g, xd, t);
	dds_run(&v->dp_convs, xd, t);
	xp = conv_new(&v->dp_proj, xd, t, NULL);
	free(xd);
	z0 = alloc_f(t);
	z1 = alloc_f(t);
	if (!xp || !z0 || !z1) {
		free(xp); free(z0); free(z1);
		return NULL;
	}
	/* torch.randn(b, 2, t): channel 0 then channel 1. */
	for (i = 0; i < t; i++)
		z0[i] = gaussian(seed) * noise_w;
	for (i = 0; i < t; i++)
		z1[i] = gaussian(seed) * noise_w;
	for (j = 0; j < v->n_dpf; j++) {
		/* Flip, then the ConvFlow backwards. */
		float *tmp = z0, *h, *p;
		z0 = z1;
		z1 = tmp;
		h = conv_new(&v->dpf[j].pre, z0, t, NULL);
		if (!h)
			break;
		for (i = 0; i < fch * t; i++)
			h[i] += xp[i];
		dds_run(&v->dpf[j].convs, h, t);
		p = conv_new(&v->dpf[j].proj, h, t, NULL);
		free(h);
		if (!p)
			break;
		bins = (v->dpf[j].proj.out + 1) / 3;
		for (i = 0; i < t; i++) {
			float uw[16], uh[16], ud[16];
			int k;
			for (k = 0; k < bins && k < 16; k++) {
				uw[k] = p[(size_t)k * t + i];
				uh[k] = p[(size_t)(bins + k) * t + i];
			}
			for (k = 0; k < bins - 1 && k < 15; k++)
				ud[k] = p[(size_t)(2 * bins + k) * t + i];
			z1[i] = spline_inverse(z1[i], uw, uh, ud, bins, 5.0f, norm);
		}
		free(p);
	}
	/* The last Flip, then the elementwise affine flow backwards. */
	{
		float *tmp = z0;
		z0 = z1;
		z1 = tmp;
	}
	for (i = 0; i < t; i++)
		z0[i] = (z0[i] - v->dp_m[0]) * v->dp_enl[0];
	free(z1);
	free(xp);
	logw = z0;
	return logw;
}

/* The plain duration predictor (models.DurationPredictor). */
static float *durations_plain(pv_voice *v, const float *xenc, int t, const float *g)
{
	int h = v->cfg[C_HIDDEN];
	float *x = alloc_f((size_t)h * t), *a, *b, *p;
	if (!x)
		return NULL;
	memcpy(x, xenc, (size_t)h * t * sizeof *x);
	if (g && v->has_dp_cond)
		add_cond(&v->dp_cond, g, x, t);
	a = conv_new(&v->d1, x, t, NULL);
	free(x);
	if (!a)
		return NULL;
	pv_relu(a, (size_t)v->d1.out * t);
	pv_layer_norm(a, v->d1.out, t, t, v->dn1g, v->dn1b);
	b = conv_new(&v->d2, a, t, NULL);
	free(a);
	if (!b)
		return NULL;
	pv_relu(b, (size_t)v->d2.out * t);
	pv_layer_norm(b, v->d2.out, t, t, v->dn2g, v->dn2b);
	p = conv_new(&v->dproj, b, t, NULL);
	free(b);
	return p;
}

/* The flow backwards: z (inter x f) in place. */
static void flow_reverse(pv_voice *v, float *z, int f, const float *g)
{
	int inter = v->cfg[C_INTER], half = inter / 2, j, l, c, i;
	int nl = v->cfg[C_FLOW_LAYERS];
	float *row = alloc_f(f);
	if (!row)
		return;
	for (j = v->n_flows - 1; j >= 0 && !stopped(v); j--) {
		const coupling *cp = &v->fl[j];
		int hid = cp->pre.out;
		float *h, *out, *gc = NULL, *m;
		/* Flip: reverse the channels. */
		for (c = 0; c < inter / 2; c++) {
			memcpy(row, z + (size_t)c * f, f * sizeof *row);
			memcpy(z + (size_t)c * f, z + (size_t)(inter - 1 - c) * f, f * sizeof *row);
			memcpy(z + (size_t)(inter - 1 - c) * f, row, f * sizeof *row);
		}
		h = conv_new(&cp->pre, z, f, NULL);        /* the first half's channels */
		out = calloc((size_t)hid * f, sizeof *out);
		if (!h || !out) {
			free(h);
			free(out);
			break;
		}
		if (g && cp->has_cond)
			gc = conv_new(&cp->cond, g, 1, NULL);
		for (l = 0; l < nl; l++) {
			float *xi = conv_new(&cp->in[l], h, f, NULL), *rs;
			if (!xi)
				break;
			if (gc)
				for (c = 0; c < 2 * hid; c++)
					for (i = 0; i < f; i++)
						xi[(size_t)c * f + i] += gc[l * 2 * hid + c];
			/* tanh(first half) * sigmoid(second half) */
			for (c = 0; c < hid; c++)
				for (i = 0; i < f; i++) {
					float a = xi[(size_t)c * f + i], b = xi[(size_t)(hid + c) * f + i];
					xi[(size_t)c * f + i] = tanhf(a) / (1.0f + expf(-b));
				}
			rs = conv_new(&cp->rs[l], xi, f, NULL);
			free(xi);
			if (!rs)
				break;
			if (l < nl - 1) {
				for (i = 0; i < hid * f; i++) {
					h[i] += rs[i];
					out[i] += rs[(size_t)hid * f + i];
				}
			} else {
				for (i = 0; i < hid * f; i++)
					out[i] += rs[i];
			}
			free(rs);
		}
		free(gc);
		free(h);
		m = conv_new(&cp->post, out, f, NULL);
		free(out);
		if (!m)
			break;
		for (i = 0; i < half * f; i++)
			z[(size_t)half * f + i] -= m[i];
		free(m);
	}
	free(row);
}

/* HiFi-GAN generator: z (inter x f) to audio. */
static float *decode(pv_voice *v, const float *z, int f, const float *g, int *n_out)
{
	int t = f, tt, i, j, nk = v->cfg[C_KERNELS];
	float *x = conv_new(&v->dec_pre, z, f, &t), *y;
	if (!x)
		return NULL;
	if (g && v->has_dec_cond)
		add_cond(&v->dec_cond, g, x, t);
	for (i = 0; i < v->cfg[C_UPS]; i++) {
		const pv_conv *up = &v->ups[i];
		float *xs = NULL;
		size_t n;
		pv_leaky_relu(x, (size_t)up->in * t, 0.1f);
		y = conv_new(up, x, t, &tt);
		free(x);
		if (!y)
			return NULL;
		x = y;
		t = tt;
		n = (size_t)up->out * t;
		for (j = 0; j < nk && !stopped(v); j++) {
			const resblock *r = &v->res[i * nk + j];
			float *a = alloc_f(n);
			int q;
			if (!a)
				break;
			memcpy(a, x, n * sizeof *a);
			for (q = 0; q < r->n; q++) {
				float *xt = alloc_f(n), *u;
				size_t s;
				if (!xt)
					break;
				memcpy(xt, a, n * sizeof *xt);
				pv_leaky_relu(xt, n, 0.1f);
				u = conv_new(&r->c[0][q], xt, t, NULL);
				free(xt);
				if (!u)
					break;
				if (v->cfg[C_RESBLOCK] == 1) {
					float *u2;
					pv_leaky_relu(u, n, 0.1f);
					u2 = conv_new(&r->c[1][q], u, t, NULL);
					free(u);
					if (!u2)
						break;
					u = u2;
				}
				for (s = 0; s < n; s++)
					a[s] += u[s];
				free(u);
			}
			if (!xs) {
				xs = a;
			} else {
				size_t s;
				for (s = 0; s < n; s++)
					xs[s] += a[s];
				free(a);
			}
		}
		free(x);
		if (!xs || stopped(v)) {
			free(xs);
			return NULL;
		}
		for (n = (size_t)up->out * t; n > 0; n--)
			xs[n - 1] /= nk;
		x = xs;
	}
	pv_leaky_relu(x, (size_t)v->dec_post.in * t, 0.01f);
	y = conv_new(&v->dec_post, x, t, &tt);
	free(x);
	if (!y)
		return NULL;
	for (i = 0; i < tt; i++)
		y[i] = tanhf(y[i]);
	*n_out = tt;
	return y;
}

/* ---- Synthesis ---------------------------------------------------------------- */

float *pv_synthesize(pv_voice *v, const int *ids, int t, const pv_options *opt, int *n_samples,
		pv_times *times)
{
	pv_options o;
	float *xenc = NULL, *m = NULL, *logs = NULL, *logw = NULL, *z = NULL, *audio = NULL;
	float g[1024], *gp = NULL;
	int inter = v->cfg[C_INTER], i, c, f, *wc = NULL;
	uint32_t seed;
	double t0, t1;
	*n_samples = 0;
	if (t <= 0)
		return NULL;
	if (opt)
		o = *opt;
	else
		pv_default_options(v, &o);
	seed = o.seed ? o.seed : (uint32_t)time(NULL) * 2654435761u + 1;
	v->stop = o.stop;
	if (v->emb_g && v->gin <= 1024) {
		int s = o.speaker >= 0 && o.speaker < v->cfg[C_SPEAKERS] ? o.speaker : 0;
		memcpy(g, v->emb_g + (size_t)s * v->gin, v->gin * sizeof *g);
		gp = g;
	}
	t0 = pv_clock();
	if (encode(v, ids, t, &xenc, &m, &logs))
		goto done;
	t1 = pv_clock();
	if (times) times->enc = t1 - t0;
	t0 = t1;

	logw = v->cfg[C_SDP] ? durations_sdp(v, xenc, t, gp, o.noise_w, &seed) : durations_plain(v, xenc, t, gp);
	if (!logw)
		goto done;
	wc = malloc(t * sizeof *wc);
	if (!wc)
		goto done;
	for (f = 0, i = 0; i < t; i++) {
		wc[i] = (int)ceilf(expf(logw[i]) * o.length_scale);
		if (wc[i] < 0)
			wc[i] = 0;
		f += wc[i];
	}
	if (f < 1)
		f = 1;
	t1 = pv_clock();
	if (times) times->dp = t1 - t0;
	t0 = t1;

	/* Each phoneme's mean and spread over its frames, plus noise. */
	z = alloc_f((size_t)inter * f);
	if (!z)
		goto done;
	{
		int fr = 0;
		for (i = 0; i < t && fr < f; i++) {
			int k;
			for (k = 0; k < wc[i] && fr < f; k++, fr++)
				for (c = 0; c < inter; c++)
					z[(size_t)c * f + fr] = m[(size_t)c * t + i];
		}
		for (; fr < f; fr++)
			for (c = 0; c < inter; c++)
				z[(size_t)c * f + fr] = 0;
		fr = 0;
		/* randn_like(m_p) is [inter][frames]: channel by channel. */
		for (c = 0; c < inter; c++) {
			fr = 0;
			for (i = 0; i < t; i++) {
				int k;
				float e = expf(logs[(size_t)c * t + i]) * o.noise_scale;
				for (k = 0; k < wc[i] && fr < f; k++, fr++)
					z[(size_t)c * f + fr] += gaussian(&seed) * e;
			}
		}
	}
	flow_reverse(v, z, f, gp);
	if (stopped(v))
		goto done;
	t1 = pv_clock();
	if (times) times->flow = t1 - t0;
	t0 = t1;

	audio = decode(v, z, f, gp, n_samples);
	if (times) times->dec = pv_clock() - t0;

	/* Keep the stages for the tests. */
	free(v->d_m); free(v->d_logs); free(v->d_logw); free(v->d_z);
	v->d_m = m; v->d_logs = logs; v->d_logw = logw; v->d_z = z;
	m = logs = logw = z = NULL;
	v->d_n = t;
	v->d_frames = f;
done:
	free(xenc); free(m); free(logs); free(logw); free(z); free(wc);
	return audio;
}
