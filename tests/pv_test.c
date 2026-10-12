/* SPDX-License-Identifier: MIT
 * Copyright (C) 2026 Andrew Youll */
/* Checks the engine against ONNX Runtime's outputs (tests/make_ref.py), with
 * the noise scales at 0 so both are deterministic:
 *   pv_test VOICE REFDIR [out.wav]
 * Each stage must match to within a small tolerance (relative to its size). */
#include "pvoice.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
static double tol_scale = 1;

static float *load_f32(const char *dir, const char *name, size_t *n)
{
	char path[512];
	FILE *f;
	long size;
	float *p;
	snprintf(path, sizeof path, "%s/%s.f32", dir, name);
	f = fopen(path, "rb");
	if (!f)
		return NULL;
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	p = malloc(size);
	if (p && fread(p, 1, size, f) != (size_t)size) {
		free(p);
		p = NULL;
	}
	fclose(f);
	*n = size / 4;
	return p;
}

/* Max absolute difference, against the reference's largest value. */
static void compare(const char *stage, const float *got, size_t ngot, const float *want, size_t nwant,
		double tol)
{
	size_t i, n = ngot < nwant ? ngot : nwant;
	double maxd = 0, maxv = 1e-9;
	checks++;
	for (i = 0; i < n; i++) {
		double d = fabs((double)got[i] - want[i]);
		if (d > maxd)
			maxd = d;
		if (fabs(want[i]) > maxv)
			maxv = fabs(want[i]);
	}
	printf("%-9s %7zu values (want %7zu): max diff %.3g of %.3g (%.2g%%)\n", stage, ngot, nwant,
		maxd, maxv, 100 * maxd / maxv);
	if (ngot != nwant || maxd / maxv > tol * tol_scale) {
		fails++;
		printf("FAIL %s\n", stage);
	}
}

static void write_wav(const char *path, const float *a, int n, int rate)
{
	FILE *f = fopen(path, "wb");
	int i, bytes = n * 2;
	unsigned char h[44];
	float peak = 0.01f;
	if (!f)
		return;
	for (i = 0; i < n; i++)
		if (fabsf(a[i]) > peak)
			peak = fabsf(a[i]);
	memcpy(h, "RIFF", 4);
#define P32(o, v) do { h[o] = (v) & 255; h[o + 1] = ((v) >> 8) & 255; h[o + 2] = ((v) >> 16) & 255; h[o + 3] = ((unsigned)(v) >> 24) & 255; } while (0)
#define P16(o, v) do { h[o] = (v) & 255; h[o + 1] = ((v) >> 8) & 255; } while (0)
	P32(4, 36 + bytes);
	memcpy(h + 8, "WAVEfmt ", 8);
	P32(16, 16); P16(20, 1); P16(22, 1); P32(24, rate); P32(28, rate * 2); P16(32, 2); P16(34, 16);
	memcpy(h + 36, "data", 4);
	P32(40, bytes);
	fwrite(h, 1, 44, f);
	for (i = 0; i < n; i++) {
		int s = (int)lrintf(a[i] / peak * 32767 * 0.9f);
		unsigned char b[2] = { s & 255, (s >> 8) & 255 };
		fwrite(b, 1, 2, f);
	}
	fclose(f);
}

int main(int argc, char **argv)
{
	char err[200], path[512];
	pv_voice *v;
	pv_options o;
	pv_times tm;
	pv_debug d;
	int ids[2048], n = 0, ns, x;
	float *audio, *ref;
	size_t nref;
	FILE *f;
	if (argc < 3) {
		fprintf(stderr, "usage: pv_test VOICE REFDIR [out.wav]\n");
		return 2;
	}
	v = pv_load(argv[1], err, sizeof err);
	if (!v) {
		printf("FAIL load: %s\n", err);
		return 1;
	}
	snprintf(path, sizeof path, "%s/ids.txt", argv[2]);
	f = fopen(path, "r");
	while (f && n < 2048 && fscanf(f, "%d", &x) == 1)
		ids[n++] = x;
	if (f)
		fclose(f);
	pv_default_options(v, &o);
	o.noise_scale = 0;
	o.noise_w = 0;
	o.length_scale = 1;
	o.seed = 1;
	audio = pv_synthesize(v, ids, n, &o, &ns, &tm);
	if (!audio) {
		printf("FAIL synthesize\n");
		return 1;
	}
	pv_last_debug(v, &d);
	/* PV_TOL=f16: a voice stored in half precision (inaudibly different). */
	{
		const char *t = getenv("PV_TOL");
		if (t && !strcmp(t, "f16"))
			tol_scale = 20;
	}
	if ((ref = load_f32(argv[2], "enc_m", &nref))) { compare("enc_m", d.enc_m, (size_t)192 * d.n, ref, nref, 1e-3); free(ref); }
	if ((ref = load_f32(argv[2], "enc_logs", &nref))) { compare("enc_logs", d.enc_logs, (size_t)192 * d.n, ref, nref, 1e-3); free(ref); }
	if ((ref = load_f32(argv[2], "logw", &nref))) { compare("logw", d.logw, d.n, ref, nref, 1e-3); free(ref); }
	if ((ref = load_f32(argv[2], "z", &nref))) { compare("z", d.z, (size_t)192 * d.frames, ref, nref, 2e-3); free(ref); }
	if ((ref = load_f32(argv[2], "audio", &nref))) { compare("audio", audio, ns, ref, nref, 1e-2); free(ref); }
	printf("%d phonemes, %d frames, %.2f s of audio at %d Hz\n", n, d.frames,
		(double)ns / pv_sample_rate(v), pv_sample_rate(v));
	printf("times: encoder %.3f s, durations %.3f s, flow %.3f s, decoder %.3f s; real-time factor %.3f\n",
		tm.enc, tm.dp, tm.flow, tm.dec, (tm.enc + tm.dp + tm.flow + tm.dec) / ((double)ns / pv_sample_rate(v)));
	if (argc > 3)
		write_wav(argv[3], audio, ns, pv_sample_rate(v));
	/* With noise: still sensible (no NaNs, a plausible length). */
	{
		int ns2, i, bad = 0;
		float *a2;
		pv_default_options(v, &o);
		o.seed = 12345;
		a2 = pv_synthesize(v, ids, n, &o, &ns2, NULL);
		checks++;
		for (i = 0; a2 && i < ns2; i++)
			if (!(a2[i] >= -1.0f && a2[i] <= 1.0f))
				bad++;
		if (!a2 || bad || ns2 < ns / 2 || ns2 > ns * 2) {
			fails++;
			printf("FAIL noisy synthesis: %d samples, %d bad\n", ns2, bad);
		}
		free(a2);
	}
	/* Stopped: gives up, and the voice still works after. */
	{
		volatile int stop = 1;
		int ns3;
		float *a3;
		pv_default_options(v, &o);
		o.stop = &stop;
		a3 = pv_synthesize(v, ids, n, &o, &ns3, NULL);
		checks++;
		if (a3) {
			fails++;
			printf("FAIL stop: synthesis went on\n");
			free(a3);
		}
		stop = 0;
		a3 = pv_synthesize(v, ids, n, &o, &ns3, NULL);
		checks++;
		if (!a3 || ns3 < ns / 2) {
			fails++;
			printf("FAIL after a stop\n");
		}
		free(a3);
	}
	free(audio);
	pv_free(v);
	printf("%d checks, %d failed\n", checks, fails);
	return fails != 0;
}
