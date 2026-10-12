/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
/* pvoice's arithmetic. Every convolution becomes matrix products (one per
 * kernel tap: the taps' inputs are shifted views of the padded signal), so
 * nearly all the time goes in pv_gemm_packed, which has NEON kernels on ARM
 * and plain C elsewhere. Its weights are packed when the voice loads so the
 * four rows of a tile share one load. */
#include "pv_math.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define PV_NEON 1
#endif

/* ---- Scratch ----------------------------------------------------------------- */

/* Padded inputs and transposed-convolution taps go in one buffer, kept and
 * grown, not allocated per layer (megabytes, dozens of times a sentence).
 * So one synthesis at a time per program (pvoice.h says so too). */
static float *scratch;
static size_t scratch_cap;

static float *get_scratch(size_t n)
{
	if (n > scratch_cap) {
		float *s = realloc(scratch, n * sizeof *s);
		if (!s)
			return NULL;
		scratch = s;
		scratch_cap = n;
	}
	return scratch;
}

/* ---- Matrix product ------------------------------------------------------ */

/* The work is tiled so it stays in cache: NB columns of the output at a
 * time (all its rows, every tap), K in blocks of KB. */
#define KB 128
#define NB 96

/* C[i][j] += A[i][p] * B[p][j] for rows m0..m1, columns n0..n1, plainly
 * (A row-major, lda apart). */
static void tail(int m0, int m1, int n0, int n1, int k0, int k1, const float *a, int lda,
		const float *b, int ldb, float *c, int ldc)
{
	int i, j, p;
	for (i = m0; i < m1; i++)
		for (p = k0; p < k1; p++) {
			float av = a[i * lda + p];
			const float *br = b + p * ldb;
			float *cr = c + i * ldc;
			for (j = n0; j < n1; j++)
				cr[j] += av * br[j];
		}
}

/* One block of a plain (unpacked) product: 4 rows x 8 columns at a time. */
static void block(int m, int n0, int n1, int k0, int k1, const float *a, int lda,
		const float *b, int ldb, float *c, int ldc)
{
	int i, j, p, m4 = m & ~3, n8 = n0 + ((n1 - n0) & ~7);
	for (i = 0; i < m4; i += 4) {
		const float *a0 = a + i * lda, *a1 = a0 + lda, *a2 = a1 + lda, *a3 = a2 + lda;
		float *c0 = c + i * ldc, *c1 = c0 + ldc, *c2 = c1 + ldc, *c3 = c2 + ldc;
		for (j = n0; j < n8; j += 8) {
			const float *bp = b + k0 * ldb + j;
#ifdef PV_NEON
			float32x4_t s00 = vld1q_f32(c0 + j), s01 = vld1q_f32(c0 + j + 4);
			float32x4_t s10 = vld1q_f32(c1 + j), s11 = vld1q_f32(c1 + j + 4);
			float32x4_t s20 = vld1q_f32(c2 + j), s21 = vld1q_f32(c2 + j + 4);
			float32x4_t s30 = vld1q_f32(c3 + j), s31 = vld1q_f32(c3 + j + 4);
			for (p = k0; p < k1; p++, bp += ldb) {
				float32x4_t b0 = vld1q_f32(bp), b1 = vld1q_f32(bp + 4);
				s00 = vmlaq_n_f32(s00, b0, a0[p]); s01 = vmlaq_n_f32(s01, b1, a0[p]);
				s10 = vmlaq_n_f32(s10, b0, a1[p]); s11 = vmlaq_n_f32(s11, b1, a1[p]);
				s20 = vmlaq_n_f32(s20, b0, a2[p]); s21 = vmlaq_n_f32(s21, b1, a2[p]);
				s30 = vmlaq_n_f32(s30, b0, a3[p]); s31 = vmlaq_n_f32(s31, b1, a3[p]);
			}
			vst1q_f32(c0 + j, s00); vst1q_f32(c0 + j + 4, s01);
			vst1q_f32(c1 + j, s10); vst1q_f32(c1 + j + 4, s11);
			vst1q_f32(c2 + j, s20); vst1q_f32(c2 + j + 4, s21);
			vst1q_f32(c3 + j, s30); vst1q_f32(c3 + j + 4, s31);
#else
			float s0[8], s1[8], s2[8], s3[8];
			int q;
			for (q = 0; q < 8; q++) {
				s0[q] = c0[j + q]; s1[q] = c1[j + q]; s2[q] = c2[j + q]; s3[q] = c3[j + q];
			}
			for (p = k0; p < k1; p++, bp += ldb) {
				float v0 = a0[p], v1 = a1[p], v2 = a2[p], v3 = a3[p];
				for (q = 0; q < 8; q++) {
					float bv = bp[q];
					s0[q] += v0 * bv; s1[q] += v1 * bv; s2[q] += v2 * bv; s3[q] += v3 * bv;
				}
			}
			for (q = 0; q < 8; q++) {
				c0[j + q] = s0[q]; c1[j + q] = s1[q]; c2[j + q] = s2[q]; c3[j + q] = s3[q];
			}
#endif
		}
		if (n8 < n1)
			tail(i, i + 4, n8, n1, k0, k1, a, lda, b, ldb, c, ldc);
	}
	if (m4 < m)
		tail(m4, m, n0, n1, k0, k1, a, lda, b, ldb, c, ldc);
}

void pv_gemm_taps(int m, int n, int k, int taps, const float *a, int lda, const float *b, int ldb,
		int tap_step, float *c, int ldc)
{
	int j0, k0, t;
	for (j0 = 0; j0 < n; j0 += NB) {
		int j1 = j0 + NB < n ? j0 + NB : n;
		for (t = 0; t < taps; t++)
			for (k0 = 0; k0 < k; k0 += KB) {
				int k1 = k0 + KB < k ? k0 + KB : k;
				block(m, j0, j1, k0, k1, a + (size_t)t * m * lda, lda, b + t * tap_step, ldb, c, ldc);
			}
	}
}

void pv_gemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb,
		float *c, int ldc)
{
	pv_gemm_taps(m, n, k, 1, a, lda, b, ldb, 0, c, ldc);
}

/* ---- Matrix product with packed weights (the convolutions) ---------------- */

/* A (m x k) packed in panels of 4 rows: panel r holds rows 4r..4r+3, as k
 * groups of 4 (one per column), so one load gives the four rows' weights
 * for a column. The m % 4 rows left over follow, row by row. */
size_t pv_packed_index(int m, int k, int row, int col)
{
	int m4 = m & ~3;
	if (row < m4)
		return ((size_t)(row >> 2) * k + col) * 4 + (row & 3);
	return (size_t)m4 * k + (size_t)(row - m4) * k + col;
}

/* ---- The convolution engine ---------------------------------------------- */

/* Every convolution (and each phase of a transposed one) is one product
 *
 *   C[i][j] = init + sum over taps t, inputs p of A_t[i][p] * B_t[p][j],
 *   B_t[p][j] = act(x[p][j + off0 + t * off_step]), 0 outside 0..t_in-1
 *
 * with A_t packed (pv_packed_index) a_tap floats apart, and init either C's
 * own values (accumulate) or the bias plus a residual signal. B is never
 * made in full: for each block of NB2 output columns and a slice of the
 * inputs (kb of them, across every tap) its part is packed into strips 12
 * columns wide, [strip][tap][p][12], applying the leaky ReLU and the zero
 * padding as it's copied. One strip (12 KB at most) is used by every
 * four-row panel of A in turn, so it stays in the L1 cache while A streams
 * from L2; each 4 x 12 tile of C stays in 12 NEON registers across the
 * whole slice. Layer by layer that's one pass over the input and one over
 * the output, where padding, bias, the product and the residual sum were
 * separate passes over signals of megabytes (the decoder's last stage is
 * 32 channels x 16000 a second). Tuned with a model of the Cortex-A72's
 * caches (32 KB 2-way L1, 1 MB L2) under qemu. */
#define NB2  96         /* output columns per block (8 strips) */
#define KQ   256        /* taps x inputs per slice */

typedef struct {
	const float *x;         /* B's source rows, xld apart, t_in long */
	int xld, t_in, off0, off_step;
	float slope;            /* leaky ReLU on the way in; 1: none */
	const float *bias;      /* init (first slice): bias[i] (or 0) + res[i][j] (or 0) */
	const float *res;
	int resld;
	int accumulate;         /* init is C's own values instead */
} pv_bsrc;

static float *bpack;
static size_t bpack_cap;

static inline float act(float v, float slope)
{
	return v < 0 ? v * slope : v;
}

static void pack_b(const pv_bsrc *s, int taps, int k0, int k1, int j0, int j1, float *bp)
{
	int j, t, p, q, kb = k1 - k0;
	for (j = j0; j < j1; j += 12) {
		int w = j1 - j < 12 ? j1 - j : 12;
		for (t = 0; t < taps; t++) {
			int u0 = j + s->off0 + t * s->off_step;    /* x column for this strip's first */
			for (p = k0; p < k1; p++) {
				const float *src = s->x + (size_t)p * s->xld;
				float *d = bp + ((size_t)((j - j0) / 12) * taps * kb + (size_t)t * kb + (p - k0)) * 12;
				if (u0 >= 0 && u0 + w <= s->t_in && w == 12) {
					const float *r = src + u0;
					if (s->slope == 1.0f) {
						memcpy(d, r, 12 * sizeof *d);
					} else {
#ifdef PV_NEON
						/* max(v, 0) + slope * min(v, 0) */
						float32x4_t k = vdupq_n_f32(s->slope), z = vdupq_n_f32(0);
						for (q = 0; q < 12; q += 4) {
							float32x4_t v = vld1q_f32(r + q);
							vst1q_f32(d + q, vmlaq_f32(vmaxq_f32(v, z), vminq_f32(v, z), k));
						}
#else
						for (q = 0; q < 12; q++)
							d[q] = act(r[q], s->slope);
#endif
					}
				} else {
					for (q = 0; q < 12; q++) {
						int u = u0 + q;
						d[q] = q < w && u >= 0 && u < s->t_in ? act(src[u], s->slope) : 0.0f;
					}
				}
			}
		}
	}
}

/* Where the first slice starts from: C, or bias + residual (or 0). */
static void init_row(const pv_bsrc *s, int first, int i, int j, int w, const float *c, float *out)
{
	int q;
	if (!first || s->accumulate) {
		for (q = 0; q < w; q++)
			out[q] = c[q];
	} else {
		float bv = s->bias ? s->bias[i] : 0.0f;
		const float *r = s->res ? s->res + (size_t)i * s->resld + j : NULL;
		for (q = 0; q < w; q++)
			out[q] = r ? bv + r[q] : bv;
	}
	for (q = w; q < 12; q++)
		out[q] = 0;
}

/* C rows i..i+3 (ldc apart), columns j..j+w-1 (w <= 12): the tile over taps x
 * kb. pa: the panel's weights for tap 0 at input k0; tap t's are a_tap
 * floats further on. */
static void kernel_4x12(const pv_bsrc *s, int first, int i, int j, const float *pa, size_t a_tap,
		int taps, int kb, const float *bs, float *c, int ldc, int w)
{
	float in[4][12];
	const float *src[4];
	float bias[4];
	int t, p, r, q, from_c = !first || s->accumulate;
	/* Each row starts from C, or bias + residual. */
	for (r = 0; r < 4; r++) {
		src[r] = from_c ? c + (size_t)r * ldc : s->res ? s->res + (size_t)(i + r) * s->resld + j : NULL;
		bias[r] = !from_c && s->bias ? s->bias[i + r] : 0.0f;
	}
	if (w < 12)
		for (r = 0; r < 4; r++)
			for (q = 0; q < 12; q++)
				in[r][q] = (q < w && src[r] ? src[r][q] : 0.0f) + bias[r];
#ifdef PV_NEON
	{
		float32x4_t s00, s01, s02, s10, s11, s12, s20, s21, s22, s30, s31, s32;
#define ROW(r, a0, a1, a2) do { \
		float32x4_t bv = vdupq_n_f32(bias[r]); \
		if (w < 12) { a0 = vld1q_f32(in[r]); a1 = vld1q_f32(in[r] + 4); a2 = vld1q_f32(in[r] + 8); } \
		else if (src[r]) { a0 = vaddq_f32(vld1q_f32(src[r]), bv); a1 = vaddq_f32(vld1q_f32(src[r] + 4), bv); \
			a2 = vaddq_f32(vld1q_f32(src[r] + 8), bv); } \
		else { a0 = a1 = a2 = bv; } } while (0)
		ROW(0, s00, s01, s02); ROW(1, s10, s11, s12); ROW(2, s20, s21, s22); ROW(3, s30, s31, s32);
#undef ROW
		for (t = 0; t < taps; t++) {
			const float *pp = pa + t * a_tap;
			for (p = 0; p < kb; p++, pp += 4, bs += 12) {
				float32x4_t av = vld1q_f32(pp);
				float32x2_t lo = vget_low_f32(av), hi = vget_high_f32(av);
				float32x4_t b0 = vld1q_f32(bs), b1 = vld1q_f32(bs + 4), b2 = vld1q_f32(bs + 8);
				s00 = vmlaq_lane_f32(s00, b0, lo, 0); s01 = vmlaq_lane_f32(s01, b1, lo, 0); s02 = vmlaq_lane_f32(s02, b2, lo, 0);
				s10 = vmlaq_lane_f32(s10, b0, lo, 1); s11 = vmlaq_lane_f32(s11, b1, lo, 1); s12 = vmlaq_lane_f32(s12, b2, lo, 1);
				s20 = vmlaq_lane_f32(s20, b0, hi, 0); s21 = vmlaq_lane_f32(s21, b1, hi, 0); s22 = vmlaq_lane_f32(s22, b2, hi, 0);
				s30 = vmlaq_lane_f32(s30, b0, hi, 1); s31 = vmlaq_lane_f32(s31, b1, hi, 1); s32 = vmlaq_lane_f32(s32, b2, hi, 1);
			}
		}
		if (w == 12) {
			float *c0 = c, *c1 = c0 + ldc, *c2 = c1 + ldc, *c3 = c2 + ldc;
			vst1q_f32(c0, s00); vst1q_f32(c0 + 4, s01); vst1q_f32(c0 + 8, s02);
			vst1q_f32(c1, s10); vst1q_f32(c1 + 4, s11); vst1q_f32(c1 + 8, s12);
			vst1q_f32(c2, s20); vst1q_f32(c2 + 4, s21); vst1q_f32(c2 + 8, s22);
			vst1q_f32(c3, s30); vst1q_f32(c3 + 4, s31); vst1q_f32(c3 + 8, s32);
			return;
		}
		vst1q_f32(in[0], s00); vst1q_f32(in[0] + 4, s01); vst1q_f32(in[0] + 8, s02);
		vst1q_f32(in[1], s10); vst1q_f32(in[1] + 4, s11); vst1q_f32(in[1] + 8, s12);
		vst1q_f32(in[2], s20); vst1q_f32(in[2] + 4, s21); vst1q_f32(in[2] + 8, s22);
		vst1q_f32(in[3], s30); vst1q_f32(in[3] + 4, s31); vst1q_f32(in[3] + 8, s32);
	}
#else
	if (w == 12)
		for (r = 0; r < 4; r++)
			for (q = 0; q < 12; q++)
				in[r][q] = (src[r] ? src[r][q] : 0.0f) + bias[r];
	for (t = 0; t < taps; t++) {
		const float *pp = pa + t * a_tap;
		for (p = 0; p < kb; p++, pp += 4, bs += 12)
			for (q = 0; q < 12; q++) {
				float bv = bs[q];
				in[0][q] += pp[0] * bv; in[1][q] += pp[1] * bv; in[2][q] += pp[2] * bv; in[3][q] += pp[3] * bv;
			}
	}
#endif
	for (r = 0; r < 4; r++)
		for (q = 0; q < w; q++)
			c[(size_t)r * ldc + q] = in[r][q];
}

/* One left-over row (i >= m4): row-major weights, k apart, at ar for tap 0. */
static void kernel_row(const pv_bsrc *s, int first, int i, int j, const float *ar, size_t a_tap,
		int taps, int kb, const float *bs, float *c, int w)
{
	float in[12];
	int t, p, q;
	init_row(s, first, i, j, w, c, in);
	for (t = 0; t < taps; t++) {
		const float *pr = ar + t * a_tap;
		for (p = 0; p < kb; p++, bs += 12) {
			float av = pr[p];
			for (q = 0; q < 12; q++)
				in[q] += av * bs[q];
		}
	}
	for (q = 0; q < w; q++)
		c[q] = in[q];
}

/* C (m x n, ldc) as described above, for a k-input product. */
static int gemm_conv(int m, int n, int k, int taps, const float *ap, size_t a_tap, const pv_bsrc *s,
		float *c, int ldc)
{
	int m4 = m & ~3, kb = KQ / taps, j0, k0, i;
	size_t need;
	if (kb < 8)
		kb = 8;
	if (kb > k)
		kb = k;
	need = (size_t)((NB2 + 11) / 12) * taps * kb * 12;
	if (need > bpack_cap) {
		float *nbp = realloc(bpack, need * sizeof *nbp);
		if (!nbp)
			return -1;
		bpack = nbp;
		bpack_cap = need;
	}
	for (j0 = 0; j0 < n; j0 += NB2) {
		int j1 = j0 + NB2 < n ? j0 + NB2 : n;
		for (k0 = 0; k0 < k; k0 += kb) {
			int k1 = k0 + kb < k ? k0 + kb : k, kn = k1 - k0, j;
			pack_b(s, taps, k0, k1, j0, j1, bpack);
			for (j = j0; j < j1; j += 12) {
				const float *bs = bpack + (size_t)((j - j0) / 12) * taps * kn * 12;
				int w = j1 - j < 12 ? j1 - j : 12;
				for (i = 0; i < m4; i += 4)
					kernel_4x12(s, k0 == 0, i, j, ap + ((size_t)(i >> 2) * k + k0) * 4, a_tap, taps, kn,
						bs, c + (size_t)i * ldc + j, ldc, w);
				for (; i < m; i++)
					kernel_row(s, k0 == 0, i, j, ap + (size_t)m4 * k + (size_t)(i - m4) * k + k0, a_tap,
						taps, kn, bs, c + (size_t)i * ldc + j, w);
			}
		}
	}
	return 0;
}

void pv_gemm_packed(int m, int n, int k, int taps, const float *ap, const float *b, int ldb,
		int tap_step, float *c, int ldc)
{
	pv_bsrc s;
	memset(&s, 0, sizeof s);
	s.x = b;
	s.xld = ldb;
	s.t_in = 0x3fffffff;           /* the caller keeps every column in range */
	s.off_step = tap_step;
	s.slope = 1.0f;
	s.accumulate = 1;
	gemm_conv(m, n, k, taps, ap, (size_t)m * k, &s, c, ldc);
}

/* ---- Convolutions -------------------------------------------------------- */

int pv_conv_out_len(const pv_conv *c, int t)
{
	if (c->transposed)
		return (t - 1) * c->stride - c->pad_l - c->pad_r + c->k;
	return (t + c->pad_l + c->pad_r - c->dil * (c->k - 1) - 1) / c->stride + 1;
}

/* Copies a row, applying a leaky ReLU (slope < 1) or not (slope 1). */
static void copy_act(float *d, const float *s, int n, float slope)
{
	int i = 0;
	if (slope == 1.0f) {
		memcpy(d, s, n * sizeof *d);
		return;
	}
#ifdef PV_NEON
	{
		float32x4_t k = vdupq_n_f32(slope), z = vdupq_n_f32(0);
		for (; i + 4 <= n; i += 4) {
			float32x4_t v = vld1q_f32(s + i);
			uint32x4_t neg = vcltq_f32(v, z);
			vst1q_f32(d + i, vbslq_f32(neg, vmulq_f32(v, k), v));
		}
	}
#endif
	for (; i < n; i++)
		d[i] = s[i] < 0 ? s[i] * slope : s[i];
}

void pv_conv_run_res(const pv_conv *c, const float *x, int t, int xld, float *y, int yld, float slope,
		const float *res, int resld)
{
	int tout = pv_conv_out_len(c, t), o, i, kk;
	pv_bsrc s;
	memset(&s, 0, sizeof s);
	s.x = x;
	s.xld = xld;
	s.t_in = t;
	s.slope = slope;
	s.bias = c->b;
	s.res = res;
	s.resld = resld;
	if (c->transposed) {
		/* Polyphase: output phase r (positions s*u + r) is an ordinary
		 * convolution of x with the taps kk = r + pad_l (mod s), spaced s
		 * apart: y[s*u + r] = sum W[kk] x[u + (r + pad_l - kk) / s]. Each
		 * phase is made into a scratch row block, then spread out. */
		int st = c->stride, r;
		for (r = 0; r < st && r < tout; r++) {
			int kk0 = ((r + c->pad_l) % st + st) % st, taps = 0, nr = (tout - r + st - 1) / st;
			float *tmp;
			for (kk = kk0; kk < c->k; kk += st)
				taps++;
			tmp = get_scratch((size_t)c->out * nr);
			if (!tmp)
				return;
			s.off0 = (r + c->pad_l - kk0) / st;
			s.off_step = -1;
			s.res = NULL;
			if (taps == 0) {
				for (o = 0; o < c->out; o++)
					for (i = 0; i < nr; i++)
						tmp[(size_t)o * nr + i] = c->b ? c->b[o] : 0.0f;
			} else if (gemm_conv(c->out, nr, c->in, taps, c->w + (size_t)kk0 * c->out * c->in,
					(size_t)st * c->out * c->in, &s, tmp, nr)) {
				return;
			}
			for (o = 0; o < c->out; o++) {
				const float *tr = tmp + (size_t)o * nr;
				float *yr = y + (size_t)o * yld + r;
				const float *rr = res ? res + (size_t)o * resld + r : NULL;
				for (i = 0; i < nr; i++)
					yr[(size_t)i * st] = rr ? tr[i] + rr[(size_t)i * st] : tr[i];
			}
		}
		return;
	}
	if (c->groups == 1 && c->stride == 1) {
		s.off0 = -c->pad_l;
		s.off_step = c->dil;
		gemm_conv(c->out, tout, c->in, c->k, c->w, (size_t)c->out * c->in, &s, y, yld);
		return;
	}
	{
		/* Depthwise (groups == channels: w is [out][k]), or strided. */
		for (o = 0; o < c->out; o++) {
			float *yr = y + (size_t)o * yld;
			for (i = 0; i < tout; i++) {
				float acc = c->b ? c->b[o] : 0.0f;
				int ci;
				if (res)
					acc += res[(size_t)o * resld + i];
				for (kk = 0; kk < c->k; kk++) {
					int u = i * c->stride - c->pad_l + kk * c->dil;
					if (u < 0 || u >= t)
						continue;
					if (c->groups > 1) {
						acc += c->w[o * c->k + kk] * act(x[(size_t)o * xld + u], slope);
					} else {
						for (ci = 0; ci < c->in; ci++)
							acc += c->w[(size_t)kk * c->out * c->in + pv_packed_index(c->out, c->in, o, ci)] *
								act(x[(size_t)ci * xld + u], slope);
					}
				}
				yr[i] = acc;
			}
		}
	}
}

void pv_conv_run_act(const pv_conv *c, const float *x, int t, int xld, float *y, int yld, float slope)
{
	pv_conv_run_res(c, x, t, xld, y, yld, slope, NULL, 0);
}

void pv_conv_run(const pv_conv *c, const float *x, int t, int xld, float *y, int yld)
{
	pv_conv_run_act(c, x, t, xld, y, yld, 1.0f);
}

/* ---- Fast exp, tanh and sigmoid ------------------------------------------------ */

/* exp(x) as 2^n * e^r, r in [-ln2/2, ln2/2], e^r by a degree-6 polynomial:
 * within 2 ulp over the range used here. The NEON and plain versions do
 * the same arithmetic. */
#define EXP_HI 88.0f
#define EXP_LO -87.0f

static float exp_fast(float x)
{
	float n, r, p;
	int ni;
	union { float f; unsigned u; } s;
	if (x > EXP_HI) x = EXP_HI;
	if (x < EXP_LO) x = EXP_LO;
	n = floorf(x * 1.44269504f + 0.5f);
	r = x - n * 0.693359375f;
	r = r + n * 2.12194440e-4f;
	p = 1.0f / 720;
	p = p * r + 1.0f / 120;
	p = p * r + 1.0f / 24;
	p = p * r + 1.0f / 6;
	p = p * r + 0.5f;
	p = p * r + 1.0f;
	p = p * r + 1.0f;
	ni = (int)n;
	s.u = (unsigned)(ni + 127) << 23;
	return p * s.f;
}

#ifdef PV_NEON
static float32x4_t exp_neon(float32x4_t x)
{
	float32x4_t n, r, p, half = vdupq_n_f32(0.5f);
	int32x4_t ni;
	x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(EXP_LO)), vdupq_n_f32(EXP_HI));
	/* floor(x*log2e + 0.5): truncate, then step down if that went up. */
	n = vmlaq_n_f32(half, x, 1.44269504f);
	ni = vcvtq_s32_f32(n);
	{
		float32x4_t t = vcvtq_f32_s32(ni);
		uint32x4_t up = vcgtq_f32(t, n);
		ni = vsubq_s32(ni, vreinterpretq_s32_u32(vandq_u32(up, vdupq_n_u32(1))));
		n = vcvtq_f32_s32(ni);
	}
	r = vmlsq_n_f32(x, n, 0.693359375f);
	r = vmlaq_n_f32(r, n, 2.12194440e-4f);
	p = vdupq_n_f32(1.0f / 720);
	p = vmlaq_f32(vdupq_n_f32(1.0f / 120), p, r);
	p = vmlaq_f32(vdupq_n_f32(1.0f / 24), p, r);
	p = vmlaq_f32(vdupq_n_f32(1.0f / 6), p, r);
	p = vmlaq_f32(vdupq_n_f32(0.5f), p, r);
	p = vmlaq_f32(vdupq_n_f32(1.0f), p, r);
	p = vmlaq_f32(vdupq_n_f32(1.0f), p, r);
	return vmulq_f32(p, vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23)));
}

/* 1/x to full precision: an estimate and two Newton steps. */
static float32x4_t recip_neon(float32x4_t x)
{
	float32x4_t e = vrecpeq_f32(x);
	e = vmulq_f32(vrecpsq_f32(x, e), e);
	e = vmulq_f32(vrecpsq_f32(x, e), e);
	return e;
}
#endif

/* a[i] = tanh(a[i]) * sigmoid(b[i]): WaveNet's gate. tanh(a) is
 * 1 - 2 / (e^2a + 1); sigmoid(b) is 1 / (1 + e^-b). */
void pv_gate(float *a, const float *b, size_t n)
{
	size_t i = 0;
#ifdef PV_NEON
	{
		float32x4_t one = vdupq_n_f32(1.0f), two = vdupq_n_f32(2.0f);
		for (; i + 4 <= n; i += 4) {
			float32x4_t ea = exp_neon(vmulq_f32(vld1q_f32(a + i), two));
			float32x4_t eb = exp_neon(vnegq_f32(vld1q_f32(b + i)));
			float32x4_t th = vsubq_f32(one, vmulq_f32(two, recip_neon(vaddq_f32(ea, one))));
			float32x4_t sg = recip_neon(vaddq_f32(one, eb));
			vst1q_f32(a + i, vmulq_f32(th, sg));
		}
	}
#endif
	for (; i < n; i++) {
		float th = 1.0f - 2.0f / (exp_fast(2.0f * a[i]) + 1.0f);
		float sg = 1.0f / (1.0f + exp_fast(-b[i]));
		a[i] = th * sg;
	}
}

void pv_tanh(float *x, size_t n)
{
	size_t i = 0;
#ifdef PV_NEON
	{
		float32x4_t one = vdupq_n_f32(1.0f), two = vdupq_n_f32(2.0f);
		for (; i + 4 <= n; i += 4) {
			float32x4_t e = exp_neon(vmulq_f32(vld1q_f32(x + i), two));
			vst1q_f32(x + i, vsubq_f32(one, vmulq_f32(two, recip_neon(vaddq_f32(e, one)))));
		}
	}
#endif
	for (; i < n; i++)
		x[i] = 1.0f - 2.0f / (exp_fast(2.0f * x[i]) + 1.0f);
}

/* ---- Elementwise ----------------------------------------------------------- */

void pv_layer_norm(float *x, int ch, int t, int ld, const float *gamma, const float *beta)
{
	float *mean = calloc((size_t)t * 2, sizeof *mean), *var;
	int c, i;
	if (!mean)
		return;
	var = mean + t;
	for (c = 0; c < ch; c++) {
		const float *r = x + (size_t)c * ld;
		for (i = 0; i < t; i++)
			mean[i] += r[i];
	}
	for (i = 0; i < t; i++)
		mean[i] /= ch;
	for (c = 0; c < ch; c++) {
		const float *r = x + (size_t)c * ld;
		for (i = 0; i < t; i++) {
			float d = r[i] - mean[i];
			var[i] += d * d;
		}
	}
	for (i = 0; i < t; i++)
		var[i] = 1.0f / sqrtf(var[i] / ch + 1e-5f);
	for (c = 0; c < ch; c++) {
		float *r = x + (size_t)c * ld;
		float g = gamma[c], b = beta[c];
		for (i = 0; i < t; i++)
			r[i] = (r[i] - mean[i]) * var[i] * g + b;
	}
	free(mean);
}

void pv_leaky_relu(float *x, size_t n, float slope)
{
	copy_act(x, x, (int)n, slope);
}

void pv_relu(float *x, size_t n)
{
	size_t i;
	for (i = 0; i < n; i++)
		if (x[i] < 0)
			x[i] = 0;
}

void pv_gelu(float *x, size_t n)
{
	size_t i;
	for (i = 0; i < n; i++)
		x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678f));
}

/* ---- Up-sampling for playback ------------------------------------------------ */

/* A "low" voice speaks at 16 kHz. The sound system plays it at its own
 * rate (48 kHz, say) by repeating or straight-line joining samples, which
 * leaves copies of the voice's spectrum above 8 kHz: a tinny, metallic
 * edge. Here it's up-sampled by a whole factor with a proper low-pass
 * filter (a Kaiser-windowed sinc, 48 taps per phase, flat to 0.44 of the
 * voice's rate, copies down by about 80 dB), so the sound system gets a
 * clean signal at a rate it plays without converting much. */
#define UP_TAPS 48
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int pv_upsample_factor(int rate)
{
	return rate <= 16000 ? 3 : rate <= 24000 ? 2 : 1;
}

static double bessel_i0(double x)
{
	double s = 1, t = 1;
	int k;
	for (k = 1; k < 40; k++) {
		t *= (x / (2 * k)) * (x / (2 * k));
		s += t;
	}
	return s;
}

float *pv_upsample(const float *in, int n, int factor, int *n_out)
{
	int len = UP_TAPS * factor, c = (len - 1) / 2, m, j, total;
	float *h, *out;
	double beta = 8.0, fc = 0.47 / factor;     /* cutoff, in cycles a sample at the new rate */
	if (factor <= 1) {
		out = malloc((n ? n : 1) * sizeof *out);
		if (out)
			memcpy(out, in, n * sizeof *out);
		*n_out = n;
		return out;
	}
	h = malloc(len * sizeof *h);
	total = n * factor;
	out = malloc((total ? total : 1) * sizeof *out);
	if (!h || !out) {
		free(h);
		free(out);
		return NULL;
	}
	for (j = 0; j < len; j++) {
		double x = j - (len - 1) / 2.0, r = 2.0 * x / (len - 1);
		double sinc = x == 0 ? 2 * fc : sin(2 * M_PI * fc * x) / (M_PI * x);
		double w = bessel_i0(beta * sqrt(fabs(1 - r * r))) / bessel_i0(beta);
		h[j] = (float)(sinc * w * factor);     /* x factor: the zeros put in between */
	}
	/* out[m] = y[m + c], y[k] = sum_j h[j] u[k - j], u = in with factor - 1
	 * zeros after each sample: only the taps landing on samples count. */
	for (m = 0; m < total; m++) {
		int k = m + c, ph = k % factor, i = k / factor;
		double acc = 0;
		for (j = ph; j < len; j += factor, i--)
			if (i >= 0 && i < n)
				acc += h[j] * in[i];
		out[m] = (float)acc;
	}
	free(h);
	*n_out = total;
	return out;
}
