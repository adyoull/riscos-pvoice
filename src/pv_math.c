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

/* Rows m4..m: the left-over rows, row-major (k apart) at rest. */
static void packed_tail_rows(int m, int n0, int n1, int k0, int k1, int k, const float *rest,
		const float *b, int ldb, float *c, int ldc)
{
	int i, j, p, m4 = m & ~3;
	for (i = m4; i < m; i++)
		for (p = k0; p < k1; p++) {
			float av = rest[(size_t)(i - m4) * k + p];
			const float *br = b + p * ldb;
			float *cr = c + i * ldc;
			for (j = n0; j < n1; j++)
				cr[j] += av * br[j];
		}
}

/* One block, 4 rows by 12 columns in registers (12 NEON accumulators, 3
 * for B, 1 for A's four weights: all 16), then by 8, then plainly. */
static void block_packed(int m, int n0, int n1, int k0, int k1, int k, const float *ap,
		const float *b, int ldb, float *c, int ldc)
{
	int i, j, p, m4 = m & ~3;
	for (i = 0; i < m4; i += 4) {
		const float *pa = ap + (size_t)(i >> 2) * k * 4;
		float *c0 = c + i * ldc, *c1 = c0 + ldc, *c2 = c1 + ldc, *c3 = c2 + ldc;
		j = n0;
#ifdef PV_NEON
		for (; j + 12 <= n1; j += 12) {
			const float *bp = b + k0 * ldb + j, *pp = pa + k0 * 4;
			float32x4_t s00 = vld1q_f32(c0 + j), s01 = vld1q_f32(c0 + j + 4), s02 = vld1q_f32(c0 + j + 8);
			float32x4_t s10 = vld1q_f32(c1 + j), s11 = vld1q_f32(c1 + j + 4), s12 = vld1q_f32(c1 + j + 8);
			float32x4_t s20 = vld1q_f32(c2 + j), s21 = vld1q_f32(c2 + j + 4), s22 = vld1q_f32(c2 + j + 8);
			float32x4_t s30 = vld1q_f32(c3 + j), s31 = vld1q_f32(c3 + j + 4), s32 = vld1q_f32(c3 + j + 8);
			for (p = k0; p < k1; p++, bp += ldb, pp += 4) {
				float32x4_t av = vld1q_f32(pp);
				float32x2_t lo = vget_low_f32(av), hi = vget_high_f32(av);
				float32x4_t b0 = vld1q_f32(bp), b1 = vld1q_f32(bp + 4), b2 = vld1q_f32(bp + 8);
				s00 = vmlaq_lane_f32(s00, b0, lo, 0); s01 = vmlaq_lane_f32(s01, b1, lo, 0); s02 = vmlaq_lane_f32(s02, b2, lo, 0);
				s10 = vmlaq_lane_f32(s10, b0, lo, 1); s11 = vmlaq_lane_f32(s11, b1, lo, 1); s12 = vmlaq_lane_f32(s12, b2, lo, 1);
				s20 = vmlaq_lane_f32(s20, b0, hi, 0); s21 = vmlaq_lane_f32(s21, b1, hi, 0); s22 = vmlaq_lane_f32(s22, b2, hi, 0);
				s30 = vmlaq_lane_f32(s30, b0, hi, 1); s31 = vmlaq_lane_f32(s31, b1, hi, 1); s32 = vmlaq_lane_f32(s32, b2, hi, 1);
			}
			vst1q_f32(c0 + j, s00); vst1q_f32(c0 + j + 4, s01); vst1q_f32(c0 + j + 8, s02);
			vst1q_f32(c1 + j, s10); vst1q_f32(c1 + j + 4, s11); vst1q_f32(c1 + j + 8, s12);
			vst1q_f32(c2 + j, s20); vst1q_f32(c2 + j + 4, s21); vst1q_f32(c2 + j + 8, s22);
			vst1q_f32(c3 + j, s30); vst1q_f32(c3 + j + 4, s31); vst1q_f32(c3 + j + 8, s32);
		}
		for (; j + 8 <= n1; j += 8) {
			const float *bp = b + k0 * ldb + j, *pp = pa + k0 * 4;
			float32x4_t s00 = vld1q_f32(c0 + j), s01 = vld1q_f32(c0 + j + 4);
			float32x4_t s10 = vld1q_f32(c1 + j), s11 = vld1q_f32(c1 + j + 4);
			float32x4_t s20 = vld1q_f32(c2 + j), s21 = vld1q_f32(c2 + j + 4);
			float32x4_t s30 = vld1q_f32(c3 + j), s31 = vld1q_f32(c3 + j + 4);
			for (p = k0; p < k1; p++, bp += ldb, pp += 4) {
				float32x4_t av = vld1q_f32(pp);
				float32x2_t lo = vget_low_f32(av), hi = vget_high_f32(av);
				float32x4_t b0 = vld1q_f32(bp), b1 = vld1q_f32(bp + 4);
				s00 = vmlaq_lane_f32(s00, b0, lo, 0); s01 = vmlaq_lane_f32(s01, b1, lo, 0);
				s10 = vmlaq_lane_f32(s10, b0, lo, 1); s11 = vmlaq_lane_f32(s11, b1, lo, 1);
				s20 = vmlaq_lane_f32(s20, b0, hi, 0); s21 = vmlaq_lane_f32(s21, b1, hi, 0);
				s30 = vmlaq_lane_f32(s30, b0, hi, 1); s31 = vmlaq_lane_f32(s31, b1, hi, 1);
			}
			vst1q_f32(c0 + j, s00); vst1q_f32(c0 + j + 4, s01);
			vst1q_f32(c1 + j, s10); vst1q_f32(c1 + j + 4, s11);
			vst1q_f32(c2 + j, s20); vst1q_f32(c2 + j + 4, s21);
			vst1q_f32(c3 + j, s30); vst1q_f32(c3 + j + 4, s31);
		}
#else
		for (; j + 8 <= n1; j += 8) {
			const float *bp = b + k0 * ldb + j, *pp = pa + k0 * 4;
			float s0[8], s1[8], s2[8], s3[8];
			int q;
			for (q = 0; q < 8; q++) {
				s0[q] = c0[j + q]; s1[q] = c1[j + q]; s2[q] = c2[j + q]; s3[q] = c3[j + q];
			}
			for (p = k0; p < k1; p++, bp += ldb, pp += 4) {
				float v0 = pp[0], v1 = pp[1], v2 = pp[2], v3 = pp[3];
				for (q = 0; q < 8; q++) {
					float bv = bp[q];
					s0[q] += v0 * bv; s1[q] += v1 * bv; s2[q] += v2 * bv; s3[q] += v3 * bv;
				}
			}
			for (q = 0; q < 8; q++) {
				c0[j + q] = s0[q]; c1[j + q] = s1[q]; c2[j + q] = s2[q]; c3[j + q] = s3[q];
			}
		}
#endif
		/* The last few columns, plainly. */
		for (; j < n1; j++)
			for (p = k0; p < k1; p++) {
				const float *pp = pa + p * 4;
				float bv = b[p * ldb + j];
				c0[j] += pp[0] * bv; c1[j] += pp[1] * bv; c2[j] += pp[2] * bv; c3[j] += pp[3] * bv;
			}
	}
	if (m4 < m)
		packed_tail_rows(m, n0, n1, k0, k1, k, ap + (size_t)m4 * k, b, ldb, c, ldc);
}

void pv_gemm_packed(int m, int n, int k, int taps, const float *ap, const float *b, int ldb,
		int tap_step, float *c, int ldc)
{
	int j0, k0, t;
	for (j0 = 0; j0 < n; j0 += NB) {
		int j1 = j0 + NB < n ? j0 + NB : n;
		for (t = 0; t < taps; t++)
			for (k0 = 0; k0 < k; k0 += KB) {
				int k1 = k0 + KB < k ? k0 + KB : k;
				block_packed(m, j0, j1, k0, k1, k, ap + (size_t)t * m * k, b + t * tap_step, ldb, c, ldc);
			}
	}
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

void pv_conv_run_act(const pv_conv *c, const float *x, int t, int xld, float *y, int yld, float slope)
{
	int tout = pv_conv_out_len(c, t), o, i, kk;
	if (c->transposed) {
		/* y[o][s*t' - pad_l + kk] += sum_i W[kk][o][i] x[i][t'], one tap at a time. */
		float *tmp = get_scratch((size_t)c->out * t);
		for (o = 0; o < c->out; o++) {
			float bv = c->b ? c->b[o] : 0.0f;
			for (i = 0; i < tout; i++)
				y[o * yld + i] = bv;
		}
		if (!tmp)
			return;
		for (kk = 0; kk < c->k; kk++) {
			int tt, first = -1, last = -1;
			memset(tmp, 0, (size_t)c->out * t * sizeof *tmp);
			pv_gemm_packed(c->out, t, c->in, 1, c->w + (size_t)kk * c->out * c->in, x, xld, 0, tmp, t);
			/* The inputs that land inside the output. */
			for (tt = 0; tt < t; tt++) {
				int pos = tt * c->stride - c->pad_l + kk;
				if (pos >= 0 && pos < tout) {
					if (first < 0)
						first = tt;
					last = tt;
				}
			}
			if (first < 0)
				continue;
			for (o = 0; o < c->out; o++) {
				float *yr = y + o * yld + first * c->stride - c->pad_l + kk;
				const float *tr = tmp + o * t;
				for (tt = first; tt <= last; tt++, yr += c->stride)
					*yr += tr[tt];
			}
		}
		return;
	}
	{
		/* Pad the input once (activating it on the way); each tap is then
		 * a shifted view of it. */
		int tp = t + c->pad_l + c->pad_r;
		float *xp = get_scratch((size_t)c->in * tp);
		if (!xp)
			return;
		for (i = 0; i < c->in; i++) {
			float *r = xp + (size_t)i * tp;
			memset(r, 0, c->pad_l * sizeof *r);
			copy_act(r + c->pad_l, x + (size_t)i * xld, t, slope);
			memset(r + c->pad_l + t, 0, c->pad_r * sizeof *r);
		}
		for (o = 0; o < c->out; o++) {
			float bv = c->b ? c->b[o] : 0.0f;
			float *yr = y + (size_t)o * yld;
			for (i = 0; i < tout; i++)
				yr[i] = bv;
		}
		if (c->groups == 1) {
			pv_gemm_packed(c->out, tout, c->in, c->k, c->w, xp, tp, c->dil, y, yld);
		} else {
			/* Depthwise (groups == channels): w is [out][k]. */
			for (o = 0; o < c->out; o++) {
				const float *xr = xp + (size_t)o * tp;
				float *yr = y + (size_t)o * yld;
				for (kk = 0; kk < c->k; kk++) {
					float wv = c->w[o * c->k + kk];
					const float *xs = xr + kk * c->dil;
					for (i = 0; i < tout; i++)
						yr[i] += wv * xs[i];
				}
			}
		}
	}
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
