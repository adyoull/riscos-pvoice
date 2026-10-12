/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
/* pvoice's arithmetic. Every convolution becomes matrix products (one per
 * kernel tap: the taps' inputs are shifted views of the padded signal), so
 * nearly all the time goes in pv_gemm, which has a NEON inner kernel on ARM
 * and plain C (that compilers vectorise) elsewhere. */
#include "pv_math.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define PV_NEON 1
#endif

/* ---- Matrix product ------------------------------------------------------ */

/* The work is tiled so it stays in cache: NB columns of the output at a
 * time (all its rows, every tap), K in blocks of KB. Inside, 4 rows by 8
 * columns are accumulated in registers (8 NEON q registers on ARM). */
#define KB 128
#define NB 96

/* C[i][j] += A[i][p] * B[p][j + off] over p, for rows m0..m1, columns
 * n0..n1, plainly. */
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

/* One block: rows 0..m, columns n0..n1 (n1 - n0 <= NB), k0..k1. */
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

/* ---- Convolutions -------------------------------------------------------- */

int pv_conv_out_len(const pv_conv *c, int t)
{
	if (c->transposed)
		return (t - 1) * c->stride - c->pad_l - c->pad_r + c->k;
	return (t + c->pad_l + c->pad_r - c->dil * (c->k - 1) - 1) / c->stride + 1;
}

void pv_conv_run(const pv_conv *c, const float *x, int t, int xld, float *y, int yld)
{
	int tout = pv_conv_out_len(c, t), o, i, kk;
	if (c->transposed) {
		/* y[o][s*t' - pad_l + kk] += sum_i W[kk][o][i] x[i][t'], one tap at a time. */
		float *tmp = malloc((size_t)c->out * t * sizeof *tmp);
		for (o = 0; o < c->out; o++) {
			float bv = c->b ? c->b[o] : 0.0f;
			for (i = 0; i < tout; i++)
				y[o * yld + i] = bv;
		}
		if (!tmp)
			return;
		for (kk = 0; kk < c->k; kk++) {
			int tt;
			memset(tmp, 0, (size_t)c->out * t * sizeof *tmp);
			pv_gemm(c->out, t, c->in, c->w + (size_t)kk * c->out * c->in, c->in, x, xld, tmp, t);
			for (o = 0; o < c->out; o++) {
				float *yr = y + o * yld;
				const float *tr = tmp + o * t;
				for (tt = 0; tt < t; tt++) {
					int pos = tt * c->stride - c->pad_l + kk;
					if (pos >= 0 && pos < tout)
						yr[pos] += tr[tt];
				}
			}
		}
		free(tmp);
		return;
	}
	{
		/* Pad the input once; each tap is then a shifted view of it. */
		int tp = t + c->pad_l + c->pad_r;
		float *xp = malloc((size_t)c->in * tp * sizeof *xp);
		if (!xp)
			return;
		for (i = 0; i < c->in; i++) {
			float *r = xp + (size_t)i * tp;
			memset(r, 0, c->pad_l * sizeof *r);
			memcpy(r + c->pad_l, x + (size_t)i * xld, t * sizeof *r);
			memset(r + c->pad_l + t, 0, c->pad_r * sizeof *r);
		}
		for (o = 0; o < c->out; o++) {
			float bv = c->b ? c->b[o] : 0.0f;
			float *yr = y + (size_t)o * yld;
			for (i = 0; i < tout; i++)
				yr[i] = bv;
		}
		if (c->groups == 1) {
			pv_gemm_taps(c->out, tout, c->in, c->k, c->w, c->in, xp, tp, c->dil, y, yld);
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
		free(xp);
	}
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
	size_t i;
	for (i = 0; i < n; i++)
		if (x[i] < 0)
			x[i] *= slope;
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
