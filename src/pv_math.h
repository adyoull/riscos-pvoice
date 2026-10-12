/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
/* pvoice's arithmetic: matrices are row-major floats; signals are
 * [channels][time] with each channel's row 'ld' floats apart. */
#ifndef PV_MATH_H
#define PV_MATH_H

#include <stddef.h>

/* C[m][n] += A[m][k] * B[k][n] (lda, ldb, ldc: row strides in floats). */
void pv_gemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb,
		float *c, int ldc);

/* The same summed over taps: C[i][j] += sum_t A_t[i][p] * B[p][j + t * tap_step],
 * where A_t = a + t * m * lda (a convolution's taps). */
void pv_gemm_taps(int m, int n, int k, int taps, const float *a, int lda, const float *b, int ldb,
		int tap_step, float *c, int ldc);

typedef struct {
	int out, in, k;           /* channels and kernel width */
	int stride, pad_l, pad_r, dil, groups, transposed;
	float *w;                 /* packed: [k][out][in] (depthwise: [out][k]) */
	float *b;                 /* out, or NULL */
} pv_conv;

/* Output length for an input of t. */
int pv_conv_out_len(const pv_conv *c, int t);
/* y[out][tout] = conv(x[in][t]); x rows xld apart, y rows yld apart (yld >=
 * tout). Uses a scratch buffer of its own. */
void pv_conv_run(const pv_conv *c, const float *x, int t, int xld, float *y, int yld);

/* LayerNorm over the channels at each time step, in place. */
void pv_layer_norm(float *x, int ch, int t, int ld, const float *gamma, const float *beta);

void pv_leaky_relu(float *x, size_t n, float slope);
void pv_gelu(float *x, size_t n);
void pv_relu(float *x, size_t n);

#endif
