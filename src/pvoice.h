/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
/* pvoice: Piper voices (VITS) in plain C, for RISC OS and anything else.
 *
 * A voice is a .voice file made from a Piper .onnx model by
 * tools/onnx2voice.py. Text becomes phonemes through eSpeak NG (pvoice_say.c
 * does that part); this file's calls turn phoneme ids into audio.
 *
 * Not thread-safe per voice: one synthesis at a time on a pv_voice. */
#ifndef PVOICE_H
#define PVOICE_H

#include <stddef.h>
#include <stdint.h>

typedef struct pv_voice pv_voice;

typedef struct {
	float noise_scale;    /* variation in the voice (Piper's default 0.667) */
	float length_scale;   /* > 1 slower, < 1 faster */
	float noise_w;        /* variation in timing (0.8) */
	int speaker;          /* multi-speaker voices */
	uint32_t seed;        /* random numbers; 0 picks one */
	volatile int *stop;   /* if not NULL: synthesis gives up (returns NULL) soon
	                       * after another thread sets *stop */
} pv_options;

/* Loads a .voice file. NULL on failure, with a message in err. */
pv_voice *pv_load(const char *path, char *err, size_t err_size);
void pv_free(pv_voice *v);

int pv_sample_rate(const pv_voice *v);
const char *pv_espeak_voice(const pv_voice *v);
/* The voice's own defaults. */
void pv_default_options(const pv_voice *v, pv_options *o);

/* Phonemes (Unicode code points, as eSpeak NG writes IPA) to the ids the
 * model wants: BOS, each known phoneme followed by PAD, EOS. Returns the
 * number of ids (up to max). */
int pv_phonemes_to_ids(const pv_voice *v, const uint32_t *phonemes, int n, int *ids, int max);

/* Speaks one sentence's ids: returns malloc'd samples in -1..1 (free them)
 * and their number, or NULL. Timing of the stages goes to *times if not
 * NULL (centiseconds or whatever pv_clock counts: encoder, durations, flow,
 * decoder). */
typedef struct { double enc, dp, flow, dec; } pv_times;
float *pv_synthesize(pv_voice *v, const int *ids, int n, const pv_options *o, int *n_samples,
		pv_times *times);

/* For the tests: the stages' outputs from the last pv_synthesize (owned by
 * the voice, valid until the next call). */
typedef struct {
	const float *enc_m, *enc_logs;   /* inter x n */
	const float *logw;               /* n */
	const float *z;                  /* inter x frames */
	int n, frames;
} pv_debug;
void pv_last_debug(const pv_voice *v, pv_debug *d);

/* A clock in seconds for the timings (the caller may replace it). */
extern double (*pv_clock)(void);

#endif
