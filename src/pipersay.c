/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
/* PiperSay: speaks text with a Piper voice, and says how long it took.
 *
 *   pipersay [-v VOICE] [-d ESPEAK_DATA] [-o OUT.wav] [-r SPEED] [-s SEED]
 *            [-n] [-i] [-p] [-f FILE] [TEXT...]
 *
 *   -v  the .voice file (default <PiperSay$Dir>.Voices.en_GB-southern_english_female-low)
 *   -d  eSpeak NG's data directory
 *   -o  write the audio as a WAV file
 *   -r  length scale: 1 normal, 1.2 slower, 0.8 faster
 *   -s  random seed (the same seed gives the same audio)
 *   -n  no noise (flat, but repeatable)
 *   -i  print each sentence's phonemes and ids
 *   -p  play it (RISC OS: through SharedSoundBuffer)
 *   -x  phonemes and ids only (no speech)
 *   -f  read the text from a file (UTF-8, or Latin-1)
 *
 * For each sentence it prints the phonemes' count, the audio's length and
 * the time synthesis took; at the end the real-time factor (synthesis time /
 * audio time: under 1 means it can speak as fast as it talks). */
#include "pvoice.h"
#include "pv_espeak.h"
#include "pv_play.h"
#include "pv_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __riscos__
#include <swis.h>
#include <unixlib/local.h>
int __riscosify_control = __RISCOSIFY_NO_PROCESS;
const char *const __dynamic_da_name = "PiperSay Heap";
int __dynamic_da_max_size = 256 * 1024 * 1024;

static double ro_clock(void)
{
	unsigned t = 0;
	_swix(OS_ReadMonotonicTime, _OUT(0), &t);
	return t / 100.0;
}
#define DEFAULT_VOICE "<PiperSay$Dir>.Voices.en_GB-southern_english_female-low"
#define DEFAULT_DATA "<PiperSay$Dir>.espeak-ng-data"
#else
#include <time.h>
static double ro_clock(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}
#define DEFAULT_VOICE "voices/en_GB-southern_english_female-low.voice"
#define DEFAULT_DATA "espeak-ng-data"
#endif

static void put16(FILE *f, int v) { fputc(v & 255, f); fputc((v >> 8) & 255, f); }
static void put32(FILE *f, unsigned v) { put16(f, v & 0xffff); put16(f, v >> 16); }

static int write_wav(const char *path, const short *s, size_t n, int rate)
{
	FILE *f = fopen(path, "wb");
	size_t i;
	if (!f)
		return -1;
	fwrite("RIFF", 1, 4, f); put32(f, (unsigned)(36 + n * 2));
	fwrite("WAVEfmt ", 1, 8, f); put32(f, 16); put16(f, 1); put16(f, 1);
	put32(f, rate); put32(f, rate * 2); put16(f, 2); put16(f, 16);
	fwrite("data", 1, 4, f); put32(f, (unsigned)(n * 2));
	for (i = 0; i < n; i++)
		put16(f, s[i]);
	return fclose(f);
}

int main(int argc, char **argv)
{
	const char *voice_path = DEFAULT_VOICE, *data = DEFAULT_DATA, *out = NULL, *file = NULL;
	char err[300], *text;
	size_t tlen = 0, total = 0, cap = 0;
	short *pcm = NULL;
	pv_voice *v;
	pv_options o;
	pv_sentence *sent;
	int i, ns, nsent, show = 0, play = 0, noise = 1, ids_only = 0;
	double t0, synth = 0, load;
	float speed = 0;
	uint32_t seed = 0;

	pv_clock = ro_clock;
	for (i = 1; i < argc && argv[i][0] == '-' && argv[i][1]; i++) {
		char c = argv[i][1];
		if (strchr("vdorsfP", c) && i + 1 >= argc) {
			fprintf(stderr, "pipersay: -%c needs a value\n", c);
			return 2;
		}
		switch (c) {
		case 'v': voice_path = argv[++i]; break;
		case 'd': data = argv[++i]; break;
		case 'o': out = argv[++i]; break;
		case 'r': speed = (float)atof(argv[++i]); break;
		case 's': seed = (uint32_t)strtoul(argv[++i], NULL, 10); break;
		case 'n': noise = 0; break;
		case 'i': show = 1; break;
		case 'p': play = 1; break;
		case 'x': ids_only = show = 1; break;
		case 'f': file = argv[++i]; break;
		case 'P': pv_prefetch = atoi(argv[++i]); break;
		default:
			fprintf(stderr, "usage: pipersay [-v voice] [-d espeak-data] [-o out.wav] [-r speed] [-s seed] [-n] [-i] [-p] [-P prefetch] text...\n");
			return 2;
		}
	}
	/* The text: the rest of the arguments, joined. */
	text = malloc(1);
	text[0] = 0;
	for (; i < argc; i++) {
		size_t l = strlen(argv[i]);
		text = realloc(text, tlen + l + 2);
		if (tlen)
			text[tlen++] = ' ';
		memcpy(text + tlen, argv[i], l + 1);
		tlen += l;
	}
	if (file) {
		FILE *f = fopen(file, "rb");
		long size;
		if (!f) {
			fprintf(stderr, "pipersay: can't open %s\n", file);
			return 1;
		}
		fseek(f, 0, SEEK_END);
		size = ftell(f);
		fseek(f, 0, SEEK_SET);
		text = realloc(text, tlen + size + 2);
		if (tlen)
			text[tlen++] = ' ';
		tlen += fread(text + tlen, 1, size, f);
		text[tlen] = 0;
		fclose(f);
	}
	if (!tlen) {
		fprintf(stderr, "pipersay: no text\n");
		return 2;
	}

	t0 = ro_clock();
	v = pv_load(voice_path, err, sizeof err);
	if (!v) {
		fprintf(stderr, "pipersay: %s\n", err);
		return 1;
	}
	if (pv_espeak_init(data, pv_espeak_voice(v), err, sizeof err)) {
		fprintf(stderr, "pipersay: %s\n", err);
		return 1;
	}
	load = ro_clock() - t0;
	if (!ids_only)
		printf("Voice %s (%d Hz, eSpeak voice %s) loaded in %.2f s\n", voice_path, pv_sample_rate(v),
		pv_espeak_voice(v), load);

	pv_default_options(v, &o);
	if (speed > 0)
		o.length_scale = speed;
	if (!noise)
		o.noise_scale = o.noise_w = 0;
	o.seed = seed;
	nsent = pv_phonemize(text, &sent);
	if (nsent < 0) {
		fprintf(stderr, "pipersay: phonemes failed\n");
		return 1;
	}
	for (i = 0; i < nsent; i++) {
		int ids[4096], n = pv_phonemes_to_ids(v, sent[i].ph, sent[i].n, ids, 4096), j;
		pv_times tm;
		float *a, peak = 0.01f, gain;
		double t = ro_clock(), dt, secs;
		if (show) {
			printf("  [%.*s]\n  %s\n  ", sent[i].end - sent[i].start, text + sent[i].start,
				pv_phonemes_text(sent[i].ph, sent[i].n));
			for (j = 0; j < n; j++)
				printf("%d ", ids[j]);
			printf("\n");
		}
		if (ids_only)
			continue;
		a = pv_synthesize(v, ids, n, &o, &ns, &tm);
		dt = ro_clock() - t;
		synth += dt;
		if (!a) {
			fprintf(stderr, "pipersay: synthesis failed\n");
			return 1;
		}
		secs = (double)ns / pv_sample_rate(v);
		printf("Sentence %d: %d phonemes, %.2f s of speech in %.2f s (x%.2f) [encoder %.2f, durations %.2f, flow %.2f, decoder %.2f]\n",
			i + 1, n, secs, dt, dt / secs, tm.enc, tm.dp, tm.flow, tm.dec);
		/* To 16 bits, as Piper does: the sentence's peak scaled to full. */
		for (j = 0; j < ns; j++)
			if (fabsf(a[j]) > peak)
				peak = fabsf(a[j]);
		gain = 32767.0f / peak;
		if (total + ns + pv_sample_rate(v) / 5 > cap) {
			cap = (total + ns + pv_sample_rate(v) / 5) * 2;
			pcm = realloc(pcm, cap * sizeof *pcm);
			if (!pcm)
				return 1;
		}
		for (j = 0; j < ns; j++)
			pcm[total++] = (short)lrintf(a[j] * gain * 0.95f);
		/* A short pause between sentences. */
		for (j = 0; j < pv_sample_rate(v) / 5; j++)
			pcm[total++] = 0;
		free(a);
	}
	if (total) {
		double secs = (double)total / pv_sample_rate(v);
		printf("Total: %.2f s of speech in %.2f s: real-time factor %.2f (under 1 keeps up)\n",
			secs, synth, synth / secs);
	}
	if (out && write_wav(out, pcm, total, pv_sample_rate(v)))
		fprintf(stderr, "pipersay: couldn't write %s\n", out);
#ifdef __riscos__
	else if (out)
		_swix(OS_File, _INR(0, 2), 18, out, 0xfb1);   /* WAVE */
#endif
	if (play && total) {
		/* Up-sampled for the sound system (pv_upsample: no tinny edge
		 * from its own rate conversion). */
		int k = pv_upsample_factor(pv_sample_rate(v)), nu = 0;
		float *fl = malloc(total * sizeof *fl), *up = NULL;
		short *pu = NULL;
		size_t i;
		if (fl) {
			for (i = 0; i < total; i++)
				fl[i] = pcm[i];
			up = pv_upsample(fl, (int)total, k, &nu);
			pu = up ? malloc((size_t)nu * sizeof *pu) : NULL;
		}
		if (pu) {
			for (i = 0; i < (size_t)nu; i++)
				pu[i] = (short)(up[i] > 32767 ? 32767 : up[i] < -32768 ? -32768 : lrintf(up[i]));
			if (pv_play(pu, (size_t)nu, pv_sample_rate(v) * k, err, sizeof err))
				fprintf(stderr, "pipersay: %s\n", err);
		} else if (pv_play(pcm, total, pv_sample_rate(v), err, sizeof err)) {
			fprintf(stderr, "pipersay: %s\n", err);
		}
		free(fl);
		free(up);
		free(pu);
	}
	pv_sentences_free(sent, nsent);
	free(pcm);
	free(text);
	pv_free(v);
	return 0;
}
