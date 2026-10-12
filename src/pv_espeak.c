/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
/* Text to phonemes with eSpeak NG (see pv_espeak.h). */
#include "pv_espeak.h"
#include <espeak-ng/speak_lib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CLAUSE_INTONATION_FULL_STOP 0x00000000
#define CLAUSE_INTONATION_COMMA 0x00001000
#define CLAUSE_INTONATION_QUESTION 0x00002000
#define CLAUSE_INTONATION_EXCLAMATION 0x00003000
#define CLAUSE_TYPE_CLAUSE 0x00040000
#define CLAUSE_TYPE_SENTENCE 0x00080000
#define CLAUSE_PERIOD (40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COMMA (20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_QUESTION (40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_EXCLAMATION (45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COLON (30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_SEMICOLON (30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)

int pv_espeak_init(const char *data_dir, const char *voice, char *err, size_t err_size)
{
	if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir, 0) < 0) {
		snprintf(err, err_size, "eSpeak NG couldn't start (data in %s)", data_dir);
		return -1;
	}
	if (espeak_SetVoiceByName(voice) != EE_OK) {
		/* Straight to the file, without eSpeak listing its voices
		 * directory: English voices are in lang.gmw. */
		char file[80];
#ifdef __riscos__
		snprintf(file, sizeof file, "gmw.%s", voice);
#else
		snprintf(file, sizeof file, "gmw/%s", voice);
#endif
		if (espeak_SetVoiceByFile(file) != EE_OK) {
			snprintf(err, err_size, "eSpeak NG has no voice %s", voice);
			return -1;
		}
	}
	return 0;
}

/* Appends UTF-8 s to the sentence as code points, leaving out "(...)". */
static int append(pv_sentence *s, int *cap, const char *p, int *in_flag)
{
	const unsigned char *u = (const unsigned char *)p;
	while (*u) {
		uint32_t c;
		int extra, i;
		if (*u < 0x80) { c = *u; extra = 0; }
		else if ((*u & 0xe0) == 0xc0) { c = *u & 0x1f; extra = 1; }
		else if ((*u & 0xf0) == 0xe0) { c = *u & 0x0f; extra = 2; }
		else if ((*u & 0xf8) == 0xf0) { c = *u & 0x07; extra = 3; }
		else { u++; continue; }
		u++;
		for (i = 0; i < extra && (*u & 0xc0) == 0x80; i++, u++)
			c = c << 6 | (*u & 0x3f);
		if (*in_flag) {
			if (c == ')')
				*in_flag = 0;
			continue;
		}
		if (c == '(') {
			*in_flag = 1;
			continue;
		}
		if (s->n == *cap) {
			int nc = *cap ? *cap * 2 : 256;
			uint32_t *np = realloc(s->ph, nc * sizeof *np);
			if (!np)
				return -1;
			s->ph = np;
			*cap = nc;
		}
		s->ph[s->n++] = c;
	}
	return 0;
}

int pv_phonemize(const char *utf8, pv_sentence **out)
{
	const void *p = utf8;
	pv_sentence *list = NULL;
	int count = 0, cap_list = 0, cap = 0, in_flag = 0, next_start = 0;
	pv_sentence cur;
	memset(&cur, 0, sizeof cur);
	*out = NULL;
	while (p) {
		int term = 0;
		const char *ph, *ts = "";
		ph = espeak_TextToPhonemesWithTerminator(&p, espeakCHARS_AUTO, espeakPHONEMES_IPA, &term);
		if (!cur.n && !cur.ph)
			cur.start = next_start;
		term &= 0x000fffff;
		if (term == CLAUSE_PERIOD) ts = ".";
		else if (term == CLAUSE_QUESTION) ts = "?";
		else if (term == CLAUSE_EXCLAMATION) ts = "!";
		else if (term == CLAUSE_COMMA) ts = ", ";
		else if (term == CLAUSE_COLON) ts = ": ";
		else if (term == CLAUSE_SEMICOLON) ts = "; ";
		if ((ph && append(&cur, &cap, ph, &in_flag)) || append(&cur, &cap, ts, &in_flag))
			goto fail;
		/* eSpeak has read one character of the next clause already. */
		if (p) {
			const char *q = (const char *)p;
			if (q > utf8) {
				q--;
				while (q > utf8 && ((unsigned char)*q & 0xc0) == 0x80)
					q--;
			}
			cur.end = (int)(q - utf8);   /* (p itself stays: eSpeak keeps that character) */
		} else {
			cur.end = (int)strlen(utf8);
		}
		if ((term & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE || !p) {
			if (cur.n) {
				if (count == cap_list) {
					int nc = cap_list ? cap_list * 2 : 16;
					pv_sentence *nl = realloc(list, nc * sizeof *nl);
					if (!nl)
						goto fail;
					list = nl;
					cap_list = nc;
				}
				list[count++] = cur;
				next_start = cur.end;
			} else {
				free(cur.ph);
			}
			if (cur.end > next_start)
				next_start = cur.end;
			memset(&cur, 0, sizeof cur);
			cap = 0;
			in_flag = 0;
		}
	}
	*out = list;
	return count;
fail:
	free(cur.ph);
	pv_sentences_free(list, count);
	return -1;
}

void pv_sentences_free(pv_sentence *s, int count)
{
	int i;
	for (i = 0; i < count; i++)
		free(s[i].ph);
	free(s);
}

const char *pv_phonemes_text(const uint32_t *ph, int n)
{
	static char buf[4096];
	int i, k = 0;
	for (i = 0; i < n && k < (int)sizeof buf - 5; i++) {
		uint32_t c = ph[i];
		if (c < 0x80) buf[k++] = (char)c;
		else if (c < 0x800) { buf[k++] = (char)(0xc0 | c >> 6); buf[k++] = (char)(0x80 | (c & 63)); }
		else if (c < 0x10000) { buf[k++] = (char)(0xe0 | c >> 12); buf[k++] = (char)(0x80 | (c >> 6 & 63)); buf[k++] = (char)(0x80 | (c & 63)); }
		else { buf[k++] = (char)(0xf0 | c >> 18); buf[k++] = (char)(0x80 | (c >> 12 & 63)); buf[k++] = (char)(0x80 | (c >> 6 & 63)); buf[k++] = (char)(0x80 | (c & 63)); }
	}
	buf[k] = 0;
	return buf;
}
