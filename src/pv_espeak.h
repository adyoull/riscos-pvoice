/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
/* Text to phonemes with eSpeak NG, as Piper does it (piper/phonemize_espeak.py):
 * clause by clause, IPA, the clause's punctuation kept, language-switch
 * flags "(xx)" dropped, split into sentences. */
#ifndef PV_ESPEAK_H
#define PV_ESPEAK_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
	uint32_t *ph;       /* phonemes (code points) */
	int n;
	int start, end;     /* the sentence's bytes in the text */
} pv_sentence;

/* data_dir: eSpeak NG's data (phontab, en_dict, lang...); voice: as named in
 * the voice file ("en-gb"). 0, or -1 with a message. Once per program. */
int pv_espeak_init(const char *data_dir, const char *voice, char *err, size_t err_size);
/* UTF-8 text to sentences (malloc'd; free with pv_sentences_free). Returns
 * the number of sentences, or -1. */
int pv_phonemize(const char *utf8, pv_sentence **out);
void pv_sentences_free(pv_sentence *s, int count);
/* The phonemes as UTF-8, for logs (static buffer). */
const char *pv_phonemes_text(const uint32_t *ph, int n);

#endif
