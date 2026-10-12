/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
#ifndef PV_PLAY_H
#define PV_PLAY_H

#include <stddef.h>

/* Plays 16-bit mono samples at rate Hz, returning when they've played.
 * 0, or -1 with a message. RISC OS only (SharedSoundBuffer). */
int pv_play(const short *s, size_t n, int rate, char *err, size_t err_size);

#endif
