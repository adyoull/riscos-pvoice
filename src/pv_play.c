/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Andrew Youll */
/* Plays 16-bit mono samples to the end (blocking), through SharedSoundBuffer
 * and StreamManager (John Duffell's modules, which mix into SharedSound),
 * the way riscos-midisynth's output_ssb.c does. The modules are loaded by
 * !PiperSay.!Run (SharedSound, StreamManager, then SharedSoundBuffer). */
#include "pv_play.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __riscos__
#include <kernel.h>
#include <swis.h>

#define XSharedSoundBuffer_OpenStream         (0x20000 | 0x55FC0)
#define XSharedSoundBuffer_CloseStream        (0x20000 | 0x55FC1)
#define XSharedSoundBuffer_Volume             (0x20000 | 0x55FC4)
#define XSharedSoundBuffer_SampleRate         (0x20000 | 0x55FC5)
#define XSharedSoundBuffer_Pause              (0x20000 | 0x55FC9)
#define XSharedSoundBuffer_ReturnStreamHandle (0x20000 | 0x55FCE)
#define XStreamManager_AddBlock               (0x20000 | 0x57282)
#define XStreamManager_SetBuffer              (0x20000 | 0x57287)
#define XStreamManager_BufferStats            (0x20000 | 0x57288)

#define BLOCK_FRAMES 1024
#define BLOCK_BYTES (BLOCK_FRAMES * 4)

static int queued(int stream)
{
	_kernel_swi_regs r;
	r.r[0] = stream;
	if (_kernel_swi(XStreamManager_BufferStats, &r, &r))
		return -1;
	return r.r[0] - r.r[1];
}

int pv_play(const short *s, size_t n, int rate, char *err, size_t err_size)
{
	_kernel_swi_regs r;
	_kernel_oserror *e;
	short block[BLOCK_FRAMES * 2];
	int ssb, stream, target = rate * 4 / 5, started = 0;   /* keep 0.2 s queued */
	size_t pos = 0;
	unsigned now = 0, since = 0;
	int last_q = -1;
	r.r[0] = 2;                     /* R2 holds the block size */
	r.r[1] = (int)"PiperSay";
	r.r[2] = BLOCK_BYTES;
	if ((e = _kernel_swi(XSharedSoundBuffer_OpenStream, &r, &r)) != NULL) {
		snprintf(err, err_size, "can't play: %s (SharedSound, StreamManager and SharedSoundBuffer "
			"are needed: run !PiperSay to load them)", e->errmess);
		return -1;
	}
	ssb = r.r[0];
	r.r[0] = ssb;
	if ((e = _kernel_swi(XSharedSoundBuffer_ReturnStreamHandle, &r, &r)) != NULL) {
		snprintf(err, err_size, "can't play (%s)", e->errmess);
		r.r[0] = ssb;
		_kernel_swi(XSharedSoundBuffer_CloseStream, &r, &r);
		return -1;
	}
	stream = r.r[0];
	if (target < BLOCK_BYTES * 2)
		target = BLOCK_BYTES * 2;
	r.r[0] = stream; r.r[1] = target * 4;
	_kernel_swi(XStreamManager_SetBuffer, &r, &r);
	r.r[0] = ssb; r.r[1] = rate * 1024;
	_kernel_swi(XSharedSoundBuffer_SampleRate, &r, &r);
	r.r[0] = ssb; r.r[1] = (int)0xffffffffu;
	_kernel_swi(XSharedSoundBuffer_Volume, &r, &r);
	r.r[0] = ssb; r.r[1] = 0;      /* paused until two blocks are queued */
	_kernel_swi(XSharedSoundBuffer_Pause, &r, &r);
	while (pos < n) {
		int q = queued(stream), i;
		if (q < 0)
			break;
		if (q >= target) {
			if (!started) {            /* full before two blocks? start anyway */
				r.r[0] = ssb; r.r[1] = 1;
				_kernel_swi(XSharedSoundBuffer_Pause, &r, &r);
				started = 1;
			}
			/* Busy-wait (a command-line test), but give up if nothing plays. */
			_swix(OS_ReadMonotonicTime, _OUT(0), &now);
			if (q != last_q) {
				last_q = q;
				since = now;
			} else if (now - since > 300) {
				snprintf(err, err_size, "the sound stopped playing");
				break;
			}
			continue;
		}
		for (i = 0; i < BLOCK_FRAMES; i++) {
			short v = pos < n ? s[pos++] : 0;
			block[2 * i] = block[2 * i + 1] = v;
		}
		r.r[0] = stream; r.r[1] = (int)block; r.r[2] = BLOCK_BYTES;
		if (_kernel_swi(XStreamManager_AddBlock, &r, &r))
			break;
		if (!started && queued(stream) >= BLOCK_BYTES * 2) {
			r.r[0] = ssb; r.r[1] = 1;
			_kernel_swi(XSharedSoundBuffer_Pause, &r, &r);
			started = 1;
		}
	}
	if (!started) {
		r.r[0] = ssb; r.r[1] = 1;
		_kernel_swi(XSharedSoundBuffer_Pause, &r, &r);
	}
	{
		/* Until it's played (with a limit, should the stream stall). */
		unsigned t0, t;
		_swix(OS_ReadMonotonicTime, _OUT(0), &t0);
		do
			_swix(OS_ReadMonotonicTime, _OUT(0), &t);
		while (queued(stream) > 0 && t - t0 < 300);
	}
	r.r[0] = ssb;
	_kernel_swi(XSharedSoundBuffer_CloseStream, &r, &r);
	return 0;
}

#else

int pv_play(const short *s, size_t n, int rate, char *err, size_t err_size)
{
	(void)s; (void)n; (void)rate;
	snprintf(err, err_size, "playing is only built for RISC OS; use -o to write a WAV file");
	return -1;
}

#endif
