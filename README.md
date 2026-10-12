# pvoice: Piper's neural voices for RISC OS

[Piper](https://github.com/OHF-Voice/piper1-gpl)'s text-to-speech voices,
run by a small engine in plain C, so they work on RISC OS (and anything else
with a C compiler). Text becomes phonemes through
[eSpeak NG](https://github.com/espeak-ng/espeak-ng), as in Piper itself.
Status: **test** (0.0.1-test5).

## How it works

A Piper voice is a VITS model exported to ONNX. Rather than port ONNX
Runtime (a large C++ library), `src/pvoice.c` runs the same network
directly, as defined in Piper's training code (`piper/train/vits`):

1. text encoder: phoneme embeddings, then a transformer with relative
   position attention;
2. stochastic duration predictor, run backwards from noise through
   rational quadratic spline flows;
3. each phoneme's frames, then the flow (residual coupling layers)
   backwards;
4. HiFi-GAN decoder to audio.

Almost all the time goes in convolutions (`src/pv_math.c`): about 2.1
billion multiply-adds per second of speech. Each is one matrix product
over all its taps. The weights are packed at load into panels of four
output channels; the input is packed as it's used, a block at a time, into
strips 12 columns wide, with the leaky ReLU and the zero padding applied on
the way. One strip stays in the L1 cache while the weights stream past it,
and each 4x12 tile of results stays in NEON registers until it's finished,
starting from the bias and the residual signal, so a layer is one pass
over its input and one over its output. Transposed convolutions (the
decoder's up-sampling) are done by phase. The decoder's resblock means are
folded into the next layer's weights. `make bench` runs one sentence under
qemu; measured with a cache model of the Cortex-A72 (32 KB L1, 1 MB L2),
0.0.1-test5 misses the L1 cache 42% less and the L2 cache 42% less than
test4.

`tools/onnx2voice.py` converts a Piper `.onnx` voice (and its `.json`) into
a `.voice` file: the weights in half precision (31 MB; the difference from
float32 is inaudible: the spectrum differs by 0.04 dB on average), the
hyperparameters, the phoneme map and the eSpeak voice.

## Checked against Piper

- With the noise turned off, every stage (encoder, durations, flow output,
  audio) matches ONNX Runtime running the original model: the audio is
  within 0.03% (`tests/pv_test.c`, references from `tests/make_ref.py`).
- Text to phoneme ids matches Python Piper exactly for the test texts
  (numbers, money, abbreviations, quotes, accented words:
  `tests/compare-ids.sh`).
- Both run on ARM Linux under `qemu-arm-aligntrap` (RISC OS's alignment
  rules) as well as on the host.

## PiperSay

A command-line program that speaks text, writes a WAV file, plays it (on
RISC OS through SharedSoundBuffer) and reports how long synthesis took:

    PiperSay -p Hello from RISC OS.
    PiperSay -f Story -o Story/wav

`!PiperSay` (from `make zip`) holds it with the Southern English Female
voice and eSpeak's English data; `Obey <PiperSay$Dir>.Test` in a TaskWindow
times it on the machine.

## Building

Needs the [riscos-crossdev](https://github.com/adyoull/riscos-crossdev) 1.4
toolchain and a riscos-crossdev checkout (set `RISCOS_TOOLCHAIN` and
`RCD_DIR` in `config.mk`), `gcc-arm-linux-gnueabihf`, `qemu-user`, `cmake`,
and python3 with `onnx` and `numpy` (and `onnxruntime` and `piper-tts` to
remake the test references).

    tools/build-espeak.sh    # eSpeak NG at Piper's commit, patched; English data
    tools/get-voices.sh      # the en-GB voice, converted
    make && make test        # host and ARM Linux builds, tests
    make zip                 # build/PiperSay-<version>.zip for RISC OS

## Licences

The engine and PiperSay are **GPL-3.0-or-later** (they include eSpeak NG,
GPL-3.0-or-later): see `COPYING`. The build scripts, Makefile, tests and
tools are MIT (`LICENSE-MIT`). Each voice has its own licence: the
packaged Southern English Female voice is CC BY-SA 4.0 (`docs/VOICES.txt`);
the tests' references in `tests/ref` were made with it.
