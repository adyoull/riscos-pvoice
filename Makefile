# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Andrew Youll
# pvoice: Piper voices in plain C, for RISC OS.
#
#   tools/build-espeak.sh   once: eSpeak NG's source (patched) and English data
#   make                    host, ARM Linux and RISC OS builds of the engine,
#                           its test and PiperSay
#   make test               the tests: host, then ARM Linux under
#                           qemu-arm-aligntrap (RISC OS's alignment rules)
#   make zip                PiperSay for RISC OS with a voice and eSpeak's
#                           data: build/PiperSay-$(VERSION).zip
#
# RCD_DIR is a riscos-crossdev checkout; RISCOS_TOOLCHAIN the riscos-crossdev
# 1.4 toolchain. Set them in config.mk (not in git) or on the command line.

-include config.mk
RCD_DIR ?= ../riscos-crossdev
RISCOS_TOOLCHAIN ?= /opt/riscos-crossdev-toolchain-1.4-x86_64-linux
VERSION ?= 0.0.1-test6
# The voice packaged (its dataset is CC BY-SA 4.0), and the one the tests'
# references were made with (tools/get-voices.sh fetches and converts both).
VOICE ?= voices/en_GB-southern_english_female-low.voice
TEST_VOICE ?= voices/en_GB-southern_english_female-low.voice
TEST_VOICE32 ?= voices/en_GB-southern_english_female-low-f32.voice

RO_CC := $(RISCOS_TOOLCHAIN)/bin/arm-riscos-gnueabihf-gcc
AL_CC := arm-linux-gnueabihf-gcc
HOST_CC ?= gcc
# NEON (every ARMv7 machine RISC OS 5 runs on has it); tuned for the Pi 4.
RO_CFLAGS := -O2 -mcpu=cortex-a8 -mtune=cortex-a72 -mfpu=neon -mfloat-abi=hard -fstack-clash-protection
AL_CFLAGS := -O2 -g -marm -march=armv7-a -mfpu=neon -mfloat-abi=hard -mtune=cortex-a72 \
	-mno-unaligned-access -fstack-clash-protection
HOST_CFLAGS := -O2 -g
WARN := -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -std=gnu99

ES := third/espeak-ng
ES_SRC := $(addprefix $(ES)/src/libespeak-ng/, common.c mnemonics.c error.c ieee80.c \
	compiledata.c compiledict.c dictionary.c encoding.c intonation.c langopts.c numbers.c \
	phoneme.c phonemelist.c readclause.c setlengths.c soundicon.c spect.c ssml.c synthdata.c \
	synthesize.c tr_languages.c translate.c translateword.c voices.c wavegen.c speech.c \
	espeak_api.c) \
	$(addprefix $(ES)/src/ucd-tools/src/, case.c categories.c ctype.c proplist.c scripts.c tostring.c)
ES_INC := -I$(ES)/src/include -I$(ES)/src/include/compat -I$(ES)/src/ucd-tools/src/include \
	-Ibuild/espeak-config -DLIBESPEAK_NG_EXPORT=1
ES_FLAGS := -fno-exceptions -fwrapv -w

PV_SRC := src/pvoice.c src/pv_math.c
PV_HDR := src/pvoice.h src/pv_math.h
SAY_SRC := src/pipersay.c src/pv_espeak.c src/pv_play.c
SAY_HDR := src/pv_espeak.h src/pv_play.h

# For linking into an application (Folio): engine, eSpeak wrapper and eSpeak.
build/ro/libpvoice.a: $(PV_SRC) $(PV_HDR) src/pv_espeak.c src/pv_espeak.h build/ro/libespeak.a
	@mkdir -p build/ro/pv
	$(RO_CC) $(RO_CFLAGS) $(WARN) -Isrc -I$(ES)/src/include -c src/pvoice.c -o build/ro/pv/pvoice.o
	$(RO_CC) $(RO_CFLAGS) $(WARN) -Isrc -I$(ES)/src/include -c src/pv_math.c -o build/ro/pv/pv_math.o
	$(RO_CC) $(RO_CFLAGS) $(WARN) -Isrc -I$(ES)/src/include -c src/pv_espeak.c -o build/ro/pv/pv_espeak.o
	cp build/ro/libespeak.a $@
	$(RISCOS_TOOLCHAIN)/bin/arm-riscos-gnueabihf-ar rs $@ build/ro/pv/pvoice.o build/ro/pv/pv_math.o build/ro/pv/pv_espeak.o
build/al/libpvoice.a: $(PV_SRC) $(PV_HDR) src/pv_espeak.c src/pv_espeak.h build/al/libespeak.a
	@mkdir -p build/al/pv
	$(AL_CC) $(AL_CFLAGS) $(WARN) -Isrc -I$(ES)/src/include -c src/pvoice.c -o build/al/pv/pvoice.o
	$(AL_CC) $(AL_CFLAGS) $(WARN) -Isrc -I$(ES)/src/include -c src/pv_math.c -o build/al/pv/pv_math.o
	$(AL_CC) $(AL_CFLAGS) $(WARN) -Isrc -I$(ES)/src/include -c src/pv_espeak.c -o build/al/pv/pv_espeak.o
	cp build/al/libespeak.a $@
	arm-linux-gnueabihf-ar rs $@ build/al/pv/pvoice.o build/al/pv/pv_math.o build/al/pv/pv_espeak.o

all: build/ro/libpvoice.a build/al/libpvoice.a build/host/pv_test build/host/pipersay build/al/pv_test build/al/pipersay build/ro/pipersay.elf

# eSpeak NG as a library, per target.
define espeak_lib
build/$(1)/espeak/%.o: $(ES)/src/%.c
	@mkdir -p $$(@D)
	$(2) $(3) $(ES_FLAGS) $(ES_INC) -c $$< -o $$@
build/$(1)/libespeak.a: $(ES_SRC:$(ES)/src/%.c=build/$(1)/espeak/%.o)
	rm -f $$@; $(4) rcs $$@ $$^
endef
$(eval $(call espeak_lib,host,$(HOST_CC),$(HOST_CFLAGS),ar))
$(eval $(call espeak_lib,al,$(AL_CC),$(AL_CFLAGS),arm-linux-gnueabihf-ar))
$(eval $(call espeak_lib,ro,$(RO_CC),$(RO_CFLAGS),$(RISCOS_TOOLCHAIN)/bin/arm-riscos-gnueabihf-ar))

build/host/pv_test: tests/pv_test.c $(PV_SRC) $(PV_HDR)
	@mkdir -p $(@D)
	$(HOST_CC) $(HOST_CFLAGS) $(WARN) -Isrc tests/pv_test.c $(PV_SRC) -lm -o $@
build/host/pipersay: $(SAY_SRC) $(SAY_HDR) $(PV_SRC) $(PV_HDR) build/host/libespeak.a
	$(HOST_CC) $(HOST_CFLAGS) $(WARN) -Isrc -I$(ES)/src/include $(SAY_SRC) $(PV_SRC) build/host/libespeak.a -lm -o $@

build/al/pv_test: tests/pv_test.c $(PV_SRC) $(PV_HDR)
	@mkdir -p $(@D)
	$(AL_CC) $(AL_CFLAGS) $(WARN) -no-pie -Isrc tests/pv_test.c $(PV_SRC) -lm -o $@
build/al/pipersay: $(SAY_SRC) $(SAY_HDR) $(PV_SRC) $(PV_HDR) build/al/libespeak.a
	$(AL_CC) $(AL_CFLAGS) $(WARN) -no-pie -Isrc -I$(ES)/src/include $(SAY_SRC) $(PV_SRC) build/al/libespeak.a -lm -o $@

# One synthesis, for counting instructions under qemu or timing on a board.
build/al/pv_bench: tests/pv_bench.c $(PV_SRC) $(PV_HDR)
	@mkdir -p $(@D)
	$(AL_CC) $(AL_CFLAGS) $(WARN) -no-pie -Isrc tests/pv_bench.c $(PV_SRC) -lm -o $@
bench: build/al/pv_bench
	time $(QEMU) build/al/pv_bench $(TEST_VOICE32) tests/ref

build/ro/pipersay.elf: $(SAY_SRC) $(SAY_HDR) $(PV_SRC) $(PV_HDR) build/ro/libespeak.a
	$(RO_CC) $(RO_CFLAGS) $(WARN) -static -Isrc -I$(ES)/src/include $(SAY_SRC) $(PV_SRC) build/ro/libespeak.a -lm -o $@

QEMU := PATH=$(RISCOS_TOOLCHAIN)/bin:$$PATH $(RCD_DIR)/qemu/aligntrap.sh
ESDATA := build/espeak-ng-data

# Strict against ONNX Runtime with the float32 voice; looser with the
# float16 one (half-precision weights: inaudible, but not bit-close).
test: build/host/pv_test build/al/pv_test build/host/pipersay build/al/pipersay
	build/host/pv_test $(TEST_VOICE32) tests/ref
	$(QEMU) build/al/pv_test $(TEST_VOICE32) tests/ref
	PV_TOL=f16 build/host/pv_test $(TEST_VOICE) tests/ref
	tests/compare-ids.sh build/host/pipersay $(TEST_VOICE) $(ESDATA)
	PATH=$(RISCOS_TOOLCHAIN)/bin:$$PATH tests/compare-ids.sh "$(RCD_DIR)/qemu/aligntrap.sh build/al/pipersay" $(TEST_VOICE) $(ESDATA)

check: build/ro/pipersay.elf
	RISCOS_TOOLCHAIN=$(RISCOS_TOOLCHAIN) $(RCD_DIR)/checks/check-all.sh build/ro/pipersay.elf

# !PiperSay: the program, a voice, eSpeak's English data, and a test script.
zip: build/ro/pipersay.elf
	RISCOS_TOOLCHAIN=$(RISCOS_TOOLCHAIN) $(RCD_DIR)/checks/check-all.sh build/ro/pipersay.elf
	rm -rf build/app && mkdir -p build/app
	cp -r app/!PiperSay build/app/
	$(RISCOS_TOOLCHAIN)/bin/arm-riscos-gnueabihf-strip -o build/ro/pipersay.stripped build/ro/pipersay.elf
	$(RISCOS_TOOLCHAIN)/bin/elf2aif -e build/ro/pipersay.stripped 'build/app/!PiperSay/PiperSay,ff8'
	mkdir -p 'build/app/!PiperSay/Voices'
	cp $(VOICE) 'build/app/!PiperSay/Voices/'"$$(basename $(VOICE) .voice)"',ffd'
	cp -r $(ESDATA) 'build/app/!PiperSay/espeak-ng-data'
	for f in phontab phonindex phondata intonations en_dict; do \
		mv 'build/app/!PiperSay/espeak-ng-data/'$$f 'build/app/!PiperSay/espeak-ng-data/'$$f',ffd'; done
	for f in 'build/app/!PiperSay/espeak-ng-data/lang/gmw/'*; do mv "$$f" "$$f,fff"; done
	cp COPYING 'build/app/!PiperSay/Licence,fff'
	cp docs/VOICES.txt 'build/app/!PiperSay/VoiceLicence,fff'
	python3 $(RCD_DIR)/package/rozip.py build/PiperSay-$(VERSION).zip -C build/app '!PiperSay'
	@ls -l build/PiperSay-$(VERSION).zip

clean:
	rm -rf build/host build/al build/ro build/app

.PHONY: all test bench check zip clean
