#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Andrew Youll
# Fetches eSpeak NG at the commit Piper uses, applies our patches, builds it
# for this machine (to compile its data), and puts the English data Piper's
# en-GB voices need in build/espeak-ng-data.
#
#   tools/build-espeak.sh            (from the top of the repository)
#
# The library itself is built for ARM Linux and RISC OS by the Makefile,
# from third/espeak-ng/src.
set -eu
COMMIT=212928b394a96e8fd2096616bfd54e17845c48f6   # 2025-03-22, as libpiper pins
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=$HERE/third/espeak-ng
if [ ! -d "$SRC/.git" ]; then
	mkdir -p "$HERE/third"
	git clone -q https://github.com/espeak-ng/espeak-ng.git "$SRC"
	(cd "$SRC" && git checkout -q $COMMIT &&
		git -c user.name=build -c user.email=build@localhost am -q "$HERE"/patches/espeak-ng/*.patch)
fi
B=$HERE/build/espeak-host
mkdir -p "$B"
(cd "$B" && cmake "$SRC" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DUSE_ASYNC=OFF \
	-DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_KLATT=OFF \
	-DUSE_SPEECHPLAYER=OFF -DENABLE_TESTS=OFF >"$B/cmake.log" 2>&1 && make -j2 >"$B/make.log" 2>&1)
# Only what English needs: phonemes, intonation, the English dictionary and
# the English voices.
D=$HERE/build/espeak-ng-data
rm -rf "$D"
mkdir -p "$D/lang/gmw"
for f in phontab phonindex phondata intonations en_dict; do
	cp "$B/espeak-ng-data/$f" "$D/"
done
cp "$B"/espeak-ng-data/lang/gmw/en* "$D/lang/gmw/"
# The library's config.h (the same for every target).
mkdir -p "$HERE/build/espeak-config"
cp "$B/src/libespeak-ng/include/config.h" "$HERE/build/espeak-config/config.h"
echo "eSpeak NG $COMMIT: data in $D ($(du -sk "$D" | cut -f1) KB)"
