#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Andrew Youll
# Fetches Piper's English (UK) "low" voice Southern English Female (dataset
# OpenSLR SLR83, CC BY-SA 4.0) from the Piper project's GitHub release v0.0.2
# and converts it to .voice files in voices/: en_GB-southern_english_female-low
# (half precision, the one packaged) and ...-f32 (float32, for the tests'
# exact comparison with ONNX Runtime).
# Needs python3 with onnx and numpy.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
URL=https://github.com/rhasspy/piper/releases/download/v0.0.2
T=$(mktemp -d)
mkdir -p "$HERE/voices"
for v in southern_english_female; do
	curl -sfL -o "$T/$v.tgz" "$URL/voice-en-gb-$v-low.tar.gz"
	mkdir -p "$T/$v"
	tar xzf "$T/$v.tgz" -C "$T/$v"
	python3 -I "$HERE/tools/onnx2voice.py" "$T/$v/en-gb-$v-low.onnx" "$HERE/voices/en_GB-$v-low.voice"
done
python3 -I "$HERE/tools/onnx2voice.py" "$T/southern_english_female/en-gb-southern_english_female-low.onnx" \
	"$HERE/voices/en_GB-southern_english_female-low-f32.voice" --f32
rm -rf "$T"
