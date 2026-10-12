#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Andrew Youll
# Fetches Piper's two English (UK) "low" voices from the Piper project's
# GitHub release v0.0.2 and converts them to .voice files in voices/:
#   en_GB-southern_english_female-low   dataset: OpenSLR SLR83, CC BY-SA 4.0
#   en_GB-alan-low                      dataset: Mycroft "apope" (all rights
#                                       reserved: for the tests only, not
#                                       to be redistributed)
# and en_GB-alan-low-f32 (float32, for the tests' exact comparison).
# Needs python3 with onnx and numpy.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
URL=https://github.com/rhasspy/piper/releases/download/v0.0.2
T=$(mktemp -d)
mkdir -p "$HERE/voices"
for v in southern_english_female alan; do
	curl -sfL -o "$T/$v.tgz" "$URL/voice-en-gb-$v-low.tar.gz"
	mkdir -p "$T/$v"
	tar xzf "$T/$v.tgz" -C "$T/$v"
	python3 -I "$HERE/tools/onnx2voice.py" "$T/$v/en-gb-$v-low.onnx" "$HERE/voices/en_GB-$v-low.voice"
done
python3 -I "$HERE/tools/onnx2voice.py" "$T/alan/en-gb-alan-low.onnx" "$HERE/voices/en_GB-alan-low-f32.voice" --f32
rm -rf "$T"
