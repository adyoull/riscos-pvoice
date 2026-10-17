#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Andrew Youll
# Fetches Piper's English (UK) "low" voice Southern English Female (dataset
# OpenSLR SLR83, CC BY-SA 4.0) from the Piper project's GitHub release v0.0.2
# and converts it to .voice files in voices/: en_GB-southern_english_female-low
# (half precision, the one packaged) and ...-f32 (float32, for the tests'
# exact comparison with ONNX Runtime).
#
# The download and both results are checked by sha256, so every build has
# the same voice. Voices already made (and right) are left alone.
# Needs python3 with pip: onnx 1.18.0 and numpy 2.1.3 (which make exactly
# these files) are installed into build/pydeps, and the conversion uses only
# those (python3 -S), whatever else the machine has.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
URL=https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-gb-southern_english_female-low.tar.gz
TGZ_SHA=7c1bbe23e61a57bdb450b137f69a83ff5358159262e1ed7d2308fa14f4924da9
F16=en_GB-southern_english_female-low.voice
F16_SHA=bcce48201e418a1c837ab745557dd9453952ca91d13f7780e1c990ce8ddb4676
F32=en_GB-southern_english_female-low-f32.voice
F32_SHA=863299bae21339ea7a181ea9174fb82e05acda25ea8e04f4d9deea306a5c27c8

ok() {   # file sha
	[ -f "$1" ] && echo "$2  $1" | sha256sum -c --quiet - 2>/dev/null
}
mkdir -p "$HERE/voices"
if ok "$HERE/voices/$F16" $F16_SHA && ok "$HERE/voices/$F32" $F32_SHA; then
	echo "get-voices.sh: voices/ is up to date"
	exit 0
fi

PY=$HERE/build/pydeps
if ! PYTHONPATH=$PY python3 -S -c 'import onnx, numpy' 2>/dev/null; then
	echo "get-voices.sh: installing onnx 1.18.0 and numpy 2.1.3 in build/pydeps (pip)"
	rm -rf "$PY"
	python3 -m pip install --quiet --no-cache-dir --disable-pip-version-check \
		--target "$PY" onnx==1.18.0 numpy==2.1.3
fi

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
curl -sfL -o "$T/voice.tgz" "$URL"
echo "$TGZ_SHA  $T/voice.tgz" | sha256sum -c --quiet - || {
	echo "get-voices.sh: the download from $URL isn't the expected file" >&2; exit 1; }
tar xzf "$T/voice.tgz" -C "$T"
ONNX="$T/en-gb-southern_english_female-low.onnx"
PYTHONPATH=$PY python3 -S "$HERE/tools/onnx2voice.py" "$ONNX" "$HERE/voices/$F16"
PYTHONPATH=$PY python3 -S "$HERE/tools/onnx2voice.py" "$ONNX" "$HERE/voices/$F32" --f32
for f in "$F16 $F16_SHA" "$F32 $F32_SHA"; do
	set -- $f
	ok "$HERE/voices/$1" "$2" || {
		echo "get-voices.sh: voices/$1 came out different from the expected file" >&2; exit 1; }
done
echo "get-voices.sh: voices/$F16 and $F32 made and checked"
