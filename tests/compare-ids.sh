#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Andrew Youll
# Text to phoneme ids through PiperSay must give what Python Piper gives
# (tests/ids-expected.txt, made from tests/texts.txt with piper-tts).
#   tests/compare-ids.sh "PIPERSAY COMMAND" VOICE ESPEAK_DATA
set -eu
cmd=$1 voice=$2 data=$3
out=$(mktemp)
while IFS= read -r line; do
	[ -n "$line" ] || continue
	$cmd -x -v "$voice" -d "$data" "$line" | grep -E '^ +[0-9]' | sed 's/^ *//; s/ *$//'
done < tests/texts.txt > "$out"
if cmp -s "$out" tests/ids-expected.txt; then
	echo "phoneme ids: $(wc -l < "$out") sentences, all as Python Piper"
	rm -f "$out"
else
	echo "FAIL phoneme ids differ from Python Piper:"
	diff "$out" tests/ids-expected.txt | head -20
	rm -f "$out"
	exit 1
fi
