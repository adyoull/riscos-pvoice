#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Andrew Youll
"""Reference outputs for the engine tests, from ONNX Runtime (host only):

    python3 tests/make_ref.py model.onnx OUTDIR [noise_scale length_scale noise_w]

Writes OUTDIR/ids.txt (phoneme ids, one line) and OUTDIR/<stage>.f32 (raw
float32): enc_m (inter x T), enc_logs, logw (T), z (inter x frames: the
flow's output, the decoder's input) and audio. With the noise scales at 0
(the default here) the model is deterministic, so the C engine must match. """
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

REF_TEXT = ("It is a truth universally acknowledged, that a single man in possession "
            "of a good fortune, must be in want of a wife.")


def main():
    model, out = sys.argv[1], sys.argv[2]
    scales = [float(x) for x in sys.argv[3:6]] if len(sys.argv) > 5 else [0.0, 1.0, 0.0]
    m = onnx.load(model)
    g = m.graph
    want = {
        "enc_split": [n for n in g.node if n.name == "/enc_p/Split"][0].output,
        "logw": [n for n in g.node if n.name == "/dp/Split"][0].output[0],
        "z": [n for n in g.node if n.name == "/dec/conv_pre/Conv"][0].input[0],
    }
    for name in [want["enc_split"][0], want["enc_split"][1], want["logw"], want["z"]]:
        g.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    # Public-domain text (Austen), through eSpeak (en-gb) as Piper does.
    from piper import PiperVoice
    v = PiperVoice.load(model)
    ids = v.phonemes_to_ids(v.phonemize(REF_TEXT)[0])
    x = np.array([ids], dtype=np.int64)
    feeds = {"input": x, "input_lengths": np.array([x.shape[1]], dtype=np.int64),
             "scales": np.array(scales, dtype=np.float32)}
    outs = s.run(None, feeds)
    names = [o.name for o in s.get_outputs()]
    r = dict(zip(names, outs))
    with open(out + "/ids.txt", "w") as f:
        f.write(" ".join(str(i) for i in ids) + "\n")

    def dump(stage, arr):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        a.tofile(out + "/" + stage + ".f32")
        print("%-8s %s" % (stage, a.shape))
    dump("enc_m", r[want["enc_split"][0]][0])
    dump("enc_logs", r[want["enc_split"][1]][0])
    dump("logw", r[want["logw"]][0, 0])
    dump("z", r[want["z"]][0])
    dump("audio", r["output"].reshape(-1))


if __name__ == "__main__":
    main()
