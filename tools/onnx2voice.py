#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Andrew Youll
"""Converts a Piper voice (model.onnx + model.onnx.json) into a .voice file
for the C engine:

    python3 -I tools/onnx2voice.py model.onnx [model.onnx.json] OUT.voice [--f32]

The .voice file is little-endian:

    "PVOI" u32 version (1) u32 count
    count x { u16 name_len, name, u8 type, u8 ndim, u32 dims[ndim], data }

type 0 = float32, 1 = float16, 2 = int32, 3 = bytes (UTF-8 text). Data is
padded to 4 bytes. Weights over 1024 values are stored as float16 unless
--f32 is given; the engine turns them back into float32 when it loads.

Tensors:
  <module>.weight / .bias / .conv     every Conv and ConvTranspose, named after
                                      its PyTorch module (the ONNX node's
                                      name); .conv is int32 [stride, pad_l,
                                      pad_r, dilation, groups, transposed]
  enc_p.emb.weight                    phoneme embeddings
  emb_g.weight                        speaker embeddings (multi-speaker voices)
  enc_p.encoder.attn_layers.N.emb_rel_k / emb_rel_v
  ...norm...gamma / beta              LayerNorms
  dp.flows.0.m, dp.flows.0.exp_neg_logs
  config                              int32: see CONFIG below
  scales                              float32 [noise_scale, length_scale, noise_w]
  phoneme_map                         int32 pairs [codepoint, id] (in order)
  espeak_voice                        bytes
"""
import json
import struct
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

CONFIG = ["version", "sample_rate", "n_speakers", "n_vocab", "hidden", "n_heads",
          "n_layers", "window", "inter", "use_sdp", "n_ups", "n_kernels",
          "resblock", "flow_layers"]


def main(argv):
    f32 = "--f32" in argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) == 2:
        model, out = args
        cfg_path = model + ".json"
    elif len(args) == 3:
        model, cfg_path, out = args
    else:
        sys.exit(__doc__)
    m = onnx.load(model)
    g = m.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    tensors = {}

    def put(name, arr):
        tensors[name] = arr

    used = set()
    for n in g.node:
        if n.op_type in ("Conv", "ConvTranspose"):
            path = n.name.strip("/").rsplit("/", 1)[0].replace("/", ".")
            a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
            w = inits[n.input[1]]
            put(path + ".weight", w.astype(np.float32))
            used.add(n.input[1])
            if len(n.input) > 2 and n.input[2]:
                put(path + ".bias", inits[n.input[2]].astype(np.float32))
                used.add(n.input[2])
            pads = a.get("pads", [0, 0])
            put(path + ".conv", np.array([
                a.get("strides", [1])[0], pads[0], pads[1], a.get("dilations", [1])[0],
                a.get("group", 1), 1 if n.op_type == "ConvTranspose" else 0], dtype=np.int32))
        elif n.op_type == "Gather" and n.name == "/enc_p/emb/Gather":
            put("enc_p.emb.weight", inits[n.input[0]].astype(np.float32))
        elif n.op_type == "Gather" and n.name.startswith("/emb_g/"):
            put("emb_g.weight", inits[n.input[0]].astype(np.float32))
        elif n.name == "/dp/flows.0/Mul" and n.input[1] in inits:
            put("dp.flows.0.exp_neg_logs", inits[n.input[1]].astype(np.float32))

    for name, arr in inits.items():
        if name.endswith((".gamma", ".beta", ".emb_rel_k", ".emb_rel_v")) or name == "dp.flows.0.m":
            put(name, arr.astype(np.float32))

    # Hyperparameters, from the weights' shapes.
    emb = tensors["enc_p.emb.weight"]
    rel = tensors["enc_p.encoder.attn_layers.0.emb_rel_k"]
    n_layers = len({k.split(".")[3] for k in tensors if k.startswith("enc_p.encoder.attn_layers.")})
    n_ups = len({k.split(".")[2] for k in tensors if k.startswith("dec.ups.")})
    n_res = len({k.split(".")[2] for k in tensors if k.startswith("dec.resblocks.")})
    resblock = 1 if any(".convs1." in k for k in tensors if k.startswith("dec.resblocks.")) else 2
    flow_layers = len({k.split(".")[5] for k in tensors if k.startswith("flow.flows.0.enc.in_layers.")})
    n_speakers = int(cfg.get("num_speakers", 1))
    values = {
        "version": 1,
        "sample_rate": int(cfg["audio"]["sample_rate"]),
        "n_speakers": n_speakers,
        "n_vocab": emb.shape[0],
        "hidden": emb.shape[1],
        "n_heads": emb.shape[1] // rel.shape[2],
        "n_layers": n_layers,
        "window": (rel.shape[1] - 1) // 2,
        "inter": tensors["enc_p.proj.weight"].shape[0] // 2,
        "use_sdp": 1 if any(k.startswith("dp.flows.") for k in tensors) else 0,
        "n_ups": n_ups,
        "n_kernels": n_res // n_ups,
        "resblock": resblock,
        "flow_layers": flow_layers,
    }
    put("config", np.array([values[k] for k in CONFIG], dtype=np.int32))
    inf = cfg.get("inference", {})
    put("scales", np.array([inf.get("noise_scale", 0.667), inf.get("length_scale", 1.0),
                            inf.get("noise_w", 0.8)], dtype=np.float32))
    pairs = []
    for ph, ids in cfg["phoneme_id_map"].items():
        if len(ph) != 1:
            continue          # every eSpeak phoneme is one code point
        for i in ids:
            pairs += [ord(ph), int(i)]
    put("phoneme_map", np.array(pairs, dtype=np.int32))
    put("espeak_voice", cfg.get("espeak", {}).get("voice", "en-us").encode("utf-8"))

    with open(out, "wb") as f:
        f.write(b"PVOI" + struct.pack("<II", 1, len(tensors)))
        total = 0
        for name, arr in tensors.items():
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            if isinstance(arr, bytes):
                f.write(struct.pack("<BBI", 3, 1, len(arr)))
                data = arr
            else:
                if arr.dtype == np.int32:
                    t, data = 2, arr.astype("<i4").tobytes()
                elif not f32 and arr.size > 1024:
                    t, data = 1, arr.astype("<f2").tobytes()
                else:
                    t, data = 0, arr.astype("<f4").tobytes()
                f.write(struct.pack("<BB", t, arr.ndim) + struct.pack("<%dI" % arr.ndim, *arr.shape))
            f.write(data + b"\0" * (-len(data) % 4))
            total += len(data)
    print("%s: %d tensors, %.1f MB; %s" % (out, len(tensors), total / 1e6,
          ", ".join("%s %d" % (k, values[k]) for k in CONFIG)))


if __name__ == "__main__":
    main(sys.argv[1:])
