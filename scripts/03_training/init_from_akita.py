#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init_from_akita.py — initialise a 1-head or 3-head Akita from the published 5-head model

`akita_train.py --restore X --trunk` loads only the 1D trunk; the whole head_hic
stack (2D map formation, dilated residual 2D blocks, final dense) is reinitialised.
That discards the part of Akita that actually learned folding.

This script instead copies EVERY layer whose weight shapes match between the
pretrained model and yours. Only the final dense differs (units=5 vs your 3 or 1),
so only that layer stays random. It writes an .h5 you then pass to
`akita_train.py --restore <file>` WITHOUT --trunk.

Usage
-----
python init_from_akita.py \
    --my_params    configs/params_iNeuroAkita.json \
    --pre_params   pretrained/akita/params.json \
    --pre_weights  pretrained/akita/model_best.h5 \
    --out          pretrained/akita/init_3head.h5

then

akita_train.py -k --restore pretrained/akita/init_3head.h5 \
    configs/params_iNeuroAkita.json model_input/fold1 -o $OUT

python init_from_akita.py --self_test    # logic check, no TensorFlow needed
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Sequence, Tuple


def shapes_match(a: Sequence, b: Sequence) -> bool:
    """True when two weight lists have identical shapes, element for element."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        return False
    return all(tuple(x.shape) == tuple(y.shape) for x, y in zip(a, b))


def plan_transfer(my_layers, pre_layers) -> Tuple[List[int], List[int], List[int]]:
    """Match layers positionally; return (copied, skipped_shape, skipped_empty) indices.

    Positional matching is correct only when the two architectures are the same
    up to the final dense, which is what the shape check below verifies. Any
    mismatch other than the last weighted layer means the params differ and the
    transfer should not be trusted.
    """
    copied, skipped_shape, skipped_empty = [], [], []
    n = min(len(my_layers), len(pre_layers))
    for i in range(n):
        mw = my_layers[i].get_weights()
        pw = pre_layers[i].get_weights()
        if not mw and not pw:
            skipped_empty.append(i)
        elif shapes_match(mw, pw):
            copied.append(i)
        else:
            skipped_shape.append(i)
    return copied, skipped_shape, skipped_empty


def self_test() -> int:
    class FakeW:
        def __init__(self, shape):
            self.shape = shape

    class FakeLayer:
        def __init__(self, name, shapes):
            self.name = name
            self._w = [FakeW(s) for s in shapes]

        def get_weights(self):
            return self._w

        def set_weights(self, w):
            self._w = w

    # trunk + 2D head identical; final dense differs (5 heads vs 3)
    common = [("conv1", [(11, 4, 96), (96,)]), ("bn1", [(96,), (96,), (96,), (96,)]),
              ("act", []), ("dil2d", [(3, 3, 48, 48), (48,)]), ("sym", [])]
    mine = [FakeLayer(n, s) for n, s in common] + [FakeLayer("dense", [(48, 3), (3,)])]
    pre = [FakeLayer(n, s) for n, s in common] + [FakeLayer("dense", [(48, 5), (5,)])]

    copied, bad, empty = plan_transfer(mine, pre)
    assert copied == [0, 1, 3], f"weighted layers copied: {copied}"
    assert bad == [5], "only the final dense should mismatch"
    assert empty == [2, 4], "weightless layers reported separately"

    # a 1-head model behaves the same way
    one = [FakeLayer(n, s) for n, s in common] + [FakeLayer("dense", [(48, 1), (1,)])]
    assert plan_transfer(one, pre)[1] == [5]

    # a genuinely different trunk must surface as an early mismatch, not silently copy
    other = [FakeLayer("conv1", [(11, 4, 64), (64,)])] + mine[1:]
    assert 0 in plan_transfer(other, pre)[1], "trunk difference must be flagged"

    # identical models: everything with weights is copied
    assert plan_transfer(pre, pre)[1] == []

    print("✅ init_from_akita self-test passed")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--my_params", help="your params.json (3-head or 1-head)")
    p.add_argument("--pre_params", help="params.json shipped with the pretrained model")
    p.add_argument("--pre_weights", help="pretrained model_best.h5 (5 heads)")
    p.add_argument("--out", help="output .h5 to pass to akita_train.py --restore")
    p.add_argument("--allow_partial", action="store_true",
                   help="proceed even if layers other than the final dense mismatch")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    for req in ("my_params", "pre_params", "pre_weights", "out"):
        if getattr(args, req) is None:
            print(f"❌ --{req} is required", file=sys.stderr)
            return 2

    from basenji import seqnn  # late import

    with open(args.my_params) as f:
        my_model = seqnn.SeqNN(json.load(f)["model"])
    with open(args.pre_params) as f:
        pre_model = seqnn.SeqNN(json.load(f)["model"])
    pre_model.restore(args.pre_weights)
    print(f"✅ pretrained loaded: {pre_model.model.output_shape}")
    print(f"✅ target model built: {my_model.model.output_shape}")

    my_layers = my_model.model.layers
    pre_layers = pre_model.model.layers
    if len(my_layers) != len(pre_layers):
        print(f"⚠️  layer counts differ ({len(my_layers)} vs {len(pre_layers)}); "
              f"the architectures are not the same up to the head", file=sys.stderr)

    copied, bad, empty = plan_transfer(my_layers, pre_layers)
    print(f"\n   {len(copied)} weighted layers match and will be copied")
    print(f"   {len(empty)} layers carry no weights")
    if bad:
        print(f"   {len(bad)} layers mismatch:")
        for i in bad:
            ms = [tuple(w.shape) for w in my_layers[i].get_weights()]
            ps = [tuple(w.shape) for w in pre_layers[i].get_weights()]
            print(f"     [{i}] {my_layers[i].name}: mine={ms} pretrained={ps}")

    expected_bad = {max(bad)} if bad else set()
    if bad and set(bad) != expected_bad and not args.allow_partial:
        print("\n❌ layers other than the final dense differ, so the trunks are not the "
              "same architecture. Align --my_params with the pretrained params.json, or "
              "rerun with --allow_partial if you know what you are doing.", file=sys.stderr)
        return 2

    for i in copied:
        my_layers[i].set_weights(pre_layers[i].get_weights())

    my_model.model.save_weights(args.out)
    print(f"\n💾 {args.out}")
    print("   the final dense stays randomly initialised (head count differs)")
    print(f"\nNext:\n  akita_train.py -k --restore {args.out} "
          f"{args.my_params} <data_dir> -o <out_dir>   # note: no --trunk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
