#!/usr/bin/env python3
"""
downsample_cool.py — binomial thinning of a .cool file to a target total count (no cooltools needed).
Each pixel count c becomes Binomial(c, p), p = target / total, streamed in chunks; bins are copied.
The output is unbalanced: run `cooler balance` on it afterwards.

  python downsample_cool.py --in A.cool --out A.ds.cool --count 123456789 --seed 1
"""
import argparse
import sys

import numpy as np
import cooler


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, required=True, help="target total contacts")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=20_000_000)
    a = ap.parse_args(argv)
    c = cooler.Cooler(a.inp)
    total = int(c.info["sum"])
    p = min(1.0, a.count / total)
    print(f"{a.inp}: {total} contacts -> p = {p:.4f}", file=sys.stderr)
    rng = np.random.default_rng(a.seed)
    nnz = int(c.info["nnz"])

    def chunks():
        for lo in range(0, nnz, a.chunksize):
            df = c.pixels()[lo:min(lo + a.chunksize, nnz)]
            df["count"] = rng.binomial(df["count"].values.astype(np.int64), p)
            yield df[df["count"] > 0]

    cooler.create_cooler(a.out, bins=c.bins()[["chrom", "start", "end"]][:], pixels=chunks(),
                         ordered=True, dtypes={"count": "int32"})
    print(f"✅ {a.out}: {int(cooler.Cooler(a.out).info['sum'])} contacts", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
