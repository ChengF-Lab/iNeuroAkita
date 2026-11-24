#!/usr/bin/env python3
"""
QC for distiller-nf output coolers -- iNeuroAkita.

Checks, per cell type:
  - presence and validity of ICE balancing weights
  - total / cis / trans coverage and long-range cis fraction
  - depth comparability across cell types (relevant because uneven depth
    across training inputs can masquerade as cell-type biology)

Read-only: does not modify the coolers. Prints the `cooler balance`
commands to run if weights are missing.

Usage:
    python qc_coolers.py --cool NPC=NPC/all_libraries.hg38.mapq_30.2048.cool \
                         --cool Mic=Mic/all_libraries.hg38.mapq_30.2048.cool
"""

import argparse
import os
import sys

import numpy as np

try:
    import cooler
except ImportError:
    sys.exit("cooler not found. Try: conda activate <env> && pip install cooler")


# Long-range cis threshold. Contacts below this are dominated by
# random-ligation and self-ligation products rather than genuine
# chromatin structure.
LONG_RANGE_BP = 20_000

def describe(label, path):
    """Print a QC block for one cooler. Returns a dict of summary numbers."""
    print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")
    print(f"path: {path}")

    if not os.path.exists(path):
        print("  MISSING -- file does not exist")
        return None

    c = cooler.Cooler(path)

    # ---- basic structure -------------------------------------------------
    print(f"  binsize   : {c.binsize:,} bp")
    print(f"  n_bins    : {c.info['nbins']:,}")
    print(f"  n_contacts: {c.info['nnz']:,} nonzero pixels")
    print(f"  n_chroms  : {len(c.chromnames)}")

    # Contigs that should not be in a training-input cooler.
    odd = [x for x in c.chromnames
           if x.endswith("_alt") or "random" in x or x.startswith("chrUn")
           or x == "chrM"]
    if odd:
        print(f"  WARNING: unexpected contigs present ({len(odd)}): "
              f"{odd[:5]}{' ...' if len(odd) > 5 else ''}")

    # ---- balancing weights ----------------------------------------------
    cols = list(c.bins().columns)
    print(f"  bin columns: {cols}")

    if "weight" not in cols:
        print("  >> NOT BALANCED. Run:")
        print(f"     cooler balance -p 8 '{path}'")
        return {"label": label, "balanced": False}

    w = c.bins()["weight"][:].values
    n_tot = len(w)
    n_nan = int(np.isnan(w).sum())
    finite = w[np.isfinite(w)]
    print(f"  balanced  : yes")
    print(f"  weights   : {n_tot - n_nan:,}/{n_tot:,} finite "
          f"({100 * n_nan / n_tot:.1f}% NaN / filtered out)")
    if finite.size:
        print(f"              median={np.median(finite):.4g}  "
              f"min={finite.min():.4g}  max={finite.max():.4g}")

    # A high NaN fraction means ICE discarded many bins as low-coverage --
    # at 2048 bp on shallow data this is the usual sign of insufficient depth.
    if n_nan / n_tot > 0.30:
        print("  WARNING: >30% of bins have no weight. At this resolution "
              "that usually means the library is too shallow.")

    # ---- coverage / cis-trans -------------------------------------------
    # Sum over the pixel table in chunks to keep memory bounded.
    tot = cis = long_cis = 0
    bins = c.bins()[:][["chrom", "start"]]
    chrom_codes = bins["chrom"].cat.codes.values if hasattr(
        bins["chrom"], "cat") else bins["chrom"].astype("category").cat.codes.values
    starts = bins["start"].values

    for lo in range(0, c.info["nnz"], 5_000_000):
        px = c.pixels()[lo:lo + 5_000_000]
        b1 = px["bin1_id"].values
        b2 = px["bin2_id"].values
        cnt = px["count"].values
        same = chrom_codes[b1] == chrom_codes[b2]
        dist = np.abs(starts[b2] - starts[b1])
        tot += int(cnt.sum())
        cis += int(cnt[same].sum())
        long_cis += int(cnt[same & (dist >= LONG_RANGE_BP)].sum())

    trans = tot - cis
    print(f"  contacts  : {tot:,} total")
    print(f"    cis           : {cis:,} ({100 * cis / tot:.1f}%)")
    print(f"    trans         : {trans:,} ({100 * trans / tot:.1f}%)")
    print(f"    cis >={LONG_RANGE_BP // 1000}kb   : {long_cis:,} "
          f"({100 * long_cis / tot:.1f}%)")

    # Rough expectations for bulk in situ Hi-C. Wide bands on purpose --
    # these flag obvious failures, not marginal quality differences.
    if trans / tot < 0.02:
        print("  WARNING: trans fraction implausibly low -- check mate pairing.")
    elif trans / tot > 0.60:
        print("  WARNING: trans fraction very high -- possible random ligation.")
    if long_cis / tot < 0.15:
        print("  WARNING: low long-range cis fraction -- limited structural signal.")

    return {"label": label, "balanced": True, "total": tot,
            "long_cis": long_cis, "nan_frac": n_nan / n_tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cool", action="append", required=True, metavar="LABEL=PATH",
                    help="cooler to check (repeat for several cell types); use file.mcool::resolutions/2048 for .mcool")
    args = ap.parse_args()
    SAMPLES = dict(c.split("=", 1) for c in args.cool)

    print(f"cooler {cooler.__version__}")

    results = []
    for label, path in SAMPLES.items():
        r = describe(label, path)
        if r:
            results.append(r)

    # ---- cross-sample depth comparison -----------------------------------
    ok = [r for r in results if r.get("balanced")]
    if len(ok) > 1:
        print(f"\n{'=' * 62}\nDEPTH COMPARISON (long-range cis contacts)\n{'=' * 62}")
        lo = min(r["long_cis"] for r in ok)
        for r in sorted(ok, key=lambda x: -x["long_cis"]):
            print(f"  {r['label']:<14} {r['long_cis']:>15,}  "
                  f"({r['long_cis'] / lo:.2f}x shallowest)")
        ratio = max(r["long_cis"] for r in ok) / lo
        if ratio > 2:
            print(f"\n  NOTE: {ratio:.1f}x depth spread across cell types. "
                  "Consider downsampling to the shallowest before building\n"
                  "  training targets, so the model does not learn depth "
                  "differences as cell-type biology.")

    print("\nDone.")


if __name__ == "__main__":
    main()
