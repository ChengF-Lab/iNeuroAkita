#!/usr/bin/env python3
"""
noise_ceiling.py — replicate-based noise ceiling for Mic_AD (4 libraries -> 2 halves) and NPC_P16
(2 libraries), compared with iNeuroAkita (A4) and Akita (A0) on the held-out windows.

  dump     read the akita_data.py TFRecords of the replicate dataset (basenji env, no model) and write
           <data_dir>/../replicate_targets.npz (targets of every replicate, chrom / start / end)
  ceiling  per window: r_half = Pearson r between the two replicate maps (sigma-smoothed, upper triangle,
           as add_smoothed_metrics.py), overall and per distance band;
           reliability of the full-depth map = Spearman-Brown 2 r_half / (1 + r_half)
           (each half has ~half the reads);  ceiling = sqrt(reliability) = the r a PERFECT predictor
           could reach against the full observed map, given its sequencing noise
           model    A4 / A0 r on the held-out windows (same smoothing), per window
           summary  medians genome-wide, per chromosome and per distance band; fraction of ceiling =
                    median model r / median ceiling

Outputs (--out, default model_trainOut/noise_ceiling)
  N1_summary.csv  N2_by_chrom.csv  N3_by_band.csv  per_window_replicates.csv.gz  per_window_models.csv.gz
  NF1_ceiling_vs_models  NF2_by_chrom  NF3_by_band  (.pdf .png .svg)

Usage
  python noise_ceiling.py dump    --data_dir independent_testData/noise_ceiling_v2/all --targets <targets.txt>
  python noise_ceiling.py ceiling --base_dir $base_DIR --npz independent_testData/noise_ceiling_v2/replicate_targets.npz
         --pair Mic=Mic_AD_A,Mic_AD_B --pair NPC=NPC_P16_r1,NPC_P16_r2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- dump (basenji env)
def dump(a):
    """Read only the TARGETS straight from the TFRecords (no basenji SeqDataset), so the way the
    sequence was stored (integer-coded vs one-hot, 'seq_1hot' in statistics.json) does not matter."""
    import glob
    import json
    import re
    import tensorflow as tf
    stats = json.load(open(f"{a.data_dir}/statistics.json"))
    n_t, t_len = int(stats["num_targets"]), int(stats["target_length"])
    seqs = pd.read_csv(f"{a.data_dir}/sequences.bed", sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chrom", "start", "end", "split"], dtype={0: str})
    names = pd.read_csv(a.targets, sep="\t")["identifier"].astype(str).tolist()
    if len(names) != n_t:
        raise SystemExit(f"❌ {len(names)} names in {a.targets} but num_targets = {n_t}")
    feats = {"sequence": tf.io.FixedLenFeature([], tf.string), "target": tf.io.FixedLenFeature([], tf.string)}

    def read(files, comp):
        out = []
        for rec in tf.data.TFRecordDataset(files, compression_type=comp):
            ex = tf.io.parse_single_example(rec, feats)
            t = tf.io.decode_raw(ex["target"], tf.float16).numpy()
            out.append(t.reshape(t_len, n_t))
        return out

    T, rows = [], []
    for split in [s for s in ("train", "valid", "test") if s in set(seqs.split)]:
        files = sorted(glob.glob(f"{a.data_dir}/tfrecords/{split}-*.tfr"),
                       key=lambda f: int(re.search(r"-(\d+)\.tfr$", f).group(1)))
        try:
            ts = read(files, "ZLIB")
        except tf.errors.DataLossError:
            ts = read(files, "")
        reg = seqs[seqs.split == split].reset_index(drop=True)
        if len(ts) != len(reg):
            raise SystemExit(f"❌ {split}: {len(ts)} TFRecord windows vs {len(reg)} sequences.bed rows")
        T.extend(ts)
        rows.append(reg)
        log(f"   {split}: {len(ts)} windows ({len(files)} files)")
    T = np.stack(T)
    R = pd.concat(rows, ignore_index=True)
    out = a.out or os.path.join(os.path.dirname(os.path.normpath(a.data_dir)), "replicate_targets.npz")
    m = (np.sqrt(8 * t_len + 1) - 1) / 2
    L = int(round(m)) + int(stats.get("diagonal_offset", a.diagonal_offset))
    np.savez_compressed(out, test_targets=T, obs_target_names=np.array(names), chrom=R.chrom.values.astype(str),
                        start=R.start.values, end=R.end.values, matrix_len=L,
                        diagonal_offset=int(stats.get("diagonal_offset", a.diagonal_offset)))
    log(f"✅ {out}  targets {T.shape}  (L = {L})")


# ----------------------------------------------------------------------------- ceiling
def ceiling(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from add_smoothed_metrics import vec_to_mat, nan_gauss
    from celltype_specific_diag import fold_inputs, FOLD_TEST, CELLS, HEAD, BIN, COL
    from stage2_eval import A0Source, selected_heads
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
    BANDS = [("0-100kb", 0, 100_000), ("100-250kb", 100_000, 250_000), ("250-500kb", 250_000, 500_000),
             (">500kb", 500_000, 10 ** 9), ("all", 0, 10 ** 9)]
    out = a.out or f"{a.base_dir}/model_trainOut/noise_ceiling"
    os.makedirs(out, exist_ok=True)

    def pr(x, y):
        k = np.isfinite(x) & np.isfinite(y)
        if k.sum() < 20 or x[k].std() == 0 or y[k].std() == 0:
            return np.nan
        return float(np.corrcoef(x[k], y[k])[0, 1])

    pairs = {}
    for s in a.pair:
        cell, reps = s.split("=")
        r1, r2 = reps.split(",")
        pairs[cell] = (r1, r2)
    z = np.load(a.npz, allow_pickle=False)
    names = [str(n) for n in z["obs_target_names"]]
    L, d = int(z["matrix_len"]), int(z["diagonal_offset"])
    iu = np.triu_indices(L, d)
    dist = (iu[1] - iu[0]) * BIN
    bm = {b: (dist >= lo) & (dist < hi) & (dist >= a.min_dist) for b, lo, hi in BANDS}
    T = z["test_targets"]
    chrom = np.array([str(c) for c in z["chrom"]])
    rows = []
    for cell, (r1, r2) in pairs.items():
        i1, i2 = names.index(r1), names.index(r2)
        for w in range(len(chrom)):
            s1 = nan_gauss(vec_to_mat(T[w, :, i1].astype(float), L, d), a.sigma)[iu]
            s2 = nan_gauss(vec_to_mat(T[w, :, i2].astype(float), L, d), a.sigma)[iu]
            for b, _, _ in BANDS:
                rh = pr(s1[bm[b]], s2[bm[b]])
                rows.append(dict(cell=cell, chrom=chrom[w], start=int(z["start"][w]), band=b, r_half=rh))
        log(f"   {cell}: {len(chrom)} windows")
    Rp = pd.DataFrame(rows)
    Rp["reliability"] = 2 * Rp.r_half / (1 + Rp.r_half)
    Rp["ceiling"] = np.sqrt(Rp.reliability.clip(lower=0))
    Rp.to_csv(f"{out}/per_window_replicates.csv.gz", index=False, float_format="%.4g")

    # models on held-out windows
    heads0 = selected_heads(a.base_dir)
    a0 = A0Source(a.base_dir)
    rows = []
    for fold in FOLD_TEST:
        got = fold_inputs(a.base_dir, "heldout", fold)
        if got is None:
            continue
        P, Tt, meta, L2, d2 = got
        keys = list(zip(meta.chrom, meta.start.astype(int)))
        for k, c in enumerate(CELLS):
            if c not in pairs:
                continue
            h0 = heads0.get((fold, HEAD[c]))
            A0 = a0.get("heldout", fold, c, h0, keys) if h0 else None
            for w in range(len(meta)):
                t = nan_gauss(vec_to_mat(Tt[c][w].astype(float), L2, d2), a.sigma)[iu]
                preds = {"A4": P[w, :, k]}
                if A0 is not None:
                    preds["A0"] = A0[w]
                for m, v in preds.items():
                    p = nan_gauss(vec_to_mat(v.astype(float), L2, d2), a.sigma)[iu]
                    for b, _, _ in BANDS:
                        rows.append(dict(cell=c, model=m, fold=fold, chrom=meta.chrom[w], start=int(meta.start[w]),
                                         band=b, r=pr(p[bm[b]], t[bm[b]])))
        log(f"   models {fold}")
    Rm = pd.DataFrame(rows)
    Rm.to_csv(f"{out}/per_window_models.csv.gz", index=False, float_format="%.4g")

    # summaries
    def summ(keys):
        a_ = Rp.groupby(keys)[["r_half", "reliability", "ceiling"]].median()
        b_ = Rm.pivot_table(index=keys, columns="model", values="r", aggfunc="median")
        b_.columns = [f"median_r_{c}" for c in b_.columns]
        s = a_.join(b_, how="outer")
        for m in ("A4", "A0"):
            if f"median_r_{m}" in s:
                s[f"frac_ceiling_{m}"] = s[f"median_r_{m}"] / s["ceiling"]
        return s.reset_index()
    N1 = summ(["cell", "band"])
    N1[N1.band == "all"].to_csv(f"{out}/N1_summary.csv", index=False, float_format="%.4g")
    N1.to_csv(f"{out}/N3_by_band.csv", index=False, float_format="%.4g")
    Rp_all, Rm_all = Rp[Rp.band == "all"], Rm[Rm.band == "all"]
    N2 = Rp_all.groupby(["cell", "chrom"]).ceiling.median().to_frame().join(
        Rm_all.pivot_table(index=["cell", "chrom"], columns="model", values="r", aggfunc="median")).reset_index()
    N2.to_csv(f"{out}/N2_by_chrom.csv", index=False, float_format="%.4g")

    def save(fig, pre):
        for ext in ("pdf", "png", "svg"):
            fig.savefig(f"{pre}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)
        log(f"✅ {pre}")
    cells = list(pairs)
    fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3), squeeze=False)
    for ax, c in zip(axes[0], cells):
        data = [Rp_all[Rp_all.cell == c].r_half.dropna(), Rp_all[Rp_all.cell == c].ceiling.dropna(),
                Rm_all[(Rm_all.cell == c) & (Rm_all.model == "A4")].r.dropna(),
                Rm_all[(Rm_all.cell == c) & (Rm_all.model == "A0")].r.dropna()]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="black"))
        for bx, cc in zip(bp["boxes"], ["#cfcfcf", "#888888", "#e34948", "#2a78d6"]):
            bx.set_facecolor(cc)
        ax.set_xticklabels(["replicate\nhalves r", "ceiling\nsqrt(SB)", "iNeuroAkita\n(A4)", "Akita\n(A0)"], fontsize=7)
        s = N1[(N1.cell == c) & (N1.band == "all")].iloc[0]
        ax.set_title(f"{c}: A4 = {100 * s.get('frac_ceiling_A4', np.nan):.0f}% of ceiling", fontsize=9)
        ax.set_ylabel("Pearson r per window (σ-smoothed)")
    save(fig, f"{out}/NF1_ceiling_vs_models")
    fig, axes = plt.subplots(1, len(cells), figsize=(5 * len(cells), 3), squeeze=False)
    order = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    for ax, c in zip(axes[0], cells):
        s = N2[N2.cell == c].set_index("chrom").reindex([o for o in order if o in set(N2.chrom)])
        x = np.arange(len(s))
        ax.plot(x, s.ceiling, "o-", color="#555555", ms=3, label="ceiling")
        for m, cc in (("A4", "#e34948"), ("A0", "#2a78d6")):
            if m in s:
                ax.plot(x, s[m], "o-", color=cc, ms=3, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([i.replace("chr", "") for i in s.index], fontsize=6)
        ax.set(title=f"{c}: per chromosome (median)", ylabel="r")
        ax.legend(frameon=False, fontsize=6)
    save(fig, f"{out}/NF2_by_chrom")
    fig, axes = plt.subplots(1, len(cells), figsize=(3.8 * len(cells), 3), squeeze=False)
    bands = [b for b, _, _ in BANDS if b != "all"]
    for ax, c in zip(axes[0], cells):
        s = N1[N1.cell == c].set_index("band").reindex(bands)
        ax.plot(bands, s.ceiling, "o-", color="#555555", label="ceiling")
        for m, cc in (("A4", "#e34948"), ("A0", "#2a78d6")):
            if f"median_r_{m}" in s:
                ax.plot(bands, s[f"median_r_{m}"], "o-", color=cc, label=m)
        ax.set(title=f"{c}: by contact distance", ylabel="median r")
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(frameon=False, fontsize=6)
    save(fig, f"{out}/NF3_by_band")
    log(f"✅ noise ceiling in {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--data_dir", required=True)
    d.add_argument("--targets", required=True, help="the targets txt given to akita_data.py")
    d.add_argument("--diagonal_offset", type=int, default=2)
    d.add_argument("--out", default=None)
    c = sub.add_parser("ceiling")
    c.add_argument("--base_dir", required=True)
    c.add_argument("--npz", required=True)
    c.add_argument("--pair", action="append", required=True, metavar="CELL=rep1,rep2")
    c.add_argument("--sigma", type=float, default=2.0)
    c.add_argument("--min_dist", type=int, default=0)
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dump(a) if a.cmd == "dump" else ceiling(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
