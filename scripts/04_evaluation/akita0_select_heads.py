#!/usr/bin/env python3
"""
akita0_select_heads.py — choose the original-Akita head per cell type on VALIDATION
chromosomes, then report it on TEST chromosomes (panel b) and on each fold's unseen
chromosomes of the independent datasets (panel d).

Inputs (Akita zero-shot predictions from akita_predict.py):
  {zs_root}/fold{k}/valid/akita0_fold{k}_valid_per_window_metrics.tsv
  {zs_root}/fold{k}/test/akita0_fold{k}_test_per_window_metrics.tsv
  {zs_root}/indep_{neuron,NPC,Mic}/akita0_{neuron,NPC,Mic}_per_window_metrics.tsv
  {data_root}/fold{k}/sequences.bed      (test chromosomes of each fold)

Outputs (in {zs_root}/summary/):
  akita0_all_heads.csv         every fold x split x cell type x head: median r
  akita0_selected_heads.csv    best head per fold x cell type (chosen on valid)
  akita0_panelb_test.csv       per-fold test median r with the selected head
  akita0_external_unseen.csv   independent data on each fold's test chroms, selected head

Selection rule (state it in Methods): for each fold and cell type, the Akita head
with the highest median per-window Pearson r on that fold's validation chromosomes.

  python akita0_select_heads.py --zs_root $ZS_ROOT --data_root $DATA_ROOT
  python akita0_select_heads.py --self_test
"""
import argparse
import glob
import os
import sys

import pandas as pd

# independent observed target -> the training cell type whose head choice it inherits
INDEP = {
    "neuron": ("Neuron_P35", "Neuron_P16"),
    "NPC": ("NPC_ctrl", "NPC_P16"),
    "Mic": ("Mic_ctrl", "Mic_AD"),
}


def summarise(df, keys):
    return (df.groupby(keys)["pearson"]
              .agg(n="size", median_r="median", mean_r="mean")
              .reset_index())


def test_chroms(data_root, fold):
    bed = pd.read_csv(os.path.join(data_root, fold, "sequences.bed"), sep="\t",
                      header=None, usecols=[0, 3], names=["chrom", "split"], dtype={0: str})
    return sorted(bed.loc[bed["split"] == "test", "chrom"].unique())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs_root")
    ap.add_argument("--data_root")
    ap.add_argument("--folds", nargs="*", default=[f"fold{k}" for k in range(1, 8)])
    ap.add_argument("--select", choices=["best", "worst", "mean"], default="best",
                    help="best = highest validation r (primary, fair baseline); "
                         "worst = lowest validation r (lower end of a range only); "
                         "mean = average over all 5 heads, no selection")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.zs_root and a.data_root):
        ap.error("--zs_root and --data_root are required")

    out_dir = os.path.join(a.zs_root, "summary")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for fold in a.folds:
        for split in ("valid", "test"):
            p = os.path.join(a.zs_root, fold, split, f"akita0_{fold}_{split}_per_window_metrics.tsv")
            if not os.path.exists(p):
                print(f"⚠️  missing {p}", file=sys.stderr)
                continue
            d = pd.read_csv(p, sep="\t")
            d["fold"], d["eval_split"] = fold, split
            rows.append(d)
    if not rows:
        print("❌ no fold metrics found", file=sys.stderr)
        return 2
    m = pd.concat(rows, ignore_index=True)

    allh = summarise(m, ["fold", "eval_split", "obs_target", "model_head"])
    allh.to_csv(os.path.join(out_dir, "akita0_all_heads.csv"), index=False)

    mode = a.select
    sfx = "" if mode == "best" else f"_{mode}"      # best keeps the original file names
    test = allh[allh["eval_split"] == "test"]

    # range over all 5 heads on test: always written, whatever the mode
    rng = (test.groupby(["fold", "obs_target"])["median_r"]
               .agg(min_r="min", max_r="max", mean_r="mean").reset_index())
    rng.to_csv(os.path.join(out_dir, "akita0_head_range_test.csv"), index=False)

    valid = allh[allh["eval_split"] == "valid"]
    if mode == "mean":
        # no selection: every head counts equally
        best = (valid.groupby(["fold", "obs_target"])["median_r"].mean().reset_index()
                     .rename(columns={"median_r": "valid_median_r"}))
        best["model_head"] = "mean_of_heads"
        t = (test.groupby(["fold", "obs_target"])
                 .agg(test_n=("n", "first"), test_median_r=("median_r", "mean"),
                      test_mean_r=("mean_r", "mean")).reset_index())
        panelb = best.merge(t, on=["fold", "obs_target"], how="left")
        panelb = panelb[["fold", "obs_target", "model_head", "valid_median_r",
                         "test_n", "test_median_r", "test_mean_r"]]
    else:
        if valid.empty:
            print("❌ no validation metrics; cannot select heads", file=sys.stderr)
            return 2
        if mode == "worst":
            print("⚠️  --select worst: the weakest Akita head is NOT a fair baseline. Use it only "
                  "as the lower end of a range shown next to the best head.", file=sys.stderr)
        best = (valid.sort_values("median_r", ascending=(mode == "worst"))
                     .groupby(["fold", "obs_target"], as_index=False).first()
                     [["fold", "obs_target", "model_head", "median_r"]]
                     .rename(columns={"median_r": "valid_median_r"}))
        panelb = best.merge(test, on=["fold", "obs_target", "model_head"], how="left")
        panelb = panelb[["fold", "obs_target", "model_head", "valid_median_r",
                         "n", "median_r", "mean_r"]].rename(
            columns={"n": "test_n", "median_r": "test_median_r", "mean_r": "test_mean_r"})
    best.to_csv(os.path.join(out_dir, f"akita0_selected_heads{sfx}.csv"), index=False)
    panelb.to_csv(os.path.join(out_dir, f"akita0_panelb_test{sfx}.csv"), index=False)

    print(f"\n🧠 Akita head per fold ({mode}, chosen on validation chromosomes)")
    print(best.pivot(index="fold", columns="obs_target", values="model_head").to_string())
    print(f"\n📊 test median r ({mode}): mean ± SD over folds")
    agg = panelb.groupby("obs_target")["test_median_r"].agg(["mean", "std", "count"]).round(4)
    print(agg.to_string())

    # ---- independent datasets on each fold's unseen chromosomes -------------
    ext = []
    for key, (obs, train_ct) in INDEP.items():
        p = os.path.join(a.zs_root, f"indep_{key}", f"akita0_{key}_per_window_metrics.tsv")
        if not os.path.exists(p):
            print(f"⚠️  missing {p}", file=sys.stderr)
            continue
        d = pd.read_csv(p, sep="\t")
        for fold in a.folds:
            sel = best[(best["fold"] == fold) & (best["obs_target"] == train_ct)]
            if sel.empty:
                continue
            head = sel["model_head"].iloc[0]
            chroms = test_chroms(a.data_root, fold)
            dd = d[d["chrom"].isin(chroms)]
            if mode != "mean":
                dd = dd[dd["model_head"] == head]
            if dd.empty:
                continue
            if mode == "mean":   # per-head median, then average over heads
                per_head = dd.groupby("model_head")["pearson"]
                med, mn, n = per_head.median().mean(), per_head.mean().mean(), int(per_head.size().iloc[0])
            else:
                med, mn, n = dd["pearson"].median(), dd["pearson"].mean(), len(dd)
            ext.append(dict(fold=fold, dataset=obs, head_from=train_ct, model_head=head,
                            test_chroms=",".join(chroms), n=n, median_r=med, mean_r=mn))
    if ext:
        ext = pd.DataFrame(ext)
        ext.to_csv(os.path.join(out_dir, f"akita0_external_unseen{sfx}.csv"), index=False)
        print("\n🌐 independent data, each fold's unseen chromosomes: median r mean ± SD")
        print(ext.groupby("dataset")["median_r"].agg(["mean", "std", "count"]).round(4).to_string())

    print(f"\n✅ tables in {out_dir}")
    return 0


def self_test():
    import tempfile
    import numpy as np
    rng = np.random.default_rng(0)
    heads = ["HFF", "H1hESC", "GM12878", "IMR90", "HCT116"]
    obs = ["NPC_P16", "Mic_AD", "Neuron_P16"]
    with tempfile.TemporaryDirectory() as td:
        zs, dr = os.path.join(td, "zs"), os.path.join(td, "data")
        for k, tchr in ((1, "chr10"), (2, "chr3")):
            fold = f"fold{k}"
            os.makedirs(os.path.join(dr, fold))
            with open(os.path.join(dr, fold, "sequences.bed"), "w") as f:
                f.write(f"chr1\t0\t1\ttrain\nchr8\t0\t1\tvalid\n{tchr}\t0\t1\ttest\n")
            for split, ch in (("valid", "chr8"), ("test", tchr)):
                rows = []
                for o in obs:
                    for h in heads:
                        # H1hESC best for NPC on valid; IMR90 best otherwise
                        base = 0.5 if (o == "NPC_P16" and h == "H1hESC") or \
                                      (o != "NPC_P16" and h == "IMR90") else 0.2
                        for _ in range(20):
                            rows.append(dict(chrom=ch, obs_target=o, model_head=h,
                                             pearson=base + rng.normal(0, .01)))
                dd = os.path.join(zs, fold, split); os.makedirs(dd)
                pd.DataFrame(rows).to_csv(
                    os.path.join(dd, f"akita0_{fold}_{split}_per_window_metrics.tsv"),
                    sep="\t", index=False)
        # independent NPC_ctrl on chr10 and chr3 (+ chr1 which is never a test chrom)
        rows = [dict(chrom=c, obs_target="NPC_ctrl", model_head=h, pearson=0.3)
                for c in ("chr1", "chr10", "chr3") for h in heads for _ in range(5)]
        os.makedirs(os.path.join(zs, "indep_NPC"))
        pd.DataFrame(rows).to_csv(os.path.join(zs, "indep_NPC", "akita0_NPC_per_window_metrics.tsv"),
                                  sep="\t", index=False)
        assert main(["--zs_root", zs, "--data_root", dr, "--folds", "fold1", "fold2"]) == 0
        sel = pd.read_csv(os.path.join(zs, "summary", "akita0_selected_heads.csv"))
        got = dict(zip(sel["fold"] + ":" + sel["obs_target"], sel["model_head"]))
        assert got["fold1:NPC_P16"] == "H1hESC" and got["fold2:Mic_AD"] == "IMR90", got
        ext = pd.read_csv(os.path.join(zs, "summary", "akita0_external_unseen.csv"))
        assert set(ext["test_chroms"]) == {"chr10", "chr3"}, "unseen chroms per fold"
        assert (ext["model_head"] == "H1hESC").all() and (ext["n"] == 5).all()

        # worst / mean modes write suffixed files and never overwrite the best ones
        assert main(["--zs_root", zs, "--data_root", dr, "--folds", "fold1", "fold2",
                     "--select", "worst"]) == 0
        w = pd.read_csv(os.path.join(zs, "summary", "akita0_panelb_test_worst.csv"))
        b = pd.read_csv(os.path.join(zs, "summary", "akita0_panelb_test.csv"))
        assert (w["test_median_r"] < b["test_median_r"]).all()
        assert "H1hESC" not in set(w[w["obs_target"] == "NPC_P16"]["model_head"])
        assert main(["--zs_root", zs, "--data_root", dr, "--folds", "fold1", "fold2",
                     "--select", "mean"]) == 0
        m = pd.read_csv(os.path.join(zs, "summary", "akita0_panelb_test_mean.csv"))
        assert (w["test_median_r"].values <= m["test_median_r"].values + 1e-9).all()
        assert (m["test_median_r"].values <= b["test_median_r"].values + 1e-9).all()
        rg = pd.read_csv(os.path.join(zs, "summary", "akita0_head_range_test.csv"))
        assert (rg["min_r"] <= rg["max_r"]).all()
    print("\n✅ akita0_select_heads self-test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
