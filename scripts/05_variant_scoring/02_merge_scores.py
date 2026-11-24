#!/usr/bin/env python
"""
02_merge_scores.py  --  merge the out-of-fold SuPreMo scores of all folds into one table per variant set

Input    variant_scoring/inputs/<SET>_master.tsv                      (from 01_prep_inputs.py)
         variant_scoring/scores/<SET>/fold*/<SET>_fold*_scores         (SuPreMo output, one row per var_index)
Scores   per head (NPC / Mic / Neuron) and metric (mse, mse_10-weighted, corr, corr_10-weighted, maxabs, l1):
         mean over the 4 augmentations (shift -1 / 0 / +1 and reverse complement; --aug 0: shift 0 only),
         *_cv = SD / |mean| over the augmentations (augmentation noise),
         composite_z_<head> = z(MSE) + 0.5 * z(MaxAbs) (z-scored within the set)
Outputs  variant_scoring/compare/<SET>_new_scores.tsv, summary.txt (agreement between heads, augmentation noise)
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--aug", default="all", choices=["0", "all"], help="all = mean of shift -1/0/+1 + revcomp (default); 0 = shift-0 only (sensitivity)")
a = ap.parse_args()
B = a.base; S = f"{B}/variant_scoring"; OUT = f"{S}/compare"; os.makedirs(OUT, exist_ok=True)
HEAD = {"HFF": "NPC", "H1hESC": "Mic", "GM12878": "Neuron"}
METRICS = ["mse", "mse_10-weighted", "corr", "corr_10-weighted", "maxabs", "l1"]
AUG = r"(0)" if a.aug == "0" else r"(-1|0|0_revcomp|1)"
log = []

def z(x): return (x - x.mean()) / x.std(ddof=0)

def load_new(setname):
    m = pd.read_csv(f"{S}/inputs/{setname}_master.tsv", sep="\t")
    parts = []
    for f in sorted(glob.glob(f"{S}/scores/{setname}/fold*/{setname}_fold*_scores")):
        d = pd.read_csv(f, sep="\t"); d["fold"] = re.search(r"(fold\d)", f).group(1); parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    out = m.merge(raw[["fold", "var_index"]], on=["fold", "var_index"], how="left", indicator=True)
    missing = out[out._merge == "left_only"].var_id.tolist()
    log.append(f"{setname}: {len(m)} variants, scored {len(m) - len(missing)}" + (f"; NOT scored: {missing}" if missing else ""))
    raw = raw.set_index(["fold", "var_index"])
    res = m.set_index(["fold", "var_index"]).copy()
    for h, cell in HEAD.items():
        for met in METRICS:
            cols = [c for c in raw.columns if re.fullmatch(rf"{re.escape(met)}_{h}_{AUG}", c)]
            if cols:
                res[f"{met}_{cell}"] = raw[cols].mean(axis=1)
                res[f"{met}_{cell}_cv"] = raw[cols].std(axis=1) / raw[cols].mean(axis=1).abs()   # augmentation noise
        if f"mse_{cell}" in res and f"maxabs_{cell}" in res:
            res[f"composite_z_{cell}"] = z(res[f"mse_{cell}"]) + 0.5 * z(res[f"maxabs_{cell}"])
    res = res.reset_index()
    res.to_csv(f"{OUT}/{setname}_new_scores.tsv", sep="\t", index=False)
    return res

snv = load_new("SNV42")
sv = load_new("SV120")

# ---- agreement between the three heads, augmentation noise, size dependence of SVs
for name, d in [("SNV42", snv), ("SV120", sv)]:
    r = d[["mse_NPC", "mse_Mic", "mse_Neuron"]].corr(method="spearman")
    log.append(f"\n== {name}: Spearman between heads (MSE)\n{r.round(2).to_string()}")
    log.append(f"   median augmentation CV of MSE (NPC/Mic/Neuron): "
               + " / ".join(f"{d[f'mse_{c}_cv'].median():.2f}" for c in ["NPC", "Mic", "Neuron"]))
log.append(f"\nSV120: Spearman(MSE_NPC, |SVLEN|) = {spearmanr(sv.svlen.abs(), sv.mse_NPC, nan_policy='omit')[0]:.2f}")
open(f"{OUT}/summary.txt", "w").write("\n".join(log) + "\n"); print("\n".join(log))
