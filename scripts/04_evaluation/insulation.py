#!/usr/bin/env python3
"""
insulation.py — TAD insulation-profile r of iNeuroAkita (A4) and Akita (A0) for ONE fold model on ALL
windows of the independent samples (the evaluation of Fig. 2a-f). stage2_eval.py scores each chromosome only
with the fold that held it out (out-of-fold); this script keeps every window of the chosen fold's npz.

Same definitions as stage2_eval.py: insulation = mean of M[i-w:i, i+1:i+w+1] on the raw log(obs/exp) map,
prediction masked where the target is NaN, Pearson r between predicted and observed profiles per window.
A0 = Akita zero-shot with the head selected on this fold's validation chromosomes (selected_heads).

Inputs  model_trainOut/NPC_Mic_Neuron_out_trunk/<fold>/predictOut_indep/indep_<NPC|neuron|Mic>/*_predictions_targets.npz
        model_trainOut/akita_zeroshot/indep_v2_<key>/*_predictions_targets.npz
Output  model_trainOut/stage2_eval/per_window_insulation_<fold>_all.csv.gz
        columns as per_window_insulation.csv.gz: dataset, fold, chrom, start, cell, predictor, w, ins_r, ...

  python insulation.py --base_dir $base_DIR --fold fold2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_smoothed_metrics import vec_to_mat
from celltype_specific_diag import CELLS, HEAD, INDEP_KEY, OBS, TRUNK, first, log
from stage2_eval import A0Source, selected_heads, insulation, boundaries, matched, pearson


def load_cell(base, fold, c):
    p = first(f"{base}/{TRUNK}/{fold}/predictOut_indep/indep_{INDEP_KEY[c]}/*_predictions_targets.npz")
    if p is None:
        raise SystemExit(f"❌ no independent npz for {c} in {fold}")
    z = np.load(p, allow_pickle=False)
    heads = [str(h) for h in z["model_head_names"]]
    obs = [str(o) for o in z["obs_target_names"]]
    meta = pd.DataFrame(dict(chrom=[str(x) for x in z["chrom"]], start=z["start"].astype(int)))
    P = z["test_preds_subset"][:, :, heads.index(HEAD[c])]
    T = z["test_targets"][:, :, obs.index(OBS["independent"][c])]
    log(f"   {c}: {p} ({len(meta)} windows)")
    return P, T, meta, int(z["matrix_len"]), int(z["diagonal_offset"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--fold", default="fold2")
    ap.add_argument("--ins_window", type=int, nargs="+", default=[10, 25])
    ap.add_argument("--prominence", type=float, default=0.5)
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    base = a.base_dir.rstrip("/")
    out = a.out or f"{base}/model_trainOut/stage2_eval/per_window_insulation_{a.fold}_all.csv.gz"
    heads0 = selected_heads(base)
    a0 = A0Source(base)
    rows = []
    for c in CELLS:
        P, T, meta, L, d = load_cell(base, a.fold, c)
        keys = list(zip(meta.chrom, meta.start))
        h = heads0.get((a.fold, HEAD[c]))
        A0 = None if h is None else a0.get("independent", a.fold, c, h, keys)
        if A0 is None:
            raise SystemExit(f"❌ A0 for {c} ({a.fold}, head {h}) not available")
        log(f"   {c}: Akita head {h}")
        for w in range(len(meta)):
            raw_t = vec_to_mat(T[w].astype(float), L, d)
            lab = dict(dataset="independent_all", fold=a.fold, chrom=meta.chrom[w], start=int(meta.start[w]),
                       cell=c, akita_head=h)
            ins_t = {iw: insulation(raw_t, iw) for iw in a.ins_window}
            bnd_t = {iw: boundaries(ins_t[iw], iw, a.prominence) for iw in a.ins_window}
            for name, v in (("A4", P[w]), ("A0", A0[w])):
                mat = vec_to_mat(v.astype(float), L, d)
                mat[~np.isfinite(raw_t)] = np.nan
                for iw in a.ins_window:
                    ins_p = insulation(mat, iw)
                    bp = boundaries(ins_p, iw, a.prominence)
                    bt = bnd_t[iw]
                    rows.append(dict(lab, predictor=name, w=iw, ins_r=pearson(ins_p, ins_t[iw]),
                                     n_obs_boundaries=len(bt), n_pred_boundaries=len(bp),
                                     obs_recovered=matched(bt, bp, a.tol), pred_supported=matched(bp, bt, a.tol)))
            if (w + 1) % 500 == 0:
                log(f"   {c}: {w + 1}/{len(meta)} windows")
        del P, T, A0
    R = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R.to_csv(out, index=False)
    s = R[R.w == a.ins_window[0]].pivot_table(index=["cell", "chrom", "start"], columns="predictor", values="ins_r")
    for c in CELLS:
        x = s.loc[c]
        log(f"   {c}: n = {len(x)}, median insulation r A0 {x.A0.median():.3f} -> A4 {x.A4.median():.3f}, "
            f"{(x.A4 > x.A0).mean():.0%} windows improved")
    log(f"✅ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
