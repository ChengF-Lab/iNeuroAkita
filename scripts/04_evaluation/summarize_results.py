#!/usr/bin/env python3
"""
summarize_results.py — collect every iNeuroAkita training / evaluation result into tables.

What it reads (nothing is recomputed; it only gathers files that already exist)
-------------------------------------------------------------------------------
Training      tfevents in <run>/train and <run>/validation (Keras TensorBoard logs)
Held-out      <run>/predictOut_v3/hold_test/*_test_per_window_metrics_smooth2.tsv  (merged)
              <run>/predictOut/hold_test/*_test_per_window_metrics_smooth2.tsv     (single-head)
              falls back to *_per_window_metrics.tsv (raw only) when no smooth2 file exists
External v2   <run>/predictOut_indep/indep_*/*_per_window_metrics_smooth2.tsv
A0            model_trainOut/akita_zeroshot/foldK/test/akita0_foldK_test_per_window_metrics[_smooth2].tsv
              model_trainOut/akita_zeroshot/indep_v2_{neuron,NPC,Mic}/akita0_*_per_window_metrics[_smooth2].tsv
              head per fold x cell type from akita_zeroshot/summary/akita0_selected_heads.csv
Random split  model_trainOut/NPC_Mic_Neuron_out/NPC_Mic_Neuron_random/predictOut/hold_test/random_test_*
              + model_input/NPC_Mic_Neuron_input_v2/sequences.bed (overlap with training windows)

A0 rows appear automatically once the Akita zero-shot predictions exist: just run this script again.

Models
------
A1 single-head, scratch        {NPC_P16,Neuron_P16,Mic_AD}/trainOut/foldK
A2 single-head, Akita-init     {NPC_P16,Neuron_P16,Mic_AD}/trainOut_trunk/foldK
A3 merged 3-head, scratch      model_trainOut/NPC_Mic_Neuron_out/foldK
A4 merged 3-head, Akita-init   model_trainOut/NPC_Mic_Neuron_out_trunk/foldK
A0 original Akita, zero-shot   model_trainOut/akita_zeroshot (head chosen on validation chromosomes)

Output (--out; one CSV per table, plus all_tables.xlsx when openpyxl is installed)
------
T1_training                 model x fold: epochs, best epoch, val r / r2 / loss at the best epoch
T1b_training_extra          random-split model, learning-rate pilot
T2_heldout_by_fold          model x cell x fold: n windows, mean / median r (raw, smooth), MSE
T3_heldout_by_chrom         model x cell x chromosome (each chromosome scored by the fold that held it out)
T4_heldout_summary          model x cell: mean +/- SD over the 7 fold medians, pooled median over windows
T5_heldout_vs_A0            each model vs A0, per cell: fold-level paired t / Wilcoxon (n = 7), window-level delta
T6_A0_all_heads             A0: every Akita head x cell x fold (selected head flagged)
T7_external_by_chrom        external v2, out-of-fold (unseen chromosomes): model x dataset x chromosome
T8_external_summary         external v2: model x dataset x {autosomes, chrX} for unseen; also seen windows
T9_external_vs_A0           external v2 unseen autosomes: median delta r, chromosome-bootstrap 95% CI,
                            % windows higher, fold-level paired tests
T10_random_split            random-split model vs chromosome-split A3 held-out, and by overlap with training
per_window_heldout.csv.gz   long table behind T2-T6 (source data)
per_window_external.csv.gz  long table behind T7-T9

Statistics: the main test is paired across the 7 folds (fold medians). Window-level numbers are
descriptive only, because neighbouring windows are correlated; the external CI resamples whole
chromosomes for the same reason.

Usage
-----
  python summarize_results.py --base_dir $base_DIR --out $base_DIR/model_trainOut/summary_tables
  python summarize_results.py ... --sigma 2 --metric pearson --skip_training
"""
import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

CELL_OF = {"NPC_P16": "NPC", "Neuron_P16": "Neuron", "Mic_AD": "Microglia",
           "NPC_ctrl": "NPC", "Neuron_P35": "Neuron", "Mic_ctrl": "Microglia"}
CELL_DIR = {"NPC": "NPC_P16", "Neuron": "Neuron_P16", "Microglia": "Mic_AD"}
DATASET_OF = {"NPC_ctrl": "NPC_ctrl", "Neuron_P35": "Neuron_P35", "Mic_ctrl": "Mic_ctrl"}
EXT_KEY = {"NPC_ctrl": "NPC", "Neuron_P35": "neuron", "Mic_ctrl": "Mic"}    # akita_zeroshot folder keys
CELL_ORDER = ["NPC", "Neuron", "Microglia"]
MODEL_ORDER = ["A0", "A1", "A2", "A3", "A4"]
MODEL_LABEL = {"A0": "Akita zero-shot", "A1": "single-head scratch", "A2": "single-head Akita-init",
               "A3": "merged scratch", "A4": "merged Akita-init"}
FOLDS = [f"fold{i}" for i in range(1, 8)]
METRICS = ["pearson", "spearman", "mse"]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- inputs
def fold_splits(base):
    """fold -> {'test': set(chroms), 'valid': set(chroms)} from the merged model input."""
    out = {}
    for f in FOLDS:
        bed = f"{base}/model_input/NPC_Mic_Neuron_input/{f}/sequences.bed"
        if not os.path.exists(bed):
            log(f"⚠️  missing {bed}; fold roles for {f} unknown")
            continue
        s = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 3], names=["chrom", "split"],
                        dtype=str)
        out[f] = {k: set(s.loc[s.split == k, "chrom"]) for k in ("test", "valid")}
    return out


def check_single_splits(base, splits):
    """Warn if a single-head model's folds use other chromosomes than the merged model."""
    for cdir in CELL_DIR.values():
        for f in FOLDS:
            bed = f"{base}/{cdir}/input/{f}/sequences.bed"
            if not os.path.exists(bed) or f not in splits:
                continue
            s = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 3], names=["chrom", "split"],
                            dtype=str)
            if set(s.loc[s.split == "test", "chrom"]) != splits[f]["test"]:
                log(f"⚠️  {cdir} {f}: test chromosomes differ from the merged model's")


def pick(files, sigma):
    """Prefer the smooth{sigma} table (raw + smooth columns), else the raw table."""
    tag = f"_per_window_metrics_smooth{sigma:g}.tsv"
    sm = sorted((f for f in files if f.endswith(tag)), key=os.path.getmtime)
    raw = sorted((f for f in files if f.endswith("_per_window_metrics.tsv")), key=os.path.getmtime)
    for group in (sm, raw):
        if len(group) > 1:
            log(f"⚠️  {len(group)} candidate files, using newest: {group[-1]}")
    return (sm[-1], True) if sm else ((raw[-1], False) if raw else (None, None))


def read_metrics(path, smooth):
    df = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    if not smooth:
        for m in METRICS:
            df[f"{m}_raw"] = df[m]
            df[f"{m}_smooth"] = np.nan
    keep = ["chrom", "start", "end", "obs_target", "model_head"] + \
           [f"{m}_{k}" for m in METRICS for k in ("raw", "smooth")]
    df = df[keep].copy()
    df["source_file"] = path
    df["has_smooth"] = smooth
    return df


def glob_dir(pattern):
    return [f for f in glob.glob(pattern) if os.path.isfile(f)]


def model_runs(base):
    """One dict per (model, cell-specific root)."""
    T = f"{base}/model_trainOut"
    runs = [dict(model="A3", task="merged", init="scratch", root=f"{T}/NPC_Mic_Neuron_out",
                 held="predictOut_v3/hold_test"),
            dict(model="A4", task="merged", init="Akita-init", root=f"{T}/NPC_Mic_Neuron_out_trunk",
                 held="predictOut_v3/hold_test")]
    for model, sub, init in (("A1", "trainOut", "scratch"), ("A2", "trainOut_trunk", "Akita-init")):
        for cell, cdir in CELL_DIR.items():
            runs.append(dict(model=model, task="single", init=init, cell=cell,
                             root=f"{base}/{cdir}/{sub}", held="predictOut/hold_test"))
    return runs


# ----------------------------------------------------------------------------- training
def read_events(d):
    """{tag: {epoch: value}} from a Keras TensorBoard log directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.util import tensor_util
    except ImportError:
        return None
    if not os.path.isdir(d):
        return {}
    ea = EventAccumulator(d, size_guidance={"scalars": 0, "tensors": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = {e.step: float(e.value) for e in ea.Scalars(tag)}
    for tag in ea.Tags().get("tensors", []):
        vals = {}
        for e in ea.Tensors(tag):
            try:
                vals[e.step] = float(np.asarray(tensor_util.make_ndarray(e.tensor_proto)).item())
            except Exception:
                pass
        if vals:
            out[tag] = vals
    return out


def find_tag(ev, *keys):
    for t in ev:
        tl = t.lower()
        if all(k in tl for k in keys) and tl.startswith("epoch"):
            return ev[t]
    return {}


def parse_log_epochs(run_dir):
    """Fallback: Keras progress lines in a *.out log next to the model."""
    rows = []
    for f in glob.glob(f"{run_dir}/*.out"):
        txt = open(f, errors="ignore").read()
        for i, m in enumerate(re.finditer(
                r"loss: ([\d.]+) - pearsonr: ([-\d.]+) - r2: ([-\d.]+) - val_loss: ([\d.]+)"
                r" - val_pearsonr: ([-\d.]+) - val_r2: ([-\d.]+)", txt)):
            rows.append(dict(epoch=i, train_loss=float(m[1]), train_r=float(m[2]),
                             val_loss=float(m[4]), val_r=float(m[5]), val_r2=float(m[6])))
        if rows:
            break
    return pd.DataFrame(rows)


def training_row(run_dir):
    tr, va = read_events(f"{run_dir}/train"), read_events(f"{run_dir}/validation")
    if tr is None and not getattr(training_row, "warned", False):
        log("⚠️  tensorboard not importable; reading *.out logs instead")
        training_row.warned = True
    df = pd.DataFrame()
    if va:
        vr, vl, v2 = find_tag(va, "pearson"), find_tag(va, "loss"), find_tag(va, "r2")
        tr_r, tr_l = find_tag(tr or {}, "pearson"), find_tag(tr or {}, "loss")
        ep = sorted(vr)
        df = pd.DataFrame(dict(epoch=ep, val_r=[vr[e] for e in ep],
                               val_loss=[vl.get(e, np.nan) for e in ep],
                               val_r2=[v2.get(e, np.nan) for e in ep],
                               train_r=[tr_r.get(e, np.nan) for e in ep],
                               train_loss=[tr_l.get(e, np.nan) for e in ep]))
    if df.empty:
        df = parse_log_epochs(run_dir)
    if df.empty:
        return None
    b = df.loc[df.val_r.idxmax()]
    return dict(n_epochs=len(df), best_epoch=int(b.epoch) + 1, best_val_pearsonr=b.val_r,
                val_r2_at_best=b.val_r2, val_loss_at_best=b.val_loss,
                train_pearsonr_at_best=b.train_r, train_loss_at_best=b.train_loss,
                min_val_loss=df.val_loss.min())


def training_tables(base):
    rows = []
    for r in model_runs(base):
        for f in FOLDS:
            d = f"{r['root']}/{f}"
            if not os.path.exists(f"{d}/model_best.h5"):
                continue
            t = training_row(d)
            if t:
                rows.append(dict(model=r["model"], label=MODEL_LABEL[r["model"]],
                                 cell=r.get("cell", "all (3 heads)"), fold=f, run_dir=d, **t))
    extra = []
    T = f"{base}/model_trainOut"
    for name, d in [("random split 80/10/10 (merged scratch)", f"{T}/NPC_Mic_Neuron_out/NPC_Mic_Neuron_random"),
                    ("A4 fold2, lr 0.0065 (main run)", f"{T}/NPC_Mic_Neuron_out_trunk/fold2"),
                    ("A4 fold2, lr 0.002", f"{T}/NPC_Mic_Neuron_out_trunk/fold2/lr0.002_out"),
                    ("A4 fold2, lr 0.0006", f"{T}/NPC_Mic_Neuron_out_trunk/fold2/lr0.0006_out")]:
        if os.path.exists(f"{d}/model_best.h5"):
            t = training_row(d)
            if t:
                extra.append(dict(run=name, run_dir=d, **t))
    return pd.DataFrame(rows), pd.DataFrame(extra)


# ----------------------------------------------------------------------------- held-out
def heldout_long(base, splits, sigma):
    parts = []
    for r in model_runs(base):
        for f in FOLDS:
            path, sm = pick(glob_dir(f"{r['root']}/{f}/{r['held']}/*_test_per_window_metrics*.tsv"), sigma)
            if not path:
                continue
            d = read_metrics(path, sm)
            d = d[d.obs_target == d.model_head]                  # matched head only
            d["cell"] = d.obs_target.map(CELL_OF)
            if r["task"] == "single":
                d = d[d.cell == r["cell"]]
            d["model"], d["fold"] = r["model"], f
            parts.append(d)
    # A0: selected head per fold x cell
    zs = f"{base}/model_trainOut/akita_zeroshot"
    sel_f = f"{zs}/summary/akita0_selected_heads.csv"
    sel = pd.read_csv(sel_f) if os.path.exists(sel_f) else None
    a0_all = []
    for f in FOLDS:
        path, sm = pick(glob_dir(f"{zs}/{f}/test/akita0_{f}_test_per_window_metrics*.tsv"), sigma)
        if not path:
            continue
        d = read_metrics(path, sm)
        d["cell"] = d.obs_target.map(CELL_OF)
        d["fold"] = f
        a0_all.append(d)
        if sel is not None:
            s = sel[sel.fold == f][["obs_target", "model_head"]]
            d = d.merge(s, on=["obs_target", "model_head"])
            d["model"] = "A0"
            parts.append(d)
    if sel is None:
        log(f"⚠️  {sel_f} missing; A0 left out of held-out comparisons")
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    a0_all = pd.concat(a0_all, ignore_index=True) if a0_all else pd.DataFrame()
    if not long.empty:
        long["test_fold_ok"] = [c in splits.get(f, {}).get("test", {c}) for c, f in zip(long.chrom, long.fold)]
        if not long.test_fold_ok.all():
            log(f"⚠️  {int((~long.test_fold_ok).sum())} held-out rows are not on their fold's test chromosomes")
    return long, a0_all, sel


def agg(df, keys, metric):
    g = df.groupby(keys, sort=False)
    out = g.agg(n_windows=(f"{metric}_raw", "size"),
                mean_r_raw=(f"{metric}_raw", "mean"), median_r_raw=(f"{metric}_raw", "median"),
                mean_r_smooth=(f"{metric}_smooth", "mean"), median_r_smooth=(f"{metric}_smooth", "median"),
                mean_mse_raw=("mse_raw", "mean"), mean_mse_smooth=("mse_smooth", "mean")).reset_index()
    return out


def order(df):
    if "model" in df:
        df["model"] = pd.Categorical(df.model, MODEL_ORDER, ordered=True)
    if "cell" in df:
        df["cell"] = pd.Categorical(df.cell, CELL_ORDER, ordered=True)
    keys = [k for k in ("model", "cell", "dataset", "role", "fold", "chrom_order") if k in df]
    return df.sort_values(keys).reset_index(drop=True) if keys else df


def chrom_key(c):
    c = str(c).replace("chr", "")
    return int(c) if c.isdigit() else 100 + ord(c[0])


def paired_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return dict(n_folds=len(a), mean_delta=np.nan, t_p=np.nan, wilcoxon_p=np.nan, folds_higher=np.nan)
    t = stats.ttest_rel(a, b).pvalue
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w = np.nan
    return dict(n_folds=len(a), mean_delta=float(np.mean(a - b)), t_p=t, wilcoxon_p=w,
                folds_higher=int((a > b).sum()))


def heldout_tables(long, a0_all, sel, metric):
    t2 = order(agg(long, ["model", "cell", "fold"], metric))
    t2.insert(1, "label", t2.model.astype(str).map(MODEL_LABEL))
    long = long.assign(chrom_order=long.chrom.map(chrom_key))
    t3 = order(agg(long, ["model", "cell", "chrom", "chrom_order", "fold"], metric)).drop(columns="chrom_order")
    rows = []
    for (m, c), d in t2.groupby(["model", "cell"], observed=True):
        lw = long[(long.model == m) & (long.cell == c)]
        rows.append(dict(model=m, label=MODEL_LABEL[m], cell=c, n_folds=len(d), n_windows=len(lw),
                         fold_median_r_raw_mean=d.median_r_raw.mean(), fold_median_r_raw_sd=d.median_r_raw.std(),
                         fold_median_r_smooth_mean=d.median_r_smooth.mean(),
                         fold_median_r_smooth_sd=d.median_r_smooth.std(),
                         pooled_median_r_raw=lw[f"{metric}_raw"].median(),
                         pooled_median_r_smooth=lw[f"{metric}_smooth"].median(),
                         fold_mean_mse_raw=d.mean_mse_raw.mean(), fold_mean_mse_smooth=d.mean_mse_smooth.mean()))
    t4 = order(pd.DataFrame(rows))
    # vs A0
    rows = []
    if (t2.model == "A0").any():
        a0 = t2[t2.model == "A0"]
        a0w = long[long.model == "A0"]
        for (m, c), d in t2[t2.model != "A0"].groupby(["model", "cell"], observed=True):
            ref = a0[a0.cell == c].set_index("fold")
            d = d.set_index("fold").reindex(ref.index)
            row = dict(model=m, label=MODEL_LABEL[m], cell=c)
            for k in ("raw", "smooth"):
                p = paired_tests(d[f"median_r_{k}"], ref[f"median_r_{k}"])
                row.update({f"{kk}_{k}": v for kk, v in p.items()})
            mw = long[(long.model == m) & (long.cell == c)]
            j = mw.merge(a0w[a0w.cell == c], on=["chrom", "start", "fold"], suffixes=("", "_a0"))
            for k in ("raw", "smooth"):
                dd = j[f"{metric}_{k}"] - j[f"{metric}_{k}_a0"]
                row[f"window_median_delta_{k}"] = dd.median()
                row[f"window_pct_higher_{k}"] = 100 * (dd > 0).mean() if dd.notna().any() else np.nan
            row["n_windows_paired"] = len(j)
            rows.append(row)
    t5 = order(pd.DataFrame(rows))
    t6 = pd.DataFrame()
    if not a0_all.empty:
        t6 = order(agg(a0_all, ["cell", "fold", "model_head"], metric))
        if sel is not None:
            s = sel.assign(cell=sel.obs_target.map(CELL_OF), selected=True)[["cell", "fold", "model_head", "selected"]]
            t6 = t6.merge(s, on=["cell", "fold", "model_head"], how="left")
            t6["selected"] = t6.selected.fillna(False).astype(bool)
    return t2, t3, t4, t5, t6


# ----------------------------------------------------------------------------- external
def role_of(chrom, fold, splits):
    s = splits.get(fold)
    if s is None:
        return "unknown"
    return "unseen" if chrom in s["test"] else ("valid" if chrom in s["valid"] else "seen")


def external_long(base, splits, sel, sigma):
    parts = []
    for r in model_runs(base):
        for f in FOLDS:
            files = glob_dir(f"{r['root']}/{f}/predictOut_indep/indep_*/*_per_window_metrics*.tsv")
            for sub in sorted({os.path.dirname(x) for x in files}):
                path, sm = pick([x for x in files if os.path.dirname(x) == sub], sigma)
                d = read_metrics(path, sm)
                d = d[d.obs_target.isin(DATASET_OF)]
                d["cell"] = d.model_head.map(CELL_OF)
                d = d[d.cell == d.obs_target.map(CELL_OF)]          # matching head only
                if r["task"] == "single":
                    d = d[d.cell == r["cell"]]
                d["model"], d["fold"] = r["model"], f
                parts.append(d)
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not long.empty:
        long["dataset"] = long.obs_target
        long["role"] = [role_of(c, f, splits) for c, f in zip(long.chrom, long.fold)]
    # A0 (one model, no folds): attach the head selected for the fold that holds the chromosome out
    zs = f"{base}/model_trainOut/akita_zeroshot"
    test_fold = {c: f for f, s in splits.items() for c in s["test"]}
    a0 = []
    for obs, key in EXT_KEY.items():
        path, sm = pick(glob_dir(f"{zs}/indep_v2_{key}/*_per_window_metrics*.tsv"), sigma)
        if not path or sel is None:
            continue
        d = read_metrics(path, sm)
        d["fold"] = d.chrom.map(test_fold)
        d["cell"] = CELL_OF[obs]
        s = sel.assign(cell=sel.obs_target.map(CELL_OF))[["fold", "cell", "model_head"]]
        d = d.merge(s, on=["fold", "cell", "model_head"])
        d["model"], d["dataset"], d["role"] = "A0", obs, "unseen"
        a0.append(d)
    if a0:
        long = pd.concat([long] + a0, ignore_index=True)
    else:
        log("ℹ️  no A0 v2 external metrics yet (run_10b); external tables have iNeuroAkita only")
    if not long.empty:
        long["chrX"] = long.chrom.eq("chrX")
    return long


def chrom_bootstrap(d, col_a, col_b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    chroms = d.chrom.unique()
    by = {c: (d.loc[d.chrom == c, col_a] - d.loc[d.chrom == c, col_b]).dropna().values for c in chroms}
    meds = []
    for _ in range(n):
        x = np.concatenate([by[c] for c in rng.choice(chroms, len(chroms), replace=True)])
        if len(x):
            meds.append(np.median(x))
    return (np.percentile(meds, 2.5), np.percentile(meds, 97.5)) if meds else (np.nan, np.nan)


def external_tables(ext, metric):
    if ext.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    un = ext[ext.role == "unseen"].assign(chrom_order=lambda x: x.chrom.map(chrom_key))
    t7 = order(agg(un, ["model", "dataset", "cell", "chrom", "chrom_order", "fold"], metric)).drop(columns="chrom_order")
    rows = []
    for role in ("unseen", "seen"):
        sub = ext[ext.role == role]
        if role == "seen":   # a seen window appears in several folds: average them first
            sub = (sub.groupby(["model", "dataset", "cell", "chrom", "start", "chrX"], observed=True)
                   [[f"{m}_{k}" for m in METRICS for k in ("raw", "smooth")]].mean().reset_index())
        for (m, ds, x), d in sub.groupby(["model", "dataset", "chrX"], observed=True):
            rows.append(dict(model=m, label=MODEL_LABEL[m], dataset=ds, cell=CELL_OF[ds], role=role,
                             chromosomes="chrX" if x else "autosomes", n_windows=len(d),
                             median_r_raw=d[f"{metric}_raw"].median(), mean_r_raw=d[f"{metric}_raw"].mean(),
                             median_r_smooth=d[f"{metric}_smooth"].median(),
                             mean_r_smooth=d[f"{metric}_smooth"].mean(),
                             mean_mse_raw=d.mse_raw.mean(), mean_mse_smooth=d.mse_smooth.mean()))
    t8 = order(pd.DataFrame(rows))
    rows = []
    a0 = un[(un.model == "A0") & ~un.chrX]
    if not a0.empty:
        for (m, ds), d in un[(un.model != "A0") & ~un.chrX].groupby(["model", "dataset"], observed=True):
            j = d.merge(a0[a0.dataset == ds], on=["chrom", "start"], suffixes=("", "_a0"))
            row = dict(model=m, label=MODEL_LABEL[m], dataset=ds, n_windows=len(j))
            for k in ("raw", "smooth"):
                a, b = f"{metric}_{k}", f"{metric}_{k}_a0"
                dd = j[a] - j[b]
                lo, hi = chrom_bootstrap(j, a, b) if dd.notna().any() else (np.nan, np.nan)
                fm = j.groupby("fold")[[a, b]].median()
                p = paired_tests(fm[a], fm[b])
                row.update({f"median_delta_{k}": dd.median(), f"ci95_lo_{k}": lo, f"ci95_hi_{k}": hi,
                            f"pct_windows_higher_{k}": 100 * (dd > 0).mean() if dd.notna().any() else np.nan,
                            f"fold_t_p_{k}": p["t_p"], f"fold_wilcoxon_p_{k}": p["wilcoxon_p"]})
            rows.append(row)
    t9 = order(pd.DataFrame(rows))
    return t7, t8, t9


# ----------------------------------------------------------------------------- random split
def gap_to_train(test, train):
    """Per test window: bp overlapping training windows, and distance (bp) to the nearest one."""
    tr = {c: d[["start", "end"]].values for c, d in train.groupby("chrom")}
    ov, gap = [], []
    for c, s, e in test[["chrom", "start", "end"]].itertuples(index=False):
        iv = tr.get(c)
        if iv is None or len(iv) == 0:
            ov.append(0); gap.append(np.nan)
            continue
        o = np.clip(np.minimum(iv[:, 1], e) - np.maximum(iv[:, 0], s), 0, None)
        g = np.maximum(0, np.maximum(iv[:, 0] - e, s - iv[:, 1]))
        ov.append(int(o.max())); gap.append(int(g.min()))
    return np.array(ov), np.array(gap, dtype=float)


def random_table(base, held_long, metric, sigma):
    T = f"{base}/model_trainOut"
    path, sm = pick(glob_dir(f"{T}/NPC_Mic_Neuron_out/NPC_Mic_Neuron_random/predictOut/hold_test/*_per_window_metrics*.tsv"), sigma)
    if not path:
        log("ℹ️  random-split predictions not found yet (run_10c)")
        return pd.DataFrame()
    d = read_metrics(path, sm)
    d = d[d.obs_target == d.model_head].copy()
    d["cell"] = d.obs_target.map(CELL_OF)
    bed = pd.read_csv(f"{base}/model_input/NPC_Mic_Neuron_input_v2/sequences.bed", sep="\t", header=None,
                      names=["chrom", "start", "end", "split"], dtype={"chrom": str})
    win = d[["chrom", "start", "end"]].drop_duplicates()
    win["max_overlap_bp"], win["gap_to_train_bp"] = gap_to_train(win, bed[bed.split == "train"])
    d = d.merge(win, on=["chrom", "start", "end"])
    # Akita's random split assigns whole contigs, so overlap is expected to be 0 bp; what remains is
    # proximity: a test window next to a training window shares nearby sequence and structure.
    d["proximity"] = np.select([d.max_overlap_bp > 0, d.gap_to_train_bp <= 1_048_576],
                               ["overlaps a training window", "within 1 Mb of a training window"],
                               "more than 1 Mb from any training window")
    rows = []
    for c, g in d.groupby("cell"):
        for grp, gg in [("all test windows", g)] + list(g.groupby("proximity")):
            rows.append(dict(setting="random split 80/10/10", cell=c, subset=grp, n_windows=len(gg),
                             median_r_raw=gg[f"{metric}_raw"].median(),
                             median_r_smooth=gg[f"{metric}_smooth"].median(),
                             max_overlap_bp=gg.max_overlap_bp.max(),
                             median_gap_to_train_kb=gg.gap_to_train_bp.median() / 1e3))
        ref = held_long[(held_long.model == "A3") & (held_long.cell == c)]
        if not ref.empty:
            rows.append(dict(setting="chromosome split, 7 folds (A3)", cell=c, subset="held-out chromosomes",
                             n_windows=len(ref), median_r_raw=ref[f"{metric}_raw"].median(),
                             median_r_smooth=ref[f"{metric}_smooth"].median(), max_overlap_bp=0))
        rho = stats.spearmanr(g.gap_to_train_bp, g[f"{metric}_raw"], nan_policy="omit")
        rows.append(dict(setting="Spearman rho: r vs distance to training (random split)", cell=c,
                         subset=f"p = {rho[1]:.3g}", n_windows=len(g), median_r_raw=rho[0]))
    out = pd.DataFrame(rows)
    out["cell"] = pd.Categorical(out.cell, CELL_ORDER, ordered=True)
    return out.sort_values(["cell"], kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--metric", choices=["pearson", "spearman"], default="pearson")
    ap.add_argument("--skip_training", action="store_true")
    a = ap.parse_args(argv)
    base, out = a.base_dir.rstrip("/"), a.out
    os.makedirs(out, exist_ok=True)
    splits = fold_splits(base)
    check_single_splits(base, splits)
    tables = {}

    if not a.skip_training:
        log("▶ training logs")
        tables["T1_training"], tables["T1b_training_extra"] = training_tables(base)

    log("▶ held-out")
    held, a0_all, sel = heldout_long(base, splits, a.sigma)
    if held.empty:
        log("❌ no held-out metrics found; check --base_dir")
        return 1
    (tables["T2_heldout_by_fold"], tables["T3_heldout_by_chrom"], tables["T4_heldout_summary"],
     tables["T5_heldout_vs_A0"], tables["T6_A0_all_heads"]) = heldout_tables(held, a0_all, sel, a.metric)
    held.to_csv(f"{out}/per_window_heldout.csv.gz", index=False)

    log("▶ external v2")
    ext = external_long(base, splits, sel, a.sigma)
    tables["T7_external_by_chrom"], tables["T8_external_summary"], tables["T9_external_vs_A0"] = \
        external_tables(ext, a.metric)
    if not ext.empty:
        ext.to_csv(f"{out}/per_window_external.csv.gz", index=False)

    log("▶ random split")
    tables["T10_random_split"] = random_table(base, held, a.metric, a.sigma)

    # coverage report: which inputs had smoothed metrics
    cov = (pd.concat([held.assign(set="held-out"), ext.assign(set="external v2") if not ext.empty else None])
           .groupby(["set", "model", "has_smooth"], observed=True).source_file.nunique()
           .rename("n_files").reset_index())
    tables["T0_inputs"] = cov

    for k, t in tables.items():
        if t is None or t.empty:
            log(f"   {k}: (empty)")
            continue
        t.to_csv(f"{out}/{k}.csv", index=False, float_format="%.4g")
        log(f"   {k}: {len(t)} rows")
    try:
        with pd.ExcelWriter(f"{out}/all_tables.xlsx") as xw:
            for k, t in sorted(tables.items()):
                if t is not None and not t.empty:
                    t.to_excel(xw, sheet_name=k[:31], index=False)
        log(f"✅ {out}/all_tables.xlsx")
    except Exception as e:
        log(f"ℹ️  xlsx not written ({e}); CSVs are in {out}")

    s = tables["T4_heldout_summary"]
    cols = ["model", "cell", "n_folds", "fold_median_r_raw_mean", "fold_median_r_smooth_mean"]
    log("\n📊 held-out, mean of fold medians\n" + s[cols].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
