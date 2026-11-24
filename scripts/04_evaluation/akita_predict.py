#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akita_predict.py

Prediction stage for a joint multi-target Akita model, for chromosome-disjoint
folds and for independent datasets that hold a single cell type.

Pipeline
  1) Load model (params.json + weights)
  2) Stream the requested split(s) of one or more data_dirs
  3) Predict, keeping model heads and observed targets indexed SEPARATELY
  4) Save {file_prefix}_predictions_targets.npz (+ region table + geometry)
     and {file_prefix}_metadata.json
  5) Write {file_prefix}_per_window_metrics.tsv (unsmoothed, per chromosome)
  6) Optional leakage audit against the training intervals

Design notes
------------
* --splits: read any split, several splits, or 'all' (an independent dataset is
  scored on all of its windows).
* --pair OBS=HEAD: model heads and observed targets are paired explicitly, so a
  3-head model can be scored on a 1-target independent dataset without comparing
  the wrong head. The script refuses to guess.
* Streaming inference: windows are predicted batch by batch (2,339 windows x
  99,681 pixels x 3 targets does not fit comfortably in RAM).
* The npz carries chrom/start/end/split plus matrix_len, diagonal_offset and
  bin_size, so the plotting stage never re-derives geometry.
* Metrics are unsmoothed. Smoothing both maps before correlating suppresses the
  high-frequency component the model predicts worst and inflates r; --sigma is
  recorded for display use downstream but is not applied to any number here.
* diagonal_offset comes from statistics.json. infer_L() at offset 0 returns 446
  for a 99,681-long vector instead of 448.

Examples
--------
# held-out test chromosomes, joint model, all three cell types
python akita_predict.py \
  --model_dir  $MODEL --model_targets configs/cool_targets.txt \
  --data_dirs  $DATA --splits test \
  --output_dir $OUT/hold_test --file_prefix fold1_test \
  --train_data $DATA --strict_leakage --save_all_heads

# independent P35, every window, scored against the Neuron head
python akita_predict.py \
  --model_dir  $MODEL --model_targets configs/cool_targets.txt \
  --data_dirs  independent_testData/neuron_P35/all --splits all \
  --output_dir $OUT/indep_P35 --file_prefix P35_all \
  --pair Neuron_P35=Neuron_P16 --train_data $DATA

python akita_predict.py --self_test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SPLIT_ORDER = ("train", "valid", "test")


# ============================
# Low-level helpers
# ============================

def assert_all_exist(paths: List[str], what: str = "file/dir") -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing {what}(s):\n  " + "\n  ".join(missing))


def read_targets_txt(path: str) -> List[str]:
    """Target identifiers from a targets.txt file, a data dir, or pooled_meta.json."""
    if os.path.isdir(path):
        d = path
        path = os.path.join(path, "targets.txt")
    else:
        d = os.path.dirname(path)
    if not os.path.exists(path):
        for cand in (os.path.join(d, "pooled_meta.json"),
                     os.path.join(os.path.dirname(os.path.normpath(d)), "pooled_meta.json")):
            if os.path.exists(cand):
                with open(cand) as f:
                    meta = json.load(f)
                if meta.get("target_names"):
                    return [str(n) for n in meta["target_names"]]
        raise FileNotFoundError(f"no targets.txt (or pooled_meta.json) for {d}")
    df = pd.read_csv(path, sep="\t")
    if "identifier" not in df.columns:
        raise ValueError(f"'identifier' column missing in {path}")
    return df["identifier"].astype(str).tolist()


def read_statistics(data_dir: str) -> dict:
    p = os.path.join(data_dir, "statistics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    for cand in (os.path.join(data_dir, "pooled_meta.json"),
                 os.path.join(os.path.dirname(os.path.normpath(data_dir)), "pooled_meta.json")):
        if os.path.exists(cand):
            with open(cand) as f:
                return dict(json.load(f).get("stats", {}))
    return {}


def infer_L(vec_len: int, diagonal_offset: int = 2) -> int:
    """vec_len = m(m+1)/2 with m = L - diagonal_offset. 99681, offset 2 -> 448."""
    m = (np.sqrt(8 * vec_len + 1) - 1) / 2
    m_int = int(round(m))
    if m_int * (m_int + 1) // 2 != vec_len:
        raise ValueError(f"vector length {vec_len} is not triangular")
    return m_int + diagonal_offset


def read_sequences_bed(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "sequences.bed")
    assert_all_exist([path], what="sequences.bed")
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chrom", "start", "end", "split"], dtype={0: str})


def parse_splits(spec: Sequence[str], available: Sequence[str]) -> List[str]:
    """['all'] -> every split present in train/valid/test order; otherwise validate."""
    if len(spec) == 1 and spec[0] == "all":
        labels = [s for s in SPLIT_ORDER if s in available]
        return labels + [s for s in sorted(available) if s not in SPLIT_ORDER]
    bad = [s for s in spec if s not in available]
    if bad:
        raise ValueError(f"split(s) {bad} not in sequences.bed (available: {sorted(available)})")
    return list(spec)


def region_table(data_dirs: Sequence[str], splits_per_dir: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Windows in the exact order the TFRecords will be read: dir by dir, split by
    split, BED order within a split."""
    frames = []
    for di, (ddir, labels) in enumerate(zip(data_dirs, splits_per_dir)):
        bed = read_sequences_bed(ddir)
        for lab in labels:
            sub = bed[bed["split"] == lab].copy()
            sub["data_index"] = di
            sub["data_dir"] = ddir
            frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    out["seq_index"] = np.arange(len(out))
    out["region"] = out["chrom"] + ":" + out["start"].astype(str) + "-" + out["end"].astype(str)
    return out


# ============================
# head <-> target pairing
# ============================

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve_index(key: str, names: List[str], what: str) -> int:
    key = str(key).strip()
    if re.fullmatch(r"\d+", key):
        i = int(key)
        if not 0 <= i < len(names):
            raise ValueError(f"{what} index {i} out of range (have {len(names)}: {names})")
        return i
    hits = [i for i, n in enumerate(names) if _norm(n) == _norm(key)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"{what} {key!r} not resolvable among {names}")


def resolve_pairs(obs_names: List[str], head_names: List[str],
                  pair_specs: Optional[Sequence[str]] = None) -> List[Tuple[int, int]]:
    """[(observed index, model head index)]. Explicit --pair wins; otherwise match
    by name; otherwise identity only when the counts agree. Never guesses."""
    if pair_specs:
        pairs = []
        for spec in pair_specs:
            if "=" not in spec:
                raise ValueError(f"--pair expects OBS=HEAD, got {spec!r}")
            o, h = spec.split("=", 1)
            pairs.append((resolve_index(o, obs_names, "observed target"),
                          resolve_index(h, head_names, "model head")))
        return pairs
    pairs, claimed = [], set()
    for oi, on in enumerate(obs_names):
        hits = [i for i, h in enumerate(head_names) if _norm(h) == _norm(on)]
        if len(hits) == 1:
            pairs.append((oi, hits[0]))
            claimed.add(oi)
    if len(claimed) == len(obs_names):
        return pairs
    if len(obs_names) == len(head_names):
        return [(i, i) for i in range(len(obs_names))]
    missing = [n for i, n in enumerate(obs_names) if i not in claimed]
    raise ValueError(
        f"cannot tell which model head to score {missing} against. Heads are {head_names}. "
        f"Pass --pair {missing[0]}=<head name or index>."
    )


# ============================
# leakage audit
# ============================

def interval_overlap_bp(a: pd.DataFrame, b: pd.DataFrame) -> int:
    total = 0
    for chrom, a_c in a.groupby("chrom"):
        b_c = b[b["chrom"] == chrom]
        if b_c.empty:
            continue
        b_s = np.sort(b_c["start"].to_numpy())
        b_e = np.sort(b_c["end"].to_numpy())
        for s, e in zip(a_c["start"].to_numpy(), a_c["end"].to_numpy()):
            lo = np.searchsorted(b_e, s, side="right")
            hi = np.searchsorted(b_s, e, side="left")
            if hi > lo:
                ov = np.minimum(e, b_e[lo:hi]) - np.maximum(s, b_s[lo:hi])
                total += int(ov[ov > 0].sum())
    return total


def leakage_audit(regions: pd.DataFrame, train_data: Optional[str]) -> pd.DataFrame:
    """Evaluated intervals vs the training dataset's train/valid intervals.

    A set drawn from the reference dataset's own split is compared with itself;
    that row is flagged self_comparison and full overlap there is expected.
    """
    if not train_data:
        return pd.DataFrame()
    bed = read_sequences_bed(train_data)
    tnorm = os.path.normpath(train_data)
    rows = []
    for ref_split in ("train", "valid"):
        ref = bed[bed["split"] == ref_split]
        if ref.empty:
            continue
        for ddir, ev in regions.groupby("data_dir"):
            self_cmp = (os.path.normpath(ddir) == tnorm) and (ref_split in set(ev["split"]))
            shared = sorted(set(ref["chrom"]) & set(ev["chrom"]))
            rows.append(dict(
                data_dir=ddir, eval_splits=",".join(sorted(set(ev["split"]))),
                reference_split=ref_split, self_comparison=self_cmp,
                n_eval_seqs=len(ev), shared_chroms=",".join(shared),
                overlapping_bp=interval_overlap_bp(ev, ref) if shared else 0,
            ))
    return pd.DataFrame(rows)


# ============================
# model + streaming prediction
# ============================

def load_model(model_dir: str, model_file: str = "model_best.h5"):
    from basenji import seqnn  # late import: --self_test needs no TF

    params_file = os.path.join(model_dir, "params.json")
    weights = os.path.join(model_dir, model_file)
    assert_all_exist([params_file, weights], what="model artifact")
    with open(params_file) as f:
        params = json.load(f)
    model = seqnn.SeqNN(params["model"])
    model.restore(weights)
    print("✅ Model restored")
    try:
        print("   output shape:", model.model.output_shape)
    except Exception:
        pass
    return model, params


def build_seq_dataset(data_dir: str, split: str, batch_size: int):
    from basenji import dataset

    try:
        return dataset.SeqDataset(data_dir, split_label=split, batch_size=batch_size, mode="eval")
    except TypeError:
        return dataset.SeqDataset(data_dir, split, batch_size)


def iter_batches(data_dirs: Sequence[str], splits_per_dir: Sequence[Sequence[str]],
                 batch_size: int):
    """(inputs, targets) batches in the same order as region_table()."""
    for ddir, labels in zip(data_dirs, splits_per_dir):
        for lab in labels:
            for x, y in build_seq_dataset(ddir, lab, batch_size).dataset:
                yield x.numpy(), y.numpy()


def predict_and_score(model, data_dirs, splits_per_dir, regions, pairs,
                      batch_size, diagonal_offset, keep_heads,
                      obs_names, head_names, save_npz=True):
    """Stream inference; return per-window metrics and (optionally) stacked maps."""
    from scipy.stats import pearsonr, spearmanr

    rows, kept_preds, kept_targs = [], [], []
    n_seen, L = 0, None
    t0 = time.time()

    for x, y in iter_batches(data_dirs, splits_per_dir, batch_size):
        yh = np.asarray(model.model.predict_on_batch(x))
        if L is None:
            L = infer_L(y.shape[1], diagonal_offset)
            need_h = max(h for _, h in pairs)
            need_o = max(o for o, _ in pairs)
            if yh.shape[-1] <= need_h:
                raise RuntimeError(f"model emits {yh.shape[-1]} heads; pair needs index {need_h}")
            if y.shape[-1] <= need_o:
                raise RuntimeError(f"dataset has {y.shape[-1]} targets; pair needs index {need_o}")
            print(f"   vector={y.shape[1]} -> L={L} bins (diagonal_offset={diagonal_offset}); "
                  f"{y.shape[-1]} observed target(s), {yh.shape[-1]} model head(s)")
        if save_npz:
            kept_preds.append(yh[..., keep_heads].astype(np.float16))
            kept_targs.append(y.astype(np.float16))

        for b in range(y.shape[0]):
            if n_seen >= len(regions):
                raise RuntimeError("more TFRecord windows than sequences.bed rows — "
                                   "region labels would be wrong; check --splits")
            reg = regions.iloc[n_seen]
            for oi, hi in pairs:
                obs = y[b, :, oi].astype(np.float64)
                pred = yh[b, :, hi].astype(np.float64)
                m = np.isfinite(obs) & np.isfinite(pred)
                if m.sum() < 3:
                    continue
                rows.append(dict(
                    seq_index=int(reg["seq_index"]), chrom=reg["chrom"],
                    start=int(reg["start"]), end=int(reg["end"]), region=reg["region"],
                    split=reg["split"], data_dir=reg["data_dir"],
                    obs_index=oi, obs_target=obs_names[oi],
                    head_index=hi, model_head=head_names[hi],
                    matched=(_norm(obs_names[oi]) == _norm(head_names[hi])),
                    n_pixels=int(m.sum()),
                    mse=float(np.mean((pred[m] - obs[m]) ** 2)),
                    pearson=float(pearsonr(pred[m], obs[m])[0]),
                    spearman=float(spearmanr(pred[m], obs[m])[0]),
                    obs_sd=float(np.std(obs[m])), pred_sd=float(np.std(pred[m])),
                ))
            n_seen += 1

    if n_seen != len(regions):
        print(f"⚠️  {n_seen} windows predicted but sequences.bed lists {len(regions)}; "
              f"per-chromosome labels are only valid if the orders match", file=sys.stderr)
    print(f"   {n_seen} windows in {time.time()-t0:.0f}s")

    preds = np.concatenate(kept_preds, axis=0) if kept_preds else None
    targs = np.concatenate(kept_targs, axis=0) if kept_targs else None
    return pd.DataFrame(rows), preds, targs, L


# ============================
# self-test (no TensorFlow)
# ============================

def self_test() -> int:
    assert infer_L(99681, 2) == 448, "Akita 448-bin case"
    for L in (12, 40, 448):
        for d in (0, 1, 2):
            assert infer_L(len(np.triu_indices(L, d)[0]), d) == L

    heads = ["NPC_P16", "Mic_AD", "Neuron_P16"]
    assert resolve_pairs(["Neuron_P35"], heads, ["Neuron_P35=Neuron_P16"]) == [(0, 2)]
    assert resolve_pairs(["Neuron_P35"], heads, ["0=2"]) == [(0, 2)]
    assert resolve_pairs(heads, heads) == [(0, 0), (1, 1), (2, 2)], "name match"
    assert resolve_pairs(["Neuron-P16", "npc p16"], heads) == [(0, 2), (1, 0)], "order-independent"
    try:
        resolve_pairs(["Neuron_P35"], heads)
    except ValueError as e:
        assert "--pair" in str(e)
    else:
        raise AssertionError("ambiguous pairing must raise")

    avail = {"train", "valid", "test"}
    assert parse_splits(["all"], avail) == ["train", "valid", "test"]
    assert parse_splits(["test"], avail) == ["test"]
    try:
        parse_splits(["fold0"], avail)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown split must raise")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d1 = os.path.join(td, "ds1"); os.makedirs(d1)
        with open(os.path.join(d1, "sequences.bed"), "w") as f:
            f.write("chr3\t0\t100\ttrain\nchr10\t0\t100\ttest\nchr8\t0\t100\tvalid\n")
        rt = region_table([d1], [["train", "valid", "test"]])
        assert rt["chrom"].tolist() == ["chr3", "chr8", "chr10"], "dir/split/BED order"
        assert rt["seq_index"].tolist() == [0, 1, 2]

        audit = leakage_audit(rt, d1)
        self_rows = audit[audit["self_comparison"]]
        assert len(self_rows) == 2, "own train and valid splits flagged as self-comparison"
        real = audit[(audit["overlapping_bp"] > 0) & (~audit["self_comparison"])]
        assert real.empty, "no true leakage in the single-dataset case"

    a = pd.DataFrame(dict(chrom=["chr1", "chr2"], start=[0, 0], end=[50, 10]))
    b = pd.DataFrame(dict(chrom=["chr1"], start=[40], end=[150]))
    assert interval_overlap_bp(a, b) == 10
    assert interval_overlap_bp(pd.DataFrame(dict(chrom=["chr10"], start=[0], end=[9])), b) == 0

    print("✅ akita_predict self-test passed")
    return 0


# ============================
# CLI & main
# ============================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Akita predict -> NPZ/JSON + per-window metrics (per chromosome)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--self_test", action="store_true", help="run internal checks and exit")
    p.add_argument("--model_dir", help="directory with params.json and the weights")
    p.add_argument("--model_file", default="model_best.h5")
    p.add_argument("--model_targets", help="targets.txt defining the model HEAD order "
                                           "(the joint training targets.txt)")
    p.add_argument("--data_dirs", nargs="+", help="one or more datasets to predict")
    p.add_argument("--splits", nargs="+", default=["test"],
                   help="'all', or split labels e.g. test / train valid test")
    p.add_argument("--output_dir")
    p.add_argument("--file_prefix", default="akita")
    p.add_argument("--pair", action="append", default=None, metavar="OBS=HEAD",
                   help="observed target = model head (name or index); required when "
                        "the dataset's targets and the model's heads differ")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--diagonal_offset", type=int, default=None, help="override statistics.json")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="recorded for DISPLAY smoothing downstream; no metric is smoothed")
    p.add_argument("--save_all_heads", action="store_true",
                   help="store every model head in the npz (needed for cross-head controls); "
                        "default stores only the paired heads to keep the file small")
    p.add_argument("--skip_npz", action="store_true", help="write metrics only, no maps")
    p.add_argument("--train_data", help="training dataset dir, for the leakage audit")
    p.add_argument("--strict_leakage", action="store_true",
                   help="abort if evaluated sequence overlaps training sequence")
    p.add_argument("--seed", type=int, default=1337)
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.self_test:
        return self_test()
    for req in ("model_dir", "model_targets", "data_dirs", "output_dir"):
        if getattr(args, req) is None:
            print(f"❌ --{req} is required", file=sys.stderr)
            return 2

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, args.file_prefix)
    assert_all_exist(list(args.data_dirs), what="data_dir")

    # ---- targets, heads, geometry -------------------------------------------
    head_names = read_targets_txt(args.model_targets)
    obs_lists = [read_targets_txt(d) for d in args.data_dirs]
    for i in range(1, len(obs_lists)):
        if obs_lists[i] != obs_lists[0]:
            print(f"❌ targets.txt differ between {args.data_dirs[0]} and {args.data_dirs[i]}; "
                  f"predict them separately", file=sys.stderr)
            return 2
    obs_names = obs_lists[0]
    print(f"🧬 model heads     : " + ", ".join(f"[{i}] {n}" for i, n in enumerate(head_names)))
    print(f"🧬 observed targets: " + ", ".join(f"[{i}] {n}" for i, n in enumerate(obs_names)))

    try:
        pairs = resolve_pairs(obs_names, head_names, args.pair)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 2
    for oi, hi in pairs:
        print(f"   observed [{oi}] {obs_names[oi]}  ->  model head [{hi}] {head_names[hi]}")

    stats = read_statistics(args.data_dirs[0])
    diag_off = args.diagonal_offset if args.diagonal_offset is not None else int(
        stats.get("diagonal_offset", 2))
    bin_size = int(stats.get("pool_width", 2048))

    if args.train_data:
        tstats = read_statistics(args.train_data)
        diffs = {k: (stats.get(k), tstats.get(k)) for k in
                 ("seq_length", "pool_width", "crop_bp", "diagonal_offset", "target_length")
                 if k in stats and k in tstats and stats[k] != tstats[k]}
        if diffs:
            print(f"⚠️  processing parameters differ from the training data: {diffs}\n"
                  f"   observed values are not on the same scale; prefer correlation over MSE",
                  file=sys.stderr)

    # ---- window table --------------------------------------------------------
    splits_per_dir = []
    for d in args.data_dirs:
        avail = set(read_sequences_bed(d)["split"].unique())
        try:
            splits_per_dir.append(parse_splits(args.splits, avail))
        except ValueError as e:
            print(f"❌ {d}: {e}", file=sys.stderr)
            return 2
    regions = region_table(args.data_dirs, splits_per_dir)
    print(f"\n📁 {len(regions)} windows | splits {dict(regions['split'].value_counts())} "
          f"| chroms {','.join(sorted(regions['chrom'].unique()))}")

    # ---- leakage audit (before GPU time) ------------------------------------
    audit = leakage_audit(regions, args.train_data)
    if not audit.empty:
        audit.to_csv(f"{prefix}_leakage_audit.tsv", sep="\t", index=False)
        print("\n🔍 Leakage audit")
        print(audit[["reference_split", "self_comparison", "shared_chroms",
                     "overlapping_bp"]].to_string(index=False))
        real = audit[(audit["overlapping_bp"] > 0) & (~audit["self_comparison"])]
        if len(real):
            if args.strict_leakage:
                print("❌ evaluated sequence overlaps training sequence; aborting "
                      "(--strict_leakage)", file=sys.stderr)
                return 2
            print("⚠️  evaluated sequence overlaps training sequence (expected when "
                  "predicting an independent sample on training chromosomes)", file=sys.stderr)
        else:
            print("✅ zero overlapping bp against train and valid intervals")

    # ---- model ---------------------------------------------------------------
    import tensorflow as tf  # noqa: E402

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    try:
        for g in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    model, params = load_model(args.model_dir, args.model_file)
    n_out = model.model.output_shape[-1]
    if n_out != len(head_names):
        print(f"❌ model emits {n_out} heads but --model_targets lists {len(head_names)}; "
              f"head names would be misaligned", file=sys.stderr)
        return 2

    keep_heads = list(range(n_out)) if args.save_all_heads else sorted({h for _, h in pairs})

    # ---- predict -------------------------------------------------------------
    print("\n▶ predicting")
    metrics, preds, targs, L = predict_and_score(
        model, args.data_dirs, splits_per_dir, regions, pairs, args.batch_size,
        diag_off, keep_heads, obs_names, head_names, save_npz=not args.skip_npz)

    metrics.to_csv(f"{prefix}_per_window_metrics.tsv", sep="\t", index=False)
    print(f"✅ {prefix}_per_window_metrics.tsv")

    summary = (metrics.groupby(["chrom", "obs_target", "model_head"])
               .agg(n=("pearson", "size"), median_r=("pearson", "median"),
                    mean_r=("pearson", "mean"), sd_r=("pearson", "std"),
                    median_mse=("mse", "median"))
               .reset_index())
    summary.to_csv(f"{prefix}_summary_by_chromosome.csv", index=False)

    # ---- npz + metadata ------------------------------------------------------
    npz_path = f"{prefix}_predictions_targets.npz"
    if preds is not None:
        np.savez_compressed(
            npz_path,
            test_preds_subset=preds,          # (N, vec, len(keep_heads))
            test_targets=targs,               # (N, vec, n_obs)
            chrom=regions["chrom"].to_numpy().astype("U16"),
            start=regions["start"].to_numpy(),
            end=regions["end"].to_numpy(),
            split=regions["split"].to_numpy().astype("U16"),
            model_head_names=np.array([head_names[h] for h in keep_heads], dtype="U64"),
            obs_target_names=np.array(obs_names, dtype="U64"),
            pair_obs_index=np.array([o for o, _ in pairs]),
            pair_head_index=np.array([keep_heads.index(h) for _, h in pairs]),
            matrix_len=np.array(L), diagonal_offset=np.array(diag_off),
            bin_size=np.array(bin_size),
        )
        print(f"💾 {npz_path}  preds={preds.shape} targets={targs.shape} (float16)")

    with open(f"{prefix}_metadata.json", "w") as f:
        json.dump(dict(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_dir=args.model_dir, model_file=args.model_file,
            model_targets=args.model_targets, model_head_names=head_names,
            data_dirs=list(args.data_dirs), splits=splits_per_dir,
            observed_targets=obs_names,
            comparisons=[dict(obs_index=o, obs_target=obs_names[o],
                              head_index=h, model_head=head_names[h]) for o, h in pairs],
            saved_heads=[head_names[h] for h in keep_heads],
            n_windows=int(len(regions)), matrix_len=int(L),
            diagonal_offset=int(diag_off), bin_size=int(bin_size),
            sigma_for_display=args.sigma, batch_size=args.batch_size,
            train_data=args.train_data, command=" ".join(sys.argv),
        ), f, indent=2)

    print("\n📊 median per-window Pearson r")
    print(summary.pivot_table(index="chrom", columns="obs_target",
                              values="median_r").round(4).to_string())
    print(f"\nNext:\n  python akita_eval.py --npz {npz_path} "
          f"--out-dir {args.output_dir}/figures --tag {args.file_prefix} --group-by chrom")
    return 0


if __name__ == "__main__":
    sys.exit(main())
