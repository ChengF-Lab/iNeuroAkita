#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_akita_predict.py

Pipeline:
  1) Load Akita model (params.json + model_best.h5)
  2) Load and concatenate test data from one or more data_dirs
  3) Predict (optionally select subset of targets)
  4) Save {file_prefix}_predictions_targets.npz and {file_prefix}_metadata.json
  5) (Optional) Build {file_prefix}_smoothed_metrics.csv (per-seq, smoothed off-diag MSE & Pearson)

Usage (examples):
  python run_akita_predict.py \
    --model_dir /path/to/model \
    --data_dirs /path/to/dataOut_E1 /path/to/dataOut_E2 \
    --output_dir /path/to/predict_out \
    --file_prefix akita_basic_DS_E_Male \
    --selected_targets 0 \
    --batch_size 2 \
    --make_smoothed_metrics \
    --sigma 2.0 \
    --add_region

Notes:
  * Expects each data_dir to contain sequences.bed, targets.txt, and 'test' TFRecords
  * selected_targets is a comma-separated list (e.g., "0,1") or the word "all"
  * Designed for Python 3.8
"""

import os
import json
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import tensorflow as tf
from basenji import dataset, seqnn


# ============================
# Low-level helpers
# ============================

def assert_all_exist(paths: List[str], what: str = "file/dir") -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing {what}(s):\n  " + "\n  ".join(missing))


def read_targets_txt(path: str) -> List[str]:
    df = pd.read_csv(path, sep="\t")
    if "identifier" not in df.columns:
        raise ValueError(f"'identifier' column missing in targets.txt at: {path}")
    return df["identifier"].tolist()


def infer_L(vec_len: int) -> int:
    # vec_len = L*(L+1)/2  -> solve for L
    return int((np.sqrt(8 * vec_len + 1) - 1) // 2)


def from_upper_triu(vec: np.ndarray, L: int) -> np.ndarray:
    mat = np.zeros((L, L), dtype=np.float32)
    iu = np.triu_indices(L)
    mat[iu] = vec
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat


def spatial_smoother(mat: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    sm = gaussian_filter(mat, sigma=sigma)
    np.fill_diagonal(sm, np.diag(mat))
    return sm


def offdiag_flat(mat: np.ndarray) -> np.ndarray:
    L = mat.shape[0]
    mask = ~np.eye(L, dtype=bool).reshape(-1)
    return mat.reshape(-1)[mask]


def load_region_labels_from_beds(bed_paths: List[str]) -> pd.Series:
    """Concatenate 'test' rows from multiple sequences.bed in same order as data_dirs."""
    regions = []
    for bed in bed_paths:
        df = pd.read_csv(bed, sep="\t", names=["chr", "start", "end", "type"])
        test = df[df["type"] == "test"].reset_index(drop=True)
        regions.extend(test.apply(lambda r: f"{r['chr']}:{r['start']}-{r['end']}", axis=1).tolist())
    return pd.Series(regions, dtype=str)


# ============================
# Core steps
# ============================

def load_model(model_dir: str) -> seqnn.SeqNN:
    params_file = os.path.join(model_dir, "params.json")
    model_file = os.path.join(model_dir, "model_best.h5")
    assert_all_exist([params_file, model_file], what="model artifact")

    with open(params_file) as f:
        params = json.load(f)
    params_model = params["model"]

    model = seqnn.SeqNN(params_model)
    model.restore(model_file)
    print("✅ Model restored")

    try:
        print("Model output shape:", model.model.output_shape)
    except Exception:
        pass

    return model


def load_concat_test(data_dirs: List[str], batch_size: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load 'test' split from each dir and concatenate."""
    assert_all_exist(data_dirs, what="data_dir")

    bed_paths = [os.path.join(dd, "sequences.bed") for dd in data_dirs]
    tgt_paths = [os.path.join(dd, "targets.txt") for dd in data_dirs]
    assert_all_exist(bed_paths, what="sequences.bed")
    assert_all_exist(tgt_paths, what="targets.txt")

    # Ensure identical targets.txt
    target_lists = [read_targets_txt(tp) for tp in tgt_paths]
    for i in range(1, len(target_lists)):
        if target_lists[i] != target_lists[0]:
            raise ValueError(f"targets.txt mismatch:\n  {tgt_paths[0]}\n  {tgt_paths[i]}")
    target_names = target_lists[0]
    print(f"✅ Targets loaded — {len(target_names)} outputs")

    combined_inputs, combined_targets = [], []
    for ddir in data_dirs:
        print(f"→ Loading test split from: {ddir}")
        ds = dataset.SeqDataset(ddir, "test", batch_size=batch_size)
        inputs, targets = ds.numpy(return_inputs=True, return_outputs=True)
        combined_inputs.append(inputs.astype(np.float32))
        combined_targets.append(targets)

    test_inputs = np.concatenate(combined_inputs, axis=0)
    test_targets = np.concatenate(combined_targets, axis=0)
    print(f"✅ Combined test inputs : {test_inputs.shape}, dtype={test_inputs.dtype}")
    print(f"✅ Combined test targets: {test_targets.shape}, dtype={test_targets.dtype}")

    return test_inputs, test_targets, bed_paths, tgt_paths


def predict_subset(model: seqnn.SeqNN,
                   test_inputs: np.ndarray,
                   test_targets: np.ndarray,
                   selected_targets: List[int],
                   batch_size: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Run prediction and slice outputs to selected targets list (or all if empty)."""
    T_total = test_targets.shape[-1]
    if not selected_targets:
        selected_targets = list(range(T_total))

    if min(selected_targets) < 0 or max(selected_targets) >= T_total:
        raise IndexError(f"selected_targets out of range. Valid 0..{T_total-1}, got {selected_targets}")

    preds_full = model.predict(test_inputs, batch_size=batch_size)  # (N, vec_len, T_total)
    preds_subset = preds_full[..., selected_targets]
    print(f"✅ Predictions generated: full={preds_full.shape}, subset={preds_subset.shape}")
    return preds_subset, preds_full, selected_targets


def save_npz_and_meta(npz_path: str,
                      meta_path: str,
                      preds_subset: np.ndarray,
                      test_targets: np.ndarray,
                      model_dir: str,
                      data_dirs: List[str],
                      bed_paths: List[str],
                      tgt_paths: List[str],
                      target_names: List[str],
                      selected_targets: List[int],
                      batch_size: int,
                      file_prefix: str,
                      output_dir: str,
                      sigma_smooth: float) -> None:
    np.savez_compressed(
        npz_path,
        test_preds_subset=preds_subset,
        test_targets=test_targets,
    )
    print(f"💾 Saved predictions/targets NPZ → {npz_path}")

    meta = {
        "model_dir": model_dir,
        "data_dirs": data_dirs,
        "bed_paths": bed_paths,
        "targets_paths": tgt_paths,
        "target_names": target_names,
        "selected_target_indices": selected_targets,
        "test_targets_shape": tuple(test_targets.shape),
        "test_preds_subset_shape": tuple(preds_subset.shape),
        "batch_size": batch_size,
        "file_prefix": file_prefix,
        "output_dir": output_dir,
        "sigma_smooth": sigma_smooth,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"💾 Saved metadata JSON → {meta_path}")


def build_smoothed_metrics_csv(npz_path: str,
                               out_csv: str,
                               bed_paths: List[str],
                               tgt_paths: List[str],
                               selected_targets: List[int],
                               sigma: float = 2.0,
                               add_region: bool = True) -> None:
    """
    Make {file_prefix}_smoothed_metrics.csv from NPZ.
    Columns: Region (optional), MSE_<target>, Pearson_<target>
    Uses Gaussian smoothing on matrices and evaluates off-diagonals.
    """
    npz = np.load(npz_path)
    preds = npz["test_preds_subset"]   # (N, vec_len, T_sel)
    targs = npz["test_targets"]        # (N, vec_len, T_total)

    N, vec_len, T_sel = preds.shape
    L = infer_L(vec_len)

    # names for selected targets
    tdf = pd.read_csv(tgt_paths[0], sep="\t")
    all_names = tdf["identifier"].tolist()
    if not selected_targets:
        selected_targets = list(range(len(all_names)))
    target_names_sel = [all_names[i] for i in selected_targets]

    # match target slice for ground truth
    targs_sel = targs[..., selected_targets]
    if targs_sel.shape[-1] != T_sel:
        raise ValueError(
            f"Mismatch: preds subset has T_sel={T_sel} but targets subset has {targs_sel.shape[-1]}."
        )

    regions = None
    if add_region:
        regions = load_region_labels_from_beds(bed_paths)
        if len(regions) != N:
            print(f"[warn] Region count ({len(regions)}) != N ({N}); omitting Region column.")
            regions = None

    out: Dict[str, List[float]] = {}
    if regions is not None:
        out["Region"] = list(regions)

    for t in range(T_sel):
        mse_list, r_list = [], []
        for i in range(N):
            X = spatial_smoother(from_upper_triu(targs_sel[i, :, t], L), sigma=sigma)
            Y = spatial_smoother(from_upper_triu(preds[i, :, t],      L), sigma=sigma)
            x = offdiag_flat(X)
            y = offdiag_flat(Y)

            if x.size and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                mse = float(np.mean((y - x) ** 2))
            else:
                mse = float("nan")
            mse_list.append(mse)

            if x.size > 1 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                xm, ym = x.mean(), y.mean()
                xv, yv = x - xm, y - ym
                denom  = np.sqrt((xv*xv).sum() * (yv*yv).sum())
                r = float((xv*yv).sum() / denom) if denom > 0 else float("nan")
            else:
                r = float("nan")
            r_list.append(r)

        out[f"MSE_{target_names_sel[t]}"]     = mse_list
        out[f"Pearson_{target_names_sel[t]}"] = r_list

    df = pd.DataFrame(out)
    df.index.name = "Sequence_Index"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ wrote {out_csv}  (N={N}, L={L}, T_sel={T_sel}, sigma={sigma})")


# ============================
# CLI & main
# ============================

def parse_selected_targets(arg: str, T_total: Optional[int] = None) -> List[int]:
    """Parse '--selected_targets' as 'all' or comma-separated integers."""
    if arg is None or arg.strip() == "" or arg.strip().lower() == "all":
        return []
    parts = [p.strip() for p in arg.split(",")]
    idxs = [int(p) for p in parts if p != ""]
    if T_total is not None and (min(idxs) < 0 or max(idxs) >= T_total):
        raise IndexError(f"selected_targets out of range 0..{T_total-1}: {idxs}")
    return idxs


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Akita predict → NPZ/JSON → (optional) smoothed metrics CSV")
    p.add_argument("--model_dir", required=True, help="Directory with params.json and model_best.h5")
    p.add_argument("--data_dirs", nargs="+", required=True, help="One or more test data directories")
    p.add_argument("--output_dir", required=True, help="Where to write outputs")
    p.add_argument("--file_prefix", required=True, help="Base name for outputs (no extension)")
    p.add_argument("--selected_targets", default="0", help="'all' or comma-separated target indices (e.g., '0,1')")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--make_smoothed_metrics", action="store_true")
    p.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma for smoothing")
    p.add_argument("--add_region", action="store_true", help="Include Region column from sequences.bed")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    return p


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Reproducibility (best-effort)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    try:
        # prevent TF from pre-allocating all GPU memory (nice on shared GPUs)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    # 1) Load model
    model = load_model(args.model_dir)

    # 2) Load data (and validate targets.txt)
    test_inputs, test_targets, bed_paths, tgt_paths = load_concat_test(args.data_dirs, args.batch_size)

    # 3) Predict subset
    selected_targets = parse_selected_targets(args.selected_targets, T_total=test_targets.shape[-1])
    preds_subset, _, selected_targets = predict_subset(
        model, test_inputs, test_targets, selected_targets, args.batch_size
    )

    # 4) Save NPZ + metadata
    npz_path  = os.path.join(args.output_dir, f"{args.file_prefix}_predictions_targets.npz")
    meta_path = os.path.join(args.output_dir, f"{args.file_prefix}_metadata.json")
    save_npz_and_meta(
        npz_path=npz_path,
        meta_path=meta_path,
        preds_subset=preds_subset,
        test_targets=test_targets,
        model_dir=args.model_dir,
        data_dirs=args.data_dirs,
        bed_paths=bed_paths,
        tgt_paths=tgt_paths,
        target_names=read_targets_txt(tgt_paths[0]),
        selected_targets=selected_targets,
        batch_size=args.batch_size,
        file_prefix=args.file_prefix,
        output_dir=args.output_dir,
        sigma_smooth=args.sigma,
    )

    # 5) Optional: build smoothed metrics CSV
    if args.make_smoothed_metrics:
        out_csv = os.path.join(args.output_dir, f"{args.file_prefix}_smoothed_metrics.csv")
        build_smoothed_metrics_csv(
            npz_path=npz_path,
            out_csv=out_csv,
            bed_paths=bed_paths,
            tgt_paths=tgt_paths,
            selected_targets=selected_targets,
            sigma=args.sigma,
            add_region=args.add_region,
        )

    # 6) Friendly next-step hint
    print("\nNext with akita_eval_utils.py / run_akita_eval.py:")
    print(f"  file_prefix   = '{args.file_prefix}'")
    print(f"  npz_path      = '{npz_path}'")
    print(f"  bed_file1     = '{bed_paths[0]}'")
    if len(bed_paths) > 1:
        print(f"  bed_file2     = '{bed_paths[1]}'")
    print(f"  targets_file  = '{tgt_paths[0]}'")
    print(f"  output_dir    = '{args.output_dir}'")
    if args.make_smoothed_metrics:
        print(f"  smoothed_metrics_csv = '{os.path.join(args.output_dir, args.file_prefix + '_smoothed_metrics.csv')}'")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
