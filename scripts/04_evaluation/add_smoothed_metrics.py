#!/usr/bin/env python3
"""
add_smoothed_metrics.py — per-window metrics after Gaussian smoothing of BOTH maps,
computed from an npz written by akita_predict.py (iNeuroAkita or Akita zero-shot).
No GPU and no re-prediction needed.

What it reports for every (window, observed target, model head) pair:
  pearson_raw / spearman_raw / mse_raw          unsmoothed (same as akita_predict.py)
  pearson_smooth / spearman_smooth / mse_smooth  both maps blurred with sigma bins first
  delta_pearson = pearson_smooth - pearson_raw

Implementation notes:
  * correct geometry: the vector is the upper triangle at diagonal_offset=2 (448 bins).
    offset 0 (446 bins) would shift every pixel by two bins and mirror the shifted map,
    so the smoothing would mix wrongly registered pixels;
  * NaN-aware smoothing (masked pixels do not pull neighbours toward 0);
  * scored on the upper triangle only (not every pixel twice);
  * the model head and observed target are paired explicitly (from the npz or --pair).

--replace writes pearson/spearman/mse = the smoothed values (raw kept as *_raw), so
akita_eval.py --metrics-tsv and compare_to_akita0.py can use the file unchanged. Use the SAME sigma for iNeuroAkita and Akita zero-shot.

  python add_smoothed_metrics.py --npz $OUT/indep_neuron/neuron_all_predictions_targets.npz \
      --sigma 2 --metrics-tsv $OUT/indep_neuron/neuron_all_per_window_metrics.tsv
  python add_smoothed_metrics.py --self_test
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def vec_to_mat(v, L, d):
    """upper-triangle vector (offset d) -> symmetric LxL, NaN within d of the diagonal."""
    m = np.full((L, L), np.nan)
    iu = np.triu_indices(L, d)
    m[iu] = v
    m[(iu[1], iu[0])] = v
    return m


def nan_gauss(m, sigma):
    """Gaussian blur that ignores NaN pixels (normalised convolution)."""
    ok = np.isfinite(m)
    num = gaussian_filter(np.where(ok, m, 0.0), sigma, mode="constant")
    den = gaussian_filter(ok.astype(float), sigma, mode="constant")
    out = np.full_like(m, np.nan)
    np.divide(num, den, out=out, where=den > 1e-6)
    out[~ok] = np.nan            # score only pixels that were observed
    return out


def scores(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() < 3 or np.std(p[m]) == 0 or np.std(o[m]) == 0:
        return np.nan, np.nan, np.nan, int(m.sum())
    return (float(pearsonr(p[m], o[m])[0]), float(spearmanr(p[m], o[m])[0]),
            float(np.mean((p[m] - o[m]) ** 2)), int(m.sum()))


def resolve(key, names):
    if re.fullmatch(r"\d+", str(key)):
        return int(key)
    hits = [i for i, n in enumerate(names) if _norm(n) == _norm(key)]
    if len(hits) != 1:
        raise ValueError(f"{key!r} not found in {list(names)}")
    return hits[0]


def run(npz_path, sigma, pair_specs=None, max_windows=None):
    z = np.load(npz_path, allow_pickle=False)
    P, T = z["test_preds_subset"], z["test_targets"]
    heads = [str(x) for x in z["model_head_names"]]
    obs = [str(x) for x in z["obs_target_names"]]
    L, d = int(z["matrix_len"]), int(z["diagonal_offset"])
    if pair_specs:
        pairs = [(resolve(s.split("=")[0], obs), resolve(s.split("=")[1], heads)) for s in pair_specs]
    else:
        pairs = list(zip(z["pair_obs_index"].tolist(), z["pair_head_index"].tolist()))
    iu = np.triu_indices(L, d)
    if len(iu[0]) != P.shape[1]:
        raise ValueError(f"vector length {P.shape[1]} != triangle of L={L}, offset={d}")
    N = P.shape[0] if max_windows is None else min(max_windows, P.shape[0])
    rows = []
    acc = {}   # pooled-pixel sums per pair, smoothed maps
    for i in range(N):
        for oi, hi in pairs:
            p = P[i, :, hi].astype(np.float64)
            o = T[i, :, oi].astype(np.float64)
            r0 = scores(p, o)
            ps = nan_gauss(vec_to_mat(p, L, d), sigma)[iu]
            os_ = nan_gauss(vec_to_mat(o, L, d), sigma)[iu]
            r1 = scores(ps, os_)
            k = np.isfinite(ps) & np.isfinite(os_)
            a_ = acc.setdefault((oi, hi), np.zeros(6))
            x, y = ps[k], os_[k]
            a_ += [k.sum(), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]
            rows.append(dict(seq_index=i, chrom=str(z["chrom"][i]), start=int(z["start"][i]),
                             end=int(z["end"][i]), split=str(z["split"][i]),
                             region=f"{z['chrom'][i]}:{int(z['start'][i])}-{int(z['end'][i])}",
                             obs_index=oi, obs_target=obs[oi],
                             head_index=hi, model_head=heads[hi],   # head_index = index in the npz
                             pearson_raw=r0[0], spearman_raw=r0[1], mse_raw=r0[2],
                             pearson_smooth=r1[0], spearman_smooth=r1[1], mse_smooth=r1[2],
                             n_pixels=r1[3], smooth_sigma=sigma))
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{N} windows", flush=True)
    df = pd.DataFrame(rows)
    # pooled over all pixels of all windows (what akita_eval shows as 'global' / 'pooled-pixel')
    for (oi, hi), (n, sx, sy, sxx, syy, sxy) in acc.items():
        mse = (sxx + syy - 2 * sxy) / n
        cov = sxy / n - (sx / n) * (sy / n)
        r = cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))
        sel = (df['obs_index'] == oi) & (df['head_index'] == hi)
        df.loc[sel, 'pooled_mse_smooth'] = mse
        df.loc[sel, 'pooled_r_smooth'] = r
    df["delta_pearson"] = df["pearson_smooth"] - df["pearson_raw"]
    return df


def self_test():
    rng = np.random.default_rng(0)
    L, d = 40, 2
    iu = np.triu_indices(L, d)
    yy, xx = np.mgrid[0:L, 0:L]
    true = np.sin(xx / 6.0) * np.cos(yy / 6.0); true = (true + true.T) / 2
    v_true = true[iu]
    obs = v_true + rng.normal(0, 0.8, v_true.size)          # noisy "Hi-C"
    pred = v_true + rng.normal(0, 0.3, v_true.size)          # decent model
    m = vec_to_mat(obs, L, d)
    assert np.allclose(m, m.T, equal_nan=True) and np.isnan(m[5, 6]) and np.isfinite(m[5, 7])
    assert np.allclose(vec_to_mat(obs, L, d)[iu], obs), "round trip"
    r_raw = scores(pred, obs)[0]
    r_s = scores(nan_gauss(vec_to_mat(pred, L, d), 2)[iu], nan_gauss(vec_to_mat(obs, L, d), 2)[iu])[0]
    assert r_s > r_raw, "smoothing noisy maps should raise r"
    # NaN handling: a masked pixel stays masked and does not drag neighbours to 0
    o2 = obs.copy(); o2[100] = np.nan
    s2 = nan_gauss(vec_to_mat(o2, L, d), 2)[iu]
    assert np.isnan(s2[100]) and np.isfinite(s2[101])
    # npz round trip with the akita_predict layout
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "x.npz")
        P = np.stack([pred, pred * 0], -1)[None].astype(np.float16)
        np.savez(f, test_preds_subset=P, test_targets=obs[None, :, None].astype(np.float16),
                 model_head_names=np.array(["NPC_P16", "Neuron_P16"]),
                 obs_target_names=np.array(["Neuron_P35"]), pair_obs_index=np.array([0]),
                 pair_head_index=np.array([0]), matrix_len=np.array(L), diagonal_offset=np.array(d),
                 chrom=np.array(["chr10"]), start=np.array([0]), end=np.array([1]),
                 split=np.array(["test"]))
        df = run(f, 2, ["Neuron_P35=NPC_P16"])
        assert len(df) == 1 and df["delta_pearson"].iloc[0] > 0
        x = nan_gauss(vec_to_mat(pred, L, d), 2)[iu]; y = nan_gauss(vec_to_mat(obs, L, d), 2)[iu]
        assert abs(df["pooled_r_smooth"].iloc[0] - pearsonr(x, y)[0]) < 1e-2, "pooled r"
        assert df["head_index"].iloc[0] == 0 and df["region"].iloc[0] == "chr10:0-1"
    print(f"✅ self-test passed (synthetic: raw r={r_raw:.3f}, smoothed r={r_s:.3f})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--npz")
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma in bins (1 bin = 2,048 bp)")
    ap.add_argument("--pair", action="append", metavar="OBS=HEAD",
                    help="default: the pairs stored in the npz")
    ap.add_argument("--metrics-tsv", help="original per-window metrics; extra columns "
                                          "(e.g. fold_role, data_dir) are carried over")
    ap.add_argument("--replace", action="store_true",
                    help="also write *_smooth{sigma}_asmain.tsv where pearson/spearman/mse = smoothed")
    ap.add_argument("--out", help="output prefix (default: npz path without _predictions_targets.npz)")
    ap.add_argument("--max_windows", type=int, help="quick test on the first N windows")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.npz:
        ap.error("--npz is required")

    out = a.out or re.sub(r"_predictions_targets\.npz$", "", a.npz)
    tag = f"smooth{a.sigma:g}"
    print(f"▶ {a.npz}  sigma={a.sigma} bins ({a.sigma*2.048:.1f} kb)")
    df = run(a.npz, a.sigma, a.pair, a.max_windows)

    if a.metrics_tsv:
        m = pd.read_csv(a.metrics_tsv, sep="\t", dtype={"chrom": str})
        extra = [c for c in m.columns if c not in df.columns and c not in
                 ("pearson", "spearman", "mse", "obs_sd", "pred_sd", "n_pixels")]
        if extra:
            key = ["chrom", "start", "obs_target", "model_head"]
            df = df.merge(m[key + extra].drop_duplicates(key), on=key, how="left")
            print(f"   carried over from metrics tsv: {extra}")

    f1 = f"{out}_per_window_metrics_{tag}.tsv"
    df.to_csv(f1, sep="\t", index=False)
    print(f"✅ {f1}")

    if a.replace:
        r = df.copy()
        for k in ("pearson", "spearman", "mse"):
            r[k] = r[f"{k}_smooth"]
        f2 = f"{out}_per_window_metrics_{tag}_asmain.tsv"
        r.to_csv(f2, sep="\t", index=False)
        print(f"✅ {f2}  (pearson/spearman/mse = smoothed; use with --metrics-tsv)")

    s = (df.groupby(["obs_target", "model_head"])
           .agg(n=("pearson_raw", "size"), median_r_raw=("pearson_raw", "median"),
                median_r_smooth=("pearson_smooth", "median"),
                median_delta=("delta_pearson", "median"),
                median_mse_raw=("mse_raw", "median"), median_mse_smooth=("mse_smooth", "median"),
                pooled_r_smooth=("pooled_r_smooth", "first"),
                pooled_mse_smooth=("pooled_mse_smooth", "first"))
           .reset_index())
    s.to_csv(f"{out}_summary_{tag}.csv", index=False)
    print("\n📊 raw vs smoothed (median over windows)")
    print(s.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
