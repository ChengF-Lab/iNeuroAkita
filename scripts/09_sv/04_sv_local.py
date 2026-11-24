#!/usr/bin/env python
"""
04_sv_local.py  --  local and rearrangement metrics of SVs from SuPreMo disruption tracks (real SVs and null SVs)

Track (448 bins, 2,048 bp): per-row MSE and per-row 1 - Spearman r between REF and ALT maps (SuPreMo returns 1 - r),
mean of the 4 augmentations (--aug 0: shift 0 only, sensitivity), per head (HFF = NPC, H1hESC = Mic, GM12878 = Neuron).

Breakpoints in map bins: SuPreMo masks the deleted/duplicated bins (NaN rows) -> the NaN run at the map centre;
SVs < ~1 bin are not masked and sit at the centre bin (224). Checked against rel_pos_map of the real SVs (report).
Local window = --flank bins (default 25 = 51 kb) on each side of each breakpoint (outside the masked run).

Metrics per SV and head
  local_frac   fraction of the total MSE-track disruption inside the local window      (locality)
  local_mse    mean MSE track inside the local window                                    (local magnitude)
  local_dcorr  mean (1 - r) track inside the local window                               (local rearrangement)
Outputs: variant_scoring/compare/sv_local/{SV120_local.tsv, SVnull_local.tsv, breakpoint_check.txt}
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN tracks of unscorable SVs

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--flank", type=int, default=25)
ap.add_argument("--aug", default="all", choices=["0", "all"], help="all = mean of shift -1/0/+1 + revcomp (default); 0 = shift-0 only (sensitivity)")
a = ap.parse_args()
AUGT = "0" if a.aug == "0" else ".*"   # track keys: ..._track_<head>_<shift>[_revcomp]
S = f"{a.base}/variant_scoring"; OUT = f"{S}/compare/sv_local"; os.makedirs(OUT, exist_ok=True)
HEAD = {"HFF": "NPC", "H1hESC": "Mic", "GM12878": "Neuron"}; NB, C = 448, 224

def mean_track(T, vi, met, h):
    ks = [k for k in T if re.fullmatch(rf"{vi}_{met}_track_{h}_{AUGT}", k)]
    return np.nanmean(np.vstack([np.asarray(T[k], float) for k in ks]), axis=0) if ks else None

def breakpoints(t):
    """(b0, b1): last unmasked bin before and first unmasked bin after the masked run at the centre; else (C, C)"""
    nan = np.isnan(t)
    if not nan[max(0, C - 3):C + 4].any(): return C, C
    i = C - 3 + int(np.argmax(nan[C - 3:C + 4]))
    lo = i
    while lo > 0 and nan[lo - 1]: lo -= 1
    hi = i
    while hi < NB - 1 and nan[hi + 1]: hi += 1
    return lo - 1, hi + 1

def local_metrics(tm, tc):
    b0, b1 = breakpoints(tm); w = np.zeros(NB, bool)
    w[max(0, b0 - a.flank + 1):b0 + 1] = True; w[b1:min(NB, b1 + a.flank)] = True
    tot = np.nansum(tm)
    return dict(b0=b0, b1=b1, local_frac=np.nansum(tm[w]) / tot if tot > 0 else np.nan,
                local_mse=np.nanmean(tm[w]), local_dcorr=np.nanmean(tc[w]) if tc is not None else np.nan)

def run(master, files_by_key, key_col):
    rows = []
    for key, f in files_by_key.items():
        T = np.load(f, allow_pickle=True).item()
        for r in master[master[key_col] == key].itertuples():
            row = {"id": r.id, "var_index": r.var_index}
            for h, cell in HEAD.items():
                tm, tc = mean_track(T, r.var_index, "mse", h), mean_track(T, r.var_index, "corr", h)
                if tm is None: continue
                mtr = local_metrics(tm, tc)
                for k, v in mtr.items(): row[f"{k}_{cell}" if k not in ("b0", "b1") else k] = v
            rows.append(row)
    return pd.DataFrame(rows)

# ---- real SVs
m = pd.read_csv(f"{S}/inputs/SV120_master.tsv", sep="\t").rename(columns={"var_id": "id"})
files = {os.path.basename(os.path.dirname(f)): f for f in glob.glob(f"{S}/scores/SV120_tracks/fold*/*_tracks.npy")}
R = run(m, files, "fold")
R = m[["id", "svtype", "svlen", "fold"]].merge(R, on="id", how="left")
R.rename(columns={"id": "var_id"}).to_csv(f"{OUT}/SV120_local.tsv", sep="\t", index=False)

# breakpoint check against SuPreMo's rel_pos_map (maps of the real SVs)
chk = []
for fold, f in files.items():
    mf = f.replace("_tracks.npy", "_maps.npy")
    if not os.path.exists(mf): continue
    M = np.load(mf, allow_pickle=True).item()
    for r in R[R.fold == fold].itertuples():
        k = f"{r.var_index}_maps_H1hESC_0"
        if k in M and not np.isnan(r.b0):
            rel = np.atleast_1d(M[k][2]); chk.append(dict(svtype=r.svtype, svlen=r.svlen, b0=r.b0, rel_ref=int(rel.min())))
ck = pd.DataFrame(chk)
msg = ["breakpoint check (real SVs): inferred start of the SV in the map vs min(SuPreMo rel_pos_map)"]
if len(ck):
    ck["d"] = (ck.b0 + 1 - ck.rel_ref).where(ck.b0 != C, ck.b0 - ck.rel_ref)
    msg.append(f"  n = {len(ck)}; |difference| <= 2 bins: {int((ck.d.abs() <= 2).sum())}; max |d| = {int(ck.d.abs().max())}")
    msg.append(ck[ck.d.abs() > 2].to_string(index=False) if (ck.d.abs() > 2).any() else "  all consistent")

# ---- null SVs
nm = pd.read_csv(f"{S}/inputs/SVnull/SVnull_master.tsv", sep="\t").rename(columns={"null_id": "id"})
files = {os.path.basename(os.path.dirname(f)): f for f in glob.glob(f"{S}/scores/SVnull_tracks/*/*_tracks.npy")}
if files:
    Nn = run(nm, files, "chunk")
    Nn = nm[["id", "sv_id", "svtype", "svlen", "chunk"]].merge(Nn, on="id", how="inner")
    Nn.rename(columns={"id": "null_id"}).to_csv(f"{OUT}/SVnull_local.tsv", sep="\t", index=False)
    msg.append(f"null SVs with tracks: {Nn.local_frac_Mic.notna().sum()}/{len(nm)} (chunks {len(files)})")
else:
    msg.append("null tracks not found yet (score the null SVs with SuPreMo --get_maps first)")
msg.append(f"real SVs with tracks: {R.local_frac_Mic.notna().sum()}/{len(R)}; flank = {a.flank} bins ({a.flank * 2.048:.0f} kb)")
open(f"{OUT}/breakpoint_check.txt", "w").write("\n".join(msg) + "\n"); print("\n".join(msg))
