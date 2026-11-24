#!/usr/bin/env python
"""
03_rank_select.py  --  rank and select SNVs (SNV42, eQTL) and SVs from the out-of-fold SuPreMo scores
(no new scoring; reads the outputs of 05_variant_scoring/02_merge_scores, 08_eqtl/03_merge_eqtl,
 09_sv/02_sv_null_stats and 09_sv/03_sv_gene_tracks)

Metrics (mean of the 4 augmentations (--aug 0: shift 0 only, sensitivity); per head NPC / Mic / Neuron)
  MSE     global disruption  = mean squared ALT-REF difference over the 448 x 448 map
  MaxAbs  local disruption   = largest absolute ALT-REF difference (one pixel)
  1-corr  structural change  = 1 - Spearman r of the two maps (SVs only; SNVs: rho(MSE, 1-corr) = 0.95)

SNV42  empirical P vs the within-gene low-PP eQTL null SNPs (same model and settings):
         p_matched: the --n_match nulls closest in distance to the nearest TSS (+ GC +/-1 kb if the FASTA is available)
         p_pool   : all null SNPs
       BH-FDR across the 42 SNVs (per head and metric)
       pass(metric) = p_matched < --p_thr  or  q_matched < --q_thr ;  selected_<head> = pass(MSE) and pass(MaxAbs)
       selected = selected in ANY head (union); n_heads_selected = number of heads
eQTL   empirical P vs all null SNPs (worst-case ALT allele for targets and nulls), BH-FDR across targets
       pass(metric) = p < --p_thr or q < --q_thr ;  selected_<head> = pass(MSE) and pass(MaxAbs);  selected = any head
       case_<head> = P < --case_p for BOTH MSE and MaxAbs in that head;  case_eligible = case in ANY head (union)
SV     all SVs scored and ranked; duplicates (same event reported by Lee and David) flagged and removed
       comparison strata: SV type x size (<1 kb, 1-10 kb, 10-100 kb, >100 kb); null SVs have the same type and length
       three dimensions, each metric -> empirical P vs null SVs of the same type with length within 1.5x (all types if < 100)
         global         MSE
         local          MaxAbs, local_frac (share of the disruption within 51 kb of the breakpoints), local_mse   [09_sv/04_sv_local.py]
         rearrangement  1 - corr (whole map), local_dcorr (1 - r of map rows within 51 kb of the breakpoints)
       dimension P = Cauchy combination (ACAT) of its metrics; BH-FDR per dimension across all unique SVs
       significant(dimension) = P < --sv_p or q < --q_thr ; ranks within stratum; top genes from 07 (gene tracks)
Outputs: variant_scoring/compare/selection/{SNV42_ranked.tsv, eQTL_ranked.tsv, SV_ranked.tsv, SV_strata_summary.tsv, selection_summary.txt}
"""
import argparse, glob, os, re
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--fa", default="ref_data/hg38.fa", help="hg38 FASTA (indexed)")
ap.add_argument("--head", default="Mic", choices=["NPC", "Mic", "Neuron"])
ap.add_argument("--p_thr", type=float, default=0.01)
ap.add_argument("--q_thr", type=float, default=0.1)
ap.add_argument("--n_match", type=int, default=500)
ap.add_argument("--case_p", type=float, default=0.001)
ap.add_argument("--sv_p", type=float, default=0.05)
ap.add_argument("--aug", default="all", choices=["0", "all"], help="all = mean of shift -1/0/+1 + revcomp (default); 0 = shift-0 only (sensitivity)")
a = ap.parse_args()
S = f"{a.base}/variant_scoring"; OUT = f"{S}/compare/selection"; os.makedirs(OUT, exist_ok=True)
SUP = os.environ.get("SUPREMO_DIR", f"{S}/SuPreMo")   # patched SuPreMo clone (see README)
HEAD = {"HFF": "NPC", "H1hESC": "Mic", "GM12878": "Neuron"}; CELLS = list(HEAD.values()); AUG = r"(0)" if a.aug == "0" else r"(-1|0|0_revcomp|1)"
H = a.head; log = [f"detail tables: {H} head (per-head lists and columns for all heads);  SNV pass = P < {a.p_thr} or q < {a.q_thr} for BOTH MSE and MaxAbs;  "
                   f"eQTL case: P < {a.case_p} on both metrics in any head;  SV significant: P < {a.sv_p} or q < {a.q_thr} per dimension"]

def bh(p):
    p = np.asarray(p, float); out = np.full(len(p), np.nan); ok = ~np.isnan(p)
    if ok.any():
        q = p[ok]; o = np.argsort(q); r = q[o] * len(q) / np.arange(1, len(q) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; t = np.empty_like(r); t[o] = np.minimum(r, 1); out[ok] = t
    return out
def emp(x, null):
    null = np.sort(np.asarray(null, float)); null = null[~np.isnan(null)]
    return np.nan if (np.isnan(x) or len(null) == 0) else (1 + len(null) - np.searchsorted(null, x, side="left")) / (len(null) + 1)
def acat(ps):
    ps = np.clip(np.asarray([p for p in ps if not np.isnan(p)], float), 1e-15, 1 - 1e-15)
    return np.nan if len(ps) == 0 else 0.5 - np.arctan(np.mean(np.tan((0.5 - ps) * np.pi))) / np.pi
def passes(df, pcol, qcol): return (df[pcol] < a.p_thr) | (df[qcol] < a.q_thr)

# ------------------------------------------------------------------ null SNPs (eQTL run)
e = pd.read_csv(f"{S}/compare/eQTL/eQTL_scores.tsv", sep="\t"); e["role"] = e["role"].fillna("null")
N = e[(e.role == "null") & e[f"mse_{H}"].notna()].copy()
T = e[(e.role == "target") & e[f"mse_{H}"].notna()].copy()

# ------------------------------------------------------------------ SNV42
snv = pd.read_csv(f"{S}/compare/SNV42_new_scores.tsv", sep="\t")
ga = pd.read_csv(f"{SUP}/data/gene_annot_hg38"); ga["tss"] = np.where(ga.strand == "+", ga.start, ga.end)
TSS = {c: np.sort(x.tss.values) for c, x in ga.groupby("chr")}
def near(c, p):
    t = TSS.get(c); i = np.searchsorted(t, p); return int(min(abs(p - t[max(i - 1, 0)]), abs(t[min(i, len(t) - 1)] - p)))
use_gc = os.path.exists(a.fa)
if use_gc:
    import pysam; fa = pysam.FastaFile(a.fa)
    def gc(c, p):
        s = fa.fetch(c, max(0, p - 1000), p + 1000).upper(); return (s.count("G") + s.count("C")) / max(len(s) - s.count("N"), 1)
for d in (snv, N):
    d["ltss"] = np.log10(np.array([near(c, p) for c, p in zip(d.chrom, d.pos)]) + 100)
    d["gc"] = [gc(c, p) for c, p in zip(d.chrom, d.pos)] if use_gc else 0.0
sd_l, sd_g = N.ltss.std(), (N.gc.std() if use_gc else 1.0)
idx = []
for r in snv.itertuples():
    dist = ((N.ltss - r.ltss) / sd_l) ** 2 + (((N.gc - r.gc) / sd_g) ** 2 if use_gc else 0)
    idx.append(dist.nsmallest(a.n_match).index)
for c in CELLS:
    for met in ["mse", "maxabs"]:
        col = f"{met}_{c}"
        snv[f"p_matched_{col}"] = [emp(x, N.loc[ix, col].values) for x, ix in zip(snv[col], idx)]
        snv[f"p_pool_{col}"] = [emp(x, N[col].values) for x in snv[col]]
        snv[f"q_matched_{col}"] = bh(snv[f"p_matched_{col}"])
for c in CELLS:
    snv[f"selected_{c}"] = passes(snv, f"p_matched_mse_{c}", f"q_matched_mse_{c}") & passes(snv, f"p_matched_maxabs_{c}", f"q_matched_maxabs_{c}")
snv["selected"] = snv[[f"selected_{c}" for c in CELLS]].any(axis=1); snv["n_heads_selected"] = snv[[f"selected_{c}" for c in CELLS]].sum(axis=1)
snv["rank_mse"] = snv[f"mse_{H}"].rank(ascending=False).astype(int)
snv = snv.sort_values("rank_mse")
keep = ["rank_mse", "var_id", "gene", "chrom", "pos", "ref", "alt", "tfbs", "fold", "selected", "n_heads_selected"] + [f"selected_{c}" for c in CELLS] + [
        f"mse_{H}", f"p_matched_mse_{H}", f"q_matched_mse_{H}", f"p_pool_mse_{H}",
        f"maxabs_{H}", f"p_matched_maxabs_{H}", f"q_matched_maxabs_{H}"] + \
       [f"{m}_{c}" for c in CELLS if c != H for m in ["mse", "maxabs"]] + [f"{m}_{H}_cv" for m in ["mse", "maxabs"] if f"{m}_{H}_cv" in snv]
snv[[k for k in keep if k in snv] + [c for c in snv.columns if c not in keep and (c.startswith("p_") or c.startswith("q_"))]] \
    .to_csv(f"{OUT}/SNV42_ranked.tsv", sep="\t", index=False)
log.append(f"\n== SNV42 (null: {len(N)} within-gene low-PP SNPs; {a.n_match} matched on TSS distance{' + GC' if use_gc else ''}; "
           f"min P matched = {1/(a.n_match+1):.1e}, pooled = {1/(len(N)+1):.1e})")
log.append(f"  selected (any head): {int(snv.selected.sum())}/42;  per head: " + ", ".join(f"{c} {int(snv[f'selected_{c}'].sum())}" for c in CELLS))
for c in CELLS: log.append(f"    {c}: " + ", ".join(f"{v} ({g})" for v, g in zip(snv[snv[f'selected_{c}']].var_id, snv[snv[f'selected_{c}']].gene)))
log.append(snv[snv.selected][["rank_mse", "var_id", "gene", "tfbs", f"mse_{H}", f"p_matched_mse_{H}", f"q_matched_mse_{H}",
                              f"p_matched_maxabs_{H}", f"q_matched_maxabs_{H}", "n_heads_selected"]].to_string(index=False))

# ------------------------------------------------------------------ eQTL
for c in CELLS:
    for met in ["mse", "maxabs"]:
        col = f"{met}_{c}"; nv = np.sort(N[col].dropna().values)
        T[f"p_{col}"] = (1 + len(nv) - np.searchsorted(nv, T[col].values, side="left")) / (len(nv) + 1)
        T[f"q_{col}"] = bh(T[f"p_{col}"])
    T[f"selected_{c}"] = passes(T, f"p_mse_{c}", f"q_mse_{c}") & passes(T, f"p_maxabs_{c}", f"q_maxabs_{c}")
T["selected"] = T[[f"selected_{c}" for c in CELLS]].any(axis=1); T["n_heads_selected"] = T[[f"selected_{c}" for c in CELLS]].sum(axis=1)
for c in CELLS: T[f"case_{c}"] = (T[f"p_mse_{c}"] < a.case_p) & (T[f"p_maxabs_{c}"] < a.case_p)
T["case_eligible"] = T[[f"case_{c}" for c in CELLS]].any(axis=1); T["n_heads_case"] = T[[f"case_{c}" for c in CELLS]].sum(axis=1)
T["rank_mse"] = T[f"mse_{H}"].rank(ascending=False).astype(int)
T = T.sort_values("rank_mse")
keep = ["rank_mse", "var_id", "genes", "PostProb_max", "chrom", "pos", "ref", f"alt_max_{H}", "n_alleles_scored", "fold", "selected",
        "n_heads_selected"] + [f"selected_{c}" for c in CELLS] + ["case_eligible", "n_heads_case"] + [f"case_{c}" for c in CELLS] + [ f"mse_{H}", f"p_mse_{H}", f"q_mse_{H}", f"maxabs_{H}", f"p_maxabs_{H}", f"q_maxabs_{H}"] + \
       [f"{m}_{c}" for c in CELLS if c != H for m in ["mse", "maxabs"]]
T[[k for k in keep if k in T] + [c for c in T.columns if c not in keep and (c.startswith("p_") or c.startswith("q_"))]] \
    .to_csv(f"{OUT}/eQTL_ranked.tsv", sep="\t", index=False)
nT = len(T)
log.append(f"\n== eQTL: {nT} targets vs {len(N)} null SNPs (min P = {1/(len(N)+1):.1e})")
log.append(f"  selected (any head): {int(T.selected.sum())};  all three heads: {int((T.n_heads_selected == 3).sum())};  per head: " + ", ".join(f"{c} {int(T[f'selected_{c}'].sum())}" for c in CELLS)
           + f";  expected by chance with P < {a.p_thr} on both metrics: <= {nT * a.p_thr:.0f}")
log.append(f"  q < {a.q_thr} (MSE, {H}): {int((T[f'q_mse_{H}'] < a.q_thr).sum())}")
log.append(f"  case_eligible (P < {a.case_p}, both metrics, in any head): {int(T.case_eligible.sum())}: " + ", ".join(f"{v} ({g}; {'/'.join(c for c in CELLS if r_[f'case_{c}'])})" for v, g, (_, r_) in zip(T[T.case_eligible].var_id, T[T.case_eligible].genes, T[T.case_eligible].iterrows())))
log.append(T[T.selected].head(40)[["rank_mse", "var_id", "genes", "PostProb_max", f"alt_max_{H}", f"mse_{H}", f"p_mse_{H}", f"q_mse_{H}",
                                   f"p_maxabs_{H}", "n_heads_selected"]].to_string(index=False))

# ------------------------------------------------------------------ SV
real = pd.read_csv(f"{S}/compare/SV120_new_scores.tsv", sep="\t")
nm = pd.read_csv(f"{S}/inputs/SVnull/SVnull_master.tsv", sep="\t")
parts = [pd.read_csv(f, sep="\t").assign(chunk=os.path.basename(f).replace("_scores", "")) for f in glob.glob(f"{S}/scores/SVnull/*/*_scores")]
raw = pd.concat(parts, ignore_index=True).set_index(["chunk", "var_index"])
agg = pd.DataFrame(index=raw.index)
for h, c in HEAD.items():
    for met in ["mse", "corr", "maxabs"]:
        cols = [x for x in raw.columns if re.fullmatch(rf"{met}_{h}_{AUG}", x)]
        agg[f"{met}_{c}"] = raw[cols].mean(axis=1)
nul = nm.set_index(["chunk", "var_index"]).join(agg).reset_index()
lr, ln = f"{S}/compare/sv_local/SV120_local.tsv", f"{S}/compare/sv_local/SVnull_local.tsv"
LOC = ["local_frac", "local_mse", "local_dcorr"]
if os.path.exists(lr): real = real.merge(pd.read_csv(lr, sep="\t")[["var_id"] + [f"{m}_{c}" for m in LOC for c in CELLS]], on="var_id", how="left")
if os.path.exists(ln): nul = nul.merge(pd.read_csv(ln, sep="\t")[["null_id"] + [f"{m}_{c}" for m in LOC for c in CELLS]], on="null_id", how="left")
have_local = os.path.exists(ln) and nul[f"local_frac_{H}"].notna().any()
nul = nul[nul[f"mse_{H}"].notna()]
for d in (real, nul):
    d["L"] = (d.end - d.pos).abs()
    for c in CELLS: d[f"dcorr_{c}"] = 1 - d[f"corr_{c}"]
    d["size_bin"] = pd.cut(d.L, [0, 1e3, 1e4, 1e5, 1e9], labels=["<1 kb", "1-10 kb", "10-100 kb", ">100 kb"], right=False).astype(str)
    d["stratum"] = d.svtype + " " + d.size_bin
real["duplicate_of"] = ""
for r in real[real.source == "David"].itertuples():
    hit = real[(real.source == "Lee") & (real.chrom == r.chrom) & (real.svtype == r.svtype) & ((real.pos - r.pos).abs() <= 50)]
    if len(hit): real.loc[r.Index, "duplicate_of"] = hit.var_id.iloc[0]
U = real[(real.duplicate_of == "") & real[f"mse_{H}"].notna()].copy()
DIMS = {"global": ["mse"], "local": ["maxabs"] + (["local_frac", "local_mse"] if have_local else []),
        "rearr": ["dcorr"] + (["local_dcorr"] if have_local else [])}
for c in CELLS:
    for met in sorted({m for v in DIMS.values() for m in v}):
        col = f"{met}_{c}"; pp = []
        for r in U.itertuples():
            pool = nul[(nul.svtype == r.svtype) & nul.L.between(r.L / 1.5, r.L * 1.5)][col].values
            if len(pool) < 100: pool = nul[nul.L.between(r.L / 1.5, r.L * 1.5)][col].values
            pp.append(emp(U.at[r.Index, col], pool))
        U[f"p_{col}"] = pp
    for dim, mets in DIMS.items():
        U[f"P_{dim}_{c}"] = [acat([r[f"p_{m}_{c}"] for m in mets]) for _, r in U.iterrows()]
        U[f"q_{dim}_{c}"] = bh(U[f"P_{dim}_{c}"])
        U[f"sig_{dim}_{c}"] = (U[f"P_{dim}_{c}"] < a.sv_p) | (U[f"q_{dim}_{c}"] < a.q_thr)
for dim in DIMS:
    U[f"rank_in_stratum_{dim}"] = U.groupby("stratum")[f"P_{dim}_{H}"].rank(method="min")
U["rank_in_stratum_mse"] = U.groupby("stratum")[f"mse_{H}"].rank(ascending=False, method="min")
U["n_in_stratum"] = U.groupby("stratum").var_id.transform("size")
U["dims_significant"] = U.apply(lambda r: "+".join(d for d in DIMS if r[f"sig_{d}_{H}"]), axis=1)
U["n_heads_any_sig"] = U.apply(lambda r: sum(any(r[f"sig_{d}_{c}"] for d in DIMS) for c in CELLS), axis=1)
gp = f"{S}/compare/tracks/SV120_gene_effects.tsv"
if os.path.exists(gp):
    ge = pd.read_csv(gp, sep="\t"); ge = ge[ge["head"] == H].sort_values("track_mse", ascending=False)
    U["top3_genes"] = U.var_id.map(ge.groupby("var_id").head(3).groupby("var_id").apply(
        lambda x: "; ".join(f"{g} (z={z:.1f}, {dd:+.0f} kb)" for g, z, dd in zip(x.gene, x.local_z, x.tss_dist_kb)))).fillna("")
U = U.sort_values(["stratum", f"P_local_{H}"])
keep = ["var_id", "gene", "source", "chrom", "pos", "end", "svtype", "svlen", "stratum", "n_in_stratum", "fold", "dims_significant",
        "n_heads_any_sig", f"mse_{H}", f"P_global_{H}", f"q_global_{H}", "rank_in_stratum_global",
        f"maxabs_{H}"] + ([f"local_frac_{H}", f"local_mse_{H}"] if have_local else []) + [f"P_local_{H}", f"q_local_{H}", "rank_in_stratum_local",
        f"dcorr_{H}"] + ([f"local_dcorr_{H}"] if have_local else []) + [f"P_rearr_{H}", f"q_rearr_{H}", "rank_in_stratum_rearr",
        "rank_in_stratum_mse", "top3_genes"]
keep = [k for k in keep if k in U]
U[keep + [c for c in U.columns if c not in keep and (c.startswith("p_") or c.startswith("P_") or c.startswith("q_") or c.startswith("sig_"))]] \
    .to_csv(f"{OUT}/SV_ranked.tsv", sep="\t", index=False)
real[real.duplicate_of != ""][["var_id", "duplicate_of"]].to_csv(f"{OUT}/SV_duplicates.tsv", sep="\t", index=False)
st = []
for s_, x in U.groupby("stratum"):
    y = nul[nul.stratum == s_]
    st.append(dict(stratum=s_, n=len(x), n_null=len(y), median_mse_real=x[f"mse_{H}"].median(), median_mse_null=y[f"mse_{H}"].median(),
                   **{f"n_sig_{d}": int(x[f"sig_{d}_{H}"].sum()) for d in DIMS}))
st = pd.DataFrame(st); st.to_csv(f"{OUT}/SV_strata_summary.tsv", sep="\t", index=False)
log.append(f"\n== SV: {len(real)} SVs, {int((real.duplicate_of != '').sum())} duplicates removed, {len(U)} unique scorable; null SVs {len(nul)}; "
           f"local metrics {'included' if have_local else 'NOT available yet (run_26 + 09)'}")
log.append("  dimensions: " + "; ".join(f"{d} = ACAT({', '.join(m)})" for d, m in DIMS.items()))
log.append("  significant per dimension (" + H + "): " + ", ".join(f"{d} {int(U[f'sig_{d}_{H}'].sum())}" for d in DIMS)
           + ";  q < 0.1: " + ", ".join(f"{d} {int((U[f'q_{d}_{H}'] < 0.1).sum())}" for d in DIMS))
log.append("\n  strata:\n" + st.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
cols = ["var_id", "gene", "stratum", "dims_significant", f"P_global_{H}", f"P_local_{H}", f"P_rearr_{H}", "rank_in_stratum_local", "top3_genes"]
hit = U[(U[f"sig_local_{H}"]) | (U[f"sig_rearr_{H}"])].sort_values(f"P_local_{H}")
log.append(f"\n  SVs significant in the local or rearrangement dimension ({len(hit)}):\n" + hit[[c for c in cols if c in hit]].to_string(index=False))
case = U[U.var_id.str.contains("chr17_46146541|chr10_122457302|chr16_81885235")]
log.append("\n  highlighted SVs (KANSL1 314 bp, ARMS2 445 bp, PLCG2 7.8 kb):\n" + case[[c for c in cols if c in case]].to_string(index=False))
open(f"{OUT}/selection_summary.txt", "w").write("\n".join(log) + "\n"); print("\n".join(log))
