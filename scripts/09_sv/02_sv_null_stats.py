#!/usr/bin/env python
"""
02_sv_null_stats.py  --  empirical significance of SV120 and SNV42 disruption scores against matched null variants

SV120  null = 30 matched null SVs per real SV (01_prep_sv_null.py; same chrom/fold, type, length, gene density)
       p_own    = (1 + #own null >= score) / (k + 1)                       (k = 30 -> minimum 0.032)
       p_pooled = vs all null SVs of the same type with length within 1.5x  (more resolution; BH-FDR across SVs)
       for MSE and for 1 - corr; heads NPC / Mic / Neuron
       strata table by type x size + redundancy flag (same event reported by Lee and David)
SNV42  null = within-gene low-PP eQTL SNPs already scored in the eQTL run (same model/settings), matched to each SNV
       on distance to the nearest TSS and GC (+/-1 kb): the 500 nearest nulls -> p_matched; plus pooled p; BH-FDR
       (skipped with a message if the eQTL scores are not there yet)
Outputs: variant_scoring/compare/null/{SV120_null_stats.tsv, SV120_strata.tsv, SNV42_null_stats.tsv, summary.txt, SV_size_vs_mse.pdf/png}
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, pysam
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--fa", default="ref_data/hg38.fa", help="hg38 FASTA (indexed)")
ap.add_argument("--aug", default="all", choices=["0", "all"], help="all = mean of shift -1/0/+1 + revcomp (default); 0 = shift-0 only (sensitivity)")
a = ap.parse_args()
B = a.base; S = f"{B}/variant_scoring"; OUT = f"{S}/compare/null"; os.makedirs(OUT, exist_ok=True)
SUP = os.environ.get("SUPREMO_DIR", f"{S}/SuPreMo")   # patched SuPreMo clone (see README)
HEAD = {"HFF": "NPC", "H1hESC": "Mic", "GM12878": "Neuron"}; AUG = r"(0)" if a.aug == "0" else r"(-1|0|0_revcomp|1)"; CELLS = list(HEAD.values()); log = []

def bh(p):
    p = np.asarray(p, float); ok = ~np.isnan(p); out = np.full(len(p), np.nan)
    if ok.sum() == 0: return out
    q = p[ok]; o = np.argsort(q); r = q[o] * len(q) / np.arange(1, len(q) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; t = np.empty_like(r); t[o] = np.minimum(r, 1); out[ok] = t; return out
def emp(x, null):
    null = np.sort(np.asarray(null)[~np.isnan(null)])
    return np.nan if (np.isnan(x) or len(null) == 0) else (1 + len(null) - np.searchsorted(null, x, side="left")) / (len(null) + 1)
def aggregate(raw, metrics):
    out = pd.DataFrame(index=raw.index)
    for h, cell in HEAD.items():
        for met in metrics:
            cols = [c for c in raw.columns if re.fullmatch(rf"{re.escape(met)}_{h}_{AUG}", c)]
            if cols: out[f"{met}_{cell}"] = raw[cols].mean(axis=1)
    return out

# =============================== SV120 ===============================
real = pd.read_csv(f"{S}/compare/SV120_new_scores.tsv", sep="\t")
for c in CELLS: real[f"dcorr_{c}"] = 1 - real[f"corr_{c}"]
nm = pd.read_csv(f"{S}/inputs/SVnull/SVnull_master.tsv", sep="\t")
parts = [pd.read_csv(f, sep="\t").assign(chunk=os.path.basename(f).replace("_scores", ""))
         for f in glob.glob(f"{S}/scores/SVnull/*/*_scores")]
if parts:
    raw = pd.concat(parts, ignore_index=True).set_index(["chunk", "var_index"])
    nul = nm.set_index(["chunk", "var_index"]).join(aggregate(raw, ["mse", "corr"])).reset_index()
    for c in CELLS: nul[f"dcorr_{c}"] = 1 - nul[f"corr_{c}"]
    ok = nul["mse_NPC"].notna()
    log.append(f"SV null: scored {ok.sum()}/{len(nul)} (chunks {len(parts)})"); nul = nul[ok]
    real["L"] = (real.end - real.pos).abs(); nul["L"] = (nul.end - nul.pos).abs()
    for c in CELLS:
        for met in ["mse", "dcorr"]:
            col = f"{met}_{c}"; po, pp = [], []
            for r in real.itertuples():
                x = getattr(r, col) if hasattr(r, col) else real.loc[r.Index, col]
                own = nul.loc[nul.sv_id == r.var_id, col].values
                pool = nul[(nul.svtype == r.svtype) & (nul.L.between(r.L / 1.5, r.L * 1.5))][col].values
                if len(pool) < 100: pool = nul[nul.L.between(r.L / 1.5, r.L * 1.5)][col].values
                po.append(emp(x, own)); pp.append(emp(x, pool))
            real[f"p_own_{col}"] = po; real[f"p_pool_{col}"] = pp; real[f"q_pool_{col}"] = bh(pp)
    # redundancy: same event in Lee and David (same type, starts within 50 bp)
    real["duplicate_of"] = ""
    for r in real[real.source == "David"].itertuples():
        h = real[(real.source == "Lee") & (real.chrom == r.chrom) & (real.svtype == r.svtype) & ((real.pos - r.pos).abs() <= 50)]
        if len(h): real.loc[r.Index, "duplicate_of"] = h.var_id.iloc[0]
    log.append(f"SV redundancy: {int((real.duplicate_of != '').sum())} David SVs duplicate a Lee SV -> {len(real) - int((real.duplicate_of != '').sum())} unique events")
    for c in CELLS:
        for met in ["mse", "dcorr"]:
            log.append(f"  {c:6s} {met:5s}: q_pool < 0.05: {int((real[f'q_pool_{met}_{c}'] < 0.05).sum())}, < 0.1: {int((real[f'q_pool_{met}_{c}'] < 0.1).sum())};"
                       f"  p_own <= 1/31: {int((real[f'p_own_{met}_{c}'] <= 1/31 + 1e-9).sum())}")
    # strata by SV type x size
    bins = [0, 1e3, 1e4, 1e5, 1e6]; labs = ["<1 kb", "1-10 kb", "10-100 kb", ">100 kb"]
    real["size_bin"] = pd.cut(real.L, bins, labels=labs); nul["size_bin"] = pd.cut(nul.L, bins, labels=labs)
    st = []
    for (t, b), x in real.groupby(["svtype", "size_bin"], observed=True):
        y = nul[(nul.svtype == t) & (nul.size_bin == b)]
        own_med = x.var_id.map(nul.groupby("sv_id").mse_Mic.median())
        st.append(dict(svtype=t, size_bin=b, n=len(x), median_mse_Mic_real=x.mse_Mic.median(), median_mse_Mic_null=y.mse_Mic.median(),
                       frac_real_above_own_null_median=(x.mse_Mic > own_med).mean(),
                       P_mannwhitney=mannwhitneyu(x.mse_Mic.dropna(), y.mse_Mic.dropna(), alternative="greater").pvalue if len(y) and len(x) > 2 else np.nan))
    st = pd.DataFrame(st); st.to_csv(f"{OUT}/SV120_strata.tsv", sep="\t", index=False)
    log.append("\n== SV strata (Mic head)\n" + st.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    case = {"PLCG2 7.8kb DEL": ("chr16", 81885235), "KANSL1 314bp DEL": ("chr17", 46146541),
            "ARMS2 445bp DEL (Lee)": ("chr10", 122457302), "ARMS2 461bp DEL (David)": ("chr10", 122457305)}
    rows = []
    for lab, (c, p) in case.items():
        r = real[(real.chrom == c) & (real.pos == p)]
        if len(r): r = r.iloc[0]; rows.append([lab] + [f"{r[f'p_own_mse_{x}']:.3f}/{r[f'q_pool_mse_{x}']:.3f}" for x in CELLS]
                                               + [f"{r['p_own_dcorr_Mic']:.3f}/{r['q_pool_dcorr_Mic']:.3f}"])
    log.append("\n== case SVs: p_own / q_pool (MSE NPC, Mic, Neuron; 1-corr Mic)\n"
               + pd.DataFrame(rows, columns=["SV", "NPC", "Mic", "Neuron", "1-corr Mic"]).to_string(index=False))
    real.to_csv(f"{OUT}/SV120_null_stats.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.scatter(np.log10(nul.L), np.log10(nul.mse_Mic), s=3, c="#BDBDBD", label="matched null")
    for t, col in [("DEL", "#73D07A"), ("DUP", "#F2B4AA"), ("INV", "#95B4D0")]:
        x = real[real.svtype == t]; ax.scatter(np.log10(x.L), np.log10(x.mse_Mic), s=14, c=col, edgecolor="k", lw=.3, label=t)
    ax.set_xlabel("log10 SV length (bp)"); ax.set_ylabel("log10 MSE (Mic head)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/SV_size_vs_mse.pdf"); fig.savefig(f"{OUT}/SV_size_vs_mse.png", dpi=200)
else:
    log.append("SV null scores not found yet (score inputs/SVnull/chunks with SuPreMo first)")

# =============================== SNV42 ===============================
ep = f"{S}/compare/eQTL/eQTL_scores.tsv"
if os.path.exists(ep):
    e = pd.read_csv(ep, sep="\t"); e["role"] = e["role"].fillna("null"); nul = e[(e.role == "null") & e.mse_Mic.notna()].copy()
    snv = pd.read_csv(f"{S}/compare/SNV42_new_scores.tsv", sep="\t")
    fa = pysam.FastaFile(a.fa); ga = pd.read_csv(f"{SUP}/data/gene_annot_hg38")
    ga["tss"] = np.where(ga.strand == "+", ga.start, ga.end); tss = {c: np.sort(x.tss.values) for c, x in ga.groupby("chr")}
    def near(c, p):
        t = tss.get(c); i = np.searchsorted(t, p); return int(min(abs(p - t[max(i - 1, 0)]), abs(t[min(i, len(t) - 1)] - p)))
    def gc(c, p):
        s = fa.fetch(c, max(0, p - 1000), p + 1000).upper(); return (s.count("G") + s.count("C")) / max(len(s) - s.count("N"), 1)
    for d in (snv, nul):
        d["tss_near"] = [near(c, p) for c, p in zip(d.chrom, d.pos)]; d["gc"] = [gc(c, p) for c, p in zip(d.chrom, d.pos)]
        d["ltss"] = np.log10(d.tss_near + 100)
    sd = nul[["ltss", "gc"]].std()
    for c in CELLS:
        pm, pp = [], []
        for r in snv.itertuples():
            dist = (((nul.ltss - r.ltss) / sd.ltss) ** 2 + ((nul.gc - r.gc) / sd.gc) ** 2)
            mt = nul.loc[dist.nsmallest(500).index, f"mse_{c}"].values
            pm.append(emp(getattr(r, f"mse_{c}"), mt)); pp.append(emp(getattr(r, f"mse_{c}"), nul[f"mse_{c}"].values))
        snv[f"p_matched_{c}"] = pm; snv[f"q_matched_{c}"] = bh(pm); snv[f"p_pool_{c}"] = pp
    log.append(f"\n== SNV42 vs eQTL null pool (n = {len(nul)}; 500 nearest on TSS distance + GC; minimum P = {1/501:.1e})")
    for c in CELLS: log.append(f"  {c:6s}: q_matched < 0.05: {int((snv[f'q_matched_{c}'] < 0.05).sum())}, < 0.1: {int((snv[f'q_matched_{c}'] < 0.1).sum())}")
    log.append(snv.sort_values("mse_Mic", ascending=False)[["var_id", "gene", "tfbs", "mse_Mic", "p_matched_Mic", "q_matched_Mic", "p_pool_Mic"]]
               .head(10).to_string(index=False))
    snv.to_csv(f"{OUT}/SNV42_null_stats.tsv", sep="\t", index=False)
else:
    log.append("\nSNV42: eQTL null scores not available yet (run 08_eqtl/03_merge_eqtl.py first) -> skipped")
open(f"{OUT}/summary.txt", "w").write("\n".join(log) + "\n"); print("\n".join(log))
