#!/usr/bin/env python
"""
03_sv_gene_tracks.py  --  attribute SV-induced 3D disruption to individual genes, from SuPreMo tracks and maps

For every SV (119 scorable) and head (NPC / Mic / Neuron):
  track      = per-bin disruption (448 bins; MSE and 1 - Spearman r of each map row, as returned by SuPreMo), mean of the 4 augmentations (--aug 0: shift 0 only, sensitivity)
  gene bins  = TSS (-1 kb) of protein-coding genes mapped to map bins with SuPreMo's own get_roi_in_map
               (handles the coordinate shift of DEL/DUP maps exactly as SuPreMo's TSS weighting does)
  per gene   = mean track over its TSS bins; local_z = (value - median track) / (1.4826 * MAD track);
               rank among the genes in the window; distance of the TSS to the SV
  per SV     = peak bin, its distance to the SV, fraction of total disruption within +/-50 kb of the SV
Outputs (variant_scoring/compare/tracks/)
  SV120_gene_effects.tsv   long table: sv x head x gene
  SV120_top_genes.tsv      top 3 genes per SV (Mic head) + peak/locality summary
  case_genes.tsv           genes of the highlighted cases (PLCG2 / KANSL1 / ARMS2)
  plots/<sv>.png|pdf       REF, ALT, ALT-REF maps (Mic head, shift 0) + disruption track with gene labels (hg38)
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--plot_head", default="Mic")
ap.add_argument("--aug", default="all", choices=["0", "all"], help="all = mean of shift -1/0/+1 + revcomp (default); 0 = shift-0 only (sensitivity)")
a = ap.parse_args()
AUGT = "0" if a.aug == "0" else ".*"   # track keys: ..._track_<head>_<shift>[_revcomp]
B = a.base; S = f"{B}/variant_scoring"; OUT = f"{S}/compare/tracks"; os.makedirs(f"{OUT}/plots", exist_ok=True)
SUP = os.environ.get("SUPREMO_DIR", f"{S}/SuPreMo")   # patched SuPreMo clone (see README)
HEAD = {"HFF": "NPC", "H1hESC": "Mic", "GM12878": "Neuron"}; INV = {v: k for k, v in HEAD.items()}

# ---- SuPreMo's own coordinate helpers (loads the model once; any fold is fine, only constants are used)
os.environ.setdefault("SUPREMO_MODEL_DIR", f"{B}/models/fold2")
sys.path.insert(0, f"{SUP}/scripts")   # patched SuPreMo (see README)
import get_Akita_scores_utils as u, get_roi_utils as gr
from pybedtools import BedTool
gr.pred_len = u.target_length_cropped * u.bin_size; gr.d = u.bin_size * 5
u.roi_coords_BED = gr.get_roi("genes", "hg38"); u.BedTool = BedTool
NB, BS = u.target_length_cropped, u.bin_size; TRI = np.triu_indices(NB, 2)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ga = pd.read_csv(f"{SUP}/data/gene_annot_hg38"); ga["tss"] = np.where(ga.strand == "+", ga.start, ga.end)
TSS = ga.drop_duplicates("gene").set_index("gene").tss.to_dict()
m = pd.read_csv(f"{S}/inputs/SV120_master.tsv", sep="\t")

def sq(v):
    z = np.full((NB, NB), np.nan); z[TRI] = v; return np.where(np.isnan(z), z.T, z)
def mad_z(x, t):
    t = t[~np.isnan(t)]; med = np.median(t); s = 1.4826 * np.median(np.abs(t - med)); return (x - med) / s if s > 0 else np.nan

rows, summ, done = [], [], []
for fd in sorted(glob.glob(f"{S}/scores/SV120_tracks/fold*")):
    fold = os.path.basename(fd)
    tf, mf = glob.glob(f"{fd}/*_tracks.npy"), glob.glob(f"{fd}/*_maps.npy")
    if not tf or not mf: print(f"{fold}: tracks/maps missing"); continue
    T = np.load(tf[0], allow_pickle=True).item(); M = np.load(mf[0], allow_pickle=True).item()
    for r in m[m.fold == fold].itertuples():
        vi = r.var_index
        mk = f"{vi}_maps_{INV[a.plot_head]}_0"
        if mk not in M: continue
        _, _, rel_pos_map, map_start = M[mk]
        roi = u.get_roi_in_map(r.chrom, map_start, rel_pos_map, r.svtype, r.svlen)
        roi = pd.DataFrame(columns=["Start", "End", "roi_id"]) if isinstance(roi, str) else roi
        done.append(r.var_id)
        for cell, h in INV.items():
            tr = {}
            for met in ["mse", "corr"]:
                ks = [k for k in T if re.fullmatch(rf"{vi}_{met}_track_{h}_{AUGT}", k)]
                if not ks: continue
                x = np.nanmean(np.vstack([np.asarray(T[k], float) for k in ks]), axis=0)
                tr[met] = x   # SuPreMo returns corr_track as 1 - Spearman r already (disruption); do not invert
            if "mse" not in tr: continue
            t = tr["mse"]; svbin = int(np.atleast_1d(rel_pos_map)[0])
            pk = int(np.nanargmax(t)); near = np.zeros(NB, bool); near[max(0, svbin - 25):svbin + 26] = True
            summ.append(dict(var_id=r.var_id, head=cell, peak_bin_offset_kb=(pk - svbin) * BS / 1e3,
                             frac_disruption_within_50kb=np.nansum(t[near]) / np.nansum(t)))
            for g in roi.itertuples():
                b0, b1 = int(g.Start), min(int(g.End), NB - 1)
                v = np.nanmean(t[b0:b1 + 1]); vc = np.nanmean(tr["corr"][b0:b1 + 1]) if "corr" in tr else np.nan
                rows.append(dict(var_id=r.var_id, gene_sv=r.gene, svtype=r.svtype, svlen=r.svlen, head=cell, gene=g.roi_id,
                                 track_mse=v, track_1mcorr=vc, local_z=mad_z(v, t), local_z_1mcorr=mad_z(vc, tr.get("corr", np.array([np.nan]))),
                                 tss_dist_kb=(TSS.get(g.roi_id, np.nan) - (r.pos + r.end) / 2) / 1e3))
G = pd.DataFrame(rows); P = pd.DataFrame(summ)
G["rank_in_window"] = G.groupby(["var_id", "head"]).track_mse.rank(ascending=False)
G.to_csv(f"{OUT}/SV120_gene_effects.tsv", sep="\t", index=False)
top = (G[G["head"] == a.plot_head].sort_values("track_mse", ascending=False).groupby("var_id").head(3)
       .groupby("var_id").apply(lambda x: "; ".join(f"{g} (z={z:.1f}, {d:+.0f} kb)" for g, z, d in zip(x.gene, x.local_z, x.tss_dist_kb))).rename("top3_genes_Mic"))
P2 = P[P["head"] == a.plot_head].set_index("var_id").join(top).reset_index()
sc = pd.read_csv(f"{S}/compare/SV120_new_scores.tsv", sep="\t")[["var_id", "gene", "svtype", "svlen", "mse_Mic"]]
P2 = sc.merge(P2, on="var_id", how="right").sort_values("mse_Mic", ascending=False)
P2.to_csv(f"{OUT}/SV120_top_genes.tsv", sep="\t", index=False)

CASE = {"Lee_DEL_chr16_81885235_81893072": ["PLCG2", "SDR42E1", "CMIP", "HSD17B2", "MPHOSPH6"],
        "Lee_DEL_chr17_46146541_46146855": ["KANSL1", "LRRC37A", "ARL17A", "ARL17B", "MAPT", "CRHR1"],
        "Lee_DEL_chr10_122457302_122457747": ["ARMS2", "HTRA1", "PLEKHA1", "BTBD16", "DMBT1", "TACC2"],
        "David_DEL_chr10_122457305_122457766": ["ARMS2", "HTRA1", "PLEKHA1", "BTBD16", "DMBT1", "TACC2"]}
cg = G[G.apply(lambda x: x.var_id in CASE and x.gene in CASE[x.var_id], axis=1)]
cg.to_csv(f"{OUT}/case_genes.tsv", sep="\t", index=False)
missing = {k: [g for g in v if g not in set(G[G.var_id == k].gene)] for k, v in CASE.items()}
print(f"SVs with tracks: {len(set(done))}/{len(m)}")
print("\n== case genes (Mic head): track MSE, local z (MAD), rank in window, TSS distance\n"
      + cg[cg["head"] == a.plot_head][["var_id", "gene", "track_mse", "local_z", "local_z_1mcorr", "rank_in_window", "tss_dist_kb"]].round(3).to_string(index=False))
print("case genes not in the protein-coding TSS list / window:", {k: v for k, v in missing.items() if v})
print("\n== top 15 SVs by MSE (Mic): top genes and locality\n"
      + P2.head(15)[["var_id", "gene", "svlen", "mse_Mic", "frac_disruption_within_50kb", "peak_bin_offset_kb", "top3_genes_Mic"]].to_string(index=False))

# ---- plots for cases + top prioritised SVs
plot_ids = list(CASE) + ["Lee_DUP_chr19_44909364_44909819", "David_DEL_chr6_32544885_32545290",
                         "Lee_DEL_chr1_236250828_236252511", "Lee_DEL_chr1_24852253_24857535", "Lee_DEL_chr16_81792449_81792587"]
for vid in plot_ids:
    r = m[m.var_id == vid]
    if r.empty: continue
    r = r.iloc[0]; fd = f"{S}/scores/SV120_tracks/{r.fold}"
    M = np.load(glob.glob(f"{fd}/*_maps.npy")[0], allow_pickle=True).item(); T = np.load(glob.glob(f"{fd}/*_tracks.npy")[0], allow_pickle=True).item()
    k = f"{r.var_index}_maps_{INV[a.plot_head]}_0"
    if k not in M: continue
    ref, alt, rel, start = M[k]; R_, A_ = sq(ref), sq(alt); D = A_ - R_
    ks = [x for x in T if re.fullmatch(rf"{r.var_index}_mse_track_{INV[a.plot_head]}_{AUGT}", x)]
    t = np.nanmean(np.vstack([np.asarray(T[x], float) for x in ks]), axis=0)
    st = start[0] if np.ndim(start) else start
    fig = plt.figure(figsize=(11, 4.6)); gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])
    for i, (mat, ttl, cm, lim) in enumerate([(R_, "REF", "RdBu_r", 2), (A_, "ALT", "RdBu_r", 2), (D, "ALT - REF", "PRGn", np.nanpercentile(np.abs(D), 99.5))]):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(mat, cmap=cm, vmin=-lim, vmax=lim); ax.set_title(f"{ttl} ({a.plot_head})", fontsize=8)
        ax.axvline(np.atleast_1d(rel)[0], c="k", lw=.4, ls=":"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :]); ax.plot(t, c="#73D07A", lw=.9); ax.set_xlim(0, NB - 1)
    ax.axvline(np.atleast_1d(rel)[0], c="k", lw=.5, ls=":"); ax.set_ylabel("MSE track", fontsize=7)
    gg = G[(G.var_id == vid) & (G["head"] == a.plot_head)].nlargest(6, "track_mse")
    roi = u.get_roi_in_map(r.chrom, start, rel, r.svtype, r.svlen)
    if not isinstance(roi, str):
        for g in roi[roi.roi_id.isin(gg.gene)].itertuples(): ax.annotate(g.roi_id, (g.Start, np.nanmax(t) * .9), fontsize=6, rotation=90, va="top")
    ax.set_xlabel(f"map bin (2,048 bp); window starts {r.chrom}:{int(st):,} (hg38)", fontsize=7); ax.tick_params(labelsize=6)
    fig.suptitle(f"{vid} ({r.gene}, {r.svtype} {r.svlen:,} bp)", fontsize=9); fig.tight_layout()
    fig.savefig(f"{OUT}/plots/{vid}.png", dpi=200); fig.savefig(f"{OUT}/plots/{vid}.pdf"); plt.close(fig)
print(f"plots -> {OUT}/plots")
