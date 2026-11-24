#!/usr/bin/env python
"""
plot_contact.py  --  unified REF / ALT / ALT-REF contact-map figure for SNVs and SVs (SuPreMo maps, out-of-fold)

Values are Akita log(observed / expected): 0 = contact as expected for that genomic distance, red (> 0) = enriched
(domains, stripes, loops), blue (< 0) = depleted (between domains / insulated). Blue is NOT "no data".

Panels (shared x axis, hg38 Mb); shift-0, forward-strand prediction only (no augmentation average)
  1-2  rotated upper-triangle REF and ALT maps (RdBu_r, +/- --vlim), full 459-kb height
  3    ALT - REF (purple = gain, green = loss; white = 0), --diff_mode:
         enrich            enrichment-weighted: D x w,  w = clip((max(REF, ALT) - --e0) / (--e1 - --e0), 0, 1)
                           -> full weight where either allele shows an enriched contact (log obs/exp >= --e1:
                           domains, stripes, loops; max(REF, ALT) so that contacts gained in ALT are kept),
                           fading to 0 at --e0 (default 0 = expected level); depleted / light-blue pixels in both
                           maps are not shown; |D| < --min_diff shown as 0
         weighted (default) the published-figure display (weights from max(|REF|,|ALT|), i.e. also from depleted pixels)
         raw               the unmodified difference
       Display only - all scores use the unweighted maps.
       Background (--diff_bg): white (default); ref_blue = faded REF in RdBu_r; ref_grey = REF drawn in grey (darker = more enriched) under the diff, so
       domains / stripes stay visible; the diff is overlaid with opacity rising with |shown D| (0 -> transparent).
       white = plain white.
  Triangle sides drawn as thin black frame lines.
  4    disruption track (per-row MSE, shift 0); shaded = local window (+/- --flank bins of the breakpoints)
  5    genes from the GENCODE GTF (--gtf; one transcript per protein-coding gene: Ensembl_canonical / MANE_Select,
       else the longest): exons as boxes, introns as lines, strand arrows; genes with the largest track value at
       their TSS in dark bold. Without a readable GTF: gene bodies from SuPreMo gene_annot_hg38.
Coordinates: frame start = SuPreMo map_start_coord ([1] for DUP, [0] otherwise); in a DUP frame positions downstream
             of the duplication are shifted by SVLEN.
Usage: python plot_contact.py --set SV120 --var_id Lee_DEL_chr17_46146541_46146855 [--diff_mode raw]
Output: variant_scoring/compare/tracks/contact/{SNV,SV}/<set>_<var_id>_<head>.(pdf|svg)  (suffix _<diff_mode>_<diff_bg> if not default)
Orientation: x = genomic coordinate increasing left to right (verified: old SuPreMo maps of rs636317 correlate r = 0.89 in this orientation, 0.34 mirrored)
--head all (default) writes NPC, Mic and Neuron figures
"""
import argparse, glob, gzip, os, re, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore", RuntimeWarning)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
REF_GREY = LinearSegmentedColormap.from_list('ref_grey', ['#fbfbfb', '#e4e4e4', '#ababab', '#6f6f6f'])   # depleted -> enriched
DIFF_CMAP = LinearSegmentedColormap.from_list('PRGn_r_white', ['#00441b', '#1b7837', '#a6dba0', '#ffffff', '#c2a5cf', '#762a83', '#40004b'])

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--set", required=True, choices=["SV120", "SNVcase"])
ap.add_argument("--var_id", required=True)
ap.add_argument("--head", default="all", choices=["all", "NPC", "Mic", "Neuron"], help="all = one figure per head")
ap.add_argument("--vlim", type=float, default=2.0)
ap.add_argument("--diff_mode", default="weighted", choices=["enrich", "weighted", "raw"])
ap.add_argument("--e0", type=float, default=0.0, help="enrich: weight 0 at or below this log(obs/exp)")
ap.add_argument("--e1", type=float, default=0.5, help="enrich: full weight at or above this log(obs/exp)")
ap.add_argument("--diff_bg", default="white", choices=["ref_grey", "ref_blue", "white"],
                help="background of the diff panel: REF in grey (default), faded REF in RdBu, or white")
ap.add_argument("--out_dir", default="tracks/contact", help="folder under compare/; files go to <out_dir>/SNV or <out_dir>/SV")
ap.add_argument("--formats", default="pdf,svg", help="comma-separated output formats (pdf, svg, png)")
ap.add_argument("--edge_tick_mb", type=float, default=0.2, help="spacing (Mb) of the genomic-coordinate ticks on the triangle sides")
ap.add_argument("--max_dist_kb", type=float, default=0, help="crop the triangles at this contact distance (0 = full 917 kb)")
ap.add_argument("--min_diff", type=float, default=0.02, help="enrich: |D| below this is shown as 0")
ap.add_argument("--min_diff_rel", type=float, default=0.05, help="weighted: |D| below this fraction of the map's max |D| is shown as 0")
ap.add_argument("--depleted_rel", type=float, default=0.15, help="weighted: where both maps are depleted, |D| below this fraction of max |D| is shown as 0")
ap.add_argument("--weight_thr", type=float, default=0.5)
ap.add_argument("--gtf", default="ref_data/gencode.v49.basic.annotation.gtf.gz", help="GENCODE basic annotation (GTF)")
ap.add_argument("--flank", type=int, default=25)
ap.add_argument("--n_label", type=int, default=6)
a = ap.parse_args()
if a.head == "all":                                                      # one figure per head
    import subprocess, sys
    for h_ in ["NPC", "Mic", "Neuron"]: subprocess.run([sys.executable] + sys.argv + ["--head", h_], check=True)
    sys.exit(0)
S = f"{a.base}/variant_scoring"; OUT = f"{S}/compare/{a.out_dir}/" + ("SNV" if a.set == "SNVcase" else "SV"); os.makedirs(OUT, exist_ok=True)
SUP = os.environ.get("SUPREMO_DIR", f"{S}/SuPreMo")   # patched SuPreMo clone (see README)
H = {"NPC": "HFF", "Mic": "H1hESC", "Neuron": "GM12878"}[a.head]; NB, BS, DIAG = 448, 2048, 2
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})

# ------------------------------------------------------------------ data
if a.set == "SV120":
    m = pd.read_csv(f"{S}/inputs/SV120_master.tsv", sep="\t"); r = m[m.var_id == a.var_id].iloc[0]
    d = f"{S}/scores/SV120_tracks/{r.fold}"; svtype, L, vend = r.svtype, int(r.svlen), int(r.end)
    title = f"{r.gene}  {r.svtype} {int(r.svlen):,} bp   {r.chrom}:{int(r.pos):,}-{int(r.end):,} (hg38)   [{a.head}]"
else:
    m = pd.read_csv(f"{S}/inputs/SNVcase/cases_master.tsv", sep="\t"); r = m[m.var_id == a.var_id].iloc[0]
    d = f"{S}/scores/SNVcase/{r.fold}"; svtype, L, vend = "SNV", 0, int(r.pos)
    title = f"{r.var_id} ({r.genes})  {r.chrom}:{int(r.pos):,} {r.ref}>{r.alt} (hg38)   [{a.head}]"
M = np.load(glob.glob(f"{d}/*_maps.npy")[0], allow_pickle=True).item()
T = np.load(glob.glob(f"{d}/*_tracks.npy")[0], allow_pickle=True).item()
vi = int(r.var_index); tri = np.triu_indices(NB, DIAG)
_, _, rel, start = M[f"{vi}_maps_{H}_0"]
x0 = int(np.atleast_1d(start)[1 if svtype == "DUP" else 0])
ref_v, alt_v = np.asarray(M[f"{vi}_maps_{H}_0"][0], float), np.asarray(M[f"{vi}_maps_{H}_0"][1], float)
d_v = alt_v - ref_v
ramp = np.clip((np.abs(d_v) - 0.5 * a.min_diff) / (0.5 * a.min_diff), 0, 1)       # soft floor for tiny changes
if a.diff_mode == "enrich":
    w = np.clip((np.fmax(ref_v, alt_v) - a.e0) / (a.e1 - a.e0), 0, 1)
    show_v = d_v * w * ramp
    dlab = f"ALT − REF, enrichment-weighted (full weight where REF or ALT ≥ {a.e1:g}, none where both ≤ {a.e0:g})"
elif a.diff_mode == "weighted":
    sig = np.fmax(np.abs(ref_v), np.abs(alt_v)); lo, hi = np.nanpercentile(sig, [10, 90])
    w = np.clip(np.power(np.clip((sig - lo) / (hi - lo + 1e-10), 0, 1), 1.0 / (a.weight_thr + 0.1)), 0.1, 1.0)
    depleted = np.fmax(ref_v, alt_v) < 0
    dmax = np.nanmax(np.abs(d_v)) if np.isfinite(d_v).any() else 0.0              # floors relative to this map's largest change
    thr = np.where(depleted, a.depleted_rel * dmax, a.min_diff_rel * dmax) + 1e-12
    show_v = d_v * w * np.clip((np.abs(d_v) - 0.5 * thr) / (0.5 * thr), 0, 1)
    dlab = "Diff (MUT-WT)"   # signal-weighted; |Δ| < min_diff_rel x max|Δ| (depleted_rel where both maps < 0) shown as 0 -> figure legend
else:
    show_v = d_v; dlab = "ALT − REF"
nz = np.abs(show_v[np.abs(show_v) > 0]); dlim = np.nanpercentile(nz, 99.5) if nz.size else 0.1
track = np.asarray(T[f"{vi}_mse_track_{H}_0"], float)
nan = np.isnan(track); C = NB // 2
if nan[C - 3:C + 4].any():
    i = C - 3 + int(np.argmax(nan[C - 3:C + 4])); lo_, hi_ = i, i
    while lo_ > 0 and nan[lo_ - 1]: lo_ -= 1
    while hi_ < NB - 1 and nan[hi_ + 1]: hi_ += 1
    b0, b1 = lo_ - 1, hi_ + 1
else:
    b0 = b1 = C
def g2b(g):   # genomic coordinate -> frame bin (float)
    g = np.asarray(g, float); return np.where((svtype == "DUP") & (g > vend), g + L - x0, g - x0) / BS
def b2mb(b): return (x0 + np.asarray(b, float) * BS) / 1e6

# ------------------------------------------------------------------ genes
def gtf_models(path, chrom, lo_bp, hi_bp):
    op = gzip.open if path.endswith(".gz") else open; ex, tx = [], {}
    alt = {chrom, chrom.replace("chr", "")}
    with op(path, "rt") as f:
        for ln in f:
            if ln.startswith("#"): continue
            p = ln.split("\t", 8)
            if len(p) < 9 or p[0] not in alt or p[2] not in ("exon", "transcript"): continue
            s, e = int(p[3]), int(p[4])
            if e < lo_bp or s > hi_bp: continue
            at = p[8]
            gt = re.search(r'gene_(?:type|biotype) "([^"]+)"', at)
            if gt and gt.group(1) != "protein_coding": continue
            gn = re.search(r'gene_name "([^"]+)"', at); tid = re.search(r'transcript_id "([^"]+)"', at)
            if not gn or not tid: continue
            if p[2] == "transcript":
                tx[tid.group(1)] = dict(gene=gn.group(1), strand=p[6], s=s, e=e,
                                        canon=int(bool(re.search(r'tag "(Ensembl_canonical|MANE_Select)"', at))))
            else:
                ex.append((tid.group(1), s, e))
    if not tx: return []
    t = pd.DataFrame.from_dict(tx, orient="index"); t["len"] = t.e - t.s
    best = t.sort_values(["canon", "len"], ascending=False).groupby("gene").head(1)
    exd = pd.DataFrame(ex, columns=["tid", "s", "e"])
    return [dict(gene=b.gene, strand=b.strand, s=b.s, e=b.e, exons=exd[exd.tid == tid][["s", "e"]].values.tolist()) for tid, b in best.iterrows()]
lo_bp, hi_bp = x0, x0 + NB * BS
genes = gtf_models(a.gtf, r.chrom, lo_bp, hi_bp) if a.gtf and os.path.exists(a.gtf) else []
if not genes:
    ga = pd.read_csv(f"{SUP}/data/gene_annot_hg38"); ga = ga[(ga.chr == r.chrom) & ~ga.gene.str.contains(r"\.") & (ga.end > lo_bp) & (ga.start < hi_bp)]
    genes = [dict(gene=g.gene, strand=g.strand, s=g.start, e=g.end, exons=[]) for g in ga.itertuples()]
for g in genes:
    g["bs"], g["be"] = float(g2b(g["s"])), float(g2b(g["e"]))
    tss = g["s"] if g["strand"] == "+" else g["e"]; tb = int(np.clip(g2b(tss), 0, NB - 1))
    g["tval"] = track[tb] if not np.isnan(track[tb]) else 0
top = {g["gene"] for g in sorted(genes, key=lambda x: -x["tval"])[:a.n_label]}

# ------------------------------------------------------------------ figure
FULL_KB = NB * BS / 1e3                                                   # 917.5 kb = largest contact distance
DMAX = FULL_KB if a.max_dist_kb <= 0 else min(a.max_dist_kb, FULL_KB)
FIG_W, LEFT, RIGHT = 7.2, 0.1, 0.86
PW = FIG_W * (RIGHT - LEFT)                                               # panel width (in)
TH = PW / 2 * DMAX / FULL_KB                                              # triangle panel height for 45-degree sides
hts = [TH, TH, TH, 0.85, 1.25]; gap = 0.32; top_m, bot_m = 0.45, 0.55
FIG_H = sum(hts) + gap * (len(hts) - 1) + top_m + bot_m
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = gridspec.GridSpec(5, 1, height_ratios=hts, hspace=gap / np.mean(hts), left=LEFT, right=RIGHT,
                       top=1 - top_m / FIG_H, bottom=bot_m / FIG_H)
edges = np.arange(NB + 1); X, Y = np.meshgrid(edges, edges)
XR, YR = (X + Y) / 2.0, (Y - X) / 2.0                                     # x = (i+j)/2 bins, height = (j-i)/2 bins
HMAX = DMAX / 2 / 1e3                                                     # in Mb (same unit as x) -> 45-degree sides
def sq(v): z = np.full((NB, NB), np.nan); z[tri] = v; return z
axes = []
for k, (vec, ttl, cmap, lim) in enumerate([(ref_v, "WT", "RdBu_r", a.vlim), (alt_v, "MUT", "RdBu_r", a.vlim), (show_v, dlab, DIFF_CMAP, dlim)]):
    ax = fig.add_subplot(gs[k]); axes.append(ax)
    XX, YY = b2mb(XR), YR * BS / 1e6
    if k < 2 or a.diff_bg == "white":
        pm = ax.pcolormesh(XX, YY, np.ma.masked_invalid(sq(vec)).T, cmap=cmap, vmin=-lim, vmax=lim, shading="flat", rasterized=True)
    else:
        bg = np.ma.masked_invalid(sq(ref_v)).T
        if a.diff_bg == "ref_grey":
            ax.pcolormesh(XX, YY, bg, cmap=REF_GREY, vmin=-0.4 * a.vlim, vmax=0.4 * a.vlim, shading="flat", rasterized=True)
        else:
            ax.pcolormesh(XX, YY, bg, cmap="RdBu_r", vmin=-a.vlim, vmax=a.vlim, alpha=0.35, shading="flat", rasterized=True)
        dz = sq(vec).T; A = np.nan_to_num(np.clip(np.abs(dz) / (0.3 * lim), 0, 1))
        ax.pcolormesh(XX, YY, np.ma.masked_invalid(dz), cmap=cmap, vmin=-lim, vmax=lim, alpha=A, shading="flat", rasterized=True)
        pm = ScalarMappable(norm=Normalize(-lim, lim), cmap=cmap)
    xa, xb = b2mb(0), b2mb(NB)                                              # triangle frame (45-degree sides)
    ax.plot([xa, xa + HMAX], [0, HMAX], c="k", lw=0.6, clip_on=False); ax.plot([xb, xb - HMAX], [0, HMAX], c="k", lw=0.6, clip_on=False)
    ax.set_ylim(0, HMAX); ax.set_xlim(b2mb(0), b2mb(NB)); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    # genomic-coordinate ticks on the two triangle sides (square-matrix convention): a pixel's contact partners are read by
    # following 45-degree lines: up-left to the left side = right anchor (B), up-right to the right side = left anchor (A)
    step = a.edge_tick_mb; L = 0.012 * (xb - xa); r2 = np.sqrt(0.5)
    for g in np.arange(np.ceil(xa / step) * step, xb + 1e-9, step):
        if g - xa <= 2 * HMAX + 1e-9:                                       # left side: points (i = 0, j = g)
            px, py = xa + (g - xa) / 2, (g - xa) / 2
            ax.plot([px, px - L * r2], [py, py + L * r2], c="k", lw=0.5, clip_on=False)
            ax.text(px - 2.2 * L * r2, py + 2.2 * L * r2, f"{g:.1f}", fontsize=5.5, rotation=45, ha="right", va="bottom",
                    rotation_mode="anchor", clip_on=False)
        if xb - g <= 2 * HMAX + 1e-9:                                       # right side: points (i = g, j = last)
            px, py = xb - (xb - g) / 2, (xb - g) / 2
            ax.plot([px, px + L * r2], [py, py + L * r2], c="k", lw=0.5, clip_on=False)
            ax.text(px + 2.2 * L * r2, py + 2.2 * L * r2, f"{g:.1f}", fontsize=5.5, rotation=-45, ha="left", va="bottom",
                    rotation_mode="anchor", clip_on=False)
    ax.set_title(ttl, loc="left", fontsize=7, pad=2)
    for b in sorted({b0, b1}): ax.axvline(b2mb(b + 0.5), c="k", lw=0.4, ls=":")
    ax.tick_params(labelbottom=False, length=2)
    for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    ax.set_facecolor("white")
    pos = ax.get_position(); cax = fig.add_axes([0.875, pos.y0 + 0.012, 0.011, pos.height - 0.024])
    cb = fig.colorbar(pm, cax=cax); cb.ax.tick_params(labelsize=6, length=2); cb.set_ticks([-lim, 0, lim])
    cb.ax.set_yticklabels([f"{-lim:.2g}", "0", f"{lim:.2g}"])
    cb.set_label("log(obs/exp)" if k < 2 else "Δ log(obs/exp)", fontsize=6)
axT = fig.add_subplot(gs[3], sharex=axes[0])
w0, w1 = max(0, b0 - a.flank + 1), min(NB, b1 + a.flank)
axT.axvspan(b2mb(w0), b2mb(w1), color="#73D07A", alpha=0.18, lw=0)
axT.plot(b2mb(np.arange(NB) + 0.5), track, c="#2F7D33", lw=0.8); axT.set_ylabel("disruption\n(MSE track)")
axT.tick_params(labelbottom=False, length=2); axT.set_ylim(bottom=0)
for s_ in ["top", "right"]: axT.spines[s_].set_visible(False)
# gene panel: greedy row packing including label width
axG = fig.add_subplot(gs[4], sharex=axes[0]); axG.set_yticks([])
span = b2mb(NB) - b2mb(0); rows_end = []
gl = sorted(genes, key=lambda g: g["bs"])
for g in gl:
    lab_w = len(g["gene"]) * 0.0105 * span
    x_s, x_e = b2mb(max(g["bs"], 0)), b2mb(min(g["be"], NB)); mid = (x_s + x_e) / 2
    left, right = min(x_s, mid - lab_w / 2), max(x_e, mid + lab_w / 2)
    for ri, re_ in enumerate(rows_end):
        if left > re_ + 0.004 * span: rows_end[ri] = right; g["row"] = ri; break
    else:
        rows_end.append(right); g["row"] = len(rows_end) - 1
nrow = max(1, len(rows_end))
for g in gl:
    y = -g["row"] * 1.0; bold = g["gene"] in top; col = "#1f4e79" if bold else "#8c9aa8"
    x_s, x_e = b2mb(max(g["bs"], 0)), b2mb(min(g["be"], NB))
    axG.plot([x_s, x_e], [y, y], c=col, lw=0.7, solid_capstyle="butt", zorder=2)
    if g["exons"]:
        for es, ee in g["exons"]:
            bs_, be_ = float(g2b(es)), float(g2b(ee))
            if be_ < 0 or bs_ > NB: continue
            axG.add_patch(Rectangle((b2mb(max(bs_, 0)), y - 0.18), max(b2mb(min(be_, NB)) - b2mb(max(bs_, 0)), span * 0.0008), 0.36,
                                    facecolor=col, edgecolor="none", zorder=3))
    else:
        axG.add_patch(Rectangle((x_s, y - 0.12), x_e - x_s, 0.24, facecolor=col, edgecolor="none", alpha=0.85, zorder=3))
    n_ch = int((x_e - x_s) / (0.018 * span))                                      # strand chevrons
    for c_ in np.linspace(x_s, x_e, n_ch + 2)[1:-1]:
        axG.text(c_, y, ">" if g["strand"] == "+" else "<", ha="center", va="center", fontsize=5, color="white" if not g["exons"] else col,
                 fontweight="bold", zorder=4)
    t = axG.text((x_s + x_e) / 2, y - 0.3, g["gene"], ha="center", va="top", fontsize=5.8, style="italic",
                 fontweight="bold" if bold else "normal", color="#000" if bold else "#666", zorder=5)
    t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
axG.set_ylim(-nrow + 0.1, 0.6)
for b in sorted({b0, b1}): axG.axvline(b2mb(b + 0.5), c="k", lw=0.4, ls=":", zorder=1)
axG.set_xlabel(f"{r.chrom} (Mb, hg38)" + ("   (downstream of the DUP shifted by its length)" if svtype == "DUP" else ""))
for s_ in ["top", "right", "left"]: axG.spines[s_].set_visible(False)
axG.text(1.0, 1.02, "genes: " + ("GTF transcript models" if (a.gtf and genes and genes[0]["exons"]) else "gene bodies (no GTF given)"),
         transform=axG.transAxes, ha="right", va="bottom", fontsize=5.5, color="#888")
fig.suptitle(title, fontsize=7.5, x=0.1, ha="left")
sfx = "" if (a.diff_mode, a.diff_bg) == ("weighted", "white") else f"_{a.diff_mode}_{a.diff_bg}"
base = f"{OUT}/{a.set}_{a.var_id}_{a.head}{sfx}"
fmts = [x.strip() for x in a.formats.split(",") if x.strip()]
for x in fmts: fig.savefig(f"{base}.{x}", dpi=300)
print("written", base + "." + "/.".join(fmts))
