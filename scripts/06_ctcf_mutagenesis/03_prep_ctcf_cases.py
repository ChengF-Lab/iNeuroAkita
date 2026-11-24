#!/usr/bin/env python
"""
03_prep_ctcf_cases.py  --  case loci (Fig. 3): single-base CTCF motif mutations to be scored WITH contact maps

Sites (NPC CTCF ChIP defines the classes, as in 01_prep_ctcf_sat.py):
  top      the --n_top strong sites with the largest per-site disruption in the saturation screen (max MSE over 57 SNVs, --head)
  old      fixed example loci (--old_loci; default: FABP5/IMPA1, S100Z, SDC3/NKAIN1): a saturation site inside the window if there is one (chr5: strong_039), otherwise
           the MA0139.1 FIMO hit (q < 0.05) in the central half of the window with the highest NPC CTCF ChIP signal
SNVs per site (2):
  core     largest predicted disruption: observed in the screen (sites in the screen) or, for new sites, the base change with the
           largest mean log10 MSE at that motif position / REF / ALT across the 148 strong sites
  control  smallest predicted disruption, chosen the same way (a base change at a poorly conserved motif position)
Outputs: variant_scoring/inputs/CTCFcase/{cases_master.tsv, case_<fold>.txt, manifest.tsv}
"""
import argparse, os
import numpy as np, pandas as pd, pysam, pyBigWig

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--fa", default="ref_data/hg38.fa", help="hg38 FASTA (indexed)")
ap.add_argument("--fimo_bed", default="ref_data/CTCF_motif/fimo_out/fimo_qval0.05.bed", help="FIMO CTCF motif hits (bed)")
ap.add_argument("--motif", default="MA0139.1")
ap.add_argument("--head", default="NPC"); ap.add_argument("--n_top", type=int, default=5)
ap.add_argument("--old_loci", default="chr8:80656384-81704960:FABP5_IMPA1,chr5:76333056-77381632:S100Z,chr1:30457856-31506432:SDC3_NKAIN1")
a = ap.parse_args()
S = f"{a.base}/variant_scoring"; OUT = f"{S}/inputs/CTCFcase"; os.makedirs(OUT, exist_ok=True)
CHIP = f"{S}/genome_tracks/input/ctcf/NPC_CTCF_ChIPseq_hg38.bw"; COMP = dict(zip("ACGT", "TGCA"))

m = pd.read_csv(f"{S}/compare/CTCF_sat/ctcf_sat_scores.tsv", sep="\t"); sites = pd.read_csv(f"{S}/inputs/CTCF_sat/sites.tsv", sep="\t")
mse = f"mse_{a.head}"; ok = m.dropna(subset=[mse]).copy(); ok["l"] = np.log10(ok[mse])
W = int(m.motif_pos.max())
LLR = pd.concat([m[["motif_pos", "ref_m", "llr_ref"]].set_axis(["p", "b", "v"], axis=1),
                 m[["motif_pos", "alt_m", "llr_alt"]].set_axis(["p", "b", "v"], axis=1)]).groupby(["p", "b"]).v.first().unstack()  # PWM log-odds
exp = ok[ok.cls == "strong"].groupby(["motif_pos", "ref_m", "alt_m"]).l.mean()       # expected effect of each base change
fold_of = pd.read_csv(f"{a.base}/models/chrom_to_fold.tsv", sep="\t").set_index("chrom")["fold"].to_dict()

def snv_row(site, kind, v, label):
    return dict(case_id=f"{site['case_id']}_{kind}", site_case=site["case_id"], label=label, role=kind, site_id=site.get("site_id", ""),
                chrom=site["chrom"], pos=int(v["pos"]), ref=v["ref"], alt=v["alt"], strand=site["strand"], motif_start=int(site["start"]),
                motif_end=int(site["end"]), motif_pos=int(v["motif_pos"]), ref_m=v["ref_m"], alt_m=v["alt_m"], dLLR=float(v["dLLR"]),
                screen_log10_mse=v.get("l", np.nan), chip=float(site["chip"]), fold=fold_of[site["chrom"]])

rows = []
# ---------------- top sites of the screen
per = ok.groupby("site_id").l.max()
top = per[per.index.str.startswith("strong")].sort_values(ascending=False).head(a.n_top)
def from_screen(sid, case_id, label):
    s = sites.set_index("site_id").loc[sid].to_dict(); s.update(site_id=sid, case_id=case_id)
    x = ok[ok.site_id == sid]
    rows.append(snv_row(s, "core", x.loc[x.l.idxmax()].to_dict(), label)); rows.append(snv_row(s, "control", x.loc[x.l.idxmin()].to_dict(), label))
for k, sid in enumerate(top.index): from_screen(sid, f"top{k + 1}", f"top {k + 1} of the saturation screen")

# ---------------- fixed example loci (--old_loci)
fa = pysam.FastaFile(a.fa); bw = pyBigWig.open(CHIP)
fb = None
for spec in a.old_loci.split(","):
    reg, name = spec.rsplit(":", 1); c, se = reg.split(":"); w0, w1 = map(int, se.split("-"))
    inside = sites[(sites.chrom == c) & (sites.start > w0) & (sites.end < w1) & (sites.cls == "strong")]
    if len(inside):
        from_screen(inside.sort_values("chip").site_id.iloc[-1], f"old_{name}", f"example locus ({name})"); continue
    if fb is None:
        fb = pd.read_csv(a.fimo_bed, sep="\t", header=None, comment="#").iloc[:, :6]; fb.columns = ["chrom", "start", "end", "motif_id", "p", "strand"]
        fb = fb[(fb.motif_id == a.motif) & (fb.end - fb.start == W)]
    q = (w1 - w0) // 4
    h = fb[(fb.chrom == c) & (fb.start > w0 + q) & (fb.end < w1 - q)].copy()
    if h.empty: print("no motif in", spec); continue
    h["chip"] = [float(bw.stats(c, max(0, r.start - 50), r.end + 50, type="max")[0] or 0) for r in h.itertuples()]
    r = h.sort_values("chip").iloc[-1]; seq = fa.fetch(c, int(r.start), int(r.end)).upper()
    site = dict(case_id=f"old_{name}", chrom=c, start=int(r.start), end=int(r.end), strand=r.strand, chip=r.chip)
    cand = []
    for i, b in enumerate(seq):
        mp = i if r.strand == "+" else W - 1 - i; bm = b if r.strand == "+" else COMP.get(b, "N")
        for alt in "ACGT":
            if alt == b or bm not in "ACGT": continue
            am = alt if r.strand == "+" else COMP[alt]; e = exp.get((mp + 1, bm, am), np.nan)
            cand.append(dict(pos=int(r.start) + i + 1, ref=b, alt=alt, motif_pos=mp + 1, ref_m=bm, alt_m=am,
                             dLLR=LLR.loc[mp + 1, am] - LLR.loc[mp + 1, bm], l=e))
    cand = pd.DataFrame(cand).dropna(subset=["l"])
    lab = f"example locus ({name}); motif site, ChIP {r.chip:.0f}"
    rows.append(snv_row(site, "core", cand.loc[cand.l.idxmax()].to_dict() | {"l": np.nan}, lab))
    rows.append(snv_row(site, "control", cand.loc[cand.l.idxmin()].to_dict() | {"l": np.nan}, lab))

C = pd.DataFrame(rows).drop_duplicates("case_id")
for (c, p, r_, al) in C[["chrom", "pos", "ref", "alt"]].itertuples(index=False):   # REF check against the genome
    assert fa.fetch(c, p - 1, p).upper() == r_, (c, p, r_)
man = []; C["var_index"] = -1
for f, x in C.sort_values(["fold", "chrom", "pos"]).groupby("fold"):
    C.loc[x.index, "var_index"] = np.arange(len(x))
    x[["chrom", "pos", "ref", "alt"]].rename(columns=str.upper).to_csv(f"{OUT}/case_{f}.txt", sep="\t", index=False)
    man.append((len(man), f, f"case_{f}", len(x)))
pd.DataFrame(man, columns=["task", "fold", "name", "n"]).to_csv(f"{OUT}/manifest.tsv", sep="\t", index=False)
C.to_csv(f"{OUT}/cases_master.tsv", sep="\t", index=False)
print(C[["case_id", "chrom", "pos", "ref", "alt", "strand", "motif_pos", "ref_m", "alt_m", "dLLR", "screen_log10_mse", "chip", "fold"]].round(2).to_string(index=False))
print(f"{len(C)} SNVs at {C.site_case.nunique()} sites; array tasks: {len(man)}")
