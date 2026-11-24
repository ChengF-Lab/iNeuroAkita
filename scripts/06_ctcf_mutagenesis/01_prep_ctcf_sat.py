#!/usr/bin/env python
"""
01_prep_ctcf_sat.py  --  single-base saturation mutagenesis of CTCF motifs: relate the predicted effect
                         of each base change to the motif PWM

Sites (hg38, autosomes, >= 600 kb from chromosome ends, >= --min_gap apart):
  strong   JASPAR --motif FIMO hits (q < 0.05) with NPC CTCF ChIP signal (max over motif +/- 50 bp) in the top 20% of bound sites
  weak     bound (signal >= --bound_min) but at or below the median signal of bound sites
  unbound  motif present, signal < --unbound_max                      (motif without binding: does the model use context?)
  control  motif-free 19-bp windows 5-50 kb from a strong site (no FIMO CTCF hit of any matrix within 500 bp, signal < --unbound_max)
Variants: every position of the site x the 3 other bases (SNVs; SuPreMo input CHROM POS REF ALT, 1-based, + strand)
Per variant: motif position (1..W, in motif orientation), REF/ALT base in motif orientation, PWM log-odds of REF and ALT
  (log2((p + 0.001) / 0.25)), dLLR = ALT - REF, information content of the position, site motif score.
Chunks per fold (out-of-fold model, as for the eQTL); score each chunk with SuPreMo (3 heads, 4 augmentations; see README).
Outputs: variant_scoring/inputs/CTCF_sat/{sites.tsv, ctcf_sat_master.tsv, chunks/, manifest.tsv, prep_report.txt}
"""
import argparse, os, re
import numpy as np, pandas as pd, pysam, pyBigWig

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--fa", default="ref_data/hg38.fa", help="hg38 FASTA (indexed)")
J = "ref_data/CTCF_motif"
ap.add_argument("--fimo_bed", default=f"{J}/fimo_out/fimo_qval0.05.bed", help="FIMO CTCF motif hits (bed: chr, start0, end, motif_id, p, strand)")
ap.add_argument("--meme", default=f"{J}/JASPAR2024_CTCF9.meme")
ap.add_argument("--motif", default="MA0139.1")
ap.add_argument("--chip", default="", help="default variant_scoring/genome_tracks/input/ctcf/NPC_CTCF_ChIPseq_hg38.bw")
ap.add_argument("--n_strong", type=int, default=150); ap.add_argument("--n_weak", type=int, default=75)
ap.add_argument("--n_unbound", type=int, default=75); ap.add_argument("--n_ctrl", type=int, default=50)
ap.add_argument("--bound_min", type=float, default=5.0); ap.add_argument("--unbound_max", type=float, default=1.0)
ap.add_argument("--min_gap", type=int, default=100000); ap.add_argument("--chunk", type=int, default=300)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
S = f"{a.base}/variant_scoring"; OUT = f"{S}/inputs/CTCF_sat"; os.makedirs(f"{OUT}/chunks", exist_ok=True)
CHIP = a.chip or f"{S}/genome_tracks/input/ctcf/NPC_CTCF_ChIPseq_hg38.bw"
rng = np.random.default_rng(a.seed); rep = []; COMP = dict(zip("ACGT", "TGCA")); EDGE = 600_000

# ---------------- PWM from the MEME file
txt = open(a.meme).read()
blk = re.search(rf"MOTIF\s+{re.escape(a.motif)}\b.*?letter-probability matrix:[^\n]*\n(.*?)(?:\n\s*\n|\nURL|\nMOTIF|\Z)", txt, re.S)
if not blk: raise SystemExit(f"{a.motif} not found in {a.meme}")
P = np.array([[float(v) for v in l.split()] for l in blk.group(1).strip().split("\n") if l.strip() and l.split()[0][0].isdigit()])
W = P.shape[0]; LLR = np.log2((P + 1e-3) / 0.25); IC = 2 + (P * np.log2(np.clip(P, 1e-9, 1))).sum(1); IDX = dict(zip("ACGT", range(4)))
rep.append(f"motif {a.motif}: width {W}, total IC {IC.sum():.1f} bits; consensus {''.join('ACGT'[i] for i in P.argmax(1))}")

# ---------------- motif hits + ChIP
fb = pd.read_csv(a.fimo_bed, sep="\t", header=None, comment="#")
fb = fb.iloc[:, :6]; fb.columns = ["chrom", "start", "end", "motif_id", "p", "strand"]
allhits = fb.copy()
fb = fb[(fb.motif_id == a.motif) & fb.chrom.isin([f"chr{i}" for i in range(1, 23)]) & (fb.end - fb.start == W)]
fb = fb.sort_values("p").drop_duplicates(["chrom", "start"]).reset_index(drop=True)
fa = pysam.FastaFile(a.fa); L = {c: fa.get_reference_length(c) for c in fb.chrom.unique()}
fb = fb[(fb.start > EDGE) & (fb.end < fb.chrom.map(L) - EDGE)]
bw = pyBigWig.open(CHIP)
def chip(c, s, e):
    try: v = bw.stats(c, max(0, s - 50), e + 50, type="max")[0]
    except RuntimeError: v = None
    return 0.0 if v is None else float(v)
fb["chip"] = [chip(r.chrom, r.start, r.end) for r in fb.itertuples()]
rep.append(f"{a.motif} hits (autosomes, away from ends): {len(fb)}; ChIP >= {a.bound_min}: {(fb.chip >= a.bound_min).sum()}; "
           f"< {a.unbound_max}: {(fb.chip < a.unbound_max).sum()}")
bound = fb[fb.chip >= a.bound_min]; q80, q50 = bound.chip.quantile(0.8), bound.chip.quantile(0.5)
pools = {"strong": fb[fb.chip >= q80], "weak": fb[(fb.chip >= a.bound_min) & (fb.chip <= q50)], "unbound": fb[fb.chip < a.unbound_max]}
rep.append(f"strata: strong ChIP >= {q80:.1f} (n={len(pools['strong'])}), weak {a.bound_min}-{q50:.1f} (n={len(pools['weak'])}), "
           f"unbound < {a.unbound_max} (n={len(pools['unbound'])})")
chosen = []
def far(c, s):
    return all(not (x["chrom"] == c and abs(x["start"] - s) < a.min_gap) for x in chosen)
for cls, n in [("strong", a.n_strong), ("weak", a.n_weak), ("unbound", a.n_unbound)]:
    pool = pools[cls].sample(frac=1, random_state=int(rng.integers(1e9)))
    k = 0
    for r in pool.itertuples():
        if k >= n: break
        if far(r.chrom, r.start):
            chosen.append(dict(site_id=f"{cls}_{k:03d}", cls=cls, chrom=r.chrom, start=int(r.start), end=int(r.end),
                               strand=r.strand, chip=r.chip, fimo_p=r.p)); k += 1
# controls: motif-free windows near strong sites
hits_by_chr = {c: x.sort_values("start") for c, x in allhits.groupby("chrom")}
strong = [x for x in chosen if x["cls"] == "strong"]; k = 0; tries = 0
while k < a.n_ctrl and tries < 50 * a.n_ctrl:
    tries += 1; s0 = strong[int(rng.integers(len(strong)))]
    s = s0["start"] + int(rng.choice([-1, 1]) * rng.integers(5000, 50000)); c = s0["chrom"]
    h = hits_by_chr.get(c)
    if h is not None and ((h.start - s).abs() < 500).any(): continue
    seq = fa.fetch(c, s, s + W).upper()
    if "N" in seq or chip(c, s, s + W) >= a.unbound_max: continue
    chosen.append(dict(site_id=f"control_{k:03d}", cls="control", chrom=c, start=s, end=s + W, strand="+", chip=chip(c, s, s + W), fimo_p=np.nan)); k += 1
sites = pd.DataFrame(chosen)
fold_of = pd.read_csv(f"{a.base}/models/chrom_to_fold.tsv", sep="\t").set_index("chrom")["fold"].to_dict()
sites["fold"] = sites.chrom.map(fold_of)
def motif_score(seq, strand):
    m = seq if strand == "+" else "".join(COMP[b] for b in reversed(seq))
    return float(sum(LLR[i, IDX[b]] for i, b in enumerate(m) if b in IDX))
sites["seq"] = [fa.fetch(r.chrom, r.start, r.end).upper() for r in sites.itertuples()]
sites["gc"] = sites.seq.map(lambda x: (x.count("G") + x.count("C")) / len(x))
sites["motif_score"] = [motif_score(r.seq, r.strand) for r in sites.itertuples()]
rep.append("sites by class:\n" + sites.groupby("cls").agg(n=("site_id", "size"), chip_median=("chip", "median"),
                                                        motif_score_median=("motif_score", "median"), gc=("gc", "mean")).round(2).to_string())

# ---------------- saturation SNVs
V = []
for r in sites.itertuples():
    for i, b in enumerate(r.seq):
        if b not in IDX: continue
        mp = i if r.strand == "+" else W - 1 - i
        bm = b if r.strand == "+" else COMP[b]
        for alt in "ACGT":
            if alt == b: continue
            am = alt if r.strand == "+" else COMP[alt]
            V.append(dict(var_id=f"{r.site_id}_p{mp + 1:02d}_{am}", site_id=r.site_id, cls=r.cls, chrom=r.chrom, pos=r.start + i + 1,
                          ref=b, alt=alt, strand=r.strand, motif_pos=mp + 1, ref_m=bm, alt_m=am, llr_ref=LLR[mp, IDX[bm]],
                          llr_alt=LLR[mp, IDX[am]], dLLR=LLR[mp, IDX[am]] - LLR[mp, IDX[bm]], ic=IC[mp], fold=r.fold,
                          chip=r.chip, motif_score=r.motif_score))
m = pd.DataFrame(V); man = []; m["chunk"] = ""; m["var_index"] = -1
for f, x in m.sort_values(["fold", "chrom", "pos"]).groupby("fold"):
    idx = x.index.to_numpy()
    for i in range(0, len(idx), a.chunk):
        sub = idx[i:i + a.chunk]; name = f"CTCFsat_{f}_c{i // a.chunk:03d}"
        m.loc[sub, "chunk"] = name; m.loc[sub, "var_index"] = np.arange(len(sub))
        m.loc[sub, ["chrom", "pos", "ref", "alt"]].rename(columns=str.upper).to_csv(f"{OUT}/chunks/{name}.txt", sep="\t", index=False)
        man.append((len(man), f, name, len(sub), "sat"))
pd.DataFrame(man, columns=["task", "fold", "chunk", "n", "role"]).to_csv(f"{OUT}/manifest.tsv", sep="\t", index=False)
m.to_csv(f"{OUT}/ctcf_sat_master.tsv", sep="\t", index=False); sites.to_csv(f"{OUT}/sites.tsv", sep="\t", index=False)
rep.append(f"variants: {len(m)} ({len(sites)} sites x {W} positions x 3); array tasks: {len(man)} (chunk {a.chunk}); "
           f"estimated CPU time {len(m) * 8 * 1.2 / 3600:.0f} CPU-h (~1.2 s per prediction, 8 predictions per SNV)")
open(f"{OUT}/prep_report.txt", "w").write("\n\n".join(rep) + "\n"); print("\n\n".join(rep))
