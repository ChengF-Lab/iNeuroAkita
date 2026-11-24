#!/usr/bin/env python
"""
01_prep_sv_null.py  --  matched null structural variants for the 120 ADRD SVs

For every real SV, --k null SVs are drawn with
  * the same chromosome      -> same A4 fold (out-of-fold is preserved)
  * the same SVTYPE and SVLEN
  * a comparable gene density: number of TSS in the 1-Mb window centred on the SV within [0.5x, 2x] of the real SV
    (real SVs were chosen near AD genes; random positions would otherwise be gene-poor and give an easy null)
  * a scorable window: <= 1% N in the 1,048,576-bp window centred on the SV
  * no overlap with any real SV +/- 500 kb, and >= 1.2 Mb from chromosome ends
Outputs (variant_scoring/inputs/SVnull/): SVnull_master.tsv (null_id, sv_id, chrom, pos, end, svtype, svlen, n_tss, fold, chunk,
          var_index), chunks/SVnull_<fold>_c<NNN>.txt (SuPreMo SV input), manifest.tsv, prep_report.txt
"""
import argparse, os
import numpy as np, pandas as pd, pysam

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--fa", default="ref_data/hg38.fa", help="hg38 FASTA (indexed)")
ap.add_argument("--k", type=int, default=30)
ap.add_argument("--chunk", type=int, default=150)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--max_tries", type=int, default=20000)
a = ap.parse_args()
B = a.base; S = f"{B}/variant_scoring"; OUT = f"{S}/inputs/SVnull"; os.makedirs(f"{OUT}/chunks", exist_ok=True)
SUP = os.environ.get("SUPREMO_DIR", f"{S}/SuPreMo")   # patched SuPreMo clone (see README)
rng = np.random.default_rng(a.seed); W = 1048576; rep = []

sv = pd.read_csv(f"{S}/inputs/SV120_master.tsv", sep="\t")
fa = pysam.FastaFile(a.fa); clen = dict(zip(fa.references, fa.lengths))
ga = pd.read_csv(f"{SUP}/data/gene_annot_hg38")
ga["tss"] = np.where(ga.strand == "+", ga.start, ga.end)
tss = {c: np.sort(x.tss.values) for c, x in ga.groupby("chr")}
def n_tss(c, mid):
    t = tss.get(c, np.array([])); return int(np.searchsorted(t, mid + W // 2) - np.searchsorted(t, mid - W // 2))
def n_frac(c, mid):
    s = fa.fetch(c, max(0, mid - W // 2), min(clen[c], mid + W // 2)).upper(); return s.count("N") / max(len(s), 1)
excl = {c: x[["pos", "end"]].values for c, x in sv.groupby("chrom")}
def clash(c, p, e): return any((p < E + 500_000) and (e > P - 500_000) for P, E in excl.get(c, []))

rows, short = [], []
for r in sv.itertuples():
    c, L = r.chrom, int(r.end - r.pos); mid0 = (r.pos + r.end) // 2
    nt0 = n_tss(c, mid0); lo, hi = (0, 2) if nt0 == 0 else (0.5 * nt0, 2 * nt0)
    got, tries = 0, 0
    while got < a.k and tries < a.max_tries:
        tries += 1
        p = int(rng.integers(1_200_000, clen[c] - 1_200_000 - L)); e = p + L; mid = (p + e) // 2
        if clash(c, p, e): continue
        nt = n_tss(c, mid)
        if not (lo <= nt <= hi): continue
        if n_frac(c, mid) > 0.01: continue
        got += 1
        rows.append(dict(null_id=f"null_{r.var_id}_{got:02d}", sv_id=r.var_id, chrom=c, pos=p, end=e, ref="-", alt="-",
                         svtype=r.svtype, svlen=r.svlen, n_tss=nt, n_tss_real=nt0, fold=r.fold))
    if got < a.k: short.append(f"{r.var_id}: {got}/{a.k}")
m = pd.DataFrame(rows)
rep.append(f"null SVs: {len(m)} for {sv.var_id.nunique()} real SVs (k = {a.k}); short: {short if short else 'none'}")
rep.append("by type:\n" + m.svtype.value_counts().to_string())

man = []; m["chunk"] = ""; m["var_index"] = -1
for f, x in m.sort_values(["fold", "chrom", "pos"]).groupby("fold"):
    idx = x.index.to_numpy()
    for i in range(0, len(idx), a.chunk):
        sub = idx[i:i + a.chunk]; name = f"SVnull_{f}_c{i // a.chunk:03d}"
        m.loc[sub, "chunk"] = name; m.loc[sub, "var_index"] = np.arange(len(sub))
        m.loc[sub, ["chrom", "pos", "end", "ref", "alt", "svtype", "svlen"]].rename(columns=str.upper) \
            .to_csv(f"{OUT}/chunks/{name}.txt", sep="\t", index=False)
        man.append((len(man), f, name, len(sub), "svnull"))
pd.DataFrame(man, columns=["task", "fold", "chunk", "n", "role"]).to_csv(f"{OUT}/manifest.tsv", sep="\t", index=False)
m.to_csv(f"{OUT}/SVnull_master.tsv", sep="\t", index=False)
rep.append(f"array tasks: {len(man)} (chunk size {a.chunk}); estimated {len(m) * 8 * 2.5 / 3600:.0f} CPU-h (~2.5 s per prediction incl. corr/ROI)")
open(f"{OUT}/prep_report.txt", "w").write("\n\n".join(rep) + "\n"); print("\n\n".join(rep))
