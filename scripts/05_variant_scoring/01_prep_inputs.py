#!/usr/bin/env python
"""
01_prep_inputs.py  --  build out-of-fold SuPreMo inputs (42 functional SNVs + 120 AD/ADRD SVs)

Inputs
  variant_scoring/raw_input/SNV42/SNV42_CRISPR_frVar_hg38.tsv      42 CRISPR-validated SNVs (columns: var_id chrom pos_hg19 pos_hg38 ref alt locus ...)
  variant_scoring/raw_input/SV120/SV120_ADRD_hg38.tsv              120 SVs (hg38; columns: var_id chrom pos end ref alt svtype svlen ...)
  models/chrom_to_fold.tsv                             chromosome -> fold that held it out
  hg38.fa (+ .fai)                                             REF check for SNVs

Outputs (variant_scoring/inputs/)
  SNV42_master.tsv, SV120_master.tsv     one row per variant: var_id, set, chrom, pos, end, ref, alt, svtype, svlen,
                                         fold, var_index (0-based row in that fold's SuPreMo file), annotations
  SNV42_fold{k}.txt                      SuPreMo SNV input: CHROM POS REF ALT
  SV120_fold{k}.txt                      SuPreMo SV input:  CHROM POS END REF ALT SVTYPE SVLEN
  prep_report.txt                        REF check, fold counts, dropped variants
"""
import argparse, os, sys
import pandas as pd
import pysam

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--fa", default="ref_data/hg38.fa", help="hg38 FASTA (indexed)")
a = ap.parse_args()

B = a.base
OUT = f"{B}/variant_scoring/inputs"; os.makedirs(OUT, exist_ok=True)
fold_of = pd.read_csv(f"{B}/models/chrom_to_fold.tsv", sep="\t").set_index("chrom")["fold"].to_dict()
fa = pysam.FastaFile(a.fa)
rep = []
COMP = dict(zip("ACGT", "TGCA"))

# ---------------- SNVs ----------------
s = pd.read_csv(f"{B}/variant_scoring/raw_input/SNV42/SNV42_CRISPR_frVar_hg38.tsv", sep="\t", dtype=str)
snv = pd.DataFrame({
    "var_id": s["var_id"], "set": "SNV42", "chrom": s["chrom"], "pos": s["pos_hg38"].astype(int),
    "ref": s["ref"].str.upper(), "alt": s["alt"].str.upper(), "pos_hg19": s["pos_hg19"],
    "gene": s["locus"], "disorder": s["disorder"], "tfbs": s["TFBS"], "gene_cell_type": s["gene_cell_type"],
    "MPRA_log2FC": s["MPRA_log2FC"], "CRISPR_q": s["CRISPR_q"]})
snv["end"] = snv["pos"]; snv["svtype"] = "SNV"; snv["svlen"] = 0
chk = []
for i, r in snv.iterrows():
    g = fa.fetch(r.chrom, r.pos - 1, r.pos).upper()
    if g == r.ref:                         chk.append("ok")
    elif g == r.alt:                       chk.append("ref_alt_swapped")
    elif g == COMP.get(r.ref, "N"):        chk.append("minus_strand")
    else:                                  chk.append(f"mismatch_genome={g}")
snv["ref_check"] = chk
rep.append("== SNV42 REF check vs hg38\n" + snv["ref_check"].value_counts().to_string())
bad = snv[snv.ref_check != "ok"]
if len(bad):
    rep.append("non-ok SNVs (scored as given only if fixable):\n" + bad[["var_id","chrom","pos","ref","alt","ref_check"]].to_string(index=False))
    # swapped alleles: use the genome base as REF so SuPreMo builds the right WT sequence
    sw = snv.ref_check == "ref_alt_swapped"
    snv.loc[sw, ["ref", "alt"]] = snv.loc[sw, ["alt", "ref"]].values
    mi = snv.ref_check == "minus_strand"
    snv.loc[mi, "ref"] = snv.loc[mi, "ref"].map(COMP); snv.loc[mi, "alt"] = snv.loc[mi, "alt"].map(COMP)
    snv = snv[~snv.ref_check.str.startswith("mismatch")]

# ---------------- SVs ----------------
sv = pd.read_csv(f"{B}/variant_scoring/raw_input/SV120/SV120_ADRD_hg38.tsv", sep="\t", dtype={"ref": str, "alt": str})
sv["set"] = "SV120"
dup = sv.duplicated(["chrom", "pos", "end", "svtype"], keep="first")
rep.append(f"== SV120: {len(sv)} SVs; " + sv.svtype.value_counts().to_dict().__str__() + f"; duplicated coordinates: {int(dup.sum())}")

# ---------------- fold assignment + per-fold files ----------------
for name, df, cols in [("SNV42", snv, ["chrom", "pos", "ref", "alt"]),
                       ("SV120", sv, ["chrom", "pos", "end", "ref", "alt", "svtype", "svlen"])]:
    df = df.copy()
    df["fold"] = df["chrom"].map(fold_of)
    nofold = df[df.fold.isna()]
    if len(nofold):
        rep.append(f"{name}: dropped (chromosome in no fold, e.g. chrY): " + ", ".join(nofold.var_id))
    df = df[df.fold.notna()].sort_values(["fold", "chrom", "pos"]).reset_index(drop=True)
    df["var_index"] = df.groupby("fold").cumcount()
    df.to_csv(f"{OUT}/{name}_master.tsv", sep="\t", index=False)
    for f, g in df.groupby("fold"):
        h = g[cols].copy()
        h.columns = [c.upper() for c in cols]
        h.to_csv(f"{OUT}/{name}_{f}.txt", sep="\t", index=False)
    rep.append(f"== {name} per fold\n" + df.fold.value_counts().sort_index().to_string())

open(f"{OUT}/prep_report.txt", "w").write("\n\n".join(rep) + "\n")
print("\n\n".join(rep)); print(f"\nwritten to {OUT}")
