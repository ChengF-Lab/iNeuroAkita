#!/usr/bin/env python
"""
02_prep_eqtl_alt.py  --  extra ALT alleles of multi-allelic eQTL / null SNPs (first run scored only the first dbSNP ALT)

For every SNP in inputs/eQTL/eQTL_master.tsv, all other SNV ALT alleles listed in dbSNP155 (same REF) are written as
additional SuPreMo rows. Merging (03_merge_eqtl.py) then keeps, per SNP and head, the allele with the largest MSE
(worst-case allele); null SNPs are treated identically.
Outputs (variant_scoring/inputs/eQTL_alt/): eQTL_alt_master.tsv, chunks/*.txt, manifest.tsv, prep_report.txt
"""
import argparse, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--chunk", type=int, default=300)
a = ap.parse_args()
S = f"{a.base}/variant_scoring"; IN = f"{S}/inputs/eQTL"; OUT = f"{S}/inputs/eQTL_alt"; os.makedirs(f"{OUT}/chunks", exist_ok=True)
m = pd.read_csv(f"{IN}/eQTL_master.tsv", sep="\t"); m["role"] = m["role"].fillna("null")
db = pd.read_csv(f"{IN}/_dbsnp_hits.txt", sep=r"\s+", header=None, names=["var_id", "c", "p", "REF", "ALT", "cls"], dtype=str)
db["REF"] = db.REF.str.upper(); db["ALT"] = db.ALT.str.upper()
db = db[(db.REF.str.len() == 1) & (db.ALT.str.len() == 1) & db.ALT.isin(list("ACGT"))].drop_duplicates(["var_id", "ALT"])
x = m[m.ref_check == "ok"][["var_id", "chrom", "pos", "ref", "alt", "role", "fold", "PostProb_max", "genes"]] \
    .merge(db[["var_id", "REF", "ALT"]], on="var_id")
x = x[(x.REF == x.ref) & (x.ALT != x.alt)].drop(columns=["REF", "alt"]).rename(columns={"ALT": "alt"}).reset_index(drop=True)
rep = [f"extra (SNP, ALT) rows: {len(x)}  " + str(x.role.value_counts().to_dict()) + f"; SNPs with extra alleles: {x.var_id.nunique()}"]
man = []; x["chunk"] = ""; x["var_index"] = -1
for role in ["target", "null"]:
    for f, y in x[x.role == role].sort_values(["fold", "chrom", "pos"]).groupby("fold"):
        idx = y.index.to_numpy()
        for i in range(0, len(idx), a.chunk):
            sub = idx[i:i + a.chunk]; name = f"eQTLalt_{role}_{f}_c{i // a.chunk:03d}"
            x.loc[sub, "chunk"] = name; x.loc[sub, "var_index"] = np.arange(len(sub))
            x.loc[sub, ["chrom", "pos", "ref", "alt"]].rename(columns=str.upper).to_csv(f"{OUT}/chunks/{name}.txt", sep="\t", index=False)
            man.append((len(man), f, name, len(sub), role))
pd.DataFrame(man, columns=["task", "fold", "chunk", "n", "role"]).to_csv(f"{OUT}/manifest.tsv", sep="\t", index=False)
x.to_csv(f"{OUT}/eQTL_alt_master.tsv", sep="\t", index=False)
nt = sum(1 for r in man if r[4] == "target")
rep.append(f"array tasks: {len(man)} (0-{nt - 1} targets, {nt}-{len(man) - 1} nulls); ~{len(x) * 8 * 1.2 / 3600:.0f} CPU-h")
open(f"{OUT}/prep_report.txt", "w").write("\n".join(rep) + "\n"); print("\n".join(rep))
