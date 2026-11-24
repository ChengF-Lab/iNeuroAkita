#!/usr/bin/env python
"""
01_prep_eqtl.py  --  SuPreMo inputs for microglia fine-mapped eQTLs (targets) + matched within-gene null SNPs

Design
  targets : unique SNPs with PostProb > --pp_target (default 0.1)
  null    : for every target, --k SNPs from the SAME gene with PostProb < --pp_null (default 0.01),
            not a target themselves, >= --min_dist bp from the target, closest in distance-to-TSS
            -> same locus / LD region / chromosome (= same fold) as the target
  alleles : dbSNP155 (matched by rsID; first SNV ALT); position = snp_end (1-based, hg38);
            REF checked against hg38 (swapped -> fixed, mismatch -> dropped)
Outputs (variant_scoring/inputs/eQTL/)
  eQTL_master.tsv        one row per scored SNP: var_id, role (target/null), chrom, pos, ref, alt, fold, chunk, var_index,
                         PostProb_max, genes, tss_dist ...
  eQTL_null_map.tsv      target_id -> null_id pairs (matching used for paired tests)
  chunks/eQTL_<fold>_c<NNN>.txt   SuPreMo SNV inputs (CHROM POS REF ALT); targets come first
  manifest.tsv           one line per array task: task, fold, chunk_file, n, role
  prep_report.txt
"""
import argparse, os, subprocess
import numpy as np, pandas as pd, pysam

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--fa", default="ref_data/hg38.fa", help="hg38 FASTA (indexed)")
ap.add_argument("--dbsnp", default="ref_data/dbsnp155_hg38.txt", help="dbSNP b155 table (see README)")
ap.add_argument("--pp_target", type=float, default=0.1)
ap.add_argument("--pp_null", type=float, default=0.01)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--min_dist", type=int, default=2048)
ap.add_argument("--chunk", type=int, default=300)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
B = a.base; OUT = f"{B}/variant_scoring/inputs/eQTL"; os.makedirs(f"{OUT}/chunks", exist_ok=True)
rng = np.random.default_rng(a.seed); rep = []

d = pd.read_csv(f"{B}/variant_scoring/raw_input/eQTL_Mic/finemapped_sign_eQTLOnly.csv",
                usecols=["Gene", "chr", "snp_end", "snpID", "PostProb", "external_gene_name", "eSNPto_GeneTSS_dist"])
d["chrom"] = "chr" + d["chr"].astype(str); d["pos"] = d["snp_end"].astype(int)
rep.append(f"input: {len(d)} SNP-gene pairs, {d.snpID.nunique()} SNPs, {d.Gene.nunique()} genes")

# ---------- targets ----------
t = d[d.PostProb > a.pp_target]
tgt_ids = set(t.snpID)
rep.append(f"targets (PP > {a.pp_target}): {len(t)} pairs, {len(tgt_ids)} SNPs")

# ---------- within-gene matched null ----------
pool = d[(d.PostProb < a.pp_null) & (~d.snpID.isin(tgt_ids))]
pool_by_gene = {g: x for g, x in pool.groupby("Gene")}
pairs, short = [], 0
for r in t.itertuples():
    c = pool_by_gene.get(r.Gene)
    if c is None: short += 1; continue
    c = c[(c.pos - r.pos).abs() >= a.min_dist]
    if len(c) == 0: short += 1; continue
    c = c.assign(dd=(c.eSNPto_GeneTSS_dist - r.eSNPto_GeneTSS_dist).abs() + rng.random(len(c)))  # random tie-break
    pick = c.nsmallest(a.k, "dd")
    if len(pick) < a.k: short += 1
    pairs += [(r.snpID, r.Gene, s) for s in pick.snpID]
nm = pd.DataFrame(pairs, columns=["target_id", "gene", "null_id"]).drop_duplicates()
null_ids = set(nm.null_id) - tgt_ids
rep.append(f"null: {len(nm)} target-null pairs, {len(null_ids)} unique null SNPs; target-gene pairs with < {a.k} nulls: {short}")

# ---------- alleles from dbSNP155 (grep by rsID) ----------
ids = sorted(tgt_ids | null_ids)
idf = f"{OUT}/_ids.txt"; hits = f"{OUT}/_dbsnp_hits.txt"
open(idf, "w").write("\n".join(ids) + "\n")
subprocess.run(f"LC_ALL=C grep -wFf {idf} {a.dbsnp} > {hits}", shell=True, check=False)
db = pd.read_csv(hits, sep=r"\s+", header=None, names=["snpID", "dchrom", "dpos", "REF", "ALT", "cls"], dtype=str)
db = db[db.snpID.isin(ids)]
def first_snv_alt(s):
    for x in str(s).split(","):
        x = x.strip().upper()
        if len(x) == 1 and x in "ACGT": return x
    return None
db["alt1"] = db.ALT.map(first_snv_alt); db["REF"] = db.REF.str.upper()
db = db[(db.REF.str.len() == 1) & db.alt1.notna()].drop_duplicates("snpID")
rep.append(f"dbSNP155: alleles found for {len(db)}/{len(ids)} SNPs (SNV only, first ALT)")

# ---------- one row per SNP ----------
g = d[d.snpID.isin(ids)].groupby("snpID").agg(chrom=("chrom", "first"), pos=("pos", "first"),
        PostProb_max=("PostProb", "max"), genes=("external_gene_name", lambda s: ";".join(sorted(set(map(str, s))))),
        tss_dist=("eSNPto_GeneTSS_dist", lambda s: int(s.abs().min()))).reset_index()
m = g.merge(db[["snpID", "REF", "alt1"]], on="snpID", how="inner").rename(columns={"snpID": "var_id", "REF": "ref", "alt1": "alt"})
m["role"] = np.where(m.var_id.isin(tgt_ids), "target", "null")

fa = pysam.FastaFile(a.fa); COMP = dict(zip("ACGT", "TGCA")); chk = []
for r in m.itertuples():
    b = fa.fetch(r.chrom, r.pos - 1, r.pos).upper()
    chk.append("ok" if b == r.ref else "swapped" if b == r.alt else "minus" if b == COMP.get(r.ref, "?") else "mismatch")
m["ref_check"] = chk
sw = m.ref_check == "swapped"; m.loc[sw, ["ref", "alt"]] = m.loc[sw, ["alt", "ref"]].values
mi = m.ref_check == "minus"; m.loc[mi, "ref"] = m.loc[mi, "ref"].map(COMP); m.loc[mi, "alt"] = m.loc[mi, "alt"].map(COMP)
rep.append("REF check vs hg38 (pos = snp_end):\n" + m.groupby(["role", "ref_check"]).size().to_string())
m = m[m.ref_check != "mismatch"]

fold_of = pd.read_csv(f"{B}/models/chrom_to_fold.tsv", sep="\t").set_index("chrom")["fold"].to_dict()
m["fold"] = m.chrom.map(fold_of); m = m[m.fold.notna()]
rep.append("scored SNPs by role:\n" + m.role.value_counts().to_string())

# ---------- chunks: targets first, then nulls ----------
man = []; m["chunk"] = ""; m["var_index"] = -1
for role in ["target", "null"]:
    for f, x in m[m.role == role].sort_values(["fold", "chrom", "pos"]).groupby("fold"):
        idx = x.index.to_numpy()
        for i in range(0, len(idx), a.chunk):
            sub = idx[i:i + a.chunk]; name = f"eQTL_{role}_{f}_c{i // a.chunk:03d}"
            m.loc[sub, "chunk"] = name; m.loc[sub, "var_index"] = np.arange(len(sub))
            m.loc[sub, ["chrom", "pos", "ref", "alt"]].rename(columns=str.upper).to_csv(f"{OUT}/chunks/{name}.txt", sep="\t", index=False)
            man.append((len(man), f, name, len(sub), role))
pd.DataFrame(man, columns=["task", "fold", "chunk", "n", "role"]).to_csv(f"{OUT}/manifest.tsv", sep="\t", index=False)
m.to_csv(f"{OUT}/eQTL_master.tsv", sep="\t", index=False)
nm[nm.target_id.isin(m.var_id) & nm.null_id.isin(m.var_id)].to_csv(f"{OUT}/eQTL_null_map.tsv", sep="\t", index=False)
nt = sum(1 for x in man if x[4] == "target")
rep.append(f"array tasks: {len(man)} (0-{nt - 1} targets, {nt}-{len(man) - 1} nulls), chunk size {a.chunk}")
rep.append(f"estimated CPU time: {len(m) * 8 * 1.2 / 3600:.0f} CPU-h total (~1.2 s per prediction, 8 predictions per SNP)")
open(f"{OUT}/prep_report.txt", "w").write("\n\n".join(rep) + "\n"); print("\n\n".join(rep))
