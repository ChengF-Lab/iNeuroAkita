#!/usr/bin/env python
"""
02_prep_snv_cases.py  --  SuPreMo inputs to re-score the selected SNVs with tracks + maps, and a null SNP subset with tracks

cases : SNV42 selected in any head (SNV42_ranked.tsv, selected) + eQTL case_eligible in any head (eQTL_ranked.tsv),
        eQTL ALT = worst-case allele of the --head (alt_max_<head>)
null  : --n_null random within-gene low-PP eQTL null SNPs (worst-case allele), tracks only -> null of the local metrics
Outputs (variant_scoring/inputs/SNVcase/): cases_master.tsv, null_master.tsv, case_<fold>.txt, null_<fold>_c<NNN>.txt, manifest.tsv
"""
import argparse, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".", help="project directory (layout: see README)")
ap.add_argument("--head", default="Mic"); ap.add_argument("--n_null", type=int, default=1000)
ap.add_argument("--chunk", type=int, default=200); ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
S = f"{a.base}/variant_scoring"; OUT = f"{S}/inputs/SNVcase"; os.makedirs(OUT, exist_ok=True)
s = pd.read_csv(f"{S}/compare/selection/SNV42_ranked.tsv", sep="\t"); s = s[s.selected.astype(str) == "True"]
e = pd.read_csv(f"{S}/compare/selection/eQTL_ranked.tsv", sep="\t"); e = e[e.case_eligible.astype(str) == "True"]
cases = pd.concat([
    pd.DataFrame({"var_id": s.var_id, "set": "SNV42", "genes": s.gene, "chrom": s.chrom, "pos": s.pos, "ref": s.ref, "alt": s.alt, "fold": s.fold}),
    pd.DataFrame({"var_id": e.var_id, "set": "eQTL", "genes": e.genes, "chrom": e.chrom, "pos": e.pos, "ref": e.ref, "alt": e[f"alt_max_{a.head}"], "fold": e.fold})],
    ignore_index=True).drop_duplicates("var_id")
q = pd.read_csv(f"{S}/compare/eQTL/eQTL_scores.tsv", sep="\t"); q["role"] = q.role.fillna("null")
q = q[(q.role == "null") & q[f"alt_max_{a.head}"].notna()]
null = q.sample(min(a.n_null, len(q)), random_state=a.seed)
null = pd.DataFrame({"var_id": null.var_id, "chrom": null.chrom, "pos": null.pos, "ref": null.ref, "alt": null[f"alt_max_{a.head}"], "fold": null.fold})
def write_if_changed(df, path):                                         # keep mtime of unchanged inputs -> run_28 skips them
    new = df.to_csv(sep="\t", index=False)
    if not (os.path.exists(path) and open(path).read() == new): open(path, "w").write(new)
man = []; cases["var_index"] = -1; null["chunk"] = ""; null["var_index"] = -1
for f, x in cases.sort_values(["fold", "chrom", "pos"]).groupby("fold"):
    cases.loc[x.index, "var_index"] = np.arange(len(x))
    write_if_changed(x[["chrom", "pos", "ref", "alt"]].rename(columns=str.upper), f"{OUT}/case_{f}.txt")
    man.append((len(man), "case", f, f"case_{f}", len(x)))
for f, x in null.sort_values(["fold", "chrom", "pos"]).groupby("fold"):
    idx = x.index.to_numpy()
    for i in range(0, len(idx), a.chunk):
        sub = idx[i:i + a.chunk]; name = f"null_{f}_c{i // a.chunk:03d}"
        null.loc[sub, "chunk"] = name; null.loc[sub, "var_index"] = np.arange(len(sub))
        write_if_changed(null.loc[sub, ["chrom", "pos", "ref", "alt"]].rename(columns=str.upper), f"{OUT}/{name}.txt")
        man.append((len(man), "null", f, name, len(sub)))
pd.DataFrame(man, columns=["task", "kind", "fold", "name", "n"]).to_csv(f"{OUT}/manifest.tsv", sep="\t", index=False)
cases.to_csv(f"{OUT}/cases_master.tsv", sep="\t", index=False); null.to_csv(f"{OUT}/null_master.tsv", sep="\t", index=False)
print(cases.to_string(index=False)); print(f"null SNPs: {len(null)}; array tasks: {len(man)}")
