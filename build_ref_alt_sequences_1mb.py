#!/usr/bin/env python3
# Build 1,048,576-bp REF/ALT sequences centered at each SNP (Akita-ready),
# writing REF and ALT to separate FASTA files. Handles multi-allelic ALT by
# emitting one ALT record per allele ("ALT=T", "ALT=A", etc.).
#
# Input CSV must include at least: chr, Pos, REF, ALT (plus snpID or ID preferred).
# Examples of accepted headers: chr|CHROM, Pos|POS|position|snp_start, REF, ALT, snpID|ID|rsID, Gene|Pair_Symbol
#
# Notes:
# - Coordinates are assumed GRCh38/hg38 (1-based). FASTA must match build and naming.
# - Alleles are expected on the + strand (as in VCF). Do NOT reverse-complement.

import pandas as pd
from pyfaidx import Fasta
from pathlib import Path
import argparse
import sys

WINDOW = 1048576   # 1 Mb
HALF   = WINDOW // 2
WRAP   = 60        # 0 -> no wrap
DEDUPE = False     # drop duplicates by (chr, pos, REF, ALT)
ALT_POLICY = "all" # "all" (one ALT per allele) or "first"

REFSEQ_FOR = {
    "1":"NC_000001.11","2":"NC_000002.12","3":"NC_000003.12","4":"NC_000004.12",
    "5":"NC_000005.10","6":"NC_000006.12","7":"NC_000007.14","8":"NC_000008.11",
    "9":"NC_000009.12","10":"NC_000010.11","11":"NC_000011.10","12":"NC_000012.12",
    "13":"NC_000013.11","14":"NC_000014.9","15":"NC_000015.10","16":"NC_000016.10",
    "17":"NC_000017.11","18":"NC_000018.10","19":"NC_000019.10","20":"NC_000020.11",
    "21":"NC_000021.9","22":"NC_000022.11","X":"NC_000023.11","Y":"NC_000024.10","MT":"NC_012920.1"
}

def wrap_seq(s, w=60):
    if not w or w <= 0: return s
    return "\n".join(s[i:i+w] for i in range(0, len(s), w))

def pick_col(cols, candidates):
    for c in candidates:
        if c in cols: return c
    return None

def map_chrom_for_fa(fa, ch):
    s = str(ch).strip()
    if s.startswith("chr"): s = s[3:]
    if s in {"M","Mt","MT","chrM"}: s = "MT"
    fa_contigs = list(fa.keys())
    fa_has_chr = any(c.startswith("chr") for c in fa_contigs)
    fa_has_nc  = any(c.startswith("NC_") for c in fa_contigs)
    if fa_has_chr: return "chr"+s
    if fa_has_nc:  return REFSEQ_FOR.get(s, s)
    return str(ch)

def clamp_window(fa, chrom, pos_1b):
    clen = len(fa[chrom])
    start = max(0, (pos_1b - 1) - HALF)
    end   = min(clen, start + WINDOW)
    if end - start < WINDOW:
        start = max(0, end - WINDOW)
        end   = min(clen, start + WINDOW)
    return start, end

def main(args):
    df = pd.read_csv(args.input,sep="\t",dtype=str).fillna("")

    # --- autodetect columns ---
    cols = set(df.columns)
    col_chr = pick_col(cols, ["Chr","chr","CHROM","Chrom","chromosome","Chromosome"])
    col_pos = pick_col(cols, ["Pos_hg38","Pos","POS","position","Position","Start","snp_start"])
    col_ref = pick_col(cols, ["REF","Ref","ref"])
    col_alt = pick_col(cols, ["ALT","Alt","alt","ALT_selected"])
    if not (col_chr and col_pos and col_ref and col_alt):
        missing = [x for x,y in [("chr",col_chr),("Pos",col_pos),("REF",col_ref),("ALT",col_alt)] if y is None]
        sys.exit(f"ERROR: Missing required columns: {missing}. Found: {sorted(cols)}")

    col_id  = pick_col(cols, ["snpID","ID","rsID","rsid"])
    col_gene= pick_col(cols, ["Gene","Pair_Symbol","gene","Symbol"])

    # clean & types
    df[col_chr] = df[col_chr].astype(str).str.strip()
    df[col_pos] = df[col_pos].astype(str).str.replace(r"[^\d]", "", regex=True)
    df = df[df[col_pos] != ""].copy()
    df[col_pos] = df[col_pos].astype(int)
    df[col_ref] = df[col_ref].str.upper()
    df[col_alt] = df[col_alt].str.upper()

    # Only SNVs (A/C/G/T single letters). If you have VarType, filter SNV.
    if "VarType" in df.columns:
        df = df[df["VarType"].str.upper().isin(["SNV","SNP","SINGLE",""]) | (df["VarType"]=="")].copy()
    df = df[df[col_ref].str.len()==1].copy()
    # Keep all rows even if ALT is multi-allelic; we'll split later.

    if DEDUPE:
        df = df.drop_duplicates(subset=[col_chr, col_pos, col_ref, col_alt]).copy()

    # open FASTA
    fa = Fasta(args.fasta, one_based_attributes=False)
    out_ref = open(args.out_ref, "w")
    out_alt = open(args.out_alt, "w")

    mismatch_rows = []
    n_total = 0
    n_pairs = 0
    n_contig_skip = 0
    n_ref_mismatch = 0
    n_shifted = 0

    for _, r in df.iterrows():
        n_total += 1
        chrom = map_chrom_for_fa(fa, r[col_chr])
        if chrom not in fa:
            n_contig_skip += 1
            mismatch_rows.append({"snpID": r.get(col_id,""),
                                  "reason":"contig_missing",
                                  "csv_chr": r[col_chr],
                                  "mapped_chrom": chrom,
                                  "pos": r[col_pos]})
            continue

        pos_1b = int(r[col_pos])
        REF    = str(r[col_ref])
        ALTs   = [a for a in str(r[col_alt]).split(",") if a] if ALT_POLICY=="all" else [str(r[col_alt]).split(",")[0]]

        # window
        start, end = clamp_window(fa, chrom, pos_1b)
        seq = fa[chrom][start:end].seq.upper()
        idx = (pos_1b - 1) - start

        # sanity: REF must match
        ok = (0 <= idx < len(seq)) and (seq[idx] == REF)
        if not ok and 0 <= idx-1 < len(seq) and seq[idx-1] == REF:
            idx -= 1
            ok = True
            n_shifted += 1
        if not ok:
            n_ref_mismatch += 1
            base = seq[idx] if 0 <= idx < len(seq) else "NA"
            mismatch_rows.append({
                "snpID": r.get(col_id,""), "reason": "REF_mismatch",
                "chrom": chrom, "pos": pos_1b, "window_start": start, "idx": idx,
                "REF": REF, "base_in_fasta": base
            })
            continue

        # header core
        sid = r.get(col_id, "")
        gene = r.get(col_gene, "")
        meta_bits = []
        if sid:  meta_bits.append(sid)
        if gene: meta_bits.append(f"gene={gene}")
        meta = "|".join(meta_bits) if meta_bits else f"{chrom}:{pos_1b}"

        # write REF once
        header_ref = f">{meta}|{chrom}:{pos_1b}|REF={REF}"
        out_ref.write(header_ref+"\n"+wrap_seq(seq, WRAP)+"\n")

        # write each ALT allele
        for alt in ALTs:
            if len(alt) != 1 or alt == REF:  # skip non-SNV or identical alt
                continue
            seq_alt = seq[:idx] + alt + seq[idx+1:]
            header_alt = f">{meta}|{chrom}:{pos_1b}|ALT={alt}"
            out_alt.write(header_alt+"\n"+wrap_seq(seq_alt, WRAP)+"\n")
            n_pairs += 1

    out_ref.close()
    out_alt.close()

    # mismatches
    if mismatch_rows:
        pd.DataFrame(mismatch_rows).to_csv(args.mismatch_tsv, sep="\t", index=False)

    print("✅ Done.")
    print(f"Input rows:            {n_total}")
    print(f"ALT records written:   {n_pairs}  (REF written once per input row)")
    print(f"Skipped (contig):      {n_contig_skip}")
    print(f"REF mismatches:        {n_ref_mismatch}")
    print(f"Off-by-one corrected:  {n_shifted}")
    print(f"REF FASTA:             {args.out_ref}")
    print(f"ALT FASTA:             {args.out_alt}")
    if mismatch_rows:
        print(f"Mismatch log:          {args.mismatch_tsv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build 1,048,576-bp REF/ALT FASTAs centered at SNPs.")
    p.add_argument("-i","--input", required=True, help="Input CSV with chr/Pos/REF/ALT (plus optional snpID/Gene)")
    p.add_argument("-f","--fasta", required=True, help="Reference FASTA (matching build; indexed by pyfaidx)")
    p.add_argument("--out-ref", default="REF_1Mb.fa", help="Output FASTA for REF sequences")
    p.add_argument("--out-alt", default="ALT_1Mb.fa", help="Output FASTA for ALT sequences (one per ALT allele)")
    p.add_argument("--mismatch-tsv", default="ref_mismatches.tsv", help="Log of sites where REF != FASTA base")
    args = p.parse_args()
    main(args)

