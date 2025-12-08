#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict Akita maps for REF/ALT 1Mb sequences and compute disruption scores.

- Pairs REF/ALT by core header (e.g., "rsID|gene=...|chr:pos").
- Reuses predictions (no duplicate inference).
- Reconstructs square maps from Akita's upper-triangular output.
- Masks ±diagonal_offset and computes L2/MSE/L1/MaxAbs.
- Writes a tidy TSV and optional PDFs (REF/ALT/DIFF).
- Supports triangle and square plot formats.
- Can annotate with gene markers.

Requires: basenji (seqnn, dna_io), numpy, pandas, matplotlib, Biopython.
"""

import os, re, json, argparse, gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from Bio import SeqIO
import tensorflow as tf

from basenji import dna_io, seqnn

# ------------------------- Utils -------------------------

def read_params(model_dir, data_dir=None):
    with open(os.path.join(model_dir, "params.json")) as f:
        P = json.load(f)
    params_model = P["model"]
    
    # Try multiple possible locations for statistics.json
    stats_paths = []
    if data_dir:
        stats_paths.append(os.path.join(data_dir, "statistics.json"))
    stats_paths.append(os.path.join(model_dir, "statistics.json"))
    stats_paths.append(os.path.join(model_dir, "..", "NPC_data", "statistics.json"))
    
    stats_path = None
    for path in stats_paths:
        if os.path.exists(path):
            stats_path = path
            break
    
    if not stats_path:
        raise FileNotFoundError(f"statistics.json not found. Tried: {stats_paths}")
    
    with open(stats_path) as f:
        S = json.load(f)
    return params_model, S

def load_model(model_dir, model_file="model_best.h5"):
    with open(os.path.join(model_dir, "params.json")) as f:
        params = json.load(f)
    params_model = params["model"]
    net = seqnn.SeqNN(params_model)
    net.restore(os.path.join(model_dir, model_file))
    return net, params

def read_fasta_to_dict(path):
    d = {}
    for rec in SeqIO.parse(path, "fasta"):
        d[rec.id] = str(rec.seq).upper()
    return d

def core_header(h):
    # Strip allele tokens so REF/ALT pair to same core
    parts = [t for t in h.split("|") if not (t.startswith("REF=") or t.startswith("ALT="))]
    return "|".join(parts)

def header_info(core):
    # Extract rsID/gene/chr/pos for reporting
    snpID = ""
    gene = ""
    chrom = ""
    pos = ""
    for t in core.split("|"):
        if t.startswith("gene="):
            gene = t.split("=",1)[1]
        elif t.startswith("chr"):
            chrom, pos = t.split(":")
        elif t.startswith("rs"):
            snpID = t
    return snpID, gene, chrom, pos

def to_square(vec, n_bins, diag_off):
    """
    vec: [n_triu_values] for triu(k=diag_off)
    returns [n_bins, n_bins] symmetric (diagonals set to 0)
    """
    z = np.zeros((n_bins, n_bins), dtype=np.float32)
    rr, cc = np.triu_indices(n_bins, diag_off)
    if vec.shape[0] != rr.shape[0]:
        raise ValueError(f"Vector length {vec.shape[0]} != expected triu count {rr.shape[0]} for n_bins={n_bins}, k={diag_off}")
    z[rr, cc] = vec
    # mirror
    z = z + z.T
    # zero masked diagonals (we'll ignore them in metrics)
    for k in range(-diag_off, diag_off+1):
        i = np.arange(n_bins)
        j = i + k
        m = (j>=0) & (j<n_bins)
        z[i[m], j[m]] = 0.0
    return z

def frob_norm(mat):
    return float(np.sqrt(np.sum(mat*mat)))

def mse(mat):
    return float(np.mean(mat*mat))

def l1_mean(mat):
    return float(np.mean(np.abs(mat)))

def max_abs(mat):
    return float(np.max(np.abs(mat)))

def ensure_len(seq, expected, name):
    if len(seq) != expected:
        raise ValueError(f"{name} length {len(seq)} != expected {expected}")

def read_bed_genes(path):
    """Read a BED/BED.gz with columns: chrom start end name score strand.
    Handles both tab-separated and space-separated files."""
    if not path:
        return pd.DataFrame()
    
    if not os.path.exists(path):
        print(f"[ERROR] BED file not found: {path}")
        return pd.DataFrame()
    
    opener = gzip.open if path.endswith(".gz") else open
    cols = ["chrom","start","end","name","score","strand"]
    rows = []
    line_num = 0
    
    try:
        with opener(path, "rt") as f:
            for line in f:
                line_num += 1
                line = line.rstrip("\n\r")
                if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                    continue
                
                # Try tab first, then split on whitespace
                if "\t" in line:
                    toks = [t.strip() for t in line.split("\t") if t.strip()]
                else:
                    # Split on any whitespace (handles multiple spaces)
                    # Use split() without arguments to handle multiple spaces/tabs
                    toks = line.split()
                
                if len(toks) < 3:
                    print(f"[WARN] Skipping line {line_num}: insufficient columns (need at least 3): {line[:50]}")
                    continue
                
                try:
                    chrom = toks[0].strip()
                    start = int(toks[1].strip())
                    end   = int(toks[2].strip())
                    name  = toks[3].strip() if len(toks) > 3 else "."
                    score = toks[4].strip() if len(toks) > 4 else "0"
                    strand= toks[5].strip() if len(toks) > 5 else "."
                    
                    # Validate
                    if start < 0 or end < start:
                        print(f"[WARN] Skipping line {line_num}: invalid coordinates (start={start}, end={end})")
                        continue
                    
                    rows.append((chrom, start, end, name, score, strand))
                except ValueError as e:
                    print(f"[WARN] Skipping line {line_num}: error parsing numbers: {e}")
                    continue
        
        if len(rows) == 0:
            print(f"[WARN] No valid gene entries found in {path}")
            print(f"[DEBUG] Checked {line_num} lines total")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=cols)
        print(f"[INFO] Successfully loaded {len(df)} genes from {path}")
        if len(df) > 0:
            print(f"[DEBUG] Sample gene: {df.iloc[0].to_dict()}")
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to read BED file {path}: {e}")
        return pd.DataFrame()

def draw_gene_track(ax, genes_df, x0, x1, pos_bp, promoter_kb=2.0,
                    gene_color="#2c7fb8", promoter_color="#f03b20", alpha=0.6, 
                    set_xlim=True):
    """Draw gene spans with promoter highlight; x-axis in bp.
    
    Args:
        set_xlim: If False, don't set xlim (useful when sharex is used)
    """
    if set_xlim:
        ax.set_xlim(x0, x1)
    ax.set_ylim(-0.1, 1.2)  # Extended ylim to accommodate labels
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("gray")
    ax.spines["bottom"].set_linewidth(0.5)
    
    # Draw SNP position line
    if pos_bp and x0 <= pos_bp <= x1:
        ax.axvline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)

    if genes_df.empty:
        ax.text((x0+x1)/2, 0.5, "no genes in view", ha="center", va="center", fontsize=9, style="italic")
        return

    for _,g in genes_df.iterrows():
        gx0, gx1 = max(g.start, x0), min(g.end, x1)
        if gx1 <= gx0: 
            continue
        
        # Draw gene body with edge for better visibility
        ax.add_patch(Rectangle((gx0, 0.35), gx1-gx0, 0.30,
                               facecolor=gene_color, edgecolor="black", 
                               linewidth=0.5, alpha=alpha, zorder=5))
        
        # promoter (± promoter_kb around TSS)
        tss = g.end if g.strand == "-" else g.start
        px0 = max(x0, tss - int(promoter_kb*1000))
        px1 = min(x1, tss + int(promoter_kb*1000))
        if px1 > px0:
            ax.add_patch(Rectangle((px0, 0.25), px1-px0, 0.50,
                                   facecolor=promoter_color, edgecolor="none", 
                                   alpha=0.4, zorder=6))
        
        # Always label genes below the axis (like Plot_ChIP_Map.py)
        # Calculate position in axes fraction for consistent placement
        mid_bp = 0.5 * (gx0 + gx1)
        x_frac = (mid_bp - x0) / (x1 - x0) if (x1 - x0) > 0 else 0.5
        label_offset_axes = 0.05  # Offset below axis (axes fraction)
        
        # Use transform=ax.transAxes for consistent placement below axis
        txt = ax.text(
            x_frac, -label_offset_axes, g.name,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#000000", clip_on=False, zorder=7,
            weight='bold'  # Make text bolder for better visibility
        )
        # Add white outline for better contrast (like Plot_ChIP_Map.py)
        txt.set_path_effects([
            pe.Stroke(linewidth=3.0, foreground="white"),
            pe.Normal()
        ])

def add_gene_labels_to_map(ax, genes_df, x0, x1, label_offset_axes=0.05, 
                           label_fontsize=9, label_outline="white", label_color="#000000"):
    """
    Add gene labels directly on contact map axes (like Plot_ChIP_Map.py).
    
    Args:
        ax: Contact map axis
        genes_df: DataFrame with gene information (columns: start, end, name)
        x0, x1: Genomic coordinates (start, end)
        label_offset_axes: Offset below axis (axes fraction)
        label_fontsize: Font size for labels
        label_outline: Outline color for labels
        label_color: Text color for labels
    """
    if genes_df.empty or (x1 - x0) <= 0:
        return
    
    for _, g in genes_df.iterrows():
        gs = max(x0, int(g.start))
        ge = min(x1, int(g.end))
        if ge <= x0 or gs >= x1:
            continue
        
        mid_bp = 0.5 * (gs + ge)
        x_frac = (mid_bp - x0) / (x1 - x0)
        name = str(g.get("name", ""))
        
        txt = ax.text(
            x_frac, -label_offset_axes, name,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=label_fontsize, color=label_color, clip_on=False, zorder=10,
            weight='bold'
        )
        if label_outline:
            txt.set_path_effects([
                pe.Stroke(linewidth=3.0, foreground=label_outline),
                pe.Normal()
            ])

def format_genomic_pos(pos_bp, include_magnitude=False):
    """
    Format genomic position: e.g., 59792925 -> "5.98" (or "5.98 (1e7)" if include_magnitude=True)
    """
    if pos_bp == 0:
        return "0"
    # Determine the order of magnitude
    magnitude = int(np.floor(np.log10(abs(pos_bp))))
    # Get the base value (e.g., 59792925 -> 5.98)
    base_value = pos_bp / (10 ** magnitude)
    # Format to 2 decimal places
    if include_magnitude:
        return f"{base_value:.2f} ({10**magnitude:.0e})"
    else:
        return f"{base_value:.2f}"

# ----------------------- Main workflow -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directory containing params.json and model_best.h5")
    ap.add_argument("--data_dir", default=None, help="Data directory containing statistics.json (optional, will search model_dir if not provided)")
    ap.add_argument("--ref_fa", required=True, help="REF fasta")
    ap.add_argument("--alt_fa", required=True, help="ALT fasta")
    ap.add_argument("--out_scores", default="disruption_scores.tsv")
    ap.add_argument("--plots_dir", default="plots_ref_alt_diff", help="Folder to save heatmaps; '' to skip plots")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--reduce", default="mean", help="'mean' over targets or an integer index (e.g., 0)")
    ap.add_argument("--vlim", type=float, default=2.0, help="Color scale |v| for plots")
    ap.add_argument("--plot_format", default="both", choices=["square", "triangle", "both"], 
                    help="Plot format: 'square', 'triangle', or 'both'")
    ap.add_argument("--genes_bed", default="", help="Optional BED file with genes (chrom start end name score strand)")
    ap.add_argument("--promoter_kb", type=float, default=2.0, help="Promoter window size in kb")
    ap.add_argument("--gene_color", default="#2c7fb8", help="Gene color")
    ap.add_argument("--promoter_color", default="#f03b20", help="Promoter color")
    ap.add_argument("--gene_alpha", type=float, default=0.35, help="Gene track alpha")
    ap.add_argument("--diff_signal_threshold", type=float, default=0.3, 
                    help="Signal strength weighting for diff map (0-1): lower values emphasize high-signal regions more. "
                         "0.0 = show all equally, 1.0 = strongly de-emphasize low-signal regions")
    args = ap.parse_args()

    # Load model and stats
    print("Loading model…")
    net, params = load_model(args.model_dir)
    params_model, stats = params["model"], None
    # stats from training data (needed for lengths/diagonals)
    # Try multiple possible locations for statistics.json
    stats_paths = []
    if args.data_dir:
        stats_paths.append(os.path.join(args.data_dir, "statistics.json"))
    stats_paths.append(os.path.join(args.model_dir, "statistics.json"))
    stats_paths.append(os.path.join(args.model_dir, "..", "NPC_data", "statistics.json"))
    
    stats_path = None
    for path in stats_paths:
        if os.path.exists(path):
            stats_path = path
            break
    
    if not stats_path:
        raise FileNotFoundError(f"statistics.json not found. Tried: {stats_paths}")
    
    with open(stats_path) as f:
        stats = json.load(f)

    seq_length   = stats["seq_length"]
    pool_width   = stats["pool_width"]
    crop_bins    = stats["crop_bp"] // pool_width
    diag_off     = stats["diagonal_offset"]
    n_bins_full  = seq_length // pool_width
    n_bins       = n_bins_full - 2*crop_bins
    HALF         = seq_length // 2
    crop_bp      = stats["crop_bp"]

    print(f"seq_length={seq_length}, pool_width={pool_width}, crop_bins={crop_bins}, diag_off={diag_off}, n_bins={n_bins}")

    # Read genes if provided
    genes_all = read_bed_genes(args.genes_bed) if args.genes_bed else pd.DataFrame()
    if args.genes_bed:
        if genes_all.empty:
            print(f"[WARN] No genes loaded from {args.genes_bed}. Check file format.")
        else:
            print(f"Loaded {len(genes_all)} genes from {args.genes_bed}")
            print(f"  Chromosomes: {sorted(genes_all['chrom'].unique())}")
            print(f"  Gene names: {', '.join(genes_all['name'].unique()[:10])}{'...' if len(genes_all['name'].unique()) > 10 else ''}")

    # Read REF/ALT FASTAs
    ref = read_fasta_to_dict(args.ref_fa)
    alt = read_fasta_to_dict(args.alt_fa)

    # Validate lengths
    for h,s in list(ref.items()) + list(alt.items()):
        ensure_len(s, seq_length, h)

    # Pair headers and build a unique sequence set to avoid duplicate predicts
    pairs = []   # list of (href, halt, core)
    needed = {}  # header -> sequence
    for href, sref in ref.items():
        k = core_header(href)
        # each matching ALT (supports multi-allelic ALT -> multiple halts)
        hits = [(halt, salt) for halt, salt in alt.items() if core_header(halt) == k]
        if not hits:
            print(f"[WARN] no ALT match for: {k}")
            continue
        for halt, salt in hits:
            pairs.append((href, halt, k))
            needed[href] = sref
            needed[halt] = salt

    if not pairs:
        raise SystemExit("No REF/ALT pairs formed. Check headers (need REF=/ALT= tokens).")

    print(f"Unique sequences to predict: {len(needed)}; Pairs: {len(pairs)}")

    # Predict all unique sequences and cache
    headers = list(needed.keys())
    seqs    = [needed[h] for h in headers]

    # Batch predict
    preds_cache = {}
    for i in range(0, len(headers), args.batch):
        batch_headers = headers[i:i+args.batch]
        batch_seqs = seqs[i:i+args.batch]
        X = np.stack([dna_io.dna_1hot(s) for s in batch_seqs], axis=0)  # [B, L, 4]
        Y = net.model.predict(X, verbose=0)                              # [B, n_vec, T]
        for j, hh in enumerate(batch_headers):
            preds_cache[hh] = Y[j]  # keep full targets

    # Reduction setting
    reduce_mode = args.reduce
    reduce_idx = None
    if reduce_mode != "mean":
        try:
            reduce_idx = int(reduce_mode)
        except Exception:
            raise ValueError("--reduce must be 'mean' or an integer channel index")

    # Score each pair
    rows = []
    if args.plots_dir:
        os.makedirs(args.plots_dir, exist_ok=True)

    for href, halt, core in pairs:
        P_ref = preds_cache[href]  # [n_vec, T]
        P_alt = preds_cache[halt]  # [n_vec, T]

        T = P_ref.shape[1] if P_ref.ndim == 2 else 1
        if P_ref.shape != P_alt.shape:
            raise ValueError(f"Shape mismatch {href} vs {halt}: {P_ref.shape} vs {P_alt.shape}")

        # Per-target disruption
        per_t = []
        diff_maps_for_plot = None

        if reduce_idx is not None:
            # single channel chosen
            vec_r = P_ref[:, reduce_idx]
            vec_a = P_alt[:, reduce_idx]
            M_ref = to_square(vec_r, n_bins, diag_off)
            M_alt = to_square(vec_a, n_bins, diag_off)
            # CHANGE: ALT - REF instead of REF - ALT
            D = M_alt - M_ref
            per_t.append({
                "target": reduce_idx,
                "L2_fro": frob_norm(D),
                "MSE": mse(D),
                "L1_mean": l1_mean(D),
                "MaxAbs": max_abs(D)
            })
            diff_maps_for_plot = (M_ref, M_alt, D)

        else:
            # mean over targets (default)
            # also compute per-target metrics (optional) and aggregate
            Ms_ref = []
            Ms_alt = []
            for t in range(T):
                vec_r = P_ref[:, t]
                vec_a = P_alt[:, t]
                Ms_ref.append(to_square(vec_r, n_bins, diag_off))
                Ms_alt.append(to_square(vec_a, n_bins, diag_off))

            # stack: [T, n, n]
            Ms_ref = np.stack(Ms_ref, axis=0)
            Ms_alt = np.stack(Ms_alt, axis=0)
            # aggregate maps by mean across targets
            M_ref = Ms_ref.mean(axis=0)
            M_alt = Ms_alt.mean(axis=0)
            # CHANGE: ALT - REF instead of REF - ALT
            D = M_alt - M_ref
            diff_maps_for_plot = (M_ref, M_alt, D)

            per_t.append({
                "target": "mean",
                "L2_fro": frob_norm(D),
                "MSE": mse(D),
                "L1_mean": l1_mean(D),
                "MaxAbs": max_abs(D)
            })

        # Parse allele letters from headers (if present)
        mR = re.search(r'REF=([ACGT])', href)
        mA = re.search(r'ALT=([ACGT])', halt)
        REF_base = mR.group(1) if mR else ""
        ALT_base = mA.group(1) if mA else ""

        snpID, gene, chrom, pos = header_info(core)
        pos_bp = int(pos) if pos else 0
        
        for row in per_t:
            rows.append({
                "key": core,
                "snpID": snpID,
                "gene": gene,
                "chr": chrom,
                "pos": pos,
                "REF": REF_base,
                "ALT": ALT_base,
                "reduce": reduce_mode,
                **row
            })

        # Plots
        if args.plots_dir:
            M_ref, M_alt, D = diff_maps_for_plot
            v = float(args.vlim)
            
            # Calculate genomic coordinates for x-axis
            win_left  = (pos_bp - HALF) if pos_bp else 0
            win_right = (pos_bp + HALF) if pos_bp else seq_length
            x0 = win_left  + crop_bp
            x1 = win_right - crop_bp
            
            # Select genes in visible window
            sub_genes = pd.DataFrame()
            if not genes_all.empty and chrom and pos_bp:
                # Normalize chromosome names for matching (handle both "chr11" and "11")
                def normalize_chr(c):
                    c = str(c).strip()
                    if not c.startswith("chr"):
                        c = "chr" + c
                    return c
                
                chrom_normalized = normalize_chr(chrom)
                genes_all_normalized = genes_all.copy()
                genes_all_normalized["chrom"] = genes_all_normalized["chrom"].apply(normalize_chr)
                
                # Filter genes: overlap with visible window
                sub_genes = genes_all_normalized[
                    (genes_all_normalized["chrom"] == chrom_normalized) &
                    (genes_all_normalized["end"]   >= x0) &
                    (genes_all_normalized["start"] <= x1)
                ].copy()
                
                if len(sub_genes) > 0:
                    print(f"  Found {len(sub_genes)} genes in window for {snpID or core}: {', '.join(sub_genes['name'].tolist())}")
                elif not genes_all.empty:
                    genes_on_chrom = genes_all_normalized[genes_all_normalized['chrom'] == chrom_normalized]
                    if len(genes_on_chrom) > 0:
                        print(f"  No genes found in window for {snpID or core} (chrom={chrom_normalized}, window={x0:,}-{x1:,})")
                        print(f"    Available genes on this chrom: {genes_on_chrom[['name', 'start', 'end']].to_dict('records')}")
                    else:
                        print(f"  No genes found on chromosome {chrom_normalized} in gene file")
            
            # Generate genomic position ticks with improved formatting
            n_ticks = 5
            tick_positions_bp = np.linspace(x0, x1, n_ticks)
            # Format labels without magnitude (will add to axis label)
            tick_labels = [format_genomic_pos(int(tp), include_magnitude=False) for tp in tick_positions_bp]
            
            # Determine magnitude for axis label (use first position)
            # Format: "Genomic position (bp) (1e+07)" - magnitude shown once in axis label
            if pos_bp and tick_positions_bp[0] > 0:
                magnitude = int(np.floor(np.log10(abs(tick_positions_bp[0]))))
                magnitude_str = f" ({10**magnitude:.0e})"
            else:
                magnitude_str = ""
            
            # IMPROVEMENT 3: Create signal-weighted diff map
            # Calculate signal strength (absolute values) in REF and ALT
            signal_ref = np.abs(M_ref)
            signal_alt = np.abs(M_alt)
            # Use the maximum of REF and ALT signal as the combined signal strength
            signal_combined = np.maximum(signal_ref, signal_alt)
            
            # Calculate signal strength percentile for weighting
            signal_percentiles = np.percentile(signal_combined, [10, 50, 90])
            signal_min, signal_med, signal_max = signal_percentiles
            
            # Create alpha/weight map: regions with stronger signal get higher weight (1.0)
            # Regions with weaker signal get lower weight (approaching 0.0)
            # Use a smooth sigmoid-like function for gradual transition
            if args.diff_signal_threshold > 0:
                # Normalize signal strength to [0, 1] range
                signal_norm = (signal_combined - signal_min) / (signal_max - signal_min + 1e-10)
                signal_norm = np.clip(signal_norm, 0, 1)
                
                # Apply threshold: regions below threshold get reduced weight
                threshold_val = args.diff_signal_threshold
                # Create weight: 1.0 for high signal, smoothly decreasing for low signal
                # Use a power function for smooth transition
                signal_weight = np.power(signal_norm, 1.0 / (threshold_val + 0.1))
                signal_weight = np.clip(signal_weight, 0.1, 1.0)  # Minimum 0.1 to keep some visibility
            else:
                # If threshold is 0, show everything with full intensity
                signal_weight = np.ones_like(signal_combined)
            
            # Store for use in plotting (we'll use this for alpha channel)
            D_weighted = D.copy().astype(float)
            signal_weight_map = signal_weight
            
            # Determine which plots to generate
            plot_square = args.plot_format in ["square", "both"]
            plot_triangle = args.plot_format in ["triangle", "both"]
            
            # SQUARE PLOTS
            if plot_square:
                has_genes = not sub_genes.empty
                n_rows = 2 if has_genes else 1
                fig_height = 5.6 if has_genes else 3.8
                
                # Use gridspec to ensure both rows have exactly the same width
                fig = plt.figure(figsize=(12, fig_height))
                if has_genes:
                    # Two rows: maps on top, genes on bottom
                    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[4, 1], 
                                          hspace=0.15, wspace=0.2, 
                                          left=0.08, right=0.95, top=0.92, bottom=0.1)
                    ax_ref = fig.add_subplot(gs[0, 0])
                    ax_alt = fig.add_subplot(gs[0, 1])
                    ax_diff = fig.add_subplot(gs[0, 2])
                else:
                    # Single row: just maps
                    gs = gridspec.GridSpec(1, 3, figure=fig, 
                                          wspace=0.2, 
                                          left=0.08, right=0.95, top=0.92, bottom=0.15)
                    ax_ref = fig.add_subplot(gs[0, 0])
                    ax_alt = fig.add_subplot(gs[0, 1])
                    ax_diff = fig.add_subplot(gs[0, 2])
                
                # Calculate extent for genomic coordinates
                extent = (x0, x1, x0, x1) if pos_bp else None
                
                im0 = ax_ref.imshow(M_ref, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                                    interpolation="none", extent=extent)
                # Explicitly set xlim and ylim to match extent for proper alignment
                if pos_bp:
                    ax_ref.set_xlim(x0, x1)
                    ax_ref.set_ylim(x0, x1)
                ax_ref.set_title("REF")
                if pos_bp:
                    ax_ref.set_xlabel(f"Genomic position (bp){magnitude_str}")
                    ax_ref.set_ylabel(f"Genomic position (bp){magnitude_str}")
                    ax_ref.set_xticks(tick_positions_bp)
                    ax_ref.set_xticklabels(tick_labels, rotation=45, ha="right")
                    ax_ref.set_yticks(tick_positions_bp)
                    ax_ref.set_yticklabels(tick_labels)
                    ax_ref.axvline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                    ax_ref.axhline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                
                im1 = ax_alt.imshow(M_alt, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                                    interpolation="none", extent=extent)
                # Explicitly set xlim and ylim to match extent for proper alignment
                if pos_bp:
                    ax_alt.set_xlim(x0, x1)
                    ax_alt.set_ylim(x0, x1)
                ax_alt.set_title("ALT")
                if pos_bp:
                    ax_alt.set_xlabel(f"Genomic position (bp){magnitude_str}")
                    ax_alt.set_ylabel(f"Genomic position (bp){magnitude_str}")
                    ax_alt.set_xticks(tick_positions_bp)
                    ax_alt.set_xticklabels(tick_labels, rotation=45, ha="right")
                    ax_alt.set_yticks(tick_positions_bp)
                    ax_alt.set_yticklabels(tick_labels)
                    ax_alt.axvline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                    ax_alt.axhline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                
                # Scale diff values by signal weight (reduces intensity of low-signal regions)
                # This keeps the map continuous while de-emphasizing low-signal regions
                D_scaled = D_weighted * signal_weight_map
                vmax = np.percentile(np.abs(D_scaled), 99.5)
                
                # Create a colormap-normalized version for the main plot
                im2 = ax_diff.imshow(D_scaled, cmap="RdBu_r", vmin=-vmax, vmax=vmax, 
                                     origin="lower", interpolation="none", extent=extent)
                # Explicitly set xlim and ylim to match extent for proper alignment
                if pos_bp:
                    ax_diff.set_xlim(x0, x1)
                    ax_diff.set_ylim(x0, x1)
                ax_diff.set_title("DIFF (ALT-REF)")
                if pos_bp:
                    ax_diff.set_xlabel(f"Genomic position (bp){magnitude_str}")
                    ax_diff.set_ylabel(f"Genomic position (bp){magnitude_str}")
                    ax_diff.set_xticks(tick_positions_bp)
                    ax_diff.set_xticklabels(tick_labels, rotation=45, ha="right")
                    ax_diff.set_yticks(tick_positions_bp)
                    ax_diff.set_yticklabels(tick_labels)
                    ax_diff.axvline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                    ax_diff.axhline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                
                # Gene tracks (bottom row) - share x-axis with map plots for alignment
                if has_genes:
                    gx_ref = fig.add_subplot(gs[1, 0], sharex=ax_ref)
                    gx_alt = fig.add_subplot(gs[1, 1], sharex=ax_alt)
                    gx_diff = fig.add_subplot(gs[1, 2], sharex=ax_diff)
                    
                    # sharex is already set when creating subplots, so limits are synchronized
                    # Don't set xlim in draw_gene_track when sharex is used (it's already synchronized)
                    draw_gene_track(gx_ref, sub_genes, x0, x1, pos_bp,
                                    promoter_kb=args.promoter_kb,
                                    gene_color=args.gene_color,
                                    promoter_color=args.promoter_color,
                                    alpha=args.gene_alpha, set_xlim=False)
                    draw_gene_track(gx_alt, sub_genes, x0, x1, pos_bp,
                                    promoter_kb=args.promoter_kb,
                                    gene_color=args.gene_color,
                                    promoter_color=args.promoter_color,
                                    alpha=args.gene_alpha, set_xlim=False)
                    draw_gene_track(gx_diff, sub_genes, x0, x1, pos_bp,
                                    promoter_kb=args.promoter_kb,
                                    gene_color=args.gene_color,
                                    promoter_color=args.promoter_color,
                                    alpha=args.gene_alpha, set_xlim=False)
                    
                    for gax in [gx_ref, gx_alt, gx_diff]:
                        gax.set_xlabel(f"Genomic position (bp){magnitude_str}")
                        gax.set_xticks(tick_positions_bp)
                        gax.set_xticklabels(tick_labels, rotation=45, ha="right")
                
                # Create colorbar - collect all axes
                all_axes = [ax_ref, ax_alt, ax_diff]
                if has_genes:
                    all_axes.extend([gx_ref, gx_alt, gx_diff])
                cbar = fig.colorbar(im2, ax=all_axes, shrink=0.8, pad=0.02)
                title = f"{snpID or core}  {chrom}:{pos}  REF={REF_base} ALT={ALT_base}"
                fig.suptitle(title, fontsize=11)
                safe = re.sub(r'[^A-Za-z0-9_.-]+','_', f"{snpID or chrom+'_'+pos}_{REF_base}_{ALT_base}")
                out_pdf = os.path.join(args.plots_dir, f"{safe}_square.pdf")
                fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
                plt.close(fig)
            
            # TRIANGLE PLOTS
            if plot_triangle:
                # IMPROVEMENT 2: Create proper triangle mask - mask the lower triangle completely
                # Create a mask for upper triangle (including diagonal offset)
                tri_mask = np.triu(np.ones((n_bins, n_bins), dtype=bool), k=diag_off)
                # Also mask the diagonal band
                for k in range(-diag_off, diag_off+1):
                    i = np.arange(n_bins)
                    j = i + k
                    m = (j>=0) & (j<n_bins)
                    tri_mask[i[m], j[m]] = False
                
                # Create masked arrays - lower triangle will be completely hidden
                M_ref_tri = np.ma.masked_array(M_ref, mask=~tri_mask)
                M_alt_tri = np.ma.masked_array(M_alt, mask=~tri_mask)
                D_tri_weighted = np.ma.masked_array(D_weighted, mask=~tri_mask)
                signal_weight_tri = np.ma.masked_array(signal_weight_map, mask=~tri_mask)
                
                has_genes = not sub_genes.empty
                n_rows = 2 if has_genes else 1
                fig_height = 5.6 if has_genes else 3.8
                
                # Use gridspec to ensure both rows have exactly the same width
                fig = plt.figure(figsize=(12, fig_height))
                if has_genes:
                    # Two rows: maps on top, genes on bottom
                    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[4, 1], 
                                          hspace=0.15, wspace=0.2, 
                                          left=0.08, right=0.95, top=0.92, bottom=0.1)
                    ax_ref = fig.add_subplot(gs[0, 0])
                    ax_alt = fig.add_subplot(gs[0, 1])
                    ax_diff = fig.add_subplot(gs[0, 2])
                else:
                    # Single row: just maps
                    gs = gridspec.GridSpec(1, 3, figure=fig, 
                                          wspace=0.2, 
                                          left=0.08, right=0.95, top=0.92, bottom=0.15)
                    ax_ref = fig.add_subplot(gs[0, 0])
                    ax_alt = fig.add_subplot(gs[0, 1])
                    ax_diff = fig.add_subplot(gs[0, 2])
                
                extent = (x0, x1, x0, x1) if pos_bp else None
                
                im0 = ax_ref.imshow(M_ref_tri, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                                    interpolation="none", extent=extent)
                # Explicitly set xlim and ylim to match extent for proper alignment
                if pos_bp:
                    ax_ref.set_xlim(x0, x1)
                    ax_ref.set_ylim(x0, x1)
                ax_ref.set_title("REF (triangle)")
                if pos_bp:
                    ax_ref.set_xlabel(f"Genomic position (bp){magnitude_str}")
                    ax_ref.set_ylabel(f"Genomic position (bp){magnitude_str}")
                    ax_ref.set_xticks(tick_positions_bp)
                    ax_ref.set_xticklabels(tick_labels, rotation=45, ha="right")
                    ax_ref.set_yticks(tick_positions_bp)
                    ax_ref.set_yticklabels(tick_labels)
                    ax_ref.axvline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                    ax_ref.axhline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                
                im1 = ax_alt.imshow(M_alt_tri, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                                    interpolation="none", extent=extent)
                # Explicitly set xlim and ylim to match extent for proper alignment
                if pos_bp:
                    ax_alt.set_xlim(x0, x1)
                    ax_alt.set_ylim(x0, x1)
                ax_alt.set_title("ALT (triangle)")
                if pos_bp:
                    ax_alt.set_xlabel(f"Genomic position (bp){magnitude_str}")
                    ax_alt.set_ylabel(f"Genomic position (bp){magnitude_str}")
                    ax_alt.set_xticks(tick_positions_bp)
                    ax_alt.set_xticklabels(tick_labels, rotation=45, ha="right")
                    ax_alt.set_yticks(tick_positions_bp)
                    ax_alt.set_yticklabels(tick_labels)
                    ax_alt.axvline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                    ax_alt.axhline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                
                # Scale diff values by signal weight for triangle plot
                D_tri_scaled = D_tri_weighted * signal_weight_tri
                
                # Convert masked array to regular for percentile calculation
                D_tri_for_percentile = np.abs(D_tri_scaled)
                if isinstance(D_tri_for_percentile, np.ma.MaskedArray):
                    D_tri_for_percentile = D_tri_for_percentile.filled(np.nan)
                vmax = np.nanpercentile(D_tri_for_percentile, 99.5)
                
                im2 = ax_diff.imshow(D_tri_scaled, cmap="RdBu_r", vmin=-vmax, vmax=vmax, 
                                     origin="lower", interpolation="none", extent=extent)
                # Explicitly set xlim and ylim to match extent for proper alignment
                if pos_bp:
                    ax_diff.set_xlim(x0, x1)
                    ax_diff.set_ylim(x0, x1)

                ax_diff.set_title("DIFF (ALT-REF, triangle)")
                if pos_bp:
                    ax_diff.set_xlabel(f"Genomic position (bp){magnitude_str}")
                    ax_diff.set_ylabel(f"Genomic position (bp){magnitude_str}")
                    ax_diff.set_xticks(tick_positions_bp)
                    ax_diff.set_xticklabels(tick_labels, rotation=45, ha="right")
                    ax_diff.set_yticks(tick_positions_bp)
                    ax_diff.set_yticklabels(tick_labels)
                    ax_diff.axvline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                    ax_diff.axhline(pos_bp, color="gray", ls="--", lw=0.8, alpha=0.8)
                
                # Gene tracks (bottom row) - share x-axis with map plots for alignment
                if has_genes:
                    gx_ref = fig.add_subplot(gs[1, 0], sharex=ax_ref)
                    gx_alt = fig.add_subplot(gs[1, 1], sharex=ax_alt)
                    gx_diff = fig.add_subplot(gs[1, 2], sharex=ax_diff)
                    
                    # sharex is already set when creating subplots, so limits are synchronized
                    # Don't set xlim in draw_gene_track when sharex is used (it's already synchronized)
                    draw_gene_track(gx_ref, sub_genes, x0, x1, pos_bp,
                                    promoter_kb=args.promoter_kb,
                                    gene_color=args.gene_color,
                                    promoter_color=args.promoter_color,
                                    alpha=args.gene_alpha, set_xlim=False)
                    draw_gene_track(gx_alt, sub_genes, x0, x1, pos_bp,
                                    promoter_kb=args.promoter_kb,
                                    gene_color=args.gene_color,
                                    promoter_color=args.promoter_color,
                                    alpha=args.gene_alpha, set_xlim=False)
                    draw_gene_track(gx_diff, sub_genes, x0, x1, pos_bp,
                                    promoter_kb=args.promoter_kb,
                                    gene_color=args.gene_color,
                                    promoter_color=args.promoter_color,
                                    alpha=args.gene_alpha, set_xlim=False)
                    
                    for gax in [gx_ref, gx_alt, gx_diff]:
                        gax.set_xlabel(f"Genomic position (bp){magnitude_str}")
                        gax.set_xticks(tick_positions_bp)
                        gax.set_xticklabels(tick_labels, rotation=45, ha="right")
                
                # Create colorbar - collect all axes
                all_axes = [ax_ref, ax_alt, ax_diff]
                if has_genes:
                    all_axes.extend([gx_ref, gx_alt, gx_diff])
                cbar = fig.colorbar(im2, ax=all_axes, shrink=0.8, pad=0.02)
                title = f"{snpID or core}  {chrom}:{pos}  REF={REF_base} ALT={ALT_base}"
                fig.suptitle(title, fontsize=11)
                safe = re.sub(r'[^A-Za-z0-9_.-]+','_', f"{snpID or chrom+'_'+pos}_{REF_base}_{ALT_base}")
                out_pdf = os.path.join(args.plots_dir, f"{safe}_triangle.pdf")
                fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
                plt.close(fig)

    # Save scores
    df = pd.DataFrame(rows)
    df.to_csv(args.out_scores, sep="\t", index=False)
    print(f"✅ Wrote scores: {args.out_scores}")
    if args.plots_dir:
        print(f"🖼  Plots in: {args.plots_dir}")

if __name__ == "__main__":
    main()
