#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akita_eval.py — evaluation figures for a joint multi-target Akita model (raw or smoothed metrics)

Prediction is done by akita_predict.py, which writes the npz this script reads
(and which handles the chromosome-disjoint folds, the model-head/observed-target
mapping and the leakage audit).

Two supported cases, same interface:

  1) held-out test chromosomes — one npz per eval set, one panel per cell type
  2) an independent dataset, all windows — one npz covering train+valid+test,
     with per-chromosome grouping so "unseen sample + unseen sequence"
     (chr10/chr17) is separable from "unseen sample, seen sequence"

Design notes
------------
* Regions come from the npz itself, so any number of splits and windows works.
* Metrics are unsmoothed; --sigma only affects displayed maps. Smoothing both
  maps before correlating inflates r.
* Observed targets and model heads are indexed separately (--pair), so the joint
  3-head model can be scored on a 1-target independent dataset.
* Correct diagonal_offset when rebuilding matrices, and crop-aware gene bands.
* The joint figure separates the pooled-pixel statistic from the median
  per-window statistic, and can colour points by chromosome or split.

Examples
--------
# 1) held-out test chromosomes, all three cell types
python akita_eval.py \
  --npz    $OUT/fold1_test_chroms_predictions.npz \
  --out-dir $OUT/figures --tag fold1_test \
  --group-by chrom --top-n 4 \
  --region chr17:45500000-46600000 \
  --gene-marks-json '[{"name":"KANSL1","start":46029916,"end":46225000}]'

# 2) independent P35, every window, matched against the Neuron head
python akita_eval.py \
  --npz    $OUT/iNeuroAkita_P35_indep_P35_predictions.npz \
  --out-dir $OUT/figures --tag P35_all \
  --pair Neuron_P35=Neuron_P16 \
  --group-by chrom --split-panels
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from akita_eval_utils import (
    OKABE_ITO, OKABE_ITO_DARK, OKABE_ITO_LIGHT,
    Bundle, colorize_gene_marks, load_predictions, per_window_metrics,
    plot_joint_kde, plot_region_map_gene, plot_top_n_maps, pooled_stats,
    resolve_pairs, self_test,
)


def _load_gene_marks(gene_marks_json=None, gene_marks_tsv=None, palette="OKABE_ITO"):
    marks = []
    if gene_marks_json:
        marks = json.loads(gene_marks_json)
        if not isinstance(marks, list):
            raise ValueError("--gene-marks-json must be a JSON list of objects")
    if gene_marks_tsv:
        df = pd.read_csv(gene_marks_tsv, sep="\t")
        need = {"name", "start", "end"}
        if not need.issubset(df.columns):
            raise ValueError(f"--gene-marks-tsv needs columns {need} (optional 'color')")
        for _, r in df.iterrows():
            marks.append(dict(name=str(r["name"]), start=int(r["start"]), end=int(r["end"]),
                              color=r["color"] if "color" in df.columns and pd.notna(r["color"]) else None))
    if not marks:
        return None
    pal = {"OKABE_ITO": OKABE_ITO[1:], "OKABE_ITO_LIGHT": OKABE_ITO_LIGHT,
           "OKABE_ITO_DARK": OKABE_ITO_DARK}.get(
        palette, [c.strip() for c in palette.split(",") if c.strip()])
    return colorize_gene_marks(marks, pal)


def _align_to_bundle(metrics, bundle):
    """A reused metrics TSV indexes heads in MODEL order (e.g. Neuron_P16 = 2), but the
    npz keeps only the saved heads (Neuron_P16 = 0 without --save_all_heads). Re-derive
    obs_index/head_index from the names so they match the npz; otherwise every
    (oi, hi) filter is empty and the PDF has no pages."""
    if {"obs_target", "model_head"} - set(metrics.columns):
        return metrics
    om = {n: i for i, n in enumerate(bundle.obs_names)}
    hm = {n: i for i, n in enumerate(bundle.head_names)}
    m = metrics.copy()
    m["obs_index"] = m["obs_target"].astype(str).map(om)
    m["head_index"] = m["model_head"].astype(str).map(hm)
    bad = m["obs_index"].isna() | m["head_index"].isna()
    if bad.any():
        print(f"⚠️  {int(bad.sum())} metric rows name a target/head not in the npz; dropped",
              file=sys.stderr)
    m = m[~bad].copy()
    m[["obs_index", "head_index"]] = m[["obs_index", "head_index"]].astype(int)
    return m


def _smoothed_ref(gref, metrics_pair):
    """Replace the unsmoothed pooled reference with the smoothed one written by
    add_smoothed_metrics.py, whatever container pooled_stats() returns."""
    if "pooled_r_smooth" not in metrics_pair or metrics_pair["pooled_r_smooth"].isna().all():
        print("⚠️  smoothed metrics without pooled_r_smooth; global reference lines dropped "
              "(re-run add_smoothed_metrics.py)", file=sys.stderr)
        return None
    mse = float(metrics_pair["pooled_mse_smooth"].iloc[0])
    r = float(metrics_pair["pooled_r_smooth"].iloc[0])
    if isinstance(gref, dict):
        new, hit = dict(gref), set()
        for k in gref:
            kl = str(k).lower()
            if "mse" in kl:
                new[k] = mse; hit.add("mse")
            elif kl in ("r", "corr", "pearson", "pearson_r", "pooled_r", "pooled_corr") or "corr" in kl:
                new[k] = r; hit.add("r")
        if hit == {"mse", "r"}:
            return new
    elif isinstance(gref, (tuple, list)) and len(gref) == 2:
        return type(gref)((mse, r))
    print(f"⚠️  could not map pooled reference {gref!r}; reference lines dropped", file=sys.stderr)
    return None


class _Relabel:
    """Wrap a PdfPages/PNG sink: rewrite 'unsmoothed' axis text to the real sigma."""
    def __init__(self, sink, sigma):
        self.sink, self.lab = sink, f"smoothed σ={sigma:g}"

    def __getattr__(self, name):          # close(), infodict(), ... pass through
        return getattr(self.sink, name)

    def savefig(self, fig=None, *args, **kw):
        if fig is None or not hasattr(fig, "axes"):
            import matplotlib.pyplot as plt
            fig = plt.gcf() if fig is None else plt.figure(fig)
        for ax in fig.axes:
            for t in [ax.xaxis.label, ax.yaxis.label, ax.title] + list(ax.texts):
                if "unsmoothed" in t.get_text():
                    t.set_text(t.get_text().replace("unsmoothed", self.lab))
        st = getattr(fig, "_suptitle", None)          # map pages: say which r/MSE is shown
        if st is not None and "MSE=" in st.get_text() and "metrics" not in st.get_text():
            st.set_text(st.get_text() + f" | metrics {self.lab}")
        return self.sink.savefig(fig, *args, **kw)


def _save_png(d, path, title, group_by, style, panel_label, common, sigma=None):
    """Render one joint plot straight to PNG (same code path as the PDF page)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    class _PngSink:  # plot_joint_kde only needs .savefig(fig, **kw)
        def savefig(self, fig, **kw):
            fig.savefig(path, dpi=200, **kw)

    sink = _PngSink() if sigma is None else _Relabel(_PngSink(), sigma)
    plot_joint_kde(d, out_pdf="", title=title, group_by=group_by, style=style,
                   panel_label=panel_label, pdf_handle=sink, **common)
    print(f"🖼️  {path}")


def _panel_title(args, bundle, oi, hi, n, chroms, n_pairs=1):
    """Classic-panel title, expanded per (target, head) pair."""
    if not args.panel_title:
        return f"{args.tag} — {bundle.obs_names[oi]}"
    fields = dict(target=bundle.obs_names[oi], head=bundle.head_names[hi],
                  tag=args.tag, n=n, chroms=chroms)
    try:
        title = args.panel_title.format(**fields)
    except (KeyError, IndexError, ValueError):
        title = args.panel_title
    if "{" not in args.panel_title and n_pairs > 1:
        title = f"{title} — {bundle.obs_names[oi]}"   # keep panels distinguishable
    return title


def build_argparser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--npz", help="prediction npz from akita_predict.py")
    p.add_argument("--data-dir", help="only needed for legacy npz without a region table")
    p.add_argument("--splits", nargs="*", default=None,
                   help="split order for a legacy npz, e.g. train valid test")
    p.add_argument("--out-dir")
    p.add_argument("--tag", default="akita_eval")
    p.add_argument("--metrics-tsv", help="reuse per-window metrics instead of recomputing")

    p.add_argument("--pair", action="append", default=None, metavar="OBS=HEAD",
                   help="observed target = model head, by name or index; "
                        "required when the counts differ")
    p.add_argument("--chroms", nargs="*", default=None, help="restrict to these chromosomes")
    p.add_argument("--only-splits", nargs="*", default=None, help="restrict to these window splits")

    p.add_argument("--group-by", choices=["none", "chrom", "split"], default="none",
                   help="colour the joint plot by chromosome or window split")
    p.add_argument("--split-panels", action="store_true",
                   help="also emit one joint plot per chromosome")
    p.add_argument("--color-mode", choices=["kde_scatter", "kde_contour_scatter", "plain_scatter"],
                   default="kde_scatter")
    p.add_argument("--style", choices=["both", "classic", "detailed"], default="both",
                   help="'classic' = density-coloured points in the original style; "
                        "'detailed' = grouped/annotated version; 'both' emits each")
    p.add_argument("--panel-title", default=None,
                   help="title for the classic panel. Supports {target} {head} {tag} {n} "
                        "{chroms}; without a placeholder the target name is appended when "
                        "there is more than one pair")
    p.add_argument("--separate-pdfs", action="store_true",
                   help="one PDF per observed target instead of one multi-page PDF")
    p.add_argument("--only-targets", nargs="*", default=None,
                   help="restrict to these observed target names")
    p.add_argument("--panel-label", default=None, help="bold corner letter, e.g. 'a'")
    p.add_argument("--hist-bins", type=int, default=40)
    p.add_argument("--hist-alpha", type=float, default=1.0,
                   help="fill opacity of the marginal histograms (0-1); "
                        "values <1 also apply to the grouped panels")
    p.add_argument("--hist-edgecolor", default="k", help="bar outline colour; 'none' to drop it")
    p.add_argument("--hist-edgewidth", type=float, default=0.4)
    p.add_argument("--kde-line-alpha", type=float, default=1.0,
                   help="opacity of the KDE curve drawn over each histogram")
    p.add_argument("--ymin", type=float, default=-0.4)
    p.add_argument("--ymax", type=float, default=1.0)
    p.add_argument("--point-size", type=int, default=12)
    p.add_argument("--alpha-pts", type=float, default=0.85)
    p.add_argument("--hist-color-mse", default="steelblue")
    p.add_argument("--hist-color-r", default="salmon")

    p.add_argument("--top-n", type=int, default=4)
    p.add_argument("--map-sets", nargs="*", default=["best", "median", "worst"],
                   choices=["best", "median", "worst"])
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Gaussian sigma for DISPLAYED maps only; metrics stay unsmoothed")
    p.add_argument("--vmin", type=float, default=-2.0)
    p.add_argument("--vmax", type=float, default=2.0)
    p.add_argument("--mirrored", action="store_true", default=True,
                   help="predicted above / observed below the diagonal in one panel")
    p.add_argument("--side-by-side", dest="mirrored", action="store_false")

    p.add_argument("--region", action="append", default=[],
                   help="locus 'chr:start-end' (all overlapping windows) or an exact window label")
    p.add_argument("--gene-marks-json", default=None)
    p.add_argument("--gene-marks-tsv", default=None)
    p.add_argument("--gene-palette", default="OKABE_ITO")
    p.add_argument("--band-color", default="#111111")
    p.add_argument("--band-alpha", type=float, default=0.22)
    p.add_argument("--label-pos", choices=["top", "bottom"], default="bottom")
    p.add_argument("--label-offset", type=float, default=0.04)
    p.add_argument("--bottom-adjust", type=float, default=0.22)
    p.add_argument("--png-dir", default=None,
                   help="also write the joint plots as PNG here (for slides/drafts)")
    p.add_argument("--crop-bp", type=int, default=None,
                   help="override the inferred crop; normally derived from L*bin_size")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.self_test:
        return self_test()
    for req in ("npz", "out_dir"):
        if getattr(args, req) is None:
            print(f"❌ --{req.replace('_','-')} is required", file=sys.stderr)
            return 2

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, args.tag)

    bundle = load_predictions(args.npz, data_dir=args.data_dir, splits=args.splits)
    print(f"📦 {bundle.preds.shape[0]} windows, L={bundle.L}, "
          f"diagonal_offset={bundle.diagonal_offset}, bin_size={bundle.bin_size}")
    print(f"   observed targets: {bundle.obs_names}")
    print(f"   model heads     : {bundle.head_names}")
    print(f"   chromosomes     : {', '.join(sorted(bundle.regions['chrom'].unique()))}")
    if "split" in bundle.regions.columns:
        counts = bundle.regions["split"].value_counts().to_dict()
        print(f"   window splits   : {counts}")

    try:
        pairs = resolve_pairs(bundle, args.pair)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 2
    for oi, hi in pairs:
        print(f"   comparing observed [{oi}] {bundle.obs_names[oi]} "
              f"vs model head [{hi}] {bundle.head_names[hi]}")

    # ---- per-window metrics (unsmoothed) ------------------------------------
    if args.metrics_tsv and not os.path.exists(args.metrics_tsv):
        print(f"❌ --metrics-tsv not found: {args.metrics_tsv}\n"
              f"   (refusing to fall back to recomputed UNSMOOTHED metrics)", file=sys.stderr)
        return 2
    if args.metrics_tsv:
        metrics = pd.read_csv(args.metrics_tsv, sep="\t", dtype={"chrom": str})
        print(f"📄 reusing metrics: {args.metrics_tsv}")
        metrics = _align_to_bundle(metrics, bundle)
        if metrics.empty:
            print("❌ no row of --metrics-tsv matches the npz targets/heads "
                  f"(npz: {bundle.obs_names} / {bundle.head_names})", file=sys.stderr)
            return 2
    else:
        metrics = per_window_metrics(bundle, pairs)
        metrics.to_csv(f"{prefix}_per_window_metrics.tsv", sep="\t", index=False)
        print(f"✅ {prefix}_per_window_metrics.tsv")

    if args.chroms:
        metrics = metrics[metrics["chrom"].isin(args.chroms)]
    if args.only_splits and "split" in metrics.columns:
        metrics = metrics[metrics["split"].isin(args.only_splits)]
    if metrics.empty:
        print("❌ no windows left after filtering", file=sys.stderr)
        return 2

    # ---- summary -------------------------------------------------------------
    summ = (metrics.groupby(["chrom", "obs_target", "model_head"])
            .agg(n=("pearson", "size"), median_r=("pearson", "median"),
                 mean_r=("pearson", "mean"), sd_r=("pearson", "std"),
                 median_mse=("mse", "median"), mean_mse=("mse", "mean"))
            .reset_index())
    summ.to_csv(f"{prefix}_summary_by_chromosome.csv", index=False)
    print("\n📊 median per-window Pearson r")
    print(summ.pivot_table(index="chrom", columns="obs_target",
                           values="median_r").round(4).to_string())

    # ---- figures -------------------------------------------------------------
    gene_marks = _load_gene_marks(args.gene_marks_json, args.gene_marks_tsv, args.gene_palette)
    pdf_path = f"{prefix}_evaluation.pdf"
    group_by = None if args.group_by == "none" else args.group_by

    if args.only_targets:
        keep = {t.lower() for t in args.only_targets}
        pairs = [(o, h) for o, h in pairs if bundle.obs_names[o].lower() in keep]
        if not pairs:
            print(f"❌ --only-targets matched nothing; available: {bundle.obs_names}",
                  file=sys.stderr)
            return 2

    shared_pdf = None if args.separate_pdfs else PdfPages(pdf_path)
    written = []
    n_plotted = 0
    try:
        for oi, hi in pairs:
            d = metrics[(metrics["obs_index"] == oi) & (metrics["head_index"] == hi)]
            if d.empty:
                print(f"⚠️  no metric rows for {bundle.obs_names[oi]} vs {bundle.head_names[hi]}; "
                      f"skipped", file=sys.stderr)
                continue
            n_plotted += 1
            if args.separate_pdfs:
                one_pdf = f"{prefix}_{bundle.obs_names[oi]}_vs_{bundle.head_names[hi]}.pdf"
                pdf = PdfPages(one_pdf)
                written.append(one_pdf)
            else:
                pdf = shared_pdf
            label = f"{bundle.obs_names[oi]}  (model head {bundle.head_names[hi]})"
            gref = pooled_stats(bundle, oi, hi)
            sm_sigma = None
            if "smooth_sigma" in d.columns and d["smooth_sigma"].notna().any():
                sm_sigma = float(d["smooth_sigma"].dropna().iloc[0])
                gref = _smoothed_ref(gref, d)
                pdf = _Relabel(pdf, sm_sigma)
                print(f"   {bundle.obs_names[oi]}: smoothed metrics (σ={sm_sigma:g}); "
                      f"pooled reference = {gref}")
            chroms = ", ".join(sorted(d["chrom"].unique()))
            common = dict(global_ref=gref, y_min=args.ymin, y_max=args.ymax,
                          color_mode=args.color_mode, point_size=args.point_size,
                          alpha_pts=args.alpha_pts, hist_color_mse=args.hist_color_mse,
                          hist_color_r=args.hist_color_r, hist_bins=args.hist_bins,
                          hist_alpha=args.hist_alpha, hist_edgecolor=args.hist_edgecolor,
                          hist_edgewidth=args.hist_edgewidth,
                          kde_line_alpha=args.kde_line_alpha)

            # classic: no chromosome colouring, density-coloured points, original style
            if args.style in ("both", "classic"):
                classic_title = _panel_title(args, bundle, oi, hi, len(d), chroms,
                                             n_pairs=len(pairs))
                plot_joint_kde(d, out_pdf="", title=classic_title, group_by=None,
                               style="classic", panel_label=args.panel_label,
                               pdf_handle=pdf, **common)
                if args.png_dir:
                    _save_png(d, os.path.join(args.png_dir,
                              f"{args.tag}_{bundle.obs_names[oi]}_classic.png"),
                              classic_title, None, "classic", args.panel_label, common, sm_sigma)

            # detailed: grouped by chromosome/split with median + pooled reference lines
            if args.style in ("both", "detailed"):
                plot_joint_kde(d, out_pdf="",
                               title=f"{args.tag} — {label}\n{len(d)} windows on {chroms}",
                               group_by=group_by, style="detailed", pdf_handle=pdf, **common)
                if args.png_dir:
                    _save_png(d, os.path.join(args.png_dir,
                              f"{args.tag}_{bundle.obs_names[oi]}_detailed.png"),
                              f"{args.tag} — {label}", group_by, "detailed", None, common, sm_sigma)
            if args.split_panels:
                for chrom in sorted(d["chrom"].unique()):
                    dc = d[d["chrom"] == chrom]
                    if len(dc) < 5:
                        continue
                    plot_joint_kde(
                        dc, out_pdf="", title=f"{args.tag} — {label} — {chrom}",
                        group_by=None, global_ref=None, y_min=args.ymin, y_max=args.ymax,
                        color_mode=args.color_mode, point_size=args.point_size,
                        alpha_pts=args.alpha_pts, hist_color_mse=args.hist_color_mse,
                        hist_color_r=args.hist_color_r, pdf_handle=pdf,
                    )
            for which in args.map_sets:
                plot_top_n_maps(bundle, metrics, oi, hi, out_pdf="", which=which,
                                top_n=args.top_n, sigma=args.sigma, vmin=args.vmin,
                                vmax=args.vmax, mirrored=args.mirrored, pdf_handle=pdf)
            for reg in args.region:
                try:
                    plot_region_map_gene(
                        bundle, metrics, reg, oi, hi, out_pdf="", sigma=args.sigma,
                        vmin=args.vmin, vmax=args.vmax, gene_marks=gene_marks,
                        band_color=args.band_color, band_alpha=args.band_alpha,
                        label_pos=args.label_pos, label_offset=args.label_offset,
                        bottom_adjust=args.bottom_adjust, crop_bp=args.crop_bp,
                        mirrored=args.mirrored, pdf_handle=pdf,
                    )
                except ValueError as e:
                    print(f"[warn] {reg}: {e}", file=sys.stderr)
            if args.separate_pdfs:
                pdf.close()
                print(f"✅ {one_pdf}")
    finally:
        if shared_pdf is not None:
            shared_pdf.close()

    if n_plotted == 0:
        if not args.separate_pdfs and os.path.exists(pdf_path):
            os.remove(pdf_path)          # an empty PdfPages file cannot be opened
        print("❌ nothing was plotted (see warnings above); no PDF written", file=sys.stderr)
        return 2
    if not args.separate_pdfs:
        print(f"\n✅ {pdf_path}")
    print(f"✅ {prefix}_summary_by_chromosome.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
