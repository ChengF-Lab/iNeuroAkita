#!/usr/bin/env python3
"""
compare_to_akita0.py — iNeuroAkita vs Akita zero-shot on held-out test chromosomes.

One iNeuroAkita configuration per run (scratch or Akita-init, merged or single-head);
run it again with another --label/--out for each configuration.

Outputs (prefix = --out)
  {out}_by_fold.pdf/.png        1) x = cell type; bar (or box) = mean over the 7 folds,
                                   dots = folds, grey lines pair the same fold,
                                   label = paired test across folds (p + stars)
  {out}_by_chrom_{cell}.pdf/.png 2) one figure per cell type; x = chromosome;
                                   boxes = per-window r of each model, paired by window,
                                   label = BH-adjusted paired Wilcoxon per chromosome
  {out}_per_window.csv          3) every window, both models, side by side (source data)
  {out}_per_fold.csv               per fold x cell type x model: n windows, mean/median r
  {out}_fold_stats.csv             paired tests across folds per cell type
  {out}_summary_by_celltype.csv    mean, SD, SEM over folds + pooled out-of-fold medians
  {out}_per_chrom.csv              per chromosome: n, medians, Δ, p, BH q

Inputs — where the per-window metrics live
------------------------------------------
--model_root SPEC (repeatable; all SPECs are pooled under one --label). SPEC is one of:
  ROOT                  ROOT/<fold>/<--pattern>
                        (--pattern default: predictOut_v3/hold_test/*_test_per_window_metrics.tsv)
  ROOT::PATTERN         same, but PATTERN for this ROOT only
                        e.g. /x/single_NPC::predictOut/hold_test/*_test_per_window_metrics.tsv
  TEMPLATE with {fold}  any layout: {fold} marks the fold folder name
                        e.g. "/x/runs/{fold}/eval/NPC/*_per_window_metrics.tsv"
                        e.g. "/x/metrics/NPC_{fold}_test_per_window_metrics.tsv"
  Fold names must match --fold_regex (default fold\\d+).
  Merged model: one SPEC. Single-head models: one SPEC per cell type, same --label.

Akita zero-shot (defaults follow akita0_select_heads.py):
  --zs_root ZS
  --akita0_template   default "{zs_root}/{fold}/test/akita0_{fold}_test_per_window_metrics.tsv"
  --akita0_heads      default "{zs_root}/summary/akita0_selected_heads[_mean|_worst].csv"
  --akita_select best|mean|worst  (best = the fair baseline)

Check what will be read before plotting:  add --list_inputs  (prints files per fold, exits)

Examples
  # merged, Akita-init
  python compare_to_akita0.py \
    --model_root $base/model_trainOut/NPC_Mic_Neuron_trunk_out \
    --label "iNeuroAkita (Akita-init, merged)" \
    --zs_root $base/model_trainOut/akita_zeroshot --out $FIG/cmp_trunk_merged
  # single-head: three roots, all laid out foldK/predictOut/hold_test/
  python compare_to_akita0.py \
    --model_root $S/NPC_single --model_root $S/Neuron_single --model_root $S/Mic_single \
    --pattern "predictOut/hold_test/*_test_per_window_metrics.tsv" \
    --label "iNeuroAkita (Akita-init, single)" \
    --zs_root $base/model_trainOut/akita_zeroshot --out $FIG/cmp_trunk_single
  python compare_to_akita0.py --self_test
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

BASE_LABEL = "Akita (zero-shot)"
CELL_ORDER = ["NPC_P16", "Neuron_P16", "Mic_AD"]
CELL_LABEL = {"NPC_P16": "NPC", "Neuron_P16": "Neuron", "Mic_AD": "Microglia"}
# default look (all overridable on the command line); blue/red pair passes the CVD check
STYLE = dict(base_color="#2a78d6", model_color="#e34948", fill_alpha=0.30,
             edge_color="#000000", dot_alpha=0.45, show_mean=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KEY = ["fold", "obs_target", "chrom", "start", "end"]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _natkey(s):
    """natural sort key (chr2 < chr10 < chrX), hashable for pandas sort keys."""
    s = str(s)
    m = re.fullmatch(r"(?:chr)?(\d+)", s)
    if m:
        return (0, int(m.group(1)), "")
    m = re.fullmatch(r"(\D*?)(\d+)", s)
    if m:
        return (1, int(m.group(2)), m.group(1))
    return (2, 0, s)


def stars(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.05 else f"p = {p:.2f}"


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]; n = len(pv); order = np.argsort(pv)
        ranked = pv[order] * n / np.arange(1, n + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n); out[order] = np.minimum(ranked, 1.0); q[ok] = out
    return q


# ============================================================== input resolution
def spec_to_template(spec, default_pattern):
    """ROOT | ROOT::PATTERN | TEMPLATE-with-{fold}  ->  TEMPLATE-with-{fold}."""
    if "{fold}" in spec:
        return spec
    if "::" in spec:
        root, pat = spec.split("::", 1)
    else:
        root, pat = spec, default_pattern
    return os.path.join(root, "{fold}", pat)


def resolve_template(template, fold_regex=r"fold\d+"):
    """{fold: [files, oldest..newest]} for a template containing {fold} (+ glob wildcards)."""
    template = os.path.expanduser(template)
    rx = re.escape(template)
    # first {fold} captures, any later {fold} must repeat the same name
    rx = rx.replace(r"\{fold\}", f"(?P<fold>{fold_regex})", 1).replace(r"\{fold\}", "(?P=fold)")
    rx = rx.replace(r"\*", r"[^/]*").replace(r"\?", r"[^/]")
    rx = re.compile("^" + rx + "$")
    found = {}
    for path in glob.glob(template.replace("{fold}", "*")):
        m = rx.match(path)
        if m:
            found.setdefault(m.group("fold"), []).append(path)
    return {f: sorted(v, key=os.path.getmtime) for f, v in
            sorted(found.items(), key=lambda kv: _natkey(kv[0]))}


def list_inputs(templates, fold_regex, what):
    print(f"\n📂 {what}")
    for t in templates:
        res = resolve_template(t, fold_regex)
        print(f"   {t}")
        if not res:
            print("      (nothing found)")
        for f, files in res.items():
            flag = f"   [{len(files)} files, newest used]" if len(files) > 1 else ""
            print(f"      {f:>7}: {files[-1]}{flag}")


# ============================================================== loading
def _keep_matched(m, path):
    """Keep rows where each observed target is scored by its own head. A single-head
    file whose head name differs from its target name (e.g. head 'NPC' vs target
    'NPC_P16') is kept as-is, since there is nothing else it could be paired with."""
    same = m["obs_target"].map(_norm) == m["model_head"].map(_norm)
    if same.any():
        return m[same]
    if m["obs_target"].nunique() == 1 and m["model_head"].nunique() == 1:
        print(f"ℹ️  {os.path.basename(path)}: single-head file, target "
              f"{m['obs_target'].iloc[0]!r} scored by head {m['model_head'].iloc[0]!r}",
              file=sys.stderr)
        return m
    print(f"⚠️  {path}: no row pairs a target with a same-named head; skipped", file=sys.stderr)
    return m.iloc[0:0]


def load_model(specs, pattern, metric, fold_regex=r"fold\d+"):
    rows = []
    for spec in specs:
        tmpl = spec_to_template(spec, pattern)
        res = resolve_template(tmpl, fold_regex)
        if not res:
            print(f"⚠️  nothing found for {tmpl}", file=sys.stderr)
            continue
        for fold, files in res.items():
            if len(files) > 1:
                print(f"⚠️  {fold}: {len(files)} files match {tmpl}; using newest "
                      f"{os.path.basename(files[-1])}", file=sys.stderr)
            m = pd.read_csv(files[-1], sep="\t", dtype={"chrom": str})
            m = _keep_matched(m, files[-1]).copy()
            m["fold"] = fold
            rows.append(m[["fold", "obs_target", "chrom", "start", "end", metric]])
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    dup = d.duplicated(KEY, keep=False)
    if dup.any():
        print(f"⚠️  {dup.sum()} duplicated model windows (same cell type in two roots?); "
              f"keeping the first", file=sys.stderr)
        d = d.drop_duplicates(KEY)
    return d


AKITA0_TEMPLATE = "{zs_root}/{fold}/test/akita0_{fold}_test_per_window_metrics.tsv"


def akita0_heads_path(zs_root, select, override=None):
    if override:
        return override.format(zs_root=zs_root)
    sfx = "" if select == "best" else f"_{select}"
    return os.path.join(zs_root, "summary", f"akita0_selected_heads{sfx}.csv")


def load_akita0(zs_root, select, metric, template=AKITA0_TEMPLATE, heads_path=None,
                fold_regex=r"fold\d+"):
    sel_path = akita0_heads_path(zs_root, select, heads_path)
    if not os.path.exists(sel_path):
        raise FileNotFoundError(f"{sel_path} — run akita0_select_heads.py "
                                f"{'' if select == 'best' else '--select ' + select}")
    sel = pd.read_csv(sel_path)
    files = resolve_template(template.replace("{zs_root}", zs_root), fold_regex)
    rows = []
    for fold in sorted(sel["fold"].unique(), key=_natkey):
        if fold not in files:
            print(f"⚠️  no Akita zero-shot metrics for {fold} "
                  f"({template.replace('{zs_root}', zs_root)})", file=sys.stderr)
            continue
        p = files[fold][-1]
        m = pd.read_csv(p, sep="\t", dtype={"chrom": str})
        m["fold"] = fold
        if select == "mean":                          # average over the 5 heads per window
            m = (m.groupby(["fold", "obs_target", "chrom", "start", "end"])[metric]
                   .mean().reset_index())
        else:
            s = sel[sel["fold"] == fold][["obs_target", "model_head"]]
            m = m.merge(s, on=["obs_target", "model_head"], how="inner")
        rows.append(m[["fold", "obs_target", "chrom", "start", "end", metric]])
    if not rows:
        raise RuntimeError("no Akita zero-shot test metrics found")
    return pd.concat(rows, ignore_index=True), sel


# ============================================================== analysis
def fold_level(pw, metric, label, fold_stat):
    out = []
    for mdl, col in ((label, f"{metric}_model"), (BASE_LABEL, f"{metric}_base")):
        g = pw.groupby(["obs_target", "fold"])[col]
        t = g.agg(n_windows="size", mean="mean", median="median").reset_index()
        t["model"] = mdl
        t["value"] = t[fold_stat]
        out.append(t)
    return pd.concat(out, ignore_index=True)


def fold_tests(pf, label, lower_is_better=False):
    from scipy import stats
    rows = []
    for ct in [c for c in CELL_ORDER if c in set(pf["obs_target"])] + \
              sorted(set(pf["obs_target"]) - set(CELL_ORDER)):
        a = pf[(pf["obs_target"] == ct) & (pf["model"] == label)].set_index("fold")["value"]
        b = pf[(pf["obs_target"] == ct) & (pf["model"] == BASE_LABEL)].set_index("fold")["value"]
        folds = sorted(set(a.index) & set(b.index), key=_natkey)
        if len(folds) < 2:
            continue
        d = (a.loc[folds] - b.loc[folds]).to_numpy()
        n = len(d)
        half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        p_t = stats.ttest_rel(a.loc[folds], b.loc[folds]).pvalue
        try:
            p_w = stats.wilcoxon(d).pvalue
        except ValueError:
            p_w = 1.0
        better = (d < 0) if lower_is_better else (d > 0)
        rows.append(dict(obs_target=ct, cell_type=CELL_LABEL.get(ct, ct), n_folds=n,
                         mean_model=a.loc[folds].mean(), sd_model=a.loc[folds].std(ddof=1),
                         mean_baseline=b.loc[folds].mean(), sd_baseline=b.loc[folds].std(ddof=1),
                         delta=d.mean(), ci95_low=d.mean() - half, ci95_high=d.mean() + half,
                         folds_better=int(better.sum()),
                         p_paired_t=p_t, p_wilcoxon=p_w))
    return pd.DataFrame(rows)


def chrom_tests(pw, metric):
    from scipy import stats
    rows = []
    for (ct, chrom), g in pw.groupby(["obs_target", "chrom"]):
        a, b = g[f"{metric}_model"].to_numpy(), g[f"{metric}_base"].to_numpy()
        d = a - b
        try:
            p = stats.wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else np.nan
        except ValueError:
            p = np.nan
        rows.append(dict(obs_target=ct, cell_type=CELL_LABEL.get(ct, ct), chrom=chrom,
                         fold=",".join(sorted(g["fold"].unique(), key=_natkey)),
                         n_windows=len(g), median_model=np.median(a), median_baseline=np.median(b),
                         median_delta=np.median(d), frac_windows_model_higher=float(np.mean(d > 0)),
                         p_wilcoxon=p))
    t = pd.DataFrame(rows)
    t["q_bh"] = np.nan
    for ct, idx in t.groupby("obs_target").groups.items():
        t.loc[idx, "q_bh"] = bh(t.loc[idx, "p_wilcoxon"])
    t["label"] = t["q_bh"].map(stars)
    return t.sort_values(["obs_target", "chrom"], key=lambda s: s.map(_natkey) if s.name == "chrom" else s)


# ============================================================== plotting
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
    })
    return plt


def _rgba(color, alpha):
    from matplotlib.colors import to_rgba
    return to_rgba(color, alpha)


def _legend(ax, label, extra=None, **kw):
    from matplotlib.patches import Patch
    S = STYLE
    h = [Patch(facecolor=_rgba(S["base_color"], S["fill_alpha"]), edgecolor=S["edge_color"],
               lw=0.6, label=BASE_LABEL),
         Patch(facecolor=_rgba(S["model_color"], S["fill_alpha"]), edgecolor=S["edge_color"],
               lw=0.6, label=label)]
    if extra:
        h += extra
    ax.legend(handles=h, frameon=False, **kw)


def plot_by_fold(pf, st, label, ylabel, out, style="bar", test="ttest", size=(3.4, 2.9)):
    plt = _style()
    from matplotlib.lines import Line2D
    cells = [c for c in CELL_ORDER if c in set(pf["obs_target"])] + \
            sorted(set(pf["obs_target"]) - set(CELL_ORDER))
    S = STYLE
    models = [BASE_LABEL, label]
    colors = {label: S["model_color"], BASE_LABEL: S["base_color"]}
    offs = {BASE_LABEL: -0.19, label: 0.19}
    w = 0.32
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=size)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    top_all = pf["value"].max()
    lo = min(0, pf["value"].min() - 0.05)
    for gi, ct in enumerate(cells):
        sub = pf[pf["obs_target"] == ct]
        folds = sorted(sub["fold"].unique(), key=_natkey)
        jit = {f: rng.uniform(-0.07, 0.07) for f in folds}
        for mdl in models:
            v = sub[sub["model"] == mdl].set_index("fold")["value"]
            x = gi + offs[mdl]
            fill = _rgba(colors[mdl], S["fill_alpha"])
            if style == "bar":
                ax.bar(x, v.mean(), width=w, color=fill, edgecolor=S["edge_color"],
                       linewidth=0.7, zorder=2)
            else:
                ax.boxplot([v.values], positions=[x], widths=w, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=fill, edgecolor=S["edge_color"], lw=0.7),
                           medianprops=dict(color=S["edge_color"], lw=1.0),
                           whiskerprops=dict(color=S["edge_color"], lw=0.6),
                           capprops=dict(color=S["edge_color"], lw=0.6), zorder=2)
            # dots semi-transparent and BELOW the error bar so the bar stays readable
            ax.scatter([x + jit[f] for f in v.index], v.values, s=11,
                       color=_rgba(colors[mdl], S["dot_alpha"]), edgecolor="none", zorder=4)
            if style == "bar":
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="none", ecolor=S["edge_color"],
                            elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=6)
            if S["show_mean"]:
                ax.text(x, lo + 0.015 * (top_all - lo), f"{v.mean():.3f}", ha="center",
                        va="bottom", rotation=90, fontsize=5.6, color=INK, zorder=7)
        for f in folds:   # paired lines
            a = sub[(sub["model"] == BASE_LABEL) & (sub["fold"] == f)]["value"]
            b = sub[(sub["model"] == label) & (sub["fold"] == f)]["value"]
            if len(a) and len(b):
                ax.plot([gi + offs[BASE_LABEL] + jit[f], gi + offs[label] + jit[f]],
                        [a.iloc[0], b.iloc[0]], color="#9a9994", lw=0.4, alpha=0.6, zorder=3)
        s = st[st["obs_target"] == ct]
        if len(s):
            s = s.iloc[0]
            p = s["p_paired_t"] if test == "ttest" else s["p_wilcoxon"]
            y = sub["value"].max() + 0.04 * max(top_all, 0.1)
            x0, x1 = gi + offs[BASE_LABEL], gi + offs[label]
            ax.plot([x0, x0, x1, x1], [y - 0.01, y, y, y - 0.01], color=INK2, lw=0.6)
            ax.text(gi, y + 0.005, f"{stars(p)}\n{fmt_p(p)}", ha="center", va="bottom",
                    fontsize=6, color=INK2, linespacing=1.0)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([CELL_LABEL.get(c, c) for c in cells])
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylim(lo, top_all * 1.25 + 0.02)
    ax.set_ylabel(ylabel)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    summ = "bar = mean ± SD over folds" if style == "bar" else "box = folds"
    tname = "paired t-test" if test == "ttest" else "Wilcoxon signed-rank"
    note = f"{summ}; number = mean" if S["show_mean"] else summ
    extra = [Line2D([], [], marker="o", ls="", ms=3.5, mfc=_rgba(INK2, S["dot_alpha"]),
                    mec="none", label="fold")]
    _legend(ax, label, extra, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1,
            handlelength=1.2)
    ax.text(0.5, -0.3, f"{note}\n{tname} across folds", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.8, color=INK2)
    # fixed axes box: the legend below must not squeeze the plot
    fig.subplots_adjust(left=0.2, right=0.97, top=0.96, bottom=0.12)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}_by_fold.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  {out}_by_fold.pdf / .png")


def plot_by_chrom(pw, ct_tab, metric, label, ylabel, out):
    plt = _style()
    written = []
    for ct in [c for c in CELL_ORDER if c in set(pw["obs_target"])] + \
              sorted(set(pw["obs_target"]) - set(CELL_ORDER)):
        sub = pw[pw["obs_target"] == ct]
        chroms = sorted(sub["chrom"].unique(), key=_natkey)
        tab = ct_tab[ct_tab["obs_target"] == ct].set_index("chrom")
        fig, ax = plt.subplots(figsize=(max(4.0, 0.3 * len(chroms) + 1.2), 2.5))
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        w = 0.36
        S = STYLE
        for col, color, off in ((f"{metric}_base", S["base_color"], -0.2),
                                (f"{metric}_model", S["model_color"], 0.2)):
            data = [sub[sub["chrom"] == c][col].dropna().values for c in chroms]
            ax.boxplot(data, positions=np.arange(len(chroms)) + off, widths=w, patch_artist=True,
                       showfliers=True,
                       flierprops=dict(marker="o", ms=1.5, mfc=_rgba(color, S["dot_alpha"]), mec="none"),
                       boxprops=dict(facecolor=_rgba(color, S["fill_alpha"]),
                                     edgecolor=S["edge_color"], lw=0.6),
                       medianprops=dict(color=S["edge_color"], lw=0.9),
                       whiskerprops=dict(color=S["edge_color"], lw=0.5),
                       capprops=dict(color=S["edge_color"], lw=0.5))
        ymax = np.nanmax(sub[[f"{metric}_base", f"{metric}_model"]].to_numpy())
        ymin = np.nanmin(sub[[f"{metric}_base", f"{metric}_model"]].to_numpy())
        span = ymax - ymin
        for i, c in enumerate(chroms):
            if c in tab.index:
                ax.text(i, ymax + 0.04 * span, tab.loc[c, "label"], ha="center", va="bottom",
                        fontsize=5.5, color=INK2)
        ax.set_xticks(range(len(chroms)))
        ax.set_xticklabels([f"{c.replace('chr', '')}\n{int(tab.loc[c, 'n_windows']) if c in tab.index else ''}"
                            for c in chroms], linespacing=1.3)
        ax.text(-0.012, -0.175, "n windows", transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color=INK2)
        ax.set_xlabel("Chromosome (held-out; each scored by the fold that held it out)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(ymin - 0.06 * span, ymax + 0.16 * span)
        ax.set_xlim(-0.6, len(chroms) - 0.4)
        ax.set_title(CELL_LABEL.get(ct, ct), fontsize=8, color=INK, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
        _legend(ax, label, loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=2, handlelength=1.0)
        ax.text(0.0, -0.43, "per-window paired Wilcoxon, BH-adjusted within cell type; "
                "windows are spatially correlated, read as descriptive",
                transform=ax.transAxes, fontsize=5.2, color=INK2, va="top")
        fig.tight_layout()
        base = f"{out}_by_chrom_{CELL_LABEL.get(ct, ct)}"
        for ext in ("pdf", "png"):
            fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(base)
        print(f"🖼️  {base}.pdf / .png")
    return written


# ============================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("iNeuroAkita input")
    g.add_argument("--model_root", "--model_input", action="append", default=[], metavar="SPEC",
                   help="ROOT | ROOT::PATTERN | template with {fold}; repeat for single-head models")
    g.add_argument("--label", default="iNeuroAkita")
    g.add_argument("--pattern", default="predictOut_v3/hold_test/*_test_per_window_metrics.tsv",
                   help="metrics path inside each fold folder, for SPECs given as a bare ROOT")
    g.add_argument("--fold_regex", default=r"fold\d+", help="what counts as a fold folder name")
    g = ap.add_argument_group("Akita zero-shot input")
    g.add_argument("--zs_root", help="akita_zeroshot folder")
    g.add_argument("--akita_select", choices=["best", "mean", "worst"], default="best")
    g.add_argument("--akita0_template", default=AKITA0_TEMPLATE,
                   help="per-window metrics; {zs_root} and {fold} are filled in")
    g.add_argument("--akita0_heads", default=None,
                   help="selected-heads CSV (default {zs_root}/summary/akita0_selected_heads*.csv)")
    g = ap.add_argument_group("look")
    g.add_argument("--base_color", default=STYLE["base_color"], help="Akita zero-shot fill")
    g.add_argument("--model_color", default=STYLE["model_color"], help="iNeuroAkita fill")
    g.add_argument("--fill_alpha", type=float, default=STYLE["fill_alpha"])
    g.add_argument("--edge_color", default=STYLE["edge_color"], help="bar/box border, error bars")
    g.add_argument("--dot_alpha", type=float, default=STYLE["dot_alpha"], help="fold dots")
    g.add_argument("--no_mean_label", action="store_true", help="hide the mean value on bars")
    ap.add_argument("--list_inputs", action="store_true",
                    help="print which files would be read for every fold, then exit")
    ap.add_argument("--metric", choices=["pearson", "spearman", "mse"], default="pearson")
    ap.add_argument("--fold_stat", choices=["median", "mean"], default="median",
                    help="how windows are summarised within a fold")
    ap.add_argument("--style", choices=["bar", "box"], default="bar")
    ap.add_argument("--test", choices=["ttest", "wilcoxon"], default="ttest",
                    help="test shown on the by-fold plot (both are in the table)")
    ap.add_argument("--out", default="iNeuroAkita_vs_akita0")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.model_root or not a.zs_root:
        ap.error("--model_root and --zs_root are required")
    STYLE.update(base_color=a.base_color, model_color=a.model_color, fill_alpha=a.fill_alpha,
                 edge_color=a.edge_color, dot_alpha=a.dot_alpha, show_mean=not a.no_mean_label)
    if a.list_inputs:
        list_inputs([spec_to_template(s, a.pattern) for s in a.model_root], a.fold_regex,
                    f"iNeuroAkita: {a.label}")
        list_inputs([a.akita0_template.replace("{zs_root}", a.zs_root)], a.fold_regex,
                    "Akita zero-shot per-window metrics")
        hp = akita0_heads_path(a.zs_root, a.akita_select, a.akita0_heads)
        print(f"\n📂 Akita selected heads: {hp}  ({'found' if os.path.exists(hp) else 'MISSING'})")
        return 0
    if a.akita_select == "worst":
        print("⚠️  worst-head baseline is not a fair comparison; use it only as a range bound",
              file=sys.stderr)
    if a.metric == "mse":
        print("⚠️  MSE depends on each dataset's depth/normalisation; Akita was trained on "
              "differently scaled maps, so prefer correlation for this comparison", file=sys.stderr)

    mdl = load_model(a.model_root, a.pattern, a.metric, a.fold_regex)
    if mdl.empty:
        print("❌ no iNeuroAkita metrics loaded — run with --list_inputs to see what "
              "the SPEC matches", file=sys.stderr); return 2
    print(f"📥 iNeuroAkita: {len(mdl)} window rows, folds "
          f"{', '.join(sorted(mdl['fold'].unique(), key=_natkey))}, cell types "
          f"{', '.join(sorted(mdl['obs_target'].unique()))}")
    base, sel = load_akita0(a.zs_root, a.akita_select, a.metric, a.akita0_template,
                            a.akita0_heads, a.fold_regex)

    pw = mdl.merge(base, on=KEY, how="outer", suffixes=("_model", "_base"), indicator=True)
    only = pw["_merge"].value_counts().to_dict()
    if only.get("left_only", 0) or only.get("right_only", 0):
        print(f"⚠️  unpaired windows: iNeuroAkita-only {only.get('left_only', 0)}, "
              f"Akita-only {only.get('right_only', 0)} — dropped; check both used the same "
              f"fold data", file=sys.stderr)
    pw = pw[pw["_merge"] == "both"].drop(columns="_merge")
    if a.akita_select != "mean":
        pw = pw.merge(sel[["fold", "obs_target", "model_head"]].rename(
            columns={"model_head": "akita_head"}), on=["fold", "obs_target"], how="left")
    pw["cell_type"] = pw["obs_target"].map(lambda c: CELL_LABEL.get(c, c))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pw.sort_values(["obs_target", "fold", "chrom", "start"],
                   key=lambda s: s.map(_natkey) if s.name == "chrom" else s
                   ).to_csv(f"{a.out}_per_window.csv", index=False)

    lower = a.metric == "mse"
    pf = fold_level(pw, a.metric, a.label, a.fold_stat)
    pf["cell_type"] = pf["obs_target"].map(lambda c: CELL_LABEL.get(c, c))
    pf[["model", "obs_target", "cell_type", "fold", "n_windows", "mean", "median", "value"]] \
        .to_csv(f"{a.out}_per_fold.csv", index=False)
    st = fold_tests(pf, a.label, lower)
    st.to_csv(f"{a.out}_fold_stats.csv", index=False)

    summ = (pf.groupby(["obs_target", "cell_type", "model"])["value"]
              .agg(n_folds="size", mean="mean", sd="std").reset_index())
    summ["sem"] = summ["sd"] / np.sqrt(summ["n_folds"])
    pooled = pd.concat([
        pw.groupby("obs_target")[f"{a.metric}_model"].agg(pooled_n="size", pooled_median="median")
          .reset_index().assign(model=a.label),
        pw.groupby("obs_target")[f"{a.metric}_base"].agg(pooled_n="size", pooled_median="median")
          .reset_index().assign(model=BASE_LABEL)])
    summ = summ.merge(pooled, on=["obs_target", "model"], how="left")
    summ.to_csv(f"{a.out}_summary_by_celltype.csv", index=False)

    ct = chrom_tests(pw, a.metric)
    ct.to_csv(f"{a.out}_per_chrom.csv", index=False)

    mlabel = {"pearson": "Pearson r", "spearman": "Spearman ρ", "mse": "MSE"}[a.metric]
    print(f"\n📊 per-fold {a.fold_stat} {mlabel}: mean ± SD over folds")
    print(summ[["cell_type", "model", "n_folds", "mean", "sd", "pooled_median"]].round(4).to_string(index=False))
    print("\n🧪 paired across folds")
    print(st[["cell_type", "n_folds", "mean_model", "mean_baseline", "delta", "ci95_low",
              "ci95_high", "folds_better", "p_paired_t", "p_wilcoxon"]].round(4).to_string(index=False))

    plot_by_fold(pf, st, a.label,
                 f"{a.fold_stat.capitalize()} per-window {mlabel}\n(held-out test chromosomes)",
                 a.out, a.style, a.test)
    plot_by_chrom(pw, ct, a.metric, a.label, f"Per-window {mlabel}", a.out)
    print(f"\n✅ tables: {a.out}_per_window.csv, _per_fold.csv, _fold_stats.csv, "
          f"_summary_by_celltype.csv, _per_chrom.csv")
    return 0


# ============================================================== self-test
def self_test():
    import tempfile
    rng = np.random.default_rng(0)
    cells = ["NPC_P16", "Mic_AD", "Neuron_P16"]
    base_r = {"NPC_P16": 0.47, "Neuron_P16": 0.44, "Mic_AD": 0.36}
    chrom_by_fold = {1: ["chr10", "chr17", "chrX"], 2: ["chr1", "chr21"], 3: ["chr2", "chr19"],
                     4: ["chr3", "chr4"], 5: ["chr5", "chr6"], 6: ["chr7", "chr22"], 7: ["chr8", "chr9"]}
    heads = ["HFF", "H1hESC", "GM12878", "IMR90", "HCT116"]
    with tempfile.TemporaryDirectory() as td:
        mroot, zs = os.path.join(td, "trunk"), os.path.join(td, "zs")
        sel = []
        for k, chroms in chrom_by_fold.items():
            fold = f"fold{k}"
            win = [(c, s * 1_000_000) for c in chroms for s in range(rng.integers(20, 60))]
            mrows, zrows = [], []
            fe = rng.normal(0, 0.02)
            for c, s in win:
                for o in cells:
                    signal = rng.normal(0, 0.12)
                    mrows.append(dict(chrom=c, start=s, end=s + 917504, obs_target=o, model_head=o,
                                      pearson=base_r[o] + 0.12 + fe + signal + rng.normal(0, .05),
                                      mse=0.15, spearman=0.5))
                    for h in heads:
                        zrows.append(dict(chrom=c, start=s, end=s + 917504, obs_target=o, model_head=h,
                                          pearson=base_r[o] + (0.0 if h == "H1hESC" else -0.1)
                                          + signal + rng.normal(0, .05), mse=0.3, spearman=0.4))
            d = os.path.join(mroot, fold, "predictOut_v3", "hold_test"); os.makedirs(d)
            pd.DataFrame(mrows).to_csv(os.path.join(d, f"trunk_{fold}_test_per_window_metrics.tsv"),
                                       sep="\t", index=False)
            d = os.path.join(zs, fold, "test"); os.makedirs(d)
            pd.DataFrame(zrows).to_csv(os.path.join(d, f"akita0_{fold}_test_per_window_metrics.tsv"),
                                       sep="\t", index=False)
            sel += [dict(fold=fold, obs_target=o, model_head="H1hESC", valid_median_r=0.4) for o in cells]
        os.makedirs(os.path.join(zs, "summary"))
        pd.DataFrame(sel).to_csv(os.path.join(zs, "summary", "akita0_selected_heads.csv"), index=False)
        out = os.path.join(os.getcwd(), "cmp_selftest")
        assert main(["--model_root", mroot, "--label", "iNeuroAkita (Akita-init, merged)",
                     "--zs_root", zs, "--out", out]) == 0
        pw = pd.read_csv(f"{out}_per_window.csv")
        assert set(pw["akita_head"]) == {"H1hESC"}, "only the selected head is used"
        assert len(pw) > 0 and pw[["pearson_model", "pearson_base"]].notna().all().all()
        st = pd.read_csv(f"{out}_fold_stats.csv")
        assert len(st) == 3 and (st["delta"] > 0).all() and (st["n_folds"] == 7).all()
        pc = pd.read_csv(f"{out}_per_chrom.csv")
        assert pc.groupby("obs_target").size().eq(15).all(), "15 synthetic chromosomes per cell type"
        for c in ("NPC", "Neuron", "Microglia"):
            assert os.path.exists(f"{out}_by_chrom_{c}.png")
        assert main(["--model_root", mroot, "--zs_root", zs, "--out", out + "_box",
                     "--style", "box", "--test", "wilcoxon"]) == 0

        # single-head layout: one root per cell type, foldK/predictOut/hold_test/,
        # head named differently from the target (must still be kept)
        full = pd.concat([pd.read_csv(p, sep="\t") for p in
                          glob.glob(os.path.join(mroot, "fold*", "predictOut_v3", "hold_test", "*.tsv"))])
        sroots = []
        for o in cells:
            r = os.path.join(td, f"single_{o}"); sroots.append(r)
            for k in chrom_by_fold:
                fold = f"fold{k}"
                src = pd.read_csv(glob.glob(os.path.join(mroot, fold, "predictOut_v3", "hold_test",
                                                         "*.tsv"))[0], sep="\t")
                s = src[src["obs_target"] == o].copy()
                s["model_head"] = o.split("_")[0]          # e.g. head 'NPC' for target 'NPC_P16'
                d = os.path.join(r, fold, "predictOut", "hold_test"); os.makedirs(d)
                s.to_csv(os.path.join(d, f"single_{fold}_test_per_window_metrics.tsv"), sep="\t", index=False)
        args = []
        for r in sroots:
            args += ["--model_root", r]
        common = args + ["--pattern", "predictOut/hold_test/*_test_per_window_metrics.tsv",
                         "--label", "iNeuroAkita (single)", "--zs_root", zs]
        assert main(common + ["--list_inputs"]) == 0
        assert main(common + ["--out", out + "_single"]) == 0
        ps = pd.read_csv(f"{out}_single_per_window.csv")
        assert len(ps) == len(pw), "single-head roots reproduce the merged windows"

        # ROOT::PATTERN and {fold} template give the same result
        assert main(["--model_root", f"{sroots[0]}::predictOut/hold_test/*_test_per_window_metrics.tsv",
                     "--model_root", os.path.join(sroots[1], "{fold}", "predictOut", "hold_test", "*.tsv"),
                     "--model_root", f"{sroots[2]}::predictOut/hold_test/*.tsv",
                     "--zs_root", zs, "--out", out + "_mixed", "--no_mean_label"]) == 0
        pm = pd.read_csv(f"{out}_mixed_per_window.csv")
        assert len(pm) == len(pw)
        assert len(full) > 0
    print("\n✅ compare_to_akita0 self-test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
