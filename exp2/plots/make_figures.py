#!/usr/bin/env python3
"""
Render all Experiment 2 figures from precomputed JSON data.

Usage:
    python make_figures.py                      # render all figures
    python make_figures.py --figs fig1 fig3     # render specific figures

Figures:
    fig1  -- Noise sweep: mean response time vs. log-normal σ for baselines
    fig2  -- Main results: per-model response time distributions (box plots)
    fig3  -- Strategy taxonomy: performance by strategy class
    fig4  -- Per-model detail: violin + strategy breakdown
    fig5  -- Response time CDF: tail behaviour under Pareto service
    fig6  -- Best LLM functions: code display + robust performance comparison
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Style constants
# --------------------------------------------------------------------------- #
PALETTE = {
    "SRPT-oracle":  "#2ca02c",   # green
    "SRPT-noisy":   "#1f77b4",   # blue
    "LC":           "#ff7f0e",   # orange
    "Random":       "#d62728",   # red
    "LLM":          "#9467bd",   # purple
    "srpt_only":    "#1f77b4",
    "lc_only":      "#ff7f0e",
    "both_signals": "#2ca02c",
    "random_like":  "#d62728",
    "unknown":      "#7f7f7f",
}

STRATEGY_LABELS = {
    "srpt_only":    "SRPT-only\n(ignores connections)",
    "lc_only":      "LC-only\n(ignores estimates)",
    "both_signals": "Both signals\n(blend)",
    "random_like":  "Random-like",
    "unknown":      "Other",
}

BASELINE_LINESTYLES = {
    "SRPT-oracle": ("--", 1.5, PALETTE["SRPT-oracle"]),
    "SRPT-noisy":  (":",  1.5, PALETTE["SRPT-noisy"]),
    "LC":          ("-.", 1.2, PALETTE["LC"]),
    "Random":      ((0, (1, 1)), 1.2, PALETTE["Random"]),  # dense dots, distinct from "--"
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
})

_BASELINE_ORDER = ["SRPT-oracle", "SRPT-noisy", "LC", "Random"]


def _load(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  MISSING: {path}  (run precompute_data.py first)", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, name):
    for ext in ("png", "svg"):
        out = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(out)
        print(f"  Saved: {out}")


def _add_baseline_hlines(ax, baseline_medians, exclude=None):
    """Draw horizontal reference lines for each baseline median."""
    exclude = exclude or set()
    for bname in _BASELINE_ORDER:
        if bname in exclude or bname not in baseline_medians:
            continue
        ls, lw, col = BASELINE_LINESTYLES.get(bname, ("--", 1.0, "gray"))
        ax.axhline(baseline_medians[bname], linestyle=ls, linewidth=lw,
                   color=col, alpha=0.75, label=f"{bname} median")


# --------------------------------------------------------------------------- #
# Fig 1 — Noise sweep
# --------------------------------------------------------------------------- #
def fig1_noise_sweep():
    data = _load("fig1.json")
    if data is None:
        return

    sigmas = data["sigmas"]
    dispatchers = data["dispatchers"]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    order = ["SRPT-oracle", "SRPT-noisy", "LC"]
    for name in order:
        if name not in dispatchers:
            continue
        d   = dispatchers[name]
        med = np.array(d["median_rt_per_sigma"])
        q25 = np.array(d["iqr25_rt_per_sigma"])
        q75 = np.array(d["iqr75_rt_per_sigma"])
        col = PALETTE.get(name, "gray")
        ls  = "--" if name == "SRPT-oracle" else "-"
        ax.plot(sigmas, med, marker="o", markersize=4, label=name, color=col, linestyle=ls)
        ax.fill_between(sigmas, q25, q75, alpha=0.12, color=col)

    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Log-normal noise σ")
    ax.set_ylabel("Median response time (ms)")
    ax.set_title("SRPT degrades with job-size estimation noise\n"
                 "(M/G/1, Pareto α=1.5, ρ=0.8, 8 workers)")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xlim(left=0)
    ax.grid(axis="y", alpha=0.3)
    # Annotate AFTER set_yscale so get_ylim() reflects log-scale bounds
    ax.text(1.02, ax.get_ylim()[1] ** 0.97, "Exp 2\nnoise σ", fontsize=8,
            color="gray", va="top")

    _save(fig, "fig1_noise_sweep")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2 — Main results: per-model box plots
# --------------------------------------------------------------------------- #
def fig2_main_results():
    data = _load("fig2.json")
    if data is None:
        return

    model_stats      = data["model_stats"]
    baseline_medians = data["baseline_medians"]
    metric_label     = data.get("metric_label", "Mean response time (ms)")

    if not model_stats:
        print("  No LLM model data; skipping fig2.", file=sys.stderr)
        return

    # Drop Qwen3-8B (only 1 valid trial — unreliable)
    model_stats = {m: v for m, v in model_stats.items() if m != "Qwen3-8B"}

    # Sort models by median response time
    models_sorted = sorted(model_stats.keys(),
                           key=lambda m: model_stats[m]["median"])
    box_data  = [model_stats[m]["values"] for m in models_sorted]
    positions = list(range(len(models_sorted)))

    fig, ax = plt.subplots(figsize=(max(6, len(models_sorted) * 1.4), 4.5))

    bp = ax.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "white", "linewidth": 2},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2})

    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["LLM"])
        patch.set_alpha(0.7)

    # Baseline reference lines (exclude Random — it is an extreme outlier)
    _add_baseline_hlines(ax, baseline_medians, exclude={"Random"})

    ax.set_xticks(positions)
    ax.set_xticklabels(models_sorted, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric_label)
    ax.set_title("LLM-generated dispatchers vs. baselines\n"
                 "(100 seeds per model, noise σ=1.0)")

    # Manually prepend the LLM box to the legend so it's labeled
    llm_handle = mpatches.Patch(facecolor=PALETTE["LLM"], alpha=0.7,
                                label="LLM-generated (box = IQR, 100 seeds)")
    baseline_handles, baseline_labels = ax.get_legend_handles_labels()
    ax.legend(handles=[llm_handle] + baseline_handles,
              labels=["LLM-generated (box = IQR, 100 seeds)"] + baseline_labels,
              loc="upper right", framealpha=0.85, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "fig2_main_results")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3 — Strategy taxonomy
# --------------------------------------------------------------------------- #
def fig3_strategy_taxonomy():
    data = _load("fig3.json")
    if data is None:
        return

    strategy_stats   = data["strategy_stats"]
    baseline_medians = data["baseline_medians"]
    labels_map       = data.get("strategy_labels", STRATEGY_LABELS)

    if not strategy_stats:
        print("  No strategy data; skipping fig3.", file=sys.stderr)
        return

    # Sort by median
    classes_sorted = sorted(strategy_stats.keys(),
                             key=lambda c: strategy_stats[c]["median"])
    labels    = [labels_map.get(c, c) for c in classes_sorted]
    box_data  = [strategy_stats[c]["values"] for c in classes_sorted]
    counts    = [strategy_stats[c]["n_trials"] for c in classes_sorted]
    colors    = [PALETTE.get(c, "#7f7f7f") for c in classes_sorted]
    positions = list(range(len(classes_sorted)))

    fig, ax = plt.subplots(figsize=(max(5, len(classes_sorted) * 1.6), 4.5))

    bp = ax.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2})

    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    # Annotate counts above each box — use p97 to clear most outlier points
    for pos, count, vals in zip(positions, counts, box_data):
        top = np.percentile(vals, 97)
        ax.text(pos, top * 1.02, f"n={count}", ha="center", va="bottom", fontsize=8)

    _add_baseline_hlines(ax, baseline_medians, exclude={"Random"})

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean response time (ms)")
    ax.set_title("Dispatch strategy class vs. performance\n"
                 "(classified by signals used in LLM function)")
    ax.legend(loc="upper right", framealpha=0.85, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "fig3_strategy_taxonomy")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4 — Per-model detail: violin + strategy breakdown
# --------------------------------------------------------------------------- #
def fig4_per_model_detail():
    data = _load("fig4.json")
    if data is None:
        return

    model_detail     = data["model_detail"]
    baseline_medians = data["baseline_medians"]

    if not model_detail:
        print("  No model detail data; skipping fig4.", file=sys.stderr)
        return

    models = sorted(model_detail.keys(),
                    key=lambda m: float(np.median(model_detail[m]["all_values"])))
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(max(8, n_models * 2.5), 4.5),
                              sharey=True)
    if n_models == 1:
        axes = [axes]

    # Global y-limits — multiplicative padding suits log scale
    all_vals = [v for m in models for v in model_detail[m]["all_values"]]
    ymin, ymax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    ypad = (ymax / ymin) ** 0.05

    strategy_order = ["srpt_only", "both_signals", "lc_only", "random_like", "unknown"]

    for ax, model in zip(axes, models):
        detail   = model_detail[model]
        all_vals = np.array(detail["all_values"])
        by_strat = detail["by_strategy"]

        # Violin for all trials
        if len(all_vals) >= 3:
            vp = ax.violinplot([all_vals], positions=[0], widths=0.7,
                               showmedians=True, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(PALETTE["LLM"])
                body.set_alpha(0.35)
            vp["cmedians"].set_color("white")
            vp["cmedians"].set_linewidth(2)

        # Scatter individual strategy classes
        for sc in strategy_order:
            if sc not in by_strat:
                continue
            vals = np.array(by_strat[sc]["values"])
            col  = PALETTE.get(sc, "gray")
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.zeros(len(vals)) + jitter, vals, s=12, color=col,
                       alpha=0.6, zorder=3, label=STRATEGY_LABELS.get(sc, sc))

        # Baseline lines (exclude Random)
        for bname in _BASELINE_ORDER:
            if bname == "Random" or bname not in baseline_medians:
                continue
            ls, lw, col = BASELINE_LINESTYLES[bname]
            ax.axhline(baseline_medians[bname], linestyle=ls, linewidth=lw,
                       color=col, alpha=0.8)

        ax.set_xticks([])
        short_name = model.replace("-", "\n").replace("_", "\n")
        ax.set_title(short_name, fontsize=8, pad=4)
        ax.set_ylim(ymin / ypad, ymax * ypad)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Mean response time (ms)")

    # Shared legend — only include strategy classes that appear in the data
    present_classes = {sc for m in model_detail.values() for sc in m["by_strategy"]}
    handles = []
    for sc in strategy_order:
        if sc not in present_classes:
            continue
        handles.append(mpatches.Patch(color=PALETTE.get(sc, "gray"), alpha=0.75,
                                      label=STRATEGY_LABELS.get(sc, sc)))
    for bname in _BASELINE_ORDER:
        if bname != "Random" and bname in baseline_medians:
            ls, lw, col = BASELINE_LINESTYLES[bname]
            handles.append(plt.Line2D([0], [0], color=col, linestyle=ls,
                                       linewidth=lw, label=f"{bname} baseline"))

    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.15), framealpha=0.9)
    fig.suptitle("Per-model trial distribution and strategy breakdown", fontsize=11)
    # Use subplots_adjust to leave room for the external legend rather than
    # tight_layout(), which doesn't account for artists outside the axes boundary.
    fig.subplots_adjust(bottom=0.22, top=0.9)

    _save(fig, "fig4_per_model_detail")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 5 — Response time CDF
# --------------------------------------------------------------------------- #
def fig5_rt_cdf():
    data = _load("fig5.json")
    if data is None:
        return

    cdf_data = data["cdf_data"]
    noise_sigma = data.get("noise_sigma", 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    draw_order = ["SRPT-oracle", "SRPT-noisy", "LC"] + [
        k for k in cdf_data if k not in ("SRPT-oracle", "SRPT-noisy", "LC", "Random")
    ]

    for name in draw_order:
        if name not in cdf_data:
            continue
        xs = np.array(cdf_data[name]["x"])
        ys = np.array(cdf_data[name]["y"])
        col = PALETTE.get(name, "#7f7f7f")
        ls  = "--" if name == "SRPT-oracle" else "-"
        lw  = 1.8 if "Best LLM" in name else 1.4
        ax.plot(xs, ys, label=name, color=col, linestyle=ls, linewidth=lw)

    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.text(1.05, 0.05, "1 ms\n(mean)", fontsize=7.5, color="gray", va="bottom")

    ax.set_xlabel("Response time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Job response time distribution — Pareto service, noise σ={noise_sigma}")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

    _save(fig, "fig5_rt_cdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 6 helpers
# --------------------------------------------------------------------------- #
def _render_code_panel(ax, fn):
    """Render a function's source code into a borderless axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    header = f"#{fn['rank']}  {fn['model']}   median = {fn['median_rt']:.3f} ms"
    ax.text(0.012, 0.97, header, transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="top", color="#222222")

    ax.text(0.012, 0.83, fn["display_source"], transform=ax.transAxes,
            fontsize=6.8, fontfamily="monospace", va="top", color="#111111",
            bbox=dict(facecolor="#f6f6f6", edgecolor="#cccccc",
                      boxstyle="round,pad=0.45", alpha=0.95))


def _render_perf_chart(ax, best_fns, baseline_stats):
    """Horizontal bar chart: baselines + top LLM functions, sorted worst→best (top)."""
    baseline_order = ["LC", "SRPT-noisy", "SRPT-oracle"]

    rows = []
    for name in baseline_order:
        if name not in baseline_stats:
            continue
        s = baseline_stats[name]
        rows.append(dict(label=name, median=s["median"],
                         q25=s["q25"], q75=s["q75"],
                         color=PALETTE.get(name, "gray"), is_llm=False, height=0.5))

    # LLM functions worst→best so the best sits at the top of the chart
    for fn in reversed(best_fns):
        rows.append(dict(label=f"#{fn['rank']} {fn['model']}",
                         median=fn["median_rt"],
                         q25=fn["q25_rt"], q75=fn["q75_rt"],
                         color=PALETTE["LLM"], is_llm=True, height=0.75))

    y_pos   = list(range(len(rows)))
    medians = [r["median"] for r in rows]
    xerr_lo = [r["median"] - r["q25"] for r in rows]
    xerr_hi = [r["q75"] - r["median"] for r in rows]
    heights = [r["height"] for r in rows]

    for y, med, col, h in zip(y_pos, medians, [r["color"] for r in rows], heights):
        ax.barh(y, med, color=col, alpha=0.72, height=h)
    ax.errorbar(medians, y_pos, xerr=[xerr_lo, xerr_hi],
                fmt="none", color="#333333", capsize=3, linewidth=1.1)

    # Dashed separator between baselines and LLM rows
    n_base = sum(1 for r in rows if not r["is_llm"])
    ax.axhline(n_base - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_xlabel("Median mean response time (ms)")
    ax.grid(axis="x", alpha=0.3)


# --------------------------------------------------------------------------- #
# Fig 6 — Best LLM-discovered functions
# --------------------------------------------------------------------------- #
def fig6_best_functions():
    data = _load("fig6.json")
    if data is None:
        return

    best_fns       = data["best_functions"]
    baseline_stats = data["baseline_stats"]

    if not best_fns:
        print("  No best-function data; skipping fig6.", file=sys.stderr)
        return

    n_code = min(2, len(best_fns))   # show code panels for top 2 only

    fig = plt.figure(figsize=(13, 5.5))
    gs  = GridSpec(n_code, 2, figure=fig,
                   width_ratios=[3, 2], hspace=0.10, wspace=0.06,
                   left=0.03, right=0.97, top=0.91, bottom=0.09)

    for i in range(n_code):
        ax = fig.add_subplot(gs[i, 0])
        _render_code_panel(ax, best_fns[i])

    ax_perf = fig.add_subplot(gs[:, 1])
    _render_perf_chart(ax_perf, best_fns, baseline_stats)

    fig.suptitle(data["text"]["title"], fontsize=11)
    _save(fig, "fig6_best_functions")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
ALL_FIGS = {
    "fig1": fig1_noise_sweep,
    "fig2": fig2_main_results,
    "fig3": fig3_strategy_taxonomy,
    "fig4": fig4_per_model_detail,
    "fig5": fig5_rt_cdf,
    "fig6": fig6_best_functions,
}


def main():
    parser = argparse.ArgumentParser(description="Render Exp2 figures")
    parser.add_argument("--figs", nargs="+", choices=list(ALL_FIGS.keys()),
                        default=list(ALL_FIGS.keys()),
                        help="Which figures to render (default: all)")
    args = parser.parse_args()

    for fig_name in args.figs:
        print(f"\n── {fig_name} ────────────────────────────────────────────")
        ALL_FIGS[fig_name]()

    print("\nDone. Figures written to:", FIG_DIR)


if __name__ == "__main__":
    main()
