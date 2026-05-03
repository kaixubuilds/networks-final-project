#!/usr/bin/env python3
"""
Generate all 8 paper figures from plots/data/fig{1..8}.json.
Output: figures/fig*.png + .svg

Usage:
    python plots/make_figures.py

Requires: matplotlib, numpy

All statistics and all label text are pre-computed in precompute_data.py.
This script is pure rendering.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np

DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        150,
    'savefig.dpi':       200,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

C_RANDOM = '#78909C'
C_PO2C   = '#1565C0'
C_JSQ    = '#C62828'

COLOR_KEY = {'random': C_RANDOM, 'po2c': C_PO2C, 'jsq': C_JSQ}

CLASS_COLORS = {
    'subtract':     '#2E7D32',
    'tiebreaker':   '#6A1B9A',
    'add_positive': '#E65100',
    'degenerate':   '#4E342E',
    'other':        '#546E7A',
}

FAMILY_STYLE = {
    'Qwen3':    dict(marker='o', color='#1565C0'),
    'DeepSeek': dict(marker='D', color='#558B2F'),
    'gpt-oss':  dict(marker='s', color='#6A1B9A'),
    'Other':    dict(marker='^', color='#78909C'),
}

WORKER_COLORS = plt.cm.tab10.colors[:8]


def _lighten(hex_color, factor=0.82):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return '#{:02X}{:02X}{:02X}'.format(
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load(name):
    path = os.path.join(DATA_DIR, f'{name}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found — run precompute_data.py first')
    with open(path) as f:
        return json.load(f)

def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.png'))
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.svg'))
    plt.close(fig)
    print(f'  Saved {name}.png / .svg')


# --------------------------------------------------------------------------- #
# Figure 1 — Thundering Herd
# --------------------------------------------------------------------------- #
def fig1(d):
    t   = d['text']
    arr = np.array(d['worker_series'])   # shape (T, 8)
    time_ms = d['time_ms']

    max_per_worker = arr.max(axis=0)
    top2 = set(np.argsort(max_per_worker)[-2:])

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for w in range(8):
        lw    = 1.6 if w in top2 else 0.8
        alpha = 1.0 if w in top2 else 0.5
        ax.plot(time_ms, arr[:, w], color=WORKER_COLORS[w], lw=lw, alpha=alpha)

    handles = [plt.Line2D([0], [0], color=WORKER_COLORS[w],
                          lw=1.6 if w in top2 else 0.8,
                          alpha=1.0 if w in top2 else 0.5,
                          label=f'W{w}{"*" if w in top2 else ""}')
               for w in range(8)]
    ax.legend(handles=handles, ncol=4, fontsize=7, loc='upper right', framealpha=0.7)
    ax.set_xlabel(t['xlabel'])
    ax.set_ylabel(t['ylabel'])
    ax.set_xlim(time_ms[0], time_ms[-1])
    ax.set_ylim(bottom=0)
    ax.set_title(t['title'])
    save(fig, 'fig1_thundering_herd')


# --------------------------------------------------------------------------- #
# Figure 2 — Baseline Stats Table
# --------------------------------------------------------------------------- #
def fig2(d):
    t     = d['text']
    rows  = d['rows']
    cols  = t['col_labels']
    cells = [[r['strategy'],
              f"{r['median']:.3f}",
              f"{r['iqr']:.3f}",
              f"{r['std']:.3f}",
              f"{r['min']:.3f}",
              f"{r['max']:.3f}"]
             for r in rows]

    fig, ax = plt.subplots(figsize=(5.2, 1.8))
    ax.axis('off')

    tbl = ax.table(cellText=cells, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(cols))))

    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#ECEFF1')
        tbl[0, j].set_text_props(fontweight='bold')
    for i, r in enumerate(rows):
        color = COLOR_KEY[r['color_key']]
        bg    = _lighten(color, 0.88)
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(bg)
        tbl[i + 1, 0].set_text_props(color=color, fontweight='bold')

    ax.set_title(t['title'], fontsize=10, pad=14)
    save(fig, 'fig2_baseline_ladder')


# --------------------------------------------------------------------------- #
# Figure 3 — No Model Beats Random
# --------------------------------------------------------------------------- #
def fig3(d):
    t         = d['text']
    models    = d['models']
    baselines = d['baselines']
    bl        = t['baseline_labels']

    labels  = [m['label']        for m in models]
    medians = [m['median_sigma'] for m in models]
    y_pos   = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Shade the "worse than Random" region to make the headline result immediate
    ax.axvspan(baselines['Random'], ax.get_xlim()[1],
               color='#FFCDD2', alpha=0.25, zorder=0, label=t['shade_label'])

    ax.hlines(y_pos, 0, medians, colors='#BDBDBD', lw=1.0)
    ax.plot(medians, y_pos, 'o', color='#263238', ms=6, zorder=5)

    for y, v in zip(y_pos, medians):
        ax.annotate(f'{v:.1f}', xy=(v, y), xytext=(4, 0),
                    textcoords='offset points', fontsize=7, va='center')

    ax.axvline(baselines['Random'],    color=C_RANDOM, ls='--', lw=1.2, alpha=0.85, label=bl['Random'])
    ax.axvline(baselines['Po2C'],      color=C_PO2C,   ls='--', lw=1.2, alpha=0.85, label=bl['Po2C'])
    ax.axvline(baselines['JSQ-stale'], color=C_JSQ,    ls='--', lw=1.2, alpha=0.85, label=bl['JSQ-stale'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(t['xlabel'])
    ax.set_title(t['title'])
    ax.set_xlim(left=0)
    ax.legend(loc='lower right', fontsize=7)
    save(fig, 'fig3_no_model_beats_random')


# --------------------------------------------------------------------------- #
# Figure 4 — Heavy-Tailed Distributions
# --------------------------------------------------------------------------- #
def fig4(d):
    t      = d['text']
    models = d['models']
    bmed   = d['baseline_medians']
    bl     = t['baseline_labels']

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    rng = np.random.default_rng(42)

    for x_pos, m in enumerate(models):
        sigmas = m['sigmas']
        jitter_scale = 0.22 * m['n'] / 100
        jitter = rng.uniform(-jitter_scale, jitter_scale, size=m['n'])
        ax.scatter(x_pos + jitter, sigmas,
                   s=8, alpha=0.55, color='#455A64', linewidths=0, zorder=3)
        ax.plot([x_pos - 0.22, x_pos + 0.22], [m['median'], m['median']],
                color='#B71C1C', lw=2.0, zorder=5)
        # n= label at fixed bottom of axes, aligned with data x position
        ax.text(x_pos, 0.02, f"n={m['n']}",
                transform=ax.get_xaxis_transform(),
                ha='center', fontsize=6.5, color='#546E7A')

    ax.axhline(bmed['Random'],    color=C_RANDOM, ls='--', lw=1.2, alpha=0.8, label=bl['Random'])
    ax.axhline(bmed['JSQ-stale'], color=C_JSQ,    ls='--', lw=1.2, alpha=0.8, label=bl['JSQ-stale'])

    ax.set_yscale('log')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m['label'] for m in models], fontsize=7.5, rotation=15, ha='right')
    ax.set_ylabel(t['ylabel'])
    ax.set_title(t['title'])
    ax.legend(loc='upper left', fontsize=7)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f'{y:.0f}' if y >= 10 else f'{y:.1f}'))
    save(fig, 'fig4_heavy_tailed_distributions')


# --------------------------------------------------------------------------- #
# Figure 5 — Parse Success Rates Table
# --------------------------------------------------------------------------- #
def fig5(d):
    t      = d['text']
    models = d['models']
    cols   = t['col_labels']
    cells  = [[m['name'], f"{m['parse_rate']:.0%}", str(m['n_valid'])]
              for m in models]

    fig, ax = plt.subplots(figsize=(4.5, 0.42 * len(models) + 0.9))
    ax.axis('off')

    tbl = ax.table(cellText=cells, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(cols))))

    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#ECEFF1')
        tbl[0, j].set_text_props(fontweight='bold')
    for i, m in enumerate(models):
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor('#FFFFFF')
        rate  = m['parse_rate']
        color = '#1565C0' if rate >= 0.6 else '#E65100' if rate < 0.3 else '#37474F'
        tbl[i + 1, 1].set_text_props(color=color, fontweight='bold')

    ax.set_title(t['title'], fontsize=10, pad=10)
    save(fig, 'fig5_parse_rates')


# --------------------------------------------------------------------------- #
# Figure 6 — Convergent Code
# --------------------------------------------------------------------------- #
def fig6(d):
    t       = d['text']
    entries = [v for k, v in d.items() if k != 'text']
    if len(entries) < 2:
        print('  fig6: insufficient data, skipping')
        return

    # Parse code for both panels up-front so we can size the figure from content
    panels = []
    for entry in entries:
        lines = entry['source'].strip().split('\n')
        if lines and lines[0].startswith('def dispatch'):
            lines = lines[1:]
        lines = [l.rstrip() for l in lines]
        while lines and not lines[-1]:
            lines.pop()
        panels.append({'lines': lines, 'code': '\n'.join(lines), 'entry': entry})

    max_chars  = max(max(len(l) for l in p['lines']) if p['lines'] else 1 for p in panels)
    max_nlines = max(len(p['lines']) for p in panels)

    # Font: fit to half the figure width; cap at 7.0pt
    FIG_W     = 7.2
    PANEL_W   = FIG_W / 2
    BOX_W_PAD = 0.08   # fraction of panel width consumed by box padding + margins
    fs_w      = ((1 - BOX_W_PAD) * PANEL_W * 72) / (max_chars * 0.62)
    fontsize  = max(4.0, min(fs_w, 7.0))

    # Figure height: derive from actual content so box is tight
    line_h_in  = fontsize * 1.4 / 72   # inches per line
    code_h_in  = max_nlines * line_h_in
    BOX_PAD_IN = 0.18                   # vertical padding inside box (top + bottom)
    TITLE_IN   = 0.50                   # space for panel title
    BOT_IN     = 0.06                   # bottom margin
    fig_h = code_h_in + BOX_PAD_IN + TITLE_IN + BOT_IN
    fig_h = max(fig_h, 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, fig_h))

    # Axes-coordinate fractions for the box
    content_frac = (code_h_in + BOX_PAD_IN) / fig_h
    bot_frac     = BOT_IN / fig_h
    box_h_ax     = content_frac
    box_y_ax     = bot_frac

    for ax, p in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, box_y_ax), 0.96, box_h_ax,
            boxstyle='round,pad=0.015',
            facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.8,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(0.05, box_y_ax + box_h_ax - 0.01, p['code'],
                transform=ax.transAxes, fontsize=fontsize, fontfamily='monospace',
                va='top', ha='left', color='#212121', wrap=False,
                linespacing=1.35)
        ax.set_title(
            f"{p['entry']['model_label']}\nmean \u03c3 = {p['entry']['mean_sigma']:.2f}",
            fontsize=8.0, color='#1B5E20', pad=4)

    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.06,
                        bottom=0.0, top=1.0)
    save(fig, 'fig6_correct_strategy')


# --------------------------------------------------------------------------- #
# Figure 7 — Strategy Class Performance
# --------------------------------------------------------------------------- #
def fig7(d):
    t       = d['text']
    classes = d['classes']
    bmed    = d['baseline_medians']
    bl      = t['baseline_labels']

    fig, ax = plt.subplots(figsize=(5.5, 0.7 * len(classes) + 1.4))
    rng = np.random.default_rng(7)

    for y_pos, cls in enumerate(classes):
        sigmas = cls['sigmas']
        color  = CLASS_COLORS.get(cls['cls'], '#607D8B')
        jitter = rng.uniform(-0.28, 0.28, size=cls['n'])
        ax.scatter(sigmas, y_pos + jitter,
                   s=9, alpha=0.55, color=color, linewidths=0, zorder=3)
        ax.plot([cls['median'], cls['median']], [y_pos - 0.35, y_pos + 0.35],
                color=color, lw=2.5, zorder=5)
        # n= label at fixed right edge of axes, aligned with data y position
        ax.text(0.99, y_pos, f" n={cls['n']}",
                transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=7, color=color)

    ax.axvline(bmed['Random'],    color=C_RANDOM, ls='--', lw=1.2, alpha=0.8, label=bl['Random'])
    ax.axvline(bmed['JSQ-stale'], color=C_JSQ,    ls='--', lw=1.2, alpha=0.8, label=bl['JSQ-stale'])

    ax.set_xscale('log')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([c['label'] for c in classes])
    ax.set_xlabel(t['xlabel'])
    ax.set_title(t['title'])
    ax.legend(loc='upper right', fontsize=7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f'{y:.0f}' if y >= 1 else f'{y:.2f}'))
    save(fig, 'fig7_strategy_class_performance')


# --------------------------------------------------------------------------- #
# Figure 8 — Scale vs. Performance
# --------------------------------------------------------------------------- #
def fig8(d):
    t      = d['text']
    models = d['models']
    bmed   = d['baseline_medians']
    bl     = t['baseline_labels']
    fl     = t['family_labels']

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    plotted_families = set()

    for m in models:
        style = FAMILY_STYLE[m['family']]
        label = fl[m['family']] if m['family'] not in plotted_families else None
        plotted_families.add(m['family'])

        ax.scatter(m['param_count'], m['median_sigma'],
                   marker=style['marker'], color=style['color'],
                   s=60, zorder=5, label=label,
                   edgecolors='#FFFFFF', linewidths=0.5)
        ax.annotate(m['point_label'],
                    xy=(m['param_count'], m['median_sigma']),
                    xytext=(5, 3), textcoords='offset points',
                    fontsize=7, color=style['color'])
        if m['uncertain_params']:
            ax.annotate(t['uncertain_note'],
                        xy=(m['param_count'], m['median_sigma']),
                        xytext=(5, -18), textcoords='offset points',
                        fontsize=6, color='#9E9E9E', style='italic')

    ax.axhline(bmed['Random'],    color=C_RANDOM, ls='--', lw=1.2, alpha=0.8, label=bl['Random'])
    ax.axhline(bmed['JSQ-stale'], color=C_JSQ,    ls='--', lw=1.2, alpha=0.8, label=bl['JSQ-stale'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(t['xlabel'])
    ax.set_ylabel(t['ylabel'])
    ax.set_title(t['title'])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M' if x >= 1e6 else str(int(x))))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f'{y:.0f}' if y >= 10 else f'{y:.1f}'))
    ax.legend(fontsize=7, loc='upper left')
    save(fig, 'fig8_scale_vs_performance')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print(f'Loading data from {DATA_DIR}/', flush=True)
    print(f'Saving figures to {FIGURES_DIR}/', flush=True)

    print('\nFigure 1: Thundering herd')
    fig1(load('fig1'))
    print('Figure 2: Baseline stats table')
    fig2(load('fig2'))
    print('Figure 3: No model beats Random')
    fig3(load('fig3'))
    print('Figure 4: Heavy-tailed distributions')
    fig4(load('fig4'))
    print('Figure 5: Parse rates table')
    fig5(load('fig5'))
    print('Figure 6: Correct strategy code')
    fig6(load('fig6'))
    print('Figure 7: Strategy class performance')
    fig7(load('fig7'))
    print('Figure 8: Scale vs. performance')
    fig8(load('fig8'))
    print('\nDone.')


if __name__ == '__main__':
    main()
