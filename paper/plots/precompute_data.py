#!/usr/bin/env python3
"""
Pre-compute all data needed for paper figures.
Output: plots/data/fig{1..8}.json

Each file contains:
  "text"  — every static string that appears on the figure
  <data>  — only the numbers/values actually plotted

Usage:
    python plots/precompute_data.py

Requires: numpy
"""

import heapq
import json
import os
import random
import re
import statistics

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')

# Canonical model order: Qwen3 ascending size, then DeepSeek, then gpt-oss
MODEL_ORDER = [
    'Qwen3-1.7B',
    'Qwen3-4B',
    'Qwen3-8B',
    'Qwen3-14B',
    'DeepSeek-R1-0528-Qwen3-8B',
    'gpt-oss-20b',
]

BASELINE_NAMES = ['Random', 'JSQ (stale)', 'Po2C (stale)']

EXCLUDED_MODELS = {'Qwen3-0.6B', 'Llama-3.1-8B'}

MODEL_LABEL = {
    'DeepSeek-R1-0528-Qwen3-8B': 'DeepSeek-R1\n(8B)',
}
def model_short(name):
    return MODEL_LABEL.get(name, name)

CLASS_ORDER = ['subtract', 'tiebreaker', 'add_positive', 'degenerate', 'other']
CLASS_LABELS = {
    'subtract':     'Subtract age (correct)',
    'tiebreaker':   'JSQ + age tiebreaker',
    'add_positive': 'Add age (wrong sign)',
    'degenerate':   'Degenerate',
    'other':        'Other / unclassified',
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def percentile(arr, p):
    arr = sorted(arr)
    i   = (len(arr) - 1) * p / 100
    lo  = int(i)
    hi  = min(lo + 1, len(arr) - 1)
    return arr[lo] + (arr[hi] - arr[lo]) * (i - lo)

def save_json(name, obj):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f'  Saved {path}')


# --------------------------------------------------------------------------- #
# Simulation for Figure 1
# --------------------------------------------------------------------------- #
N_WORKERS   = 8
LAMBDA      = 6.4
H           = 20.0
P_LO, P_HI  = 1.0, 5.0
MEAS        = 10.0
DEPARTURE   = 0; HEARTBEAT = 1; ARRIVAL = 2

def _dispatch_jsq(reported_queue, ages, n_workers):
    min_q = min(reported_queue)
    return random.choice([i for i, q in enumerate(reported_queue) if q == min_q])

def run_sim_worker_series(seed=0, max_jobs=30000):
    rng    = np.random.default_rng(seed)
    hb_rng = random.Random(seed)
    random.seed(seed)

    queue_lengths  = [0]   * N_WORKERS
    worker_free_at = [0.0] * N_WORKERS
    reported_queue = [0]   * N_WORKERS
    last_heard     = [-H]  * N_WORKERS
    time_ms        = []
    worker_series  = []
    next_measure   = MEAS

    heap = []
    for w in range(N_WORKERS):
        t0 = (w + 1) * H / N_WORKERS + hb_rng.uniform(P_LO, P_HI)
        heapq.heappush(heap, (t0, HEARTBEAT, w))
    heapq.heappush(heap, (float(rng.exponential(1.0 / LAMBDA)), ARRIVAL, -1))

    jobs_dispatched = 0
    while heap and jobs_dispatched < max_jobs:
        t, etype, widx = heapq.heappop(heap)
        while next_measure <= t:
            time_ms.append(next_measure)
            worker_series.append(list(queue_lengths))
            next_measure += MEAS
        if etype == DEPARTURE:
            queue_lengths[widx] = max(0, queue_lengths[widx] - 1)
        elif etype == HEARTBEAT:
            reported_queue[widx] = queue_lengths[widx]
            last_heard[widx]     = t
            heapq.heappush(heap, (t + H + hb_rng.uniform(P_LO, P_HI), HEARTBEAT, widx))
        else:
            jobs_dispatched += 1
            ages   = [t - lh for lh in last_heard]
            worker = _dispatch_jsq(list(reported_queue), ages, N_WORKERS)
            queue_lengths[worker] += 1
            finish = max(t, worker_free_at[worker]) + float(rng.exponential(1.0))
            worker_free_at[worker] = finish
            heapq.heappush(heap, (finish, DEPARTURE, worker))
            if jobs_dispatched < max_jobs:
                heapq.heappush(heap, (t + float(rng.exponential(1.0 / LAMBDA)), ARRIVAL, -1))

    return time_ms, worker_series


# --------------------------------------------------------------------------- #
# Strategy classifier for Figure 7
# --------------------------------------------------------------------------- #
def classify_strategy(src):
    if src is None:
        return None
    if re.search(r'if\s+\w+\s*>\s*max_value', src):            return 'degenerate'
    if re.search(r'\.index\(max\(queue_lengths\)', src):        return 'degenerate'
    if re.search(r'sort\s*\(.*key\s*=\s*lambda\s*\w+\s*:\s*\w+\[2\]', src): return 'degenerate'
    if re.search(r'if queue_lengths\[\w+\] > 0:\s*\n\s*return', src):        return 'degenerate'
    if re.search(r'return\s+max\b.*queue', src):                return 'degenerate'
    if re.search(r'[-]\s*queue_ages_ms\[', src):                return 'subtract'
    if re.search(r'\bq\s*-\s*a\b|\bql\s*-\s*qa\b', src):       return 'subtract'
    if re.search(r'[-]\s*age\b', src) and not re.search(r'\b(?:min|max|best|current|old)_age\b', src):
        return 'subtract'
    if re.search(r'[-]\s*served\b', src):                       return 'subtract'
    if re.search(r'est\s*=\s*q\s*-\s*\w+', src) and re.search(r'served|completed', src):
        return 'subtract'
    if re.search(r'\(queue_lengths\[\w+\],\s*queue_ages_ms\[\w+\]\)', src): return 'tiebreaker'
    if re.search(r'key\s*=\s*lambda\s*\w+:\s*\(queue_lengths', src):        return 'tiebreaker'
    if re.search(r'\b(?:min_age|best_age)\s*=\s*float\(', src):             return 'tiebreaker'
    if re.search(r'elif.*queue_lengths\[\w+\]\s*==', src):                  return 'tiebreaker'
    if re.search(r'elif\s+\w+\s*==\s*\w+_queue\w*\b', src):                return 'tiebreaker'
    if re.search(r'and\s+\w+\s*<\s*(?:min_age|best_age)\b', src):          return 'tiebreaker'
    if re.search(r'ql\s*\+\s*\w*\s*\*?\s*qa\b', src):                      return 'add_positive'
    if re.search(r'q\s*\+\s*[\d.]+\s*\*\s*age\b', src):                    return 'add_positive'
    if re.search(r'ql\s*\+\s*\w+\s*\*\s*age\b', src):                      return 'add_positive'
    if re.search(r'\+\s*\d[\d.]*\s*\*\s*queue_ages_ms\[', src):            return 'add_positive'
    if re.search(r'\+\s*queue_ages_ms\[', src):                             return 'add_positive'
    if re.search(r'\+\s*\w+\s*\*\s*(?:queue_ages_ms|qa)\b', src):          return 'add_positive'
    if re.search(r'\+\s*age\b', src):                                        return 'add_positive'
    return 'other'


# --------------------------------------------------------------------------- #
# Results loader
# --------------------------------------------------------------------------- #
def load_all_results():
    results = {}
    for d in os.listdir(RESULTS_DIR):
        summary_path = os.path.join(RESULTS_DIR, d, 'summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            data = json.load(f)
        name = data['metadata']['model_name']
        results[name] = data
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print('Loading results...', flush=True)
    all_results = load_all_results()

    any_result = next(iter(all_results.values()))
    def baseline_raw(name):
        entry = any_result['results'].get(name, {}) or {}
        return [t['mean_sigma'] for t in (entry.get('per_trial_scalars') or []) if t]

    raw_baselines = {n: baseline_raw(n) for n in BASELINE_NAMES}
    bmed = {
        'Random':    statistics.median(raw_baselines['Random']),
        'Po2C':      statistics.median(raw_baselines['Po2C (stale)']),
        'JSQ-stale': statistics.median(raw_baselines['JSQ (stale)']),
    }

    model_rows = []
    for mname in MODEL_ORDER:
        if mname not in all_results:
            print(f'  WARNING: {mname} not found in results', flush=True)
            continue
        d         = all_results[mname]
        llm_entry = d['results'].get('LLM') or {}
        per_trial = llm_entry.get('per_trial_scalars') or []
        sigmas    = [t['mean_sigma'] for t in per_trial if t is not None]
        model_rows.append({
            'name':        mname,
            'label':       model_short(mname),
            'param_count': d['llm'].get('param_count'),
            'parse_rate':  llm_entry.get('parse_success_rate', 0.0),
            'sigmas':      sigmas,
        })
        print(f'  {mname}: {len(sigmas)} valid trials, '
              f'parse={llm_entry.get("parse_success_rate", 0):.0%}', flush=True)

    # ── Figure 1 ─────────────────────────────────────────────────────────── #
    print('\nFigure 1: JSQ-stale simulation...', flush=True)
    time_ms, worker_series = run_sim_worker_series(seed=0, max_jobs=30000)
    cutoff   = 400.0
    crop_idx = next((i for i, t in enumerate(time_ms) if t > cutoff), len(time_ms))
    save_json('fig1', {
        'text': {
            'title':  'JSQ-stale: per-worker queue depth over time (thundering herd)',
            'xlabel': 'Simulation time (ms)',
            'ylabel': 'Queue length (jobs)',
        },
        'time_ms':       time_ms[:crop_idx],
        'worker_series': worker_series[:crop_idx],
    })

    # ── Figure 2 ─────────────────────────────────────────────────────────── #
    print('Figure 2: baseline stats table...', flush=True)
    fig2_rows = []
    for display, raw_key, color_key in [
        ('Random',       'Random',       'random'),
        ('Po2C (stale)', 'Po2C (stale)', 'po2c'),
        ('JSQ (stale)',  'JSQ (stale)',  'jsq'),
    ]:
        v   = raw_baselines[raw_key]
        med = statistics.median(v)
        iqr = percentile(v, 75) - percentile(v, 25)
        std = statistics.stdev(v)
        fig2_rows.append({
            'strategy':  display,
            'color_key': color_key,
            'median':    round(med, 4),
            'iqr':       round(iqr, 4),
            'std':       round(std, 4),
            'min':       round(min(v), 4),
            'max':       round(max(v), 4),
        })
    save_json('fig2', {
        'text': {
            'title':      'Baseline performance statistics (n=100 seeds each)',
            'col_labels': ['Strategy', 'Median \u03c3', 'IQR', 'Std Dev', 'Min', 'Max'],
        },
        'rows': fig2_rows,
    })

    # ── Figure 3 ─────────────────────────────────────────────────────────── #
    print('Figure 3: model medians vs baselines...', flush=True)
    fig3_models = []
    for m in model_rows:
        if not m['sigmas']:
            continue
        fig3_models.append({
            'name':         m['name'],
            'label':        m['label'],
            'median_sigma': round(statistics.median(m['sigmas']), 4),
        })
    save_json('fig3', {
        'text': {
            'title':  'LLM median performance vs. baselines',
            'xlabel': 'Median mean \u03c3 across valid trials',
            'baseline_labels': {
                'Random':    'Random',
                'Po2C':      'Po2C',
                'JSQ-stale': 'JSQ-stale',
            },
            'shade_label': 'worse than Random',
        },
        'models': fig3_models,
        'baselines': {
            'Random':    round(bmed['Random'],    4),
            'Po2C':      round(bmed['Po2C'],      4),
            'JSQ-stale': round(bmed['JSQ-stale'], 4),
        },
    })

    # ── Figure 4 ─────────────────────────────────────────────────────────── #
    print('Figure 4: per-trial sigma distributions...', flush=True)
    fig4_models = []
    for m in model_rows:
        if not m['sigmas']:
            continue
        fig4_models.append({
            'name':   m['name'],
            'label':  m['label'],
            'sigmas': m['sigmas'],
            'median': round(statistics.median(m['sigmas']), 4),
            'n':      len(m['sigmas']),
        })
    save_json('fig4', {
        'text': {
            'title':  'Distribution of LLM trial outcomes (red line = median)',
            'ylabel': 'Per-trial mean \u03c3 (log scale)',
            'baseline_labels': {
                'Random':    'Random median',
                'JSQ-stale': 'JSQ-stale median',
            },
        },
        'models': fig4_models,
        'baseline_medians': {
            'Random':    round(bmed['Random'],    4),
            'JSQ-stale': round(bmed['JSQ-stale'], 4),
        },
    })

    # ── Figure 5 ─────────────────────────────────────────────────────────── #
    print('Figure 5: parse rates...', flush=True)
    fig5_models = []
    for m in model_rows:
        fig5_models.append({
            'name':       m['name'],
            'parse_rate': round(m['parse_rate'], 4),
            'n_valid':    len(m['sigmas']),
        })
    save_json('fig5', {
        'text': {
            'title':      'Parse success rate per model',
            'col_labels': ['Model', 'Parse Rate', 'Valid Trials'],
        },
        'models': fig5_models,
    })

    # ── Figure 6 ─────────────────────────────────────────────────────────── #
    print('Figure 6: code examples...', flush=True)
    fig6 = {
        'text': {
            'suptitle': 'Both models independently converge on: '
                        'adj\u00a0=\u00a0max(q\u00a0\u2212\u00a0age,\u00a00),\u00a0return\u00a0argmin(adj)',
        },
    }
    gpt_data = all_results.get('gpt-oss-20b')
    if gpt_data:
        t = gpt_data['llm']['trials'][53]
        p = gpt_data['results']['LLM']['per_trial_scalars'][53]
        fig6['gpt_oss_20b'] = {
            'source':      t.get('function_source', ''),
            'mean_sigma':  round(p['mean_sigma'], 4) if p else None,
            'model_label': 'gpt-oss-20b, trial 53',
        }
    qwen14_data = all_results.get('Qwen3-14B')
    if qwen14_data:
        t = qwen14_data['llm']['trials'][84]
        p = qwen14_data['results']['LLM']['per_trial_scalars'][84]
        fig6['qwen3_14b'] = {
            'source':      t.get('function_source', ''),
            'mean_sigma':  round(p['mean_sigma'], 4) if p else None,
            'model_label': 'Qwen3-14B, trial 84',
        }
    save_json('fig6', fig6)

    # ── Figure 7 ─────────────────────────────────────────────────────────── #
    print('Figure 7: strategy classification...', flush=True)
    by_class = {}
    for mname, d in all_results.items():
        if mname in EXCLUDED_MODELS:
            continue
        trials    = d['llm']['trials']
        per_trial = (d['results'].get('LLM') or {}).get('per_trial_scalars') or []
        for trial, perf in zip(trials, per_trial):
            if not trial.get('parse_success') or perf is None:
                continue
            cls = classify_strategy(trial.get('function_source')) or 'other'
            by_class.setdefault(cls, []).append(perf['mean_sigma'])

    fig7_classes = []
    for cls in CLASS_ORDER:
        if cls not in by_class:
            continue
        sigs = by_class[cls]
        med  = statistics.median(sigs)
        print(f'  {cls:15s} n={len(sigs):3d}  median={med:.1f}  '
              f'range=[{min(sigs):.1f}, {max(sigs):.1f}]')
        fig7_classes.append({
            'cls':    cls,
            'label':  CLASS_LABELS[cls],
            'sigmas': sigs,
            'median': round(med, 4),
            'n':      len(sigs),
        })
    save_json('fig7', {
        'text': {
            'title':  'Performance by strategy class (vertical line = median)',
            'xlabel': 'Per-trial mean \u03c3 (log scale)',
            'baseline_labels': {
                'Random':    'Random',
                'JSQ-stale': 'JSQ-stale',
            },
        },
        'classes': fig7_classes,
        'baseline_medians': {
            'Random':    round(bmed['Random'],    4),
            'JSQ-stale': round(bmed['JSQ-stale'], 4),
        },
    })

    # ── Figure 8 ─────────────────────────────────────────────────────────── #
    print('Figure 8: scale vs. performance...', flush=True)
    def family(name):
        if name.startswith('Qwen3'):  return 'Qwen3'
        if 'DeepSeek' in name:        return 'DeepSeek'
        if 'gpt-oss' in name:         return 'gpt-oss'
        return 'Other'

    fig8_models = []
    for m in model_rows:
        if not m['sigmas'] or m['param_count'] is None:
            continue
        short = m['name'].replace('Qwen3-', '').replace('DeepSeek-R1-0528-', 'R1-')
        fig8_models.append({
            'name':             m['name'],
            'point_label':      short,
            'param_count':      m['param_count'],
            'median_sigma':     round(statistics.median(m['sigmas']), 4),
            'family':           family(m['name']),
            'uncertain_params': m['name'] == 'gpt-oss-20b',
        })
    save_json('fig8', {
        'text': {
            'title':  'Model scale vs. load-balancing performance',
            'xlabel': 'Parameter count (log scale)',
            'ylabel': 'Median mean \u03c3 (log scale)',
            'baseline_labels': {
                'Random':    'Random',
                'JSQ-stale': 'JSQ-stale',
            },
            'family_labels': {
                'Qwen3':    'Qwen3',
                'DeepSeek': 'DeepSeek-R1',
                'gpt-oss':  'gpt-oss',
                'Other':    'Other',
            },
            'uncertain_note': '(param count uncertain)',
        },
        'models': fig8_models,
        'baseline_medians': {
            'Random':    round(bmed['Random'],    4),
            'JSQ-stale': round(bmed['JSQ-stale'], 4),
        },
    })

    print('\nDone.')


if __name__ == '__main__':
    main()
