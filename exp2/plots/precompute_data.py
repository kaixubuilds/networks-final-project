#!/usr/bin/env python3
"""
Precompute figure data for Experiment 2.

Modes:
  sweep      -- run noise-σ sweep simulations (no LLM; for fig1)
  aggregate  -- load results/ LLM run directories (for fig2–fig5)
  all        -- both

Outputs JSON files to plots/data/ for consumption by make_figures.py.

Usage:
    python precompute_data.py --mode sweep
    python precompute_data.py --mode aggregate --results-dir ../results
    python precompute_data.py --mode all --results-dir ../results
"""

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Allow importing from exp2/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2 import (
    run_simulation,
    dispatch_random,
    dispatch_srpt_noisy,
    dispatch_lc,
    NOISE_SIGMA,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Picklable worker functions (must be module-level for ProcessPoolExecutor)
# --------------------------------------------------------------------------- #
def _run_mean_rt(fn, oracle, seed, sigma):
    r = run_simulation(fn, seed=seed, noise_sigma=sigma, oracle=oracle)
    return r["mean_response_time"]


def _run_response_times(fn, oracle, seed, sigma):
    r = run_simulation(fn, seed=seed, noise_sigma=sigma, oracle=oracle,
                       store_response_times=True)
    return r["response_times"]


def _run_response_times_from_src(fn_src, seed, sigma):
    """Re-compile a dynamically-generated dispatch function in the worker process.

    exec'd functions aren't picklable, so we pass the source string instead
    and reconstruct the function here.
    """
    ns = {"random": random, "math": math, "np": np}
    exec(fn_src, ns)  # noqa: S102
    fn = ns["dispatch"]
    r = run_simulation(fn, seed=seed, noise_sigma=sigma, store_response_times=True)
    return r["response_times"]


def _run_mean_rt_from_src(fn_src, seed, sigma):
    ns = {"random": random, "math": math, "np": np}
    exec(fn_src, ns)  # noqa: S102
    fn = ns["dispatch"]
    r = run_simulation(fn, seed=seed, noise_sigma=sigma)
    return r["mean_response_time"]


# --------------------------------------------------------------------------- #
# Fig 1: noise sweep — mean response time vs. σ for baselines
# --------------------------------------------------------------------------- #
def run_noise_sweep(n_seeds=20, sigmas=None, workers=None):
    if sigmas is None:
        sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    dispatchers = [
        ("SRPT-oracle", None,                True),
        ("SRPT-noisy",  dispatch_srpt_noisy, False),
        ("LC",          dispatch_lc,         False),
        ("Random",      dispatch_random,     False),
    ]

    # Accumulate: name -> sigma -> [mean_rt, ...]
    acc = {name: {s: [] for s in sigmas} for name, _, _ in dispatchers}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_mean_rt, fn, oracle, seed, sigma): (name, sigma)
            for name, fn, oracle in dispatchers
            for sigma in sigmas
            for seed in range(n_seeds)
        }
        for fut in as_completed(futures):
            name, sigma = futures[fut]
            acc[name][sigma].append(fut.result())

    out = {
        "text": {
            "title":      "SRPT degrades with job-size estimation noise\n(M/G/1, Pareto α=1.5, ρ=0.8, 8 workers)",
            "xlabel":     "Log-normal noise σ",
            "ylabel":     "Median response time (ms)",
            "annotation": "Exp 2\nnoise σ",
        },
        "sigmas": sigmas,
        "dispatchers": {},
    }
    for name, _, _ in dispatchers:
        out["dispatchers"][name] = {
            "median_rt_per_sigma": [],
            "iqr25_rt_per_sigma":  [],
            "iqr75_rt_per_sigma":  [],
        }
        for sigma in sigmas:
            rts = acc[name][sigma]
            med = float(np.median(rts))
            out["dispatchers"][name]["median_rt_per_sigma"].append(med)
            out["dispatchers"][name]["iqr25_rt_per_sigma"].append(float(np.percentile(rts, 25)))
            out["dispatchers"][name]["iqr75_rt_per_sigma"].append(float(np.percentile(rts, 75)))
            print(f"  {name:16s}  σ={sigma:.2f}  median_rt={med:.3f}", flush=True)

    path = os.path.join(DATA_DIR, "fig1.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {path}")
    return out


# --------------------------------------------------------------------------- #
# Load and aggregate LLM results
# --------------------------------------------------------------------------- #
def load_all_results(results_dir):
    """
    Walk results_dir, load summary.json files.
    Returns list of (model_name, summary_dict) pairs.
    """
    runs = []
    if not os.path.isdir(results_dir):
        print(f"WARNING: results dir not found: {results_dir}", file=sys.stderr)
        return runs
    for entry in sorted(os.listdir(results_dir)):
        summary_path = os.path.join(results_dir, entry, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path) as f:
            data = json.load(f)
        runs.append((data["metadata"]["model_name"], data))
    print(f"Loaded {len(runs)} run(s) from {results_dir}")
    return runs


def collect_per_trial_scalars(runs, metric="mean_response_time"):
    """
    Returns dict: model_name -> list[float] (one per valid LLM trial across all seeds).
    Also returns baseline medians from the first run that has them.
    """
    model_trials = {}
    baseline_medians = {}
    baseline_runs = {}

    for model_name, data in runs:
        results = data.get("results", {})

        # Collect baselines from the first run that has each one.
        # Do NOT pool across model runs — the same simulation repeated N times
        # gives spurious confidence and can drift from the sweep's reference values.
        for bname in ("Random", "SRPT-noisy", "SRPT-oracle", "LC"):
            if bname in baseline_runs:
                continue   # already have it
            if bname in results and results[bname]:
                pts = results[bname].get("per_trial_scalars", [])
                vals = [t[metric] for t in pts if t and metric in t]
                if vals:
                    baseline_runs[bname] = vals

        # Collect LLM trials
        llm_pts = results.get("LLM", {})
        if llm_pts is None:
            continue
        llm_per_trial = llm_pts.get("per_trial_scalars", [])
        llm_trial_info = data.get("llm", {}).get("trials", [])

        for i, (scalar, trial) in enumerate(
            zip(llm_per_trial, llm_trial_info + [{}] * len(llm_per_trial))
        ):
            if scalar is None or metric not in scalar:
                continue
            if scalar.get("fallback_dispatch_rate", 0.0) == 1.0:
                continue
            if model_name not in model_trials:
                model_trials[model_name] = []
            model_trials[model_name].append({
                "value":           scalar[metric],
                "strategy_class":  trial.get("strategy_class"),
                "function_source": trial.get("function_source"),
                "parse_success":   trial.get("parse_success", False),
            })

    for bname, vals in baseline_runs.items():
        baseline_medians[bname] = float(np.median(vals))

    return model_trials, baseline_medians


# --------------------------------------------------------------------------- #
# Fig 2: main results — per-model response time distribution
# --------------------------------------------------------------------------- #
def compute_fig2_main_results(runs):
    model_trials, baseline_medians = collect_per_trial_scalars(runs)

    model_stats = {}
    for model_name, trials in model_trials.items():
        vals = [t["value"] for t in trials]
        if not vals:
            continue
        model_stats[model_name] = {
            "values":   vals,
            "median":   float(np.median(vals)),
            "mean":     float(np.mean(vals)),
            "q25":      float(np.percentile(vals, 25)),
            "q75":      float(np.percentile(vals, 75)),
            "min":      float(np.min(vals)),
            "max":      float(np.max(vals)),
            "n_trials": len(vals),
        }

    out = {
        "text": {
            "title":     "LLM-generated dispatchers vs. baselines\n(100 seeds per model, noise σ=1.0)",
            "ylabel":    "Mean response time (ms)",
            "llm_label": "LLM-generated (box = IQR, 100 seeds)",
        },
        "model_stats":      model_stats,
        "baseline_medians": baseline_medians,
        "metric":           "mean_response_time",
        "metric_label":     "Mean response time (ms)",
    }
    path = os.path.join(DATA_DIR, "fig2.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {path}")
    return out


# --------------------------------------------------------------------------- #
# Fig 3: strategy taxonomy — performance by strategy class
# --------------------------------------------------------------------------- #
def compute_fig3_strategy_taxonomy(runs):
    model_trials, baseline_medians = collect_per_trial_scalars(runs)

    strategy_groups = {}
    for model_name, trials in model_trials.items():
        for t in trials:
            sc = t.get("strategy_class") or "unknown"
            if sc not in strategy_groups:
                strategy_groups[sc] = []
            strategy_groups[sc].append(t["value"])

    strategy_stats = {}
    for sc, vals in strategy_groups.items():
        strategy_stats[sc] = {
            "values":   vals,
            "median":   float(np.median(vals)),
            "mean":     float(np.mean(vals)),
            "q25":      float(np.percentile(vals, 25)),
            "q75":      float(np.percentile(vals, 75)),
            "n_trials": len(vals),
        }

    out = {
        "text": {
            "title":  "Dispatch strategy class vs. performance\n(classified by signals used in LLM function)",
            "ylabel": "Mean response time (ms)",
        },
        "strategy_stats":    strategy_stats,
        "baseline_medians":  baseline_medians,
        "strategy_labels": {
            "srpt_only":    "SRPT-only\n(ignores connections)",
            "lc_only":      "LC-only\n(ignores estimates)",
            "both_signals": "Both signals\n(blend)",
            "random_like":  "Random-like\n(neither)",
            "unknown":      "Other",
        },
    }
    path = os.path.join(DATA_DIR, "fig3.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {path}")
    return out


# --------------------------------------------------------------------------- #
# Fig 4: per-model detail — distribution violin + strategy breakdown
# --------------------------------------------------------------------------- #
def compute_fig4_per_model_detail(runs):
    model_trials, baseline_medians = collect_per_trial_scalars(runs)

    model_detail = {}
    for model_name, trials in model_trials.items():
        by_class = {}
        for t in trials:
            sc = t.get("strategy_class") or "unknown"
            if sc not in by_class:
                by_class[sc] = []
            by_class[sc].append(t["value"])
        model_detail[model_name] = {
            "all_values":  [t["value"] for t in trials],
            "by_strategy": {sc: {"values": vals, "n": len(vals)}
                            for sc, vals in by_class.items()},
        }

    param_counts = {}
    for model_name, data in runs:
        pc = data.get("llm", {}).get("param_count")
        if pc:
            param_counts[model_name] = pc

    out = {
        "text": {
            "title":  "Per-model trial distribution and strategy breakdown",
            "ylabel": "Mean response time (ms)",
        },
        "model_detail":     model_detail,
        "param_counts":     param_counts,
        "baseline_medians": baseline_medians,
    }
    path = os.path.join(DATA_DIR, "fig4.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {path}")
    return out


# --------------------------------------------------------------------------- #
# Fig 5: response time CDF — run fresh sims for key strategies
# --------------------------------------------------------------------------- #
def compute_fig5_rt_cdf(runs, n_seeds=5, workers=None):
    strategies = [
        ("SRPT-oracle", None,                True),
        ("SRPT-noisy",  dispatch_srpt_noisy, False),
        ("LC",          dispatch_lc,         False),
        ("Random",      dispatch_random,     False),
    ]

    # Find best performing LLM function across all loaded runs
    best_fn_src   = None
    best_fn_model = "unknown"
    best_rt       = float("inf")
    for model_name, data in runs:
        llm_pts = data.get("results", {}).get("LLM")
        if llm_pts is None:
            continue
        per_trial = llm_pts.get("per_trial_scalars", [])
        llm_info  = data.get("llm", {}).get("trials", [])
        for scalar, trial in zip(per_trial, llm_info + [{}] * len(per_trial)):
            if scalar is None:
                continue
            rt = scalar.get("mean_response_time", float("inf"))
            if rt < best_rt and trial.get("function_source"):
                best_rt       = rt
                best_fn_src   = trial["function_source"]
                best_fn_model = model_name

    acc = {name: [] for name, _, _ in strategies}
    if best_fn_src:
        acc[f"Best LLM ({best_fn_model})"] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Baseline strategies: functions are module-level, so they're picklable.
        futures = {
            pool.submit(_run_response_times, fn, oracle, seed, NOISE_SIGMA): (name, seed)
            for name, fn, oracle in strategies
            for seed in range(n_seeds)
        }

        # Best LLM: pass source string; worker re-execs it to avoid pickling issue.
        if best_fn_src:
            llm_label = f"Best LLM ({best_fn_model})"
            for seed in range(n_seeds):
                fut = pool.submit(_run_response_times_from_src, best_fn_src, seed, NOISE_SIGMA)
                futures[fut] = (llm_label, seed)

        for fut in as_completed(futures):
            name, _ = futures[fut]
            acc[name].extend(fut.result())

    cdf_data = {}
    for name, rts in acc.items():
        arr = np.array(rts)
        cdf_data[name] = _compute_cdf_points(arr)
        suffix = f"  mean_rt={best_rt:.3f}" if "Best LLM" in name else ""
        print(f"  CDF: {name}  n={len(arr)}{suffix}")

    out = {
        "text": {
            "title":      f"Job response time distribution — Pareto service, noise σ={NOISE_SIGMA}",
            "xlabel":     "Response time (ms, log scale)",
            "ylabel":     "CDF",
            "annotation": "1 ms\n(mean)",
        },
        "cdf_data":   cdf_data,
        "noise_sigma": NOISE_SIGMA,
    }
    path = os.path.join(DATA_DIR, "fig5.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {path}")
    return out


# --------------------------------------------------------------------------- #
# Fig 6: best LLM functions — code display + robust performance comparison
# --------------------------------------------------------------------------- #
def compute_fig6_best_functions(runs, top_n=5, n_seeds=20, workers=None):
    import textwrap

    # Collect all valid non-fallback LLM trials across all runs
    all_trials = []
    for model_name, data in runs:
        llm = data.get("results", {}).get("LLM")
        if llm is None:
            continue
        pts = llm.get("per_trial_scalars", [])
        trial_info = data.get("llm", {}).get("trials", [])
        for scalar, trial in zip(pts, trial_info + [{}] * len(pts)):
            if scalar is None:
                continue
            if scalar.get("fallback_dispatch_rate", 0.0) == 1.0:
                continue
            src = trial.get("function_source", "")
            if not src:
                continue
            all_trials.append({
                "model":          model_name,
                "mean_rt_single": scalar["mean_response_time"],
                "strategy":       trial.get("strategy_class", "unknown"),
                "source":         src,
            })

    all_trials.sort(key=lambda x: x["mean_rt_single"])

    # Deduplicate by exact source, keep top_n unique functions
    seen = set()
    unique = []
    for t in all_trials:
        key = t["source"].strip()
        if key not in seen:
            seen.add(key)
            unique.append(t)
        if len(unique) >= top_n:
            break

    print(f"  Benchmarking top {len(unique)} unique functions over {n_seeds} seeds each...")

    # Robustly evaluate each function in parallel
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_mean_rt_from_src, t["source"], seed, NOISE_SIGMA): (i, seed)
            for i, t in enumerate(unique)
            for seed in range(n_seeds)
        }
        fn_acc = {i: [] for i in range(len(unique))}
        for fut in as_completed(futures):
            i, _ = futures[fut]
            try:
                fn_acc[i].append(fut.result())
            except Exception as e:
                print(f"  WARNING: fn {i} failed on a seed: {e}", file=sys.stderr)

    # Baselines over same n_seeds for a consistent comparison
    baselines = [
        ("SRPT-oracle", None,                True),
        ("SRPT-noisy",  dispatch_srpt_noisy, False),
        ("LC",          dispatch_lc,         False),
        ("Random",      dispatch_random,     False),
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        bfutures = {
            pool.submit(_run_mean_rt, fn, oracle, seed, NOISE_SIGMA): (name, seed)
            for name, fn, oracle in baselines
            for seed in range(n_seeds)
        }
        bacc = {name: [] for name, _, _ in baselines}
        for fut in as_completed(bfutures):
            name, _ = bfutures[fut]
            bacc[name].append(fut.result())

    baseline_stats = {
        name: {
            "median": float(np.median(vals)),
            "q25":    float(np.percentile(vals, 25)),
            "q75":    float(np.percentile(vals, 75)),
        }
        for name, vals in bacc.items()
    }

    # Build per-function records
    best_functions = []
    for i, t in enumerate(unique):
        rts = fn_acc[i]
        if not rts:
            continue

        # Clean source for figure display: normalize indentation, truncate long lines
        display_src = textwrap.dedent(t["source"]).strip()
        display_lines = []
        for line in display_src.split("\n"):
            display_lines.append(line[:76] + "…" if len(line) > 77 else line)
        display_src = "\n".join(display_lines)

        best_functions.append({
            "rank":           i + 1,
            "model":          t["model"],
            "strategy":       t["strategy"],
            "source":         t["source"],
            "display_source": display_src,
            "mean_rt_single": t["mean_rt_single"],
            "median_rt":      float(np.median(rts)),
            "q25_rt":         float(np.percentile(rts, 25)),
            "q75_rt":         float(np.percentile(rts, 75)),
        })
        print(f"  Rank {i+1}: {t['model']:30s}  median_rt={np.median(rts):.3f}")

    out = {
        "text": {
            "title": "Best LLM-discovered dispatch functions vs. baselines",
            "ylabel": "Median mean response time (ms)",
            "note":  f"Robust evaluation: {n_seeds} seeds each",
        },
        "best_functions": best_functions,
        "baseline_stats": baseline_stats,
        "n_seeds":        n_seeds,
    }
    path = os.path.join(DATA_DIR, "fig6.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {path}")
    return out


def _compute_cdf_points(arr, n_points=2000):
    """Return (x, y) for a CDF plot, downsampled to n_points."""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    if n <= n_points:
        xs = sorted_arr
        ys = np.arange(1, n + 1) / n
    else:
        idx = np.linspace(0, n - 1, n_points, dtype=int)
        xs = sorted_arr[idx]
        ys = (idx + 1) / n
    return {"x": [float(v) for v in xs], "y": [float(v) for v in ys]}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Precompute Exp2 figure data")
    parser.add_argument("--mode", choices=["sweep", "aggregate", "all"], default="all",
                        help="What to compute (default: all)")
    parser.add_argument("--results-dir", default=os.path.join(
                            os.path.dirname(os.path.dirname(__file__)), "results"),
                        help="Path to exp2/results/ directory")
    parser.add_argument("--sweep-seeds", type=int, default=50,
                        help="Seeds per σ value for noise sweep (default: 50)")
    parser.add_argument("--cdf-seeds", type=int, default=5,
                        help="Seeds for CDF figure (default: 5)")
    parser.add_argument("--best-seeds", type=int, default=20,
                        help="Seeds per function for best-functions figure (default: 20)")
    parser.add_argument("--best-n", type=int, default=5,
                        help="Number of unique top functions to benchmark (default: 5)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: all CPUs)")
    args = parser.parse_args()

    runs = []
    if args.mode in ("aggregate", "all"):
        runs = load_all_results(args.results_dir)

    if args.mode in ("sweep", "all"):
        print("\n── Fig 1: noise sweep ──────────────────────────────────────")
        run_noise_sweep(n_seeds=args.sweep_seeds, workers=args.workers)

    if args.mode in ("aggregate", "all") and runs:
        print("\n── Fig 2: main results ─────────────────────────────────────")
        compute_fig2_main_results(runs)

        print("\n── Fig 3: strategy taxonomy ────────────────────────────────")
        compute_fig3_strategy_taxonomy(runs)

        print("\n── Fig 4: per-model detail ─────────────────────────────────")
        compute_fig4_per_model_detail(runs)

        print("\n── Fig 5: response time CDF ────────────────────────────────")
        compute_fig5_rt_cdf(runs, n_seeds=args.cdf_seeds, workers=args.workers)

        print("\n── Fig 6: best functions ───────────────────────────────────")
        compute_fig6_best_functions(runs, top_n=args.best_n,
                                    n_seeds=args.best_seeds, workers=args.workers)

    elif args.mode in ("aggregate", "all") and not runs:
        print("No LLM runs found; skipping figs 2–5. Run experiment2.py first.")


if __name__ == "__main__":
    main()
