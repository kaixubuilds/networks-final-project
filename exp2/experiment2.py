#!/usr/bin/env python3
"""
Experiment 2: SRPT Dispatch Under Noisy Job Size Estimates

Algorithm: SRPT-dispatch (Schrage 1968) — provably optimal for M/G/1 dispatching
           when job sizes are known exactly.
Setting change: Job sizes estimated with log-normal noise (σ=1.0) instead of known exactly.
LLM task: Given SRPT code + noisy estimates + exact connection counts, adapt SRPT to be
          robust to estimation error.

The correct adaptation requires blending estimated_remaining_work (noisy) with connections
(exact) as a sanity check. This is not a named algorithm — requires genuine reasoning.
"""

import argparse
import datetime
import heapq
import json
import math
import os
import random
import re
import sys
import textwrap

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N_WORKERS        = 8
LAMBDA           = 6.4          # jobs/ms  (ρ = 0.8 per worker with mean service = 1ms)
PARETO_ALPHA     = 1.5          # shape parameter (α > 1 → finite mean)
PARETO_MEAN      = 1.0          # mean service time ms
PARETO_XM        = PARETO_MEAN * (PARETO_ALPHA - 1.0) / PARETO_ALPHA  # ≈ 0.333 ms scale
NOISE_SIGMA      = 1.0          # log-normal noise std (σ=1 → CV ≈ 1.31)
N_JOBS           = 100_000
N_SEEDS          = 100
MEASURE_INTERVAL = 10.0         # ms between σ snapshots

# Event priorities (lower value fires first at equal timestamps)
DEPARTURE = 0
ARRIVAL   = 1

LLM_SYSTEM_PROMPT = "You are an expert in distributed systems and load balancing."

LLM_USER_PROMPT = """\
You are improving a job dispatcher for a server farm with {num_workers} workers.

The current dispatcher uses SRPT (Shortest Remaining Processing Time): it routes each
incoming job to the worker with the smallest estimated total remaining work:

    # baseline — the function you are replacing
    scores = [estimated_remaining_work[w] + job_size_estimate
              for w in range(num_workers)]
    min_s = min(scores)
    return random.choice([w for w, s in enumerate(scores) if s == min_s])

System parameters:
- Poisson arrivals, total utilisation ≈ 80%, {num_workers} workers
- Job service times follow a Pareto distribution (α=1.5, mean ≈ 1 ms) — heavy-tailed,
  high variance, many large jobs
- Job sizes are estimated with log-normal noise (σ=1.0): estimates can be off by 10×

The failure mode of the current dispatcher:
- When a large job (say, 50 ms) is severely underestimated at arrival (say, 0.3 ms),
  estimated_remaining_work for that worker stays near 0 for the entire 50 ms service.
- estimated_remaining_work[w] is a frozen accumulated sum: it is increased by the noisy
  estimate when a job ARRIVES and decreased by that same estimate when the job DEPARTS.
  It does NOT decrease as a job runs — so a wildly underestimated job makes the worker
  look nearly empty throughout its real service time.
- Result: more and more jobs get routed to the secretly-overloaded worker.

Your task: write an improved dispatch function body that reduces mean job response time.

Function signature:
    def dispatch(connections: list[int], estimated_remaining_work: list[float],
                 job_size_estimate: float, num_workers: int) -> int

    connections[w]              -- EXACT count of active jobs at worker w (no noise, no delay)
    estimated_remaining_work[w] -- noisy accumulated estimate of remaining work at w (ms)
                                   (mean job size ≈ 1 ms; estimates are biased and unreliable)
    job_size_estimate           -- noisy estimate of the incoming job's size (ms)
    num_workers                 -- number of workers

Rules:
- Return the index (0 to num_workers-1) of the worker to receive this job
- Write ONLY the function body (no def line, no imports)
- `random`, `math`, and `numpy` (as `np`) are available
- Wrap your solution code in <code> and </code> tags"""


# --------------------------------------------------------------------------- #
# Baseline dispatchers
# --------------------------------------------------------------------------- #
def dispatch_random(connections, estimated_remaining_work, job_size_estimate, num_workers):
    return random.randrange(num_workers)


def dispatch_srpt_noisy(connections, estimated_remaining_work, job_size_estimate, num_workers):
    """SRPT using noisy estimates — the established algorithm being tested."""
    scores = [estimated_remaining_work[w] + job_size_estimate for w in range(num_workers)]
    min_s = min(scores)
    return random.choice([w for w, s in enumerate(scores) if s == min_s])


def dispatch_lc(connections, estimated_remaining_work, job_size_estimate, num_workers):
    """Least Connections — exact signal, ignores estimates entirely."""
    min_c = min(connections)
    return random.choice([w for w, c in enumerate(connections) if c == min_c])


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #
def run_simulation(dispatch_fn, seed=42, n_jobs=N_JOBS, n_workers=N_WORKERS,
                   noise_sigma=NOISE_SIGMA, oracle=False, store_response_times=False):
    """
    Discrete-event M/G/1 simulation with Pareto service times and log-normal noise.

    Args:
        dispatch_fn:          callable(connections, erw, job_size_est, num_workers) -> int
        oracle:               if True, bypass dispatch_fn and use true remaining work
        store_response_times: if True, include full response_times list in result (heavy)
    Returns:
        dict of metrics
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)

    connections              = [0]   * n_workers
    estimated_remaining_work = [0.0] * n_workers
    true_remaining_work      = [0.0] * n_workers
    worker_free_at           = [0.0] * n_workers

    # Per-job size estimates stored for ERW correction on departure
    job_estimated_sizes = {}   # job_id -> estimated_service_time
    job_true_sizes      = {}   # job_id -> true_service_time

    response_times  = []
    sigma_series    = []
    jobs_per_worker = [0] * n_workers
    next_measure    = MEASURE_INTERVAL

    jobs_dispatched     = 0
    jobs_completed      = 0
    job_id_counter      = 0
    fallback_dispatches = 0
    sim_end_time        = 0.0

    # heap event: (time, etype, worker, job_id)
    heap = []
    heapq.heappush(heap, (float(rng.exponential(1.0 / LAMBDA)), ARRIVAL, -1, -1))

    while heap:
        t, etype, worker, job_id = heapq.heappop(heap)
        sim_end_time = t

        while next_measure <= t:
            sigma_series.append(float(np.std(connections)))
            next_measure += MEASURE_INTERVAL

        if etype == DEPARTURE:
            jobs_completed += 1
            connections[worker] -= 1
            est = job_estimated_sizes.pop(job_id, 0.0)
            tru = job_true_sizes.pop(job_id, 0.0)
            estimated_remaining_work[worker] = max(0.0, estimated_remaining_work[worker] - est)
            true_remaining_work[worker]      = max(0.0, true_remaining_work[worker]      - tru)

        else:  # ARRIVAL
            if jobs_dispatched >= n_jobs:
                continue
            jobs_dispatched += 1

            # Sample true service time (Pareto with shape α, scale x_m)
            true_service_time = PARETO_XM * (1.0 + float(rng.pareto(PARETO_ALPHA)))

            # Sample noisy estimate: true_size * exp(N(0, σ))
            if noise_sigma > 0:
                estimated_service_time = true_service_time * math.exp(
                    float(rng.normal(0.0, noise_sigma))
                )
            else:
                estimated_service_time = true_service_time

            # Dispatch decision
            if oracle:
                # Use exact remaining queue time (worker_free_at - t), not accumulated sizes.
                # This correctly accounts for partial completion of the currently running job.
                scores = [max(0.0, worker_free_at[w] - t) + true_service_time
                          for w in range(n_workers)]
                min_s = min(scores)
                w_idx = random.choice([w for w, s in enumerate(scores) if s == min_s])
            else:
                try:
                    w_idx = int(dispatch_fn(
                        list(connections),
                        list(estimated_remaining_work),
                        estimated_service_time,
                        n_workers,
                    )) % n_workers
                except Exception:
                    w_idx = random.randrange(n_workers)
                    fallback_dispatches += 1

            jobs_per_worker[w_idx] += 1

            # Finish time is deterministic once service time is sampled
            finish = max(t, worker_free_at[w_idx]) + true_service_time
            worker_free_at[w_idx] = finish
            response_times.append(finish - t)

            # Update running state
            jid = job_id_counter
            job_id_counter += 1
            connections[w_idx]              += 1
            job_estimated_sizes[jid]         = estimated_service_time
            job_true_sizes[jid]              = true_service_time
            estimated_remaining_work[w_idx] += estimated_service_time
            true_remaining_work[w_idx]      += true_service_time

            heapq.heappush(heap, (finish, DEPARTURE, w_idx, jid))

            if jobs_dispatched < n_jobs:
                heapq.heappush(heap, (
                    t + float(rng.exponential(1.0 / LAMBDA)), ARRIVAL, -1, -1
                ))

    rt  = np.array(response_times, dtype=np.float64) if response_times else np.array([0.0])
    sig = np.array(sigma_series,   dtype=np.float64) if sigma_series   else np.array([0.0])
    result = {
        "mean_response_time":   float(np.mean(rt)),
        "p95_response_time":    float(np.percentile(rt, 95)),
        "p99_response_time":    float(np.percentile(rt, 99)),
        "mean_sigma":           float(np.mean(sig)),
        "p95_sigma":            float(np.percentile(sig, 95)),
        "throughput":           float(jobs_completed / sim_end_time) if sim_end_time > 0 else 0.0,
        "jobs_per_worker":      jobs_per_worker,
        "fallback_dispatch_rate": (fallback_dispatches / jobs_dispatched
                                   if jobs_dispatched > 0 else 0.0),
    }
    if store_response_times:
        result["response_times"] = [float(x) for x in rt]
    return result


# --------------------------------------------------------------------------- #
# LLM inference & parsing
# --------------------------------------------------------------------------- #
def get_param_count(model_path):
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(cfg, "num_parameters"):
            return cfg.num_parameters
        h      = getattr(cfg, "hidden_size",       None)
        layers = getattr(cfg, "num_hidden_layers", None)
        vocab  = getattr(cfg, "vocab_size",        None)
        ffn    = getattr(cfg, "intermediate_size", None)
        if all(v is not None for v in (h, layers, vocab, ffn)):
            return vocab * h + layers * (4 * h * h + 3 * h * ffn)
    except Exception:
        pass
    return None


def load_llm_responses(model_path, n_trials, num_workers=N_WORKERS):
    """Sample n_trials completions in one batch; return (responses, param_count)."""
    from vllm import LLM, SamplingParams

    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user",   "content": LLM_USER_PROMPT.format(num_workers=num_workers)},
    ]
    print(f"Loading model from {model_path} ...", file=sys.stderr)
    llm     = LLM(model=model_path)
    outputs = llm.chat([messages], SamplingParams(
        n=n_trials, max_tokens=8192, temperature=1, top_p=1.0, top_k=-1
    ))
    return [o.text for o in outputs[0].outputs], get_param_count(model_path)


def parse_llm_dispatcher(response):
    """Extract function body from LLM response; return (callable, source)."""
    text = response.strip()
    m = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    lines = text.split("\n")
    if lines and re.match(r"\s*def\s+dispatch\s*\(", lines[0]):
        lines = lines[1:]

    body = textwrap.dedent("\n".join(lines))
    if not body.strip():
        raise ValueError("Empty function body")

    indented = "\n".join(("    " + l) if l.strip() else "" for l in body.split("\n"))
    func_code = (
        "def dispatch(connections, estimated_remaining_work, "
        "job_size_estimate, num_workers):\n"
        f"{indented}\n"
    )

    ns = {"random": random, "math": math, "np": np}
    exec(func_code, ns)  # noqa: S102
    fn = ns.get("dispatch")
    if not callable(fn):
        raise ValueError("exec() did not produce a callable")
    return fn, func_code


# --------------------------------------------------------------------------- #
# Strategy classification
# --------------------------------------------------------------------------- #
def classify_strategy(source_code):
    """
    Classify a parsed dispatch function by which signals it uses.

    Returns one of:
      srpt_only      — uses estimated_remaining_work, not connections
      lc_only        — uses connections, not estimated_remaining_work
      both_signals   — uses both
      random_like    — uses neither (or trivially random)
    """
    src = source_code or ""
    # Search only the function body (skip the def line which lists both as params)
    lines = src.split("\n")
    body = "\n".join(lines[1:]) if lines and re.match(r"\s*def\s+dispatch\s*\(", lines[0]) else src
    uses_erw  = bool(re.search(r"\bestimated_remaining_work\b", body))
    uses_conn = bool(re.search(r"\bconnections\b", body))
    if uses_erw and uses_conn:
        return "both_signals"
    if uses_erw:
        return "srpt_only"
    if uses_conn:
        return "lc_only"
    return "random_like"


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
SCALAR_KEYS = ("mean_response_time", "p95_response_time", "p99_response_time",
               "mean_sigma", "p95_sigma", "throughput", "fallback_dispatch_rate")
HEAVY_KEYS  = ("response_times",)


MAX_FALLBACK_RATE = 0.05   # discard trials where >5% of dispatches fell back to random


def aggregate_results(per_trial, parse_success_rate=1.0):
    valid = [r for r in per_trial
             if r is not None
             and r.get("fallback_dispatch_rate", 0.0) <= MAX_FALLBACK_RATE]
    if not valid:
        return None
    n_discarded = sum(1 for r in per_trial
                      if r is not None
                      and r.get("fallback_dispatch_rate", 0.0) > MAX_FALLBACK_RATE)
    agg = {
        "per_trial":          per_trial,
        "parse_success_rate": parse_success_rate,
        "n_discarded_fallback": n_discarded,
    }
    for key in SCALAR_KEYS:
        vals = [r[key] for r in valid if key in r]
        if vals:
            agg[f"{key}_mean"]   = float(np.mean(vals))
            agg[f"{key}_median"] = float(np.median(vals))
            agg[f"{key}_std"]    = float(np.std(vals))
            agg[f"{key}_q25"]    = float(np.percentile(vals, 25))
            agg[f"{key}_q75"]    = float(np.percentile(vals, 75))
    return agg


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_results(run_dir, metadata, llm_info, results):
    os.makedirs(run_dir, exist_ok=True)
    summary, series = {}, {}
    for name, agg in results.items():
        if agg is None:
            summary[name] = series[name] = None
            continue
        per_trial = agg.get("per_trial", [])
        summary[name] = {k: v for k, v in agg.items() if k != "per_trial"}
        summary[name]["per_trial_scalars"] = [
            {k: t[k] for k in SCALAR_KEYS if k in t} if t else None
            for t in per_trial
        ]
        series[name] = [
            ({k: t[k] for k in HEAVY_KEYS if k in t} if t else None)
            for t in per_trial
        ]

    summary_path = os.path.join(run_dir, "summary.json")
    series_path  = os.path.join(run_dir, "series.json")
    with open(summary_path, "w") as f:
        json.dump({"metadata": metadata, "llm": llm_info, "results": summary}, f, indent=2)
    with open(series_path, "w") as f:
        json.dump(series, f)
    return summary_path, series_path


def print_table(results_dict, n_trials):
    header = (
        f"{'Dispatcher':<24} | {'med(RT)':>8} {'[q25–q75]':>12} | "
        f"{'p95(RT)':>7} | {'med(σ)':>7} | {'parse%':>6} | {'discard':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(f"  n_trials={n_trials}  |  RT = response time (ms, median across seeds)  |  σ = queue std-dev")
    print(sep)
    print(header)
    print(sep)
    for name, agg in results_dict.items():
        if agg is None:
            print(f"{name:<24} | {'FAIL':>50} |")
            continue
        rate_str = (f"{agg['parse_success_rate']*100:.0f}%"
                    if agg.get("parse_success_rate", 1.0) < 1.0 else "-")
        nd = agg.get("n_discarded_fallback", 0)
        discard_str = f"{nd}" if nd > 0 else "-"
        q25 = agg.get("mean_response_time_q25", float("nan"))
        q75 = agg.get("mean_response_time_q75", float("nan"))
        print(
            f"{name:<24} | "
            f"{agg.get('mean_response_time_median', float('nan')):>8.3f} "
            f"[{q25:>5.2f}–{q75:<5.2f}] | "
            f"{agg.get('p95_response_time_median', float('nan')):>7.3f} | "
            f"{agg.get('mean_sigma_median', float('nan')):>7.3f} | "
            f"{rate_str:>6} | {discard_str:>7}"
        )
    print(sep)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: SRPT adaptation under noisy job size estimates"
    )
    parser.add_argument("--models", nargs="+", required=True, metavar="MODEL",
                        help="HuggingFace model paths")
    parser.add_argument("--n-trials", type=int, default=N_SEEDS, metavar="N",
                        help=f"LLM samples and simulation seeds per model (default: {N_SEEDS})")
    parser.add_argument("--noise-sigma", type=float, default=NOISE_SIGMA,
                        help=f"Log-normal noise σ (default: {NOISE_SIGMA})")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="Directory to write results (default: results/)")
    args = parser.parse_args()

    for model_path in args.models:
        model_name = os.path.basename(model_path.rstrip("/"))
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}", file=sys.stderr)

        # ── LLM inference ──────────────────────────────────────────────── #
        llm_fns    = [None] * args.n_trials
        llm_trials = []
        param_count = None
        try:
            responses, param_count = load_llm_responses(model_path, args.n_trials)
            for i, resp in enumerate(responses):
                trial = {
                    "raw_response":    resp,
                    "function_source": None,
                    "parse_success":   False,
                    "strategy_class":  None,
                    "error":           None,
                }
                try:
                    fn, src = parse_llm_dispatcher(resp)
                    llm_fns[i] = fn
                    trial.update({
                        "function_source": src,
                        "parse_success":   True,
                        "strategy_class":  classify_strategy(src),
                    })
                except Exception as exc:
                    trial["error"] = str(exc)
                llm_trials.append(trial)
            n_ok = sum(t["parse_success"] for t in llm_trials)
            parse_success_rate = n_ok / args.n_trials
            print(f"{n_ok}/{args.n_trials} LLM samples parsed successfully.", file=sys.stderr)
        except Exception as exc:
            print(f"LLM inference failed: {exc}", file=sys.stderr)
            parse_success_rate = 0.0
            llm_trials = [{"raw_response": None, "function_source": None,
                           "parse_success": False, "strategy_class": None,
                           "error": str(exc)}
                          for _ in range(args.n_trials)]

        # ── Baselines ──────────────────────────────────────────────────── #
        results = {}
        print("Running baselines ...", file=sys.stderr)

        for name, fn, is_oracle in [
            ("Random",       dispatch_random,     False),
            ("SRPT-noisy",   dispatch_srpt_noisy, False),
            ("SRPT-oracle",  None,                True),
            ("LC",           dispatch_lc,         False),
        ]:
            per_trial = []
            for seed in range(args.n_trials):
                print(f"  {name} seed={seed} ...", file=sys.stderr, end="\r")
                try:
                    per_trial.append(run_simulation(
                        fn, seed=seed, noise_sigma=args.noise_sigma, oracle=is_oracle,
                        store_response_times=(seed == 0),
                    ))
                except Exception as exc:
                    print(f"\n  {name} seed={seed} failed: {exc}", file=sys.stderr)
                    per_trial.append(None)
            print(f"  {name} done.        ", file=sys.stderr)
            results[name] = aggregate_results(per_trial)

        # ── LLM simulations ────────────────────────────────────────────── #
        llm_per_trial = []
        for i, fn in enumerate(llm_fns):
            if fn is None:
                llm_per_trial.append(None)
                continue
            print(f"  LLM trial={i} ...", file=sys.stderr, end="\r")
            try:
                llm_per_trial.append(run_simulation(
                    fn, seed=i, noise_sigma=args.noise_sigma,
                    store_response_times=(i == 0),
                ))
            except Exception as exc:
                print(f"\n  LLM trial={i} failed: {exc}", file=sys.stderr)
                llm_per_trial.append(None)
        print(f"  LLM done.      ", file=sys.stderr)
        results["LLM"] = aggregate_results(llm_per_trial,
                                           parse_success_rate=parse_success_rate)

        print_table(results, args.n_trials)

        # ── Save ────────────────────────────────────────────────────────── #
        metadata = {
            "model":               model_path,
            "model_name":          model_name,
            "n_trials":            args.n_trials,
            "n_jobs":              N_JOBS,
            "n_workers":           N_WORKERS,
            "lambda_per_ms":       LAMBDA,
            "pareto_alpha":        PARETO_ALPHA,
            "pareto_xm":           PARETO_XM,
            "pareto_mean_ms":      PARETO_MEAN,
            "noise_sigma":         args.noise_sigma,
            "measure_interval_ms": MEASURE_INTERVAL,
            "timestamp":           timestamp,
            "system_prompt":       LLM_SYSTEM_PROMPT,
            "user_prompt":         LLM_USER_PROMPT.format(num_workers=N_WORKERS),
        }
        llm_info = {"param_count": param_count, "trials": llm_trials}
        run_dir = os.path.join(args.out_dir, f"{model_name}_{timestamp}")
        summary_path, series_path = save_results(run_dir, metadata, llm_info, results)
        print(f"\nSummary → {summary_path}", file=sys.stderr)
        print(f"Series  → {series_path}",   file=sys.stderr)


if __name__ == "__main__":
    main()
