#!/usr/bin/env python3
"""Stale-State Dispatcher Experiment"""

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
N_WORKERS          = 8
LAMBDA             = 6.4   # jobs / ms  (ρ = 0.8 per worker)
SERVICE_TIME       = 1.0   # mean ms per job — exponential distribution
N_JOBS             = 100_000
MEASURE_INTERVAL   = 10.0  # ms
HEARTBEAT_INTERVAL = 20.0  # ms between heartbeats per worker
PROP_DELAY_LO      = 1.0   # ms — heartbeat propagation delay bounds
PROP_DELAY_HI      = 5.0   # ms

# Event priorities (lower fires first on equal timestamps)
DEPARTURE = 0
HEARTBEAT = 1
ARRIVAL   = 2

LLM_SYSTEM_PROMPT = "You are an expert in distributed systems and load balancing."

LLM_USER_PROMPT = """\
Write the body of a dispatch function for a distributed system with {num_workers} workers.

System parameters:
- Poisson job arrivals, total utilisation ≈ 80%
- Exponential service times, mean 1 ms
- Each worker independently sends a heartbeat every ~20 ms; heartbeats arrive
  1–5 ms later, so each worker's reported state has a different age

Function signature:
  def dispatch(queue_lengths: list[int], queue_ages_ms: list[float], num_workers: int) -> int

  queue_lengths[w]   -- queue depth reported in worker w's last heartbeat
  queue_ages_ms[w]   -- ms elapsed since that heartbeat arrived
  num_workers        -- number of workers

Rules:
- Ages typically range from ~1 ms (just after a heartbeat) to ~25 ms (just before the next)
- Return the index of the worker to route the next job to
- Write ONLY the function body (no def line, no imports; `random` and `math` are available)
- Wrap the code in <code> and </code> tags
- No text outside the tags"""


# --------------------------------------------------------------------------- #
# Baseline dispatchers
# --------------------------------------------------------------------------- #
def dispatch_random(queue_lengths, queue_ages_ms, num_workers):
    return random.randrange(num_workers)


def dispatch_jsq(queue_lengths, queue_ages_ms, num_workers):
    min_q = min(queue_lengths)
    return random.choice([i for i, q in enumerate(queue_lengths) if q == min_q])


def dispatch_po2c(queue_lengths, queue_ages_ms, num_workers):
    a, b = random.sample(range(num_workers), 2)
    return a if queue_lengths[a] <= queue_lengths[b] else b


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #
def run_simulation(dispatch_fn, seed=42, n_jobs=N_JOBS, n_workers=N_WORKERS):
    """Run discrete-event simulation; return metrics dict."""
    random.seed(seed)                     # dispatcher randomness
    rng    = np.random.default_rng(seed)  # inter-arrival + service times
    hb_rng = random.Random(seed)          # heartbeat jitter — isolated

    queue_lengths  = [0]   * n_workers
    worker_free_at = [0.0] * n_workers

    # Per-worker stale state available to the dispatcher
    reported_queue = [0]                     * n_workers
    last_heard     = [-HEARTBEAT_INTERVAL]   * n_workers  # age = H at t=0

    sigma_series    = []
    jobs_per_worker = [0] * n_workers
    max_queue_seen  = 0
    next_measure    = MEASURE_INTERVAL

    heap = []
    # Stagger initial heartbeats evenly across the first interval
    for w in range(n_workers):
        t0 = (w + 1) * HEARTBEAT_INTERVAL / n_workers \
             + hb_rng.uniform(PROP_DELAY_LO, PROP_DELAY_HI)
        heapq.heappush(heap, (t0, HEARTBEAT, w))
    heapq.heappush(heap, (float(rng.exponential(1.0 / LAMBDA)), ARRIVAL, -1))

    jobs_dispatched = 0
    jobs_completed  = 0
    sim_end_time    = 0.0

    while heap:
        t, etype, widx = heapq.heappop(heap)
        sim_end_time = t

        while next_measure <= t:
            sigma_series.append(float(np.std(queue_lengths)))
            mq = max(queue_lengths)
            if mq > max_queue_seen:
                max_queue_seen = mq
            next_measure += MEASURE_INTERVAL

        if etype == DEPARTURE:
            queue_lengths[widx] = max(0, queue_lengths[widx] - 1)
            jobs_completed += 1

        elif etype == HEARTBEAT:
            reported_queue[widx] = queue_lengths[widx]
            last_heard[widx]     = t
            if jobs_dispatched < n_jobs:
                next_hb = t + HEARTBEAT_INTERVAL + hb_rng.uniform(PROP_DELAY_LO, PROP_DELAY_HI)
                heapq.heappush(heap, (next_hb, HEARTBEAT, widx))

        else:  # ARRIVAL
            if jobs_dispatched >= n_jobs:
                continue
            jobs_dispatched += 1

            ages   = [t - lh for lh in last_heard]
            worker = int(dispatch_fn(list(reported_queue), ages, n_workers)) % n_workers
            jobs_per_worker[worker] += 1

            queue_lengths[worker] += 1
            finish = max(t, worker_free_at[worker]) + float(rng.exponential(SERVICE_TIME))
            worker_free_at[worker] = finish
            heapq.heappush(heap, (finish, DEPARTURE, worker))

            if jobs_dispatched < n_jobs:
                heapq.heappush(heap, (t + float(rng.exponential(1.0 / LAMBDA)), ARRIVAL, -1))

    throughput = jobs_completed / sim_end_time if sim_end_time > 0 else 0.0
    sigma_arr  = np.array(sigma_series) if sigma_series else np.array([0.0])
    return {
        "mean_sigma":      float(np.mean(sigma_arr)),
        "p95_sigma":       float(np.percentile(sigma_arr, 95)),
        "throughput":      throughput,
        "max_queue":       max_queue_seen,
        "jobs_per_worker": jobs_per_worker,
        "sigma_series":    sigma_series,
    }


# --------------------------------------------------------------------------- #
# LLM dispatcher
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


def load_llm_dispatcher(model_path, n_trials, num_workers=N_WORKERS):
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
    """Extract and exec function body; return (callable, source)."""
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

    indented  = "\n".join(("    " + l) if l.strip() else "" for l in body.split("\n"))
    func_code = f"def dispatch(queue_lengths, queue_ages_ms, num_workers):\n{indented}\n"

    ns = {"random": random, "math": math}
    exec(func_code, ns)  # noqa: S102
    fn = ns.get("dispatch")
    if not callable(fn):
        raise ValueError("exec() did not produce a callable")
    return fn, func_code


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_results(per_trial, parse_success_rate=1.0):
    valid = [r for r in per_trial if r is not None]
    if not valid:
        return None
    agg = {"per_trial": per_trial, "parse_success_rate": parse_success_rate}
    for key in ("mean_sigma", "p95_sigma", "throughput", "max_queue"):
        vals = [r[key] for r in valid]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"]  = float(np.std(vals))
    return agg


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_table(agg_results, model_name, n_trials):
    print(f"\n(n={n_trials} trial{'s' if n_trials > 1 else ''})")
    header = (
        f"{'Dispatcher':<16} | {'mean(σ)':>7} {'±std':>6} | {'p95(σ)':>6} | "
        f"{'throughput':>10} | {'max_queue':>9} | {'parse%':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, r in zip(["Random", "JSQ (stale)", "Po2C (stale)", f"LLM ({model_name})"],
                       agg_results):
        if r is None:
            print(f"{name:<16} | {'FAIL':>36} |")
            continue
        rate_str = f"{r['parse_success_rate']*100:.0f}%" if r["parse_success_rate"] < 1.0 else "-"
        print(
            f"{name:<16} | {r['mean_sigma_mean']:>7.3f} ±{r['mean_sigma_std']:>5.3f} | "
            f"{r['p95_sigma_mean']:>6.3f} | {r['throughput_mean']:>10.4f} | "
            f"{r['max_queue_mean']:>9.1f} | {rate_str:>6}"
        )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
HEAVY_KEYS  = ("sigma_series",)
SCALAR_KEYS = ("mean_sigma", "p95_sigma", "throughput", "max_queue", "jobs_per_worker")


def save_results(run_dir, metadata, llm_info, results):
    os.makedirs(run_dir, exist_ok=True)
    summary, series = {}, {}
    for name, agg in results.items():
        if agg is None:
            summary[name] = series[name] = None
            continue
        per_trial      = agg.get("per_trial", [])
        summary[name]  = {k: v for k, v in agg.items() if k != "per_trial"}
        summary[name]["per_trial_scalars"] = [
            {k: t[k] for k in SCALAR_KEYS if k in t} if t else None
            for t in per_trial
        ]
        series[name] = [
            {k: t[k] for k in HEAVY_KEYS if k in t} if t else None
            for t in per_trial
        ]

    summary_path = os.path.join(run_dir, "summary.json")
    series_path  = os.path.join(run_dir, "series.json")
    with open(summary_path, "w") as f:
        json.dump({"metadata": metadata, "llm": llm_info, "results": summary}, f, indent=2)
    with open(series_path, "w") as f:
        json.dump(series, f)
    return summary_path, series_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Stale-State Dispatcher Experiment")
    parser.add_argument("--models", nargs="+", required=True, metavar="MODEL",
                        help="HuggingFace model paths")
    parser.add_argument("--n-trials", type=int, default=1, metavar="N",
                        help="LLM samples and simulation seeds per model (default: 1)")
    args = parser.parse_args()

    for model_path in args.models:
        model_name = os.path.basename(model_path.rstrip("/"))
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}", file=sys.stderr)

        # ── LLM inference ────────────────────────────────────────────────── #
        llm_fns    = [None] * args.n_trials
        llm_trials = []
        param_count = None
        try:
            responses, param_count = load_llm_dispatcher(model_path, args.n_trials)
            for i, resp in enumerate(responses):
                trial = {"raw_response": resp, "function_source": None,
                         "parse_success": False, "error": None}
                try:
                    fn, src = parse_llm_dispatcher(resp)
                    llm_fns[i] = fn
                    trial.update({"function_source": src, "parse_success": True})
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
                           "parse_success": False, "error": str(exc)}] * args.n_trials

        # ── Simulations ──────────────────────────────────────────────────── #
        results     = {}
        agg_results = []

        for name, fn in [("Random",       dispatch_random),
                         ("JSQ (stale)",  dispatch_jsq),
                         ("Po2C (stale)", dispatch_po2c)]:
            per_trial = []
            for seed in range(args.n_trials):
                print(f"Running {name} trial={seed} ...", file=sys.stderr)
                try:
                    per_trial.append(run_simulation(fn, seed=seed))
                except Exception as exc:
                    print(f"  {name} trial={seed} failed: {exc}", file=sys.stderr)
                    per_trial.append(None)
            agg = aggregate_results(per_trial)
            agg_results.append(agg)
            results[name] = agg

        llm_per_trial = []
        for i, fn in enumerate(llm_fns):
            if fn is None:
                llm_per_trial.append(None)
                continue
            print(f"Running LLM trial={i} ...", file=sys.stderr)
            try:
                llm_per_trial.append(run_simulation(fn, seed=i))
            except Exception as exc:
                print(f"  LLM trial={i} failed: {exc}", file=sys.stderr)
                llm_per_trial.append(None)
        agg = aggregate_results(llm_per_trial, parse_success_rate=parse_success_rate)
        agg_results.append(agg)
        results["LLM"] = agg

        print_table(agg_results, model_name, args.n_trials)

        # ── Save ─────────────────────────────────────────────────────────── #
        metadata = {
            "model":                 model_path,
            "model_name":            model_name,
            "n_trials":              args.n_trials,
            "n_jobs":                N_JOBS,
            "n_workers":             N_WORKERS,
            "lambda_per_ms":         LAMBDA,
            "service_time_mean_ms":  SERVICE_TIME,
            "service_time_dist":     "exponential",
            "heartbeat_interval_ms": HEARTBEAT_INTERVAL,
            "prop_delay_lo_ms":      PROP_DELAY_LO,
            "prop_delay_hi_ms":      PROP_DELAY_HI,
            "measure_interval_ms":   MEASURE_INTERVAL,
            "timestamp":             timestamp,
            "system_prompt":         LLM_SYSTEM_PROMPT,
            "user_prompt":           LLM_USER_PROMPT.format(num_workers=N_WORKERS),
        }
        llm_info = {"param_count": param_count, "trials": llm_trials}
        summary_path, series_path = save_results(
            os.path.join("results", f"{model_name}_{timestamp}"),
            metadata, llm_info, results,
        )
        print(f"\nSummary → {summary_path}", file=sys.stderr)
        print(f"Series  → {series_path}",  file=sys.stderr)


if __name__ == "__main__":
    main()
