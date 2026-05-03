#!/usr/bin/env bash
# Submit inference jobs for stale-state dispatcher experiment.
# One job per model; logs written to logs/<jobname>-<jobid>.out
#
# Usage: bash submit_slurm.sh

set -euo pipefail

OVERLAY=/scratch/kx2217/overlays/networks-project.ext3
IMAGE=/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif
WORKDIR=/scratch/kx2217/networks-project

LOGDIR="${WORKDIR}/logs"
mkdir -p "$LOGDIR"

submit() {
    local job_name="$1"
    local model="$2"

    echo "Submitting: ${job_name}  (${model})"
    sbatch \
        --nodes=1 \
        --mem=400G \
        --gres=gpu:1 \
        --cpus-per-task=16 \
        --ntasks-per-node=1 \
        --time="2:00:00" \
        --constraint="h200" \
        --account=torch_pr_221_courant \
        --job-name="${job_name}" \
        --output="${LOGDIR}/${job_name}-%j.out" \
        --wrap "singularity exec --nv --fakeroot --overlay ${OVERLAY}:ro ${IMAGE} /bin/bash -c 'source /ext3/env.sh && conda activate networks-project && cd ${WORKDIR} && HF_TOKEN=${HF_TOKEN} python experiment.py --models ${model} --n-trials 100'"
}

# ── Qwen3 base (v1 suite) ──────────────────────────────────────────────────── #
submit "qwen3-1.7b"  "Qwen/Qwen3-1.7B"
submit "qwen3-4b"    "Qwen/Qwen3-4B"
submit "qwen3-8b"    "Qwen/Qwen3-8B"
submit "qwen3-14b"   "Qwen/Qwen3-14B"

# ── Other families ─────────────────────────────────────────────────────────── #
submit "deepseek-r1-0528-8b"  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
submit "gpt-oss-20b"          "openai/gpt-oss-20b"

echo ""
echo "All 6 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Watch logs with:  tail -f ${LOGDIR}/<jobname>-<jobid>.out"
