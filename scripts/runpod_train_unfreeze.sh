#!/usr/bin/env bash
# Launch the "unfreeze" Turkish training approach on a RunPod pod.
#
# Trains voicepacks + predictor + text_encoder + bert_encoder (decoder frozen).
# Uses predicted F0/N — predictor learns Turkish prosody directly.
#
# Usage (run from repo root):
#   HF_TOKEN=hf_xxx bash scripts/runpod_train_unfreeze.sh
#
# Override any default with env vars, e.g.:
#   HF_TOKEN=hf_xxx SPEAKER=female_speaker MAX_ROWS=500 bash scripts/runpod_train_unfreeze.sh

set -euo pipefail

export APPROACH=unfreeze
export SPEAKER="${SPEAKER:-both}"
export MAX_ROWS="${MAX_ROWS:-0}"
export STAGE2_STEPS="${STAGE2_STEPS:-0}"

exec bash scripts/runpod_train_kokoro_turkish.sh "$@"
