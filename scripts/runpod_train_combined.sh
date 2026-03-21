#!/usr/bin/env bash
# Launch the "combined" Turkish training run on a RunPod pod.
#
# Initializes from a gt_bootstrap checkpoint (good intelligibility), then trains
# with the unfreeze config (voicepack + predictor + text_encoder + bert_encoder)
# to add voice quality on top.
#
# Usage (run from repo root):
#   INIT_CHECKPOINT=/path/to/checkpoint_step_11999.pt \
#   HF_TOKEN=hf_xxx \
#   bash scripts/runpod_train_combined.sh
#
# INIT_CHECKPOINT can be a local path — the script will upload it to the pod.
# Or if it's already on the pod, set INIT_CHECKPOINT_REMOTE instead.
#
# Optional overrides:
#   SPEAKER=both MAX_ROWS=0 BATCH_SIZE=2 bash scripts/runpod_train_combined.sh

set -euo pipefail

SSH_TARGET="${SSH_TARGET:?Set SSH_TARGET=root@<host>}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# Load HF_TOKEN from .env if not set
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [[ -z "${HF_TOKEN:-}" && -f "$ENV_FILE" ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "$ENV_FILE" | head -1 | sed 's/HF_TOKEN=//;s/"//g')"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required." >&2; exit 1
fi

# Upload local checkpoint to pod if INIT_CHECKPOINT is a local file
REMOTE_CKPT="${INIT_CHECKPOINT_REMOTE:-}"
if [[ -n "${INIT_CHECKPOINT:-}" && -f "$INIT_CHECKPOINT" ]]; then
  REMOTE_CKPT="~/gt_bootstrap_init.pt"
  echo "==> uploading $INIT_CHECKPOINT → pod:$REMOTE_CKPT"
  scp -i "$SSH_KEY" -P "$SSH_PORT" "$INIT_CHECKPOINT" "${SSH_TARGET}:${REMOTE_CKPT}"
fi
if [[ -z "$REMOTE_CKPT" ]]; then
  echo "Set INIT_CHECKPOINT=/path/to/checkpoint.pt or INIT_CHECKPOINT_REMOTE=~/path/on/pod" >&2
  exit 1
fi

export APPROACH=combined
export SPEAKER="${SPEAKER:-both}"
export MAX_ROWS="${MAX_ROWS:-0}"
export STAGE2_STEPS="${STAGE2_STEPS:-0}"
export INIT_CHECKPOINT="$REMOTE_CKPT"

ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_TARGET" << ENDSSH
set -e
cd ~/reverse_eng
git pull
export CUDA_VISIBLE_DEVICES=0
nohup env CUDA_VISIBLE_DEVICES=0 \
  HF_TOKEN=$HF_TOKEN \
  APPROACH=combined \
  SPEAKER=${SPEAKER} \
  MAX_ROWS=${MAX_ROWS} \
  STAGE2_STEPS=0 \
  INIT_CHECKPOINT=$REMOTE_CKPT \
  bash scripts/runpod_train_kokoro_turkish.sh \
  </dev/null > ~/combined.log 2>&1 &
echo "launched pid=\$!"
sleep 5
tail -20 ~/combined.log
ENDSSH

echo ""
echo "==> training launched. Watch with:"
echo "    ssh -i $SSH_KEY -p $SSH_PORT $SSH_TARGET 'tail -f ~/combined.log'"
