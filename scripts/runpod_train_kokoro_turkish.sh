#!/usr/bin/env bash
set -euo pipefail

# Run from repo root after cloning reverse_eng onto a CUDA RunPod machine.
#
# Example:
#   HF_TOKEN=hf_xxx SPEAKER=female_speaker MAX_ROWS=0 BATCH_SIZE=2 \
#   bash scripts/runpod_train_kokoro_turkish.sh
#
# Notes:
# - Kokoro voicepacks support lengths up to 510, but training on full-length
#   utterances near that limit is usually a VRAM problem. Keep MAX_PHONEMES
#   below 510 unless you add segment cropping/bucketing.
# - MAX_ROWS=0 means "use the full speaker subset".

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

set_default() {
  local var_name="$1"
  local default_value="$2"
  if [[ -z "${!var_name+x}" ]]; then
    printf -v "$var_name" "%s" "$default_value"
  fi
}

set_default PYTHON_BIN python3
set_default VENV_DIR .venv-runpod
set_default DEVICE cuda
set_default SPEAKER female_speaker
set_default APPROACH direct
set_default MAX_ROWS 0
set_default NUM_WORKERS 0
set_default PIN_MEMORY 0

case "$APPROACH" in
  direct)
    set_default TRAIN_SCRIPT kokoro/training/train_turkish_approach_direct.py
    set_default MAX_PHONEMES 140
    set_default MAX_STEPS 12000
    set_default BATCH_SIZE 2
    set_default LR 2e-5
    set_default GRAD_CLIP 0.5
    set_default SAVE_AUDIO_EVERY 200
    set_default SAVE_CHECKPOINT_EVERY 500
    ;;
  direct_bert)
    set_default TRAIN_SCRIPT kokoro/training/train_turkish_approach_direct_bert.py
    set_default MAX_PHONEMES 140
    set_default MAX_STEPS 12000
    set_default BATCH_SIZE 2
    set_default LR 1e-5
    set_default GRAD_CLIP 0.5
    set_default SAVE_AUDIO_EVERY 200
    set_default SAVE_CHECKPOINT_EVERY 500
    ;;
  unfreeze_decoder)
    set_default TRAIN_SCRIPT kokoro/training/train_turkish_approach_unfreeze_decoder.py
    set_default MAX_PHONEMES 100
    set_default MAX_STEPS 8000
    set_default BATCH_SIZE 1
    set_default LR 1e-5
    set_default GRAD_CLIP 0.3
    set_default SAVE_AUDIO_EVERY 100
    set_default SAVE_CHECKPOINT_EVERY 250
    ;;
  gt_bootstrap)
    set_default TRAIN_SCRIPT kokoro/training/train_turkish_approach_gt_bootstrap.py
    set_default MAX_PHONEMES 140
    set_default MAX_STEPS 12000
    set_default BATCH_SIZE 2
    set_default LR 2e-5
    set_default GRAD_CLIP 0.5
    set_default SAVE_AUDIO_EVERY 200
    set_default SAVE_CHECKPOINT_EVERY 500
    set_default LAMBDA_F0 3.0
    set_default LAMBDA_NORM 3.0
    # Stage 2: unfreeze bert_encoder, fresh optimizer, shorter run
    set_default STAGE2_STEPS 6000
    set_default STAGE2_LR 1e-5
    set_default STAGE2_TRAIN_CONFIG voicepack_predictor_text_bertenc
    ;;
  voicepack_bootstrap)
    set_default TRAIN_SCRIPT kokoro/training/train_turkish_approach_voicepack_bootstrap.py
    set_default MAX_PHONEMES 140
    set_default MAX_STEPS 12000
    set_default BATCH_SIZE 2
    set_default LR 2e-5
    set_default GRAD_CLIP 0.5
    set_default SAVE_AUDIO_EVERY 200
    set_default SAVE_CHECKPOINT_EVERY 500
    ;;
  unfreeze)
    set_default TRAIN_SCRIPT kokoro/training/train_turkish_approach_unfreeze.py
    set_default MAX_PHONEMES 140
    set_default MAX_STEPS 12000
    set_default BATCH_SIZE 1
    set_default LR 1e-5
    set_default GRAD_CLIP 0.5
    set_default SAVE_AUDIO_EVERY 200
    set_default SAVE_CHECKPOINT_EVERY 500
    ;;
  *)
    echo "Unknown APPROACH=$APPROACH" >&2
    exit 1
    ;;
esac

set_default LAMBDA_F0 1.0
set_default LAMBDA_NORM 1.0
set_default RUN_NAME "${APPROACH}_${SPEAKER}_p${MAX_PHONEMES}_rows${MAX_ROWS}"
set_default RUN_DIR "kokoro/training/runpod_runs/${RUN_NAME}"
HF_DATASET_REPO="${HF_DATASET_REPO:-vsqrd/styletts2-turkish}"
HF_MODEL_REPO="${HF_MODEL_REPO:-hexgrad/Kokoro-82M}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "==> repo root: $ROOT_DIR"
echo "==> run dir:   $RUN_DIR"
echo "==> approach:  $APPROACH"
echo "==> speaker:   $SPEAKER"
echo "==> device:    $DEVICE"
echo "==> max phon:  $MAX_PHONEMES"
echo "==> max rows:  $MAX_ROWS"
echo "==> max steps: $MAX_STEPS"
echo "==> save aud:  $SAVE_AUDIO_EVERY"
echo "==> save ckpt: $SAVE_CHECKPOINT_EVERY"

if [[ -z "${HF_TOKEN:-}" ]]; then
  read -r -s -p "Enter Hugging Face token: " HF_TOKEN
  echo
  if [[ -z "$HF_TOKEN" ]]; then
    echo "HF token is required to download dataset artifacts." >&2
    exit 1
  fi
fi

if command -v apt-get >/dev/null 2>&1; then
  echo "==> installing system packages"
  apt-get update
  apt-get install -y ffmpeg libsndfile1 git git-lfs
  git lfs install || true
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [[ "$DEVICE" == "cuda" ]]; then
  echo "==> installing CUDA PyTorch"
  python -m pip install --upgrade \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
else
  echo "==> installing CPU PyTorch"
  python -m pip install --upgrade torch torchaudio torchvision
fi

echo "==> installing Python deps"
python -m pip install --upgrade \
  attrs \
  soundfile \
  huggingface_hub[cli] \
  hf_transfer \
  loguru \
  numpy \
  transformers \
  munch \
  pydub \
  pyyaml \
  librosa \
  nltk \
  matplotlib \
  accelerate \
  einops \
  einops-exts

DOWNLOAD_DIR="$ROOT_DIR/.downloads/styletts2_turkish"
mkdir -p "$DOWNLOAD_DIR" "$HF_HOME" "$RUN_DIR"

if [[ -f "$DOWNLOAD_DIR/combined_dataset.tar.gz" && -f "$DOWNLOAD_DIR/alignments.tar.gz" ]]; then
  echo "==> dataset tarballs already downloaded, skipping"
else
  echo "==> downloading dataset tarballs from Hugging Face"
  HF_ARGS=(download "$HF_DATASET_REPO" --repo-type dataset --local-dir "$DOWNLOAD_DIR")
  if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ARGS+=(--token "$HF_TOKEN")
  fi
  hf "${HF_ARGS[@]}" combined_dataset.tar.gz alignments.tar.gz
fi

if [[ -d "$ROOT_DIR/combined_dataset" && -f "$ROOT_DIR/combined_dataset/manifest.csv" && -d "$ROOT_DIR/alignments" ]]; then
  echo "==> extracted dataset already present, skipping"
else
  echo "==> extracting dataset"
  tar -xzf "$DOWNLOAD_DIR/combined_dataset.tar.gz" -C "$ROOT_DIR"
  tar -xzf "$DOWNLOAD_DIR/alignments.tar.gz" -C "$ROOT_DIR"
fi

if [[ -f "$ROOT_DIR/kokoro/weights/kokoro-v1_0.pth" && -n "$(find "$ROOT_DIR/kokoro/weights/voices" -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)" ]]; then
  echo "==> Kokoro weights already present, skipping"
else
  echo "==> downloading Kokoro weights and voices"
  MODEL_ARGS=(download "$HF_MODEL_REPO" --local-dir "$ROOT_DIR/kokoro/weights")
  if [[ -n "${HF_TOKEN:-}" ]]; then
    MODEL_ARGS+=(--token "$HF_TOKEN")
  fi
  hf "${MODEL_ARGS[@]}" kokoro-v1_0.pth
  mkdir -p "$ROOT_DIR/kokoro/weights/voices"
  hf "${MODEL_ARGS[@]}" --include "voices/*.pt"
fi

echo "==> building cleaned manifest"
python comparisons/build_turkish_training_manifest.py

if [[ -d "$ROOT_DIR/alignments_kokoro_tr" && -n "$(find "$ROOT_DIR/alignments_kokoro_tr" -maxdepth 1 -name '*.pt' -print -quit)" ]]; then
  echo "==> canonical alignments already present, skipping"
else
  echo "==> canonicalizing alignments"
  python comparisons/canonicalize_turkish_alignments.py
fi

echo "==> building training manifest (speaker=${SPEAKER})"
SPEAKER_ARGS=()
if [[ "$SPEAKER" != "both" ]]; then
  SPEAKER_ARGS+=(--speaker "$SPEAKER")
fi

if [[ "$MAX_ROWS" != "0" ]]; then
  python kokoro/training/build_turkish_subset_manifest.py \
    "${SPEAKER_ARGS[@]}" \
    --max-phonemes "$MAX_PHONEMES" \
    --max-rows "$MAX_ROWS" \
    --out "$RUN_DIR/${RUN_NAME}.csv"
else
  python kokoro/training/build_turkish_subset_manifest.py \
    "${SPEAKER_ARGS[@]}" \
    --max-phonemes "$MAX_PHONEMES" \
    --out "$RUN_DIR/${RUN_NAME}.csv"
fi

echo "==> launching training"
PIN_ARGS=()
if [[ "$PIN_MEMORY" == "1" ]]; then
  PIN_ARGS+=(--pin-memory)
fi

RESUME_ARGS=()
if [[ -d "$RUN_DIR/checkpoints" && -n "$(find "$RUN_DIR/checkpoints" -maxdepth 1 -name 'checkpoint_step_*.pt' -print -quit 2>/dev/null)" ]]; then
  RESUME_ARGS+=(--resume)
fi

SPEAKER_LABEL_ARGS=()
if [[ "$SPEAKER" != "both" ]]; then
  SPEAKER_LABEL_ARGS+=(--speaker-label "$SPEAKER")
fi

echo "==> stage 1 training"
python "$TRAIN_SCRIPT" \
  --manifest "$RUN_DIR/${RUN_NAME}.csv" \
  --alignment-dir alignments_kokoro_tr \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --max-samples "$MAX_ROWS" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --grad-clip "$GRAD_CLIP" \
  --lambda-f0 "$LAMBDA_F0" \
  --lambda-norm "$LAMBDA_NORM" \
  --save-dir "$RUN_DIR/checkpoints" \
  --save-audio-every "$SAVE_AUDIO_EVERY" \
  --save-checkpoint-every "$SAVE_CHECKPOINT_EVERY" \
  "${SPEAKER_LABEL_ARGS[@]}" \
  "${PIN_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/train.log"

if [[ -n "${STAGE2_STEPS:-}" && "${STAGE2_STEPS}" != "0" ]]; then
  STAGE2_CKPT="$(python3 -c "
import re, sys
from pathlib import Path
ckpts = sorted(Path('$RUN_DIR/checkpoints').glob('checkpoint_step_*.pt'))
if not ckpts: sys.exit(1)
best = max(ckpts, key=lambda p: int(re.search(r'checkpoint_step_(\d+)\.pt', p.name).group(1)))
print(best)
")"
  if [[ -z "$STAGE2_CKPT" ]]; then
    echo "==> no stage 1 checkpoint found, skipping stage 2" >&2
  else
    STAGE2_DIR="$RUN_DIR/stage2"
    mkdir -p "$STAGE2_DIR"
    echo "==> stage 2 training (config=${STAGE2_TRAIN_CONFIG:-voicepack_predictor_text_bertenc}, from $STAGE2_CKPT)"
    python "$TRAIN_SCRIPT" \
      --manifest "$RUN_DIR/${RUN_NAME}.csv" \
      --alignment-dir alignments_kokoro_tr \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --max-steps "${STAGE2_STEPS}" \
      --max-samples "$MAX_ROWS" \
      --num-workers "$NUM_WORKERS" \
      --lr "${STAGE2_LR:-1e-5}" \
      --grad-clip "$GRAD_CLIP" \
      --lambda-f0 "$LAMBDA_F0" \
      --lambda-norm "$LAMBDA_NORM" \
      --train-config "${STAGE2_TRAIN_CONFIG:-voicepack_predictor_text_bertenc}" \
      --resume-weights-only "$STAGE2_CKPT" \
      "${SPEAKER_LABEL_ARGS[@]}" \
      --save-dir "$STAGE2_DIR/checkpoints" \
      --save-audio-every "$SAVE_AUDIO_EVERY" \
      --save-checkpoint-every "$SAVE_CHECKPOINT_EVERY" \
      "${PIN_ARGS[@]}" \
      2>&1 | tee "$STAGE2_DIR/train.log"
  fi
fi
