#!/usr/bin/env bash
set -euo pipefail

# Run from repo root after cloning reverse_eng onto a CUDA RunPod machine.
#
# Example:
#   HF_TOKEN=hf_xxx SPEAKER=female_speaker MAX_ROWS=12000 BATCH_SIZE=4 \
#   bash scripts/runpod_train_kokoro_turkish.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-runpod}"
DEVICE="${DEVICE:-cuda}"
SPEAKER="${SPEAKER:-female_speaker}"
MAX_PHONEMES="${MAX_PHONEMES:-80}"
MAX_ROWS="${MAX_ROWS:-12000}"
MAX_STEPS="${MAX_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-5}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"
SAVE_EVERY="${SAVE_EVERY:-50}"
RUN_NAME="${RUN_NAME:-${SPEAKER}_p${MAX_PHONEMES}_rows${MAX_ROWS}}"
RUN_DIR="${RUN_DIR:-kokoro/training/runpod_runs/${RUN_NAME}}"
HF_DATASET_REPO="${HF_DATASET_REPO:-vsqrd/styletts2-turkish}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DISABLE_TELEMETRY=1

echo "==> repo root: $ROOT_DIR"
echo "==> run dir:   $RUN_DIR"
echo "==> speaker:   $SPEAKER"
echo "==> device:    $DEVICE"

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
  sudo apt-get update
  sudo apt-get install -y ffmpeg libsndfile1 git git-lfs
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
  python -m pip install --upgrade torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
else
  echo "==> installing CPU PyTorch"
  python -m pip install --upgrade torch torchaudio torchvision
fi

echo "==> installing Python deps"
python -m pip install --upgrade \
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

echo "==> downloading dataset tarballs from Hugging Face"
HF_ARGS=(download "$HF_DATASET_REPO" --repo-type dataset --local-dir "$DOWNLOAD_DIR")
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ARGS+=(--token "$HF_TOKEN")
fi
hf "${HF_ARGS[@]}" combined_dataset.tar.gz alignments.tar.gz

echo "==> extracting dataset"
tar -xzf "$DOWNLOAD_DIR/combined_dataset.tar.gz" -C "$ROOT_DIR"
tar -xzf "$DOWNLOAD_DIR/alignments.tar.gz" -C "$ROOT_DIR"

echo "==> building cleaned manifest"
python comparisons/build_turkish_training_manifest.py

echo "==> canonicalizing alignments"
python comparisons/canonicalize_turkish_alignments.py

echo "==> building single-speaker subset manifest"
python kokoro/training/build_turkish_subset_manifest.py \
  --speaker "$SPEAKER" \
  --max-phonemes "$MAX_PHONEMES" \
  --max-rows "$MAX_ROWS" \
  --out "$RUN_DIR/${RUN_NAME}.csv"

echo "==> launching training"
python kokoro/training/train_kokoro_turkish.py \
  --manifest "$RUN_DIR/${RUN_NAME}.csv" \
  --alignment-dir alignments_kokoro_tr \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --max-samples "$MAX_ROWS" \
  --lr "$LR" \
  --grad-clip "$GRAD_CLIP" \
  --train-config voicepack_predictor_text \
  --voicepack-init mean \
  --save-dir "$RUN_DIR/checkpoints" \
  --save-every "$SAVE_EVERY" \
  2>&1 | tee "$RUN_DIR/train.log"
