# RunPod Minimal Files

These are the repo paths the current RunPod Turkish training job depends on.

## Required repo paths

- `scripts/runpod_train_kokoro_turkish.sh`
- `comparisons/build_turkish_training_manifest.py`
- `comparisons/canonicalize_turkish_alignments.py`
- `kokoro/training/build_turkish_subset_manifest.py`
- `kokoro/training/common.py`
- `kokoro/training/dataset.py`
- `kokoro/training/losses.py`
- `kokoro/training/train_kokoro_turkish.py`
- `kokoro/weights/config.json`
- `kokoro/weights/kokoro-v1_0.pth`
- `kokoro/kokoro/custom_stft.py`
- `kokoro/kokoro/istftnet.py`
- `kokoro/kokoro/modules.py`
- `kokoro/kokoro/model.py`
- `StyleTTS2/losses.py`
- `StyleTTS2/Utils/JDC/model.py`
- `StyleTTS2/Utils/JDC/bst.t7`

## Downloaded at runtime

These do not need to be committed into the repo before sending to RunPod:

- `combined_dataset/`
- `alignments/`
- `alignments_kokoro_tr/`
- `.downloads/`
- `.hf_home/`
- `kokoro/training/runpod_runs/`

## Practical note

The simplest path is still to clone the whole repo onto the pod, because the
training stack uses local relative imports across `kokoro/`, `StyleTTS2/`, and
`comparisons/`.
