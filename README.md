
• I added five separate training entrypoints and updated the RunPod launcher to
  select them with APPROACH.

  New training scripts:

  - train_turkish_approach_direct.py
  - train_turkish_approach_direct_bert.py
  - train_turkish_approach_unfreeze_decoder.py
  - train_turkish_approach_gt_bootstrap.py
  - train_turkish_approach_voicepack_bootstrap.py

  Shared trainer updates are in train_kokoro_turkish.py. It now supports:

  - decoder-unfrozen configs
  - GT F0/N decoder conditioning
  - speaker-aware voicepack bootstrap via auto_gender

  The RunPod launcher is updated in runpod_train_kokoro_turkish.sh. Use it like:

  APPROACH=direct bash scripts/runpod_train_kokoro_turkish.sh
  APPROACH=direct_bert bash scripts/runpod_train_kokoro_turkish.sh
  APPROACH=unfreeze_decoder bash scripts/runpod_train_kokoro_turkish.sh
  APPROACH=gt_bootstrap bash scripts/runpod_train_kokoro_turkish.sh
  APPROACH=voicepack_bootstrap bash scripts/runpod_train_kokoro_turkish.sh