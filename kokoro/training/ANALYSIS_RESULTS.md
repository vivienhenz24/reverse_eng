# Kokoro Turkish Adaptation — Base Weight Analysis Results

Run date: 2026-03-20
Model: `kokoro/weights/kokoro-v1_0.pth` (82M params)
Data: `combined_dataset/kokoro_turkish_manifest.csv`, 24kHz mono

---

## 1. Weight Norms (`analyze_weight_norms.py`)

### Component Summary

| Component               |      Params | Mean\|W\| | Std\|W\| | Min\|W\| | Max\|W\| | Sparse% |
|-------------------------|------------:|----------:|---------:|---------:|---------:|--------:|
| bert                    |   6,292,480 |   23.0700 |  26.1352 |   0.0000 |  83.0028 |    0.02 |
| bert_encoder            |     393,728 |   23.4822 |  22.8082 |   0.6740 |  46.2905 |    0.00 |
| text_encoder            |   5,606,400 |   41.1997 |  64.1387 |   0.9774 | 308.5065 |    0.00 |
| predictor               |  16,203,828 |   20.7953 |  24.0973 |   0.0000 |  78.9277 |    0.03 |
| predictor.text_encoder  |   5,913,600 |   28.5006 |  28.1855 |   2.1459 |  68.7633 |    0.00 |
| predictor.lstm          |   1,839,104 |   39.8393 |  36.8668 |   2.9678 |  78.9277 |    0.00 |
| predictor.duration_proj |      25,650 |   39.3540 |  37.7342 |   1.6198 |  77.0882 |    0.00 |
| predictor.shared        |   1,839,104 |   31.2859 |  29.4682 |   2.0340 |  66.8700 |    0.00 |
| predictor.F0            |   3,292,928 |   16.4723 |  18.0233 |   0.0000 |  73.3658 |    0.07 |
| predictor.N             |   3,292,928 |   16.1092 |  17.9848 |   0.0000 |  72.8464 |    0.07 |
| decoder                 |  53,313,586 |   16.3444 |  24.6029 |   0.0000 | 152.7718 |    0.04 |
| **TOTAL**               |  81,810,022 |   18.4211 |  27.3547 |   0.0000 | 308.5065 |    0.03 |

### LR Scaling Recommendation

Higher norm = stronger prior = use lower relative LR multiplier.

The norms are tightly clustered (16–41), so the inverse-scaling heuristic collapses to ~1.0x across the board. In practice, this means **norm alone doesn't strongly differentiate LR needs** — use gradient sensitivity results (Analysis 2) to guide per-component LR ratios instead.

### Key Observations

- **text_encoder** has the highest mean norm (41.2) and the highest single-parameter norm (308.5, the token embedding matrix). The embedding has strong learned English phoneme priors.
- **predictor.lstm / duration_proj** are also high-norm (39–40), indicating well-converged, strong priors for English prosody.
- **decoder** is the largest component (53M params, 65% of total) but has the lowest mean norm (16.3), consistent with it being a general-purpose acoustic decoder.
- **predictor.F0 / predictor.N** have the most sparsity (0.07%), suggesting some dead units — these may be the easiest subcomponents to retrain.

### text_encoder Layer Breakdown (notable layers)

| Layer                        | Shape       | \|W\| L2  |
|------------------------------|-------------|----------:|
| embedding.weight             | 178×512     | 308.5065  |
| cnn.0–2 (parametrized conv)  | 512×512×5   | ~94       |
| cnn.0–2 (norm gamma/beta)    | 512         | ~20 / ~11 |
| lstm.weight_ih_l0            | 1024×512    | 62.7      |
| lstm.weight_hh_l0            | 1024×256    | 61.0      |

The token embedding dominates the text_encoder norm. The LSTM weights are substantial (~62), meaning the text encoder has strong directional priors that may resist Turkish patterns.

---

## 2. Gradient Sensitivity (`analyze_gradient_sensitivity.py`)

Batch: 4 Turkish samples, F0 loss disabled (STFT + duration only).

### Loss Per Config

| Config                                   |  Total |   STFT | Duration |
|------------------------------------------|-------:|-------:|---------:|
| voicepack_only                           | 1.9970 | 1.0284 |   0.9686 |
| voicepack_predictor                      | 2.0017 | 1.0331 |   0.9686 |
| voicepack_predictor_text                 | 1.9948 | 1.0262 |   0.9686 |
| voicepack_predictor_text_bertenc         | 1.9999 | 1.0312 |   0.9686 |
| voicepack_predictor_text_decoder         | 1.9973 | 1.0286 |   0.9686 |
| voicepack_predictor_text_bertenc_decoder | 1.9965 | 1.0279 |   0.9686 |

### Gradient Norms Per Component

`frozen` = component not in this config's training set.

| Config                                   | voicepacks | bert_enc | text_enc | predictor | decoder |  bert  |
|------------------------------------------|:----------:|:--------:|:--------:|:---------:|:-------:|:------:|
| voicepack_only                           |   0.3949   |  frozen  |  frozen  |  frozen   | frozen  | frozen |
| voicepack_predictor                      |   0.3979   |  frozen  |  frozen  |  1.0018   | frozen  | frozen |
| voicepack_predictor_text                 |   0.3966   |  frozen  |  0.0367  |  1.0015   | frozen  | frozen |
| voicepack_predictor_text_bertenc         |   0.3949   |  0.3132  |  0.0344  |  0.9948   | frozen  | frozen |
| voicepack_predictor_text_decoder         |   0.3985   |  frozen  |  0.0316  |  1.0009   |  0.3244 | frozen |
| voicepack_predictor_text_bertenc_decoder |   0.3958   |  0.3128  |  0.0308  |  0.9915   |  0.3120 | frozen |

### Component Ranking (full config)

1. **predictor**    0.9915  ████████████████████████████████████████
2. **voicepacks**   0.3958  ███████████████
3. **bert_encoder** 0.3128  ████████████
4. **decoder**      0.3120  ████████████
5. **text_encoder** 0.0308  █

### Interpretation

- **predictor is the main bottleneck** — 3× higher gradient response than any other component. Turkish phoneme sequences are producing large gradient signal through the duration/F0 prediction path.
- **text_encoder has very low gradient norm (0.03)** — it's barely responding to Turkish input. The token embeddings (already trained on the full IPA vocab) are largely passing the correct representations through; the issue is downstream in the predictor.
- **decoder and bert_encoder are roughly equal (~0.31)** — moderate sensitivity, both safe to add but neither is urgent.
- **decoder/predictor ratio = 0.31x**: unfreezing the decoder early is **not high-risk** in terms of gradient magnitude, but also unlikely to drive the main improvement.
- **voicepacks gradient (0.40)** is consistent regardless of how many other components are unfrozen — the voicepack table always absorbs a meaningful gradient.

---

## 3. Activation Statistics (`analyze_activation_stats.py`)

8 Turkish samples, pretrained model only (no finetuning), voicepack init = mean.

### Per-Sample Results

| # | Text (truncated)                                  | F0 MAE | F0 r   | Dur MAE | Dur r  | Norm MAE | Norm r  | Voiced |
|---|---------------------------------------------------|-------:|-------:|--------:|-------:|---------:|--------:|-------:|
| 0 | Merhaba Metin Bey, Barış ben, hoş geldiniz        |  67.97 | -0.008 |    1.23 | -0.144 |    3.961 |  -0.177 | 97.1%  |
| 1 | Hesabınızı bugün açmamı ister misiniz?            |  78.67 | -0.054 |    1.27 | +0.579 |    3.924 |  +0.068 | 99.3%  |
| 2 | BNP Paribas Cardif Türkiye adına arıyorum         |  50.27 | +0.177 |    1.17 | +0.713 |    4.150 |  +0.029 | 89.8%  |
| 3 | Görüşmeyi özetleyip özel çözüm söylememi          |  71.15 | -0.023 |    1.08 | +0.640 |    4.040 |  -0.032 | 100.0% |
| 4 | Hesabı kapatmamı da talep ediyor musunuz          |  49.05 | +0.354 |    1.20 | +0.646 |    4.356 |  -0.090 | 92.4%  |
| 5 | Bağlantıyı hemen sağlayayım.                      |  84.66 | +0.273 |    2.51 | +0.967 |    7.248 |  -0.046 | 100.0% |
| 6 | Cari hesap kapatma sürecinde, cevap bekliyorum    |  48.13 | -0.037 |    1.00 | +0.751 |    4.115 |  -0.166 | 90.8%  |
| 7 | Özet geçeyim: böylece çözüm net görünsün          |  46.70 | +0.343 |    1.14 | -0.113 |    4.050 |  +0.085 | 89.9%  |
| — | **AVERAGE**                                       |  62.08 | +0.128 |    1.32 | +0.505 |    4.480 |  -0.041 | 94.9%  |

### Interpretation

| Head | Avg r  | Verdict |
|------|-------:|---------|
| F0   | +0.128 | ✗ Poor — significant finetuning needed |
| Dur  | +0.505 | ✓ Already correlates — model generalizes reasonably to Turkish duration |
| Norm | −0.041 | ✗ Poor — significant finetuning needed |

- **Duration prediction already works** (r = 0.50) out of the box. Turkish phoneme length sequences map well to English-trained duration priors. This is consistent with the gradient analysis: the predictor has a large gradient, but duration may already have a useful starting point to refine from.
- **F0 prediction is essentially random** (r = 0.13). The model has no idea about Turkish pitch patterns. This is the highest priority head to train.
- **Norm prediction is anti-correlated** (r = −0.04). The model's energy envelope estimates are noise for Turkish.
- **Voiced coverage is 94.9%** — the model is not collapsing to silence; it's producing sound, just with wrong pitch and energy contour.

---

## 4. Voicepack PCA (`analyze_voicepack_pca.py`)

54 released voicepacks, shape per pack: [510, 256].

### Voice Identity PCA (mean-pooled, [54, 256])

| Metric              | Value |
|---------------------|------:|
| Effective rank      | 41.14 |
| Dims for 90% var    |    24 |
| Dims for 95% var    |    32 |
| Dims for 99% var    |    44 |

| PC   | Singular val | % var | Cumul % |
|------|-------------:|------:|--------:|
| PC1  |       6.3764 | 15.3% |   15.3% |
| PC2  |       5.7751 | 12.5% |   27.8% |
| PC3  |       4.6836 |  8.3% |   36.1% |
| PC4  |       4.4711 |  7.5% |   43.6% |
| PC5  |       4.0014 |  6.0% |   49.6% |
| PC6  |       3.8764 |  5.7% |   55.3% |
| PC7  |       3.3721 |  4.3% |   59.5% |
| PC8  |       2.9609 |  3.3% |   62.8% |
| PC9  |       2.8211 |  3.0% |   65.8% |
| PC10 |       2.5769 |  2.5% |   68.3% |

### Length-Aware PCA ([54×510, 256])

| Metric              | Value |
|---------------------|------:|
| Effective rank      | 70.47 |
| Dims for 90% var    |    29 |
| Dims for 95% var    |    39 |
| Dims for 99% var    |    59 |

### Gender Clustering (PC1 / PC2)

| Gender | Centroid PC1 | Centroid PC2 | Within-cluster spread |
|--------|-------------:|-------------:|----------------------:|
| Female |        +0.229|        +0.292|                 0.876 |
| Male   |        −0.519|        −0.122|                 0.893 |

- **Centroid separation: 0.855**
- **Separation ratio: 0.97×** — clusters overlap substantially. Gender alone does not cleanly partition the voice space.

### Turkish Init Distance from Existing Clusters

| Init strategy        | Dist→female | Dist→male | Dist→all |
|----------------------|------------:|----------:|---------:|
| mean (all voices)    |      0.9082 |    1.0720 |   1.0174 |
| af_heart (female)    |      0.9014 |    1.2364 |   1.0560 |
| pm_alex (male)       |      1.2114 |    0.9345 |   1.1600 |

All three init strategies land roughly equidistant from the existing clusters (~0.9–1.2). No init is clearly "inside" the existing voice distribution.

### Recommendation

- **Effective rank = 41.1 out of 54 voices** — the voicepack space is high-rank. Each of the 54 voices occupies a largely distinct direction.
- **Gender-matched init (voicepack_bootstrap) is not strongly justified by geometry alone** — af_heart is no closer to the female cluster than the mean init.
- **gt_bootstrap or smooth-regularized random init are more appropriate** for reaching unexplored regions of voice space where Turkish may live.
- **At least 24 dimensions should remain trainable** in the voicepack table (90% variance threshold).

---

## Summary: Approach Recommendations

| Finding | Implication |
|---------|-------------|
| predictor has 3× higher grad norm than decoder | Train predictor first; decoder can wait |
| text_encoder grad norm is ~30× lower than predictor | text_encoder is already fine for Turkish phonemes |
| Duration r = +0.50 out of box | Duration prediction needs refinement, not rebuilding |
| F0 r = +0.13, Norm r = −0.04 out of box | F0 and norm heads need the most training |
| Voicepack effective rank = 41 | gt_bootstrap > voicepack_bootstrap for Turkish; mean init is equally good as gender-matched |
| Gender cluster separation ratio = 0.97× | Gender-matched voicepack init has no geometric advantage |

### Revised approach priority order

1. **voicepack_predictor** — highest-signal components, lowest risk
2. **voicepack_predictor_text_bertenc** — add bert_encoder for richer phoneme context; text_encoder optional
3. **voicepack_predictor_text_decoder** — add decoder only after F0/norm heads have converged
4. **gt_bootstrap** — preferred over voicepack_bootstrap given high voicepack rank; use as the voicepack init strategy rather than gender-matched
