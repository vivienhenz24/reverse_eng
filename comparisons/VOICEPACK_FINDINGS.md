# Kokoro Voicepack Findings

## Current Best Answer

The strongest current explanation is:

- Kokoro voicepacks are trained per-voice, per-length style tables.
- At inference, Kokoro does a direct lookup:
  `ref_s = voicepack[voice_id, phoneme_length - 1]`
- The released `.pt` files are not best explained as stacks of reference utterance embeddings.
- They are also not best explained as raw per-length diffusion samples.

In practical terms, the voicepack is most likely a learned parameter table of shape:

`(num_voices, max_phoneme_length, 256)`

For v1.0, that is:

`(54, 510, 256)`

which is `7,050,240` parameters, or about `26.89 MiB` in fp32.

## Why This Is The Best Explanation

### 1. It matches the released inference contract exactly

Kokoro runtime uses:

`pack[len(phonemes) - 1]`

So the shipped file is already being treated as a length-indexed style table.

### 2. Public evidence explicitly says "trained voicepacks"

This wording is more consistent with a learned artifact than with a hidden one-shot exporter from a single reference embedding.

### 3. Legacy Kokoro already used full voice tables

In v0.19:

- voicepacks already existed as full `(511, 1, 256)` tensors
- `af.pt` was publicly documented as the exact average of `af_bella.pt` and `af_sarah.pt`

That is strong evidence that voicepacks were first-class artifacts, not opaque encoder outputs.

### 4. The released checkpoint contains no voice/style export machinery

The released `kokoro-v1_0.pth` contains only:

- `bert`
- `bert_encoder`
- `text_encoder`
- `predictor`
- `decoder`

There is no released:

- `style_encoder`
- `predictor_encoder`
- `diffusion`
- speaker embedding table
- voice export module

So voicepacks are structurally separate from the released backbone.

### 5. Exact stored-file recovery resists compact exporter-style models

Compact low-rank models explain the geometry very well, but they do not reproduce the exact stored float32 files.

From [exact_voicepack_roundtrip.py](/Users/vivienhenz/reverse_eng/comparisons/exact_voicepack_roundtrip.py):

- exact float32 round-trip requires terminal-centered rank `k=509`

So compact decompositions are good summaries, but not the literal stored artifact.

## What Was Eliminated

These are no longer good primary explanations for the released files:

- stacks of 510 independent reference utterance embeddings
- raw per-length diffusion samples
- simple interpolation from first slot to last slot
- simple two-curve decoder/prosody interpolation
- a compact one-shot `ref_s -> full table` exporter as the main mechanism

## What Still Holds True Structurally

Even if the table is trained directly, the released files still have strong structure:

- adjacent rows are extremely similar
- the length axis is smooth and saturating
- the table lies on a low-rank shared manifold across voices

So the trained-table hypothesis and the low-rank/dynamical observations are compatible:

- training likely produced a rich full table
- the resulting table happens to have strong shared smooth structure

## Reconstructed Training-Time Hypothesis

The most plausible training-time mechanism is:

1. Keep a trainable table:
   `voicepack_table[voice_id, phoneme_length] -> 256-d style`
2. For each training sample:
   - get `voice_id`
   - get phoneme sequence length
   - fetch `ref_s` from the table
3. Run Kokoro exactly through the released backbone:
   - BERT / text encoding
   - duration prediction
   - F0 / noise prediction
   - decoder
4. Backprop into:
   - model weights
   - the voicepack table
5. Optionally regularize neighboring length slots for smoothness

This is implemented as a concrete scaffold in [prototype_voicepack_training_loop.py](/Users/vivienhenz/reverse_eng/comparisons/prototype_voicepack_training_loop.py).

## What Remains Unknown

One important question is still open:

- was the voice table learned from scratch
- or initialized from StyleTTS2-like style features and then optimized

But the strongest current answer is no longer “hidden exporter.”

It is:

`voicepacks are trained voice-length parameter tables saved as .pt files`
