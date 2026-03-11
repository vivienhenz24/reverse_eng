# Turkish Adaptation Plan

## Current Best Path

The strongest current path to add Turkish to Kokoro is:

1. Add a Turkish front end using `espeak-ng` Turkish.
2. Normalize Turkish IPA into Kokoro's existing vocab.
3. Train new Turkish voicepack tables.
4. Finetune Kokoro on Turkish audio/text with those tables as trainable parameters.

This path is based on the current best voicepack finding:

`voicepack[voice_id, phoneme_length - 1] -> 256-d learned style`

not on a hidden post-hoc exporter.

## What We Know Already

### 1. Turkish front-end support is close

From [audit_turkish_frontend.py](/Users/vivienhenz/reverse_eng/comparisons/audit_turkish_frontend.py)
and [audit_turkish_dataset.py](/Users/vivienhenz/reverse_eng/comparisons/audit_turkish_dataset.py):

- local `espeak-ng` already supports Turkish (`tr`)
- the full Turkish dataset phoneme manifest is representable by Kokoro's current vocab after:
  - replacing line breaks with spaces
  - collapsing repeated whitespace
  - mapping `ɫ -> l`
- dataset status:
  - `47,624` wavs
  - `47,624` alignment `.pt` files
  - `2` speakers
  - normalized phoneme max length `285`
  - one mangled phonemized filename row, repairable from the base manifest
- `47,433` rows already fall into the safe first-pass alignment regime where:
  - `alignment_rows - len(phonemes) in {0, 1, 2}`

So Turkish does **not** appear to require immediate vocab expansion on the audit set.

### 2. Voicepacks are best treated as trainable tables

From [prototype_voicepack_training_loop.py](/Users/vivienhenz/reverse_eng/comparisons/prototype_voicepack_training_loop.py):

- a trainable table of shape `(54, 510, 256)` plugs directly into Kokoro's released forward path
- no extra exporter layer is required

### 3. Kokoro inference already matches this structure

At inference:

`ref_s = pack[len(phonemes) - 1]`

So Turkish support can be added by training new Turkish packs and teaching the backbone to use them well.

## Minimum Turkish Front-End Changes

### A. Add Turkish language code

In [pipeline.py](/Users/vivienhenz/reverse_eng/kokoro/kokoro/pipeline.py):

- add alias:
  - `'tr' -> 't'` or directly `'tr'`
- add `LANG_CODES` entry for Turkish using espeak-ng:
  - `'t': 'tr'`

### B. Normalize espeak Turkish IPA

Before token-to-id lookup, apply:

- line breaks -> single spaces
- collapse repeated whitespace
- `ɫ -> l`

This is enough for the current Turkish manifest to fit the existing vocab.

### C. Use the cleaned manifest, not the raw phonemized CSV

Build it with:

```bash
python comparisons/build_turkish_training_manifest.py
```

This repairs the one broken filename row and filters the rare alignment outliers.

### D. Canonicalize the alignment tensors

Build Kokoro-shaped alignments with:

```bash
python comparisons/canonicalize_turkish_alignments.py
```

Current best deterministic conversion rule:

- `alignment_rows - len(phonemes) == 0`: pad zero rows on both sides
- `alignment_rows - len(phonemes) == 1`: append one zero row on the right
- `alignment_rows - len(phonemes) == 2`: keep as-is

Rare corruption:

- `80 / 47,433` kept tensors have a malformed final frame column with multiple active rows
- collapse that final column onto the last active row before training

## Recommended Training Order

### Stage 1: Single-speaker Turkish proof of concept

Goal:

- prove intelligible Turkish is possible at all

Train:

- new Turkish voicepack table entry or entries
- `text_encoder`
- `bert_encoder`
- `predictor`

Initially freeze:

- `decoder`

Reason:

- decoder is the most language-agnostic part
- text/predictor side is where Turkish sequence modeling enters first

### Stage 2: If pronunciation/prosody is weak

Unfreeze:

- `decoder`

Then continue finetuning end-to-end.

### Stage 3: Multi-speaker Turkish

After single-speaker works:

- add more Turkish voice IDs
- train additional voicepack table rows
- keep the same Turkish front end

## Candidate Training Loop

For each batch item:

1. read rows from the cleaned Turkish manifest
2. convert normalized phonemes to token ids with current Kokoro vocab
3. compute `length_index = len(ps) - 1`
4. fetch `ref_s = voicepack_table[voice_id, length_index]`
5. load the canonicalized alignment matrix from `alignments_kokoro_tr/`
6. use that matrix for duration supervision
7. run Kokoro forward
8. backprop into:
   - model weights
   - Turkish voicepack table

For the first pass:

- keep only rows with `alignment_rows - len(phonemes) in {0, 1, 2}`
- canonicalize them into `[len(phonemes) + 2, frames]`
- quarantine the remaining `191 / 47,624` rows

## What To Build Next In Repo

1. Turkish front-end adapter
- add a small IPA normalization helper for Turkish

2. Training script
- start from [prototype_voicepack_training_loop.py](/Users/vivienhenz/reverse_eng/comparisons/prototype_voicepack_training_loop.py)
- turn it into a real finetuning script with dataset loading and losses

3. Dataset loader
- consume `combined_dataset/kokoro_turkish_manifest.csv`
- map `speaker_id -> speaker_index`
- load `alignments/*.pt` as duration/alignment supervision

4. Evaluation set
- fixed Turkish sentences for intelligibility/prosody checks

## Biggest Remaining Unknown

The largest unresolved detail is no longer the data path.

It is the exact training objective mix:

- audio reconstruction losses
- duration loss against the provided alignments
- optional F0 / noise losses
- whether adversarial losses are needed

## Practical Conclusion

The current best answer is:

- Turkish looks feasible without immediate vocab expansion
- the first concrete path is:
  - add Turkish espeak front end
  - normalize `ɫ` and zero-width joiners
  - train Turkish voicepack tables
  - finetune Kokoro text/prosody side first

This is now specific enough to implement.
