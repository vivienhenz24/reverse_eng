# Kokoro Voicepack Findings

## Summary

The released Kokoro voicepacks are best explained as offline-compiled, length-conditioned style tables, not as stacks of independent reference embeddings.

The strongest evidence is:

- Kokoro inference uses `pack[len(phonemes)-1]`, so the first axis is treated as phoneme length.
- Adjacent rows in each voicepack are almost identical, with mean cosine about `0.99998`.
- First vs last row cosine is much lower, about `0.53`, so the axis is not constant.
- Almost every per-dimension curve over length is monotonic or nearly monotonic.
- The full `(54 voices, 510 lengths, 256 dims)` tensor lies on a very low-rank shared length manifold.

## Key Results

From [discover_voicepack_basis.py](/Users/vivienhenz/reverse_eng/comparisons/discover_voicepack_basis.py):

- Rank 1 shared length basis explains about `95.63%` of residual energy.
- Rank 2 explains about `98.98%`.
- Rank 4 explains about `99.98%`.
- Rank 4 reconstruction reaches mean cosine about `0.999995`.

From [reconstruct_voicepack_exporter.py](/Users/vivienhenz/reverse_eng/comparisons/reconstruct_voicepack_exporter.py):

- A compact exporter form
  `pack[v,l,d] = mean[v,d] + sum_k coeff[v,k,d] * basis[k,l]`
  reconstructs the shipped voicepacks almost exactly with `k=4`.

From [prototype_voicepack_exporter.py](/Users/vivienhenz/reverse_eng/comparisons/prototype_voicepack_exporter.py):

- A StyleTTS2-inspired simulator using:
  - `ref_s` anchoring at the first slot
  - a shared rank-4 proposal over length
  - `s_prev`-style recurrence
  fits best with:
  - `alpha = 1.0`
  - `beta = 1.0`
  - `t_dec = 0.9`
  - `t_pro = 0.9`
- That simulator reaches mean cosine about `0.998658`.

From [compare_proposal_families.py](/Users/vivienhenz/reverse_eng/comparisons/compare_proposal_families.py):

- The discovered rank-4 proposal geometry is:
  - nearly monotonic
  - very low curvature
  - very low residual noise
- Proposal-family ranking is:
  1. `teacher_forced`
  2. `hybrid_smoothed`
  3. `diffusion_like`

This is the strongest current evidence that the coefficient proposal was more likely
deterministic or heavily smoothed than raw diffusion-sampled per length slot.

## Interpretation

This strongly suggests:

1. Kokoro was trained in a StyleTTS2-like 256-dim style space.
2. An extra offline export step converted each voice into a 510-slot lookup table.
3. That lookup table is structured by a small shared set of length-dependent basis curves.
4. The closest local-code analogue is a proposal-driven exporter with strong
   recurrence smoothing across length, not raw per-length diffusion sampling.
5. The proposal itself looks more teacher-forced / deterministic than diffusion-like.

The remaining unknown is the upstream exporter logic:

- deterministic interpolation / compilation
- teacher-forced extraction over many text lengths
- diffusion-conditioned export
- or some hybrid of those

But the released artifacts strongly support the compiled-table conclusion.

## Practical Implication

If the goal is Turkish finetuning, the missing piece is not just the training loop. It is also the post-training voicepack export step that turns a trained style representation into Kokoro’s inference-time table format.

## Current Best Answer

The brief version is:

- Kokoro voicepacks are compiled inference tables, not raw stacks of sampled styles.
- Their 510-length axis is governed by a tiny shared low-rank basis across voices.
- The final table is best explained by a strong smoothing / recurrence step over length.
- The upstream proposal feeding that smoothing looks more teacher-forced or deterministic than diffusion-like.

So the most likely pipeline is:

1. Extract a StyleTTS2-like 256-dim reference style.
2. Build a deterministic or teacher-forced proposal over phoneme lengths.
3. Smooth / compile that proposal into the shipped 510-slot voicepack.
