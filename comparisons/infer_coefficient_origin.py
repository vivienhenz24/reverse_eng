"""
Infer how Kokoro's voicepack coefficients were most likely produced.

This script combines:
  - source-mined StyleTTS2 leads
  - Kokoro runtime behavior
  - discovered low-rank voicepack structure

to rank coefficient-origin mechanisms:
  - deterministic reference-style export
  - teacher-forced export
  - diffusion-conditioned export
  - hybrid diffusion/reference export
  - longitudinal-smoothed hybrid export
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sep(title="", width=88, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


def find_line(path, needle):
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return i, line.strip()
    return None, None


anchors = [
    ("Utterance style target", ROOT / "StyleTTS2/train_second.py", "s_trg = torch.cat([gs, s_dur], dim=-1).detach()"),
    ("Reference style features", ROOT / "StyleTTS2/train_second.py", "ref_s = torch.cat([ref_ss, ref_sp], dim=1)"),
    ("Diffusion sample", ROOT / "StyleTTS2/train_second.py", "s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(texts.device),"),
    ("Alpha/beta reference mix", ROOT / "StyleTTS2/Demo/Inference_LibriTTS.ipynb", "ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]"),
    ("Alpha/beta prosody mix", ROOT / "StyleTTS2/Demo/Inference_LibriTTS.ipynb", "s = beta * s + (1 - beta)  * ref_s[:, 128:]"),
    ("No-diffusion reference setting", ROOT / "StyleTTS2/Colab/StyleTTS2_Demo_LibriTTS.ipynb", "This setting uses 100% of the reference timbre and prosody and do not use the diffusion model at all."),
    ("No-diffusion call", ROOT / "StyleTTS2/Colab/StyleTTS2_Demo_LibriTTS.ipynb", "wav = inference(text, ref_s, diffusion_steps=10, alpha=0, beta=0, embedding_scale=1)"),
    ("Save ref_s with weights", ROOT / "StyleTTS2/Colab/StyleTTS2_Finetune_Demo.ipynb", "this style vector ref_s can be saved as a parameter together with the model weights"),
    ("Long-form previous style smoothing", ROOT / "StyleTTS2/Demo/Inference_LibriTTS.ipynb", "s_pred = t * s_prev + (1 - t) * s_pred"),
    ("Stage-2 predictor init from style encoder", ROOT / "StyleTTS2/train_second.py", "model.predictor_encoder = copy.deepcopy(model.style_encoder)"),
    ("Kokoro pack lookup", ROOT / "kokoro/kokoro/pipeline.py", "return model(ps, pack[len(ps)-1], speed, return_output=True)"),
]


hypotheses = [
    {
        "name": "Pure deterministic reference export",
        "mechanism": "Compute ref_s once from reference audio, then compile a length table without diffusion.",
        "pros": [
            "Supported by demos that explicitly allow alpha=0, beta=0 with no diffusion.",
            "Supported by note that ref_s can be saved with the model weights.",
            "Fits a stable low-rank compiled table better than per-call random sampling.",
        ],
        "cons": [
            "Raw compute_style output is utterance-level, so an extra compiler is still needed for the 510-length axis.",
        ],
        "score": 0.84,
    },
    {
        "name": "Teacher-forced deterministic export",
        "mechanism": "Use utterance/reference style encoders plus text/predictor paths over many controlled lengths, no diffusion sampling.",
        "pros": [
            "Directly grounded in training-time s_trg and predictor usage.",
            "Deterministic enough to produce the observed very low-rank shared length manifold.",
            "Explains stronger prosody sensitivity over the length axis.",
        ],
        "cons": [
            "No direct exporter code is present in StyleTTS2.",
        ],
        "score": 0.88,
    },
    {
        "name": "Pure diffusion-conditioned export",
        "mechanism": "Sample style with diffusion for each length slot, conditioned on text and reference features.",
        "pros": [
            "Directly supported by multispeaker StyleTTS2 inference path.",
        ],
        "cons": [
            "Unseeded diffusion sampling would normally inject more stochastic variation than the shipped low-rank tables show.",
            "Harder to reconcile with the near-monotonic, almost deterministic length curves.",
        ],
        "score": 0.46,
    },
    {
        "name": "Reference-plus-diffusion hybrid export",
        "mechanism": "Generate s_pred with diffusion, then mix with ref_s using alpha/beta and save the result.",
        "pros": [
            "Directly matches the LibriTTS demo inference formulas.",
            "Could preserve speaker identity while injecting text-conditioned length variation.",
        ],
        "cons": [
            "Still likely noisier than the released tables unless diffusion was heavily suppressed or averaged.",
        ],
        "score": 0.74,
    },
    {
        "name": "Long-form smoothed hybrid export",
        "mechanism": "Use diffusion/reference hybrid plus previous-style smoothing (s_prev) over an ordered length sweep.",
        "pros": [
            "The long-form inference demos already smooth style across successive calls with s_prev.",
            "That kind of recursion can naturally generate a low-rank, monotonic trajectory over length.",
            "Best available source lead for turning per-call style generation into a smooth table.",
        ],
        "cons": [
            "Still not directly shown as a training/export utility.",
            "Would need an ordering choice over lengths and a deterministic seed policy.",
        ],
        "score": 0.91,
    },
]


print("=" * 88)
print("  COEFFICIENT ORIGIN INFERENCE")
print("=" * 88)

sep("SOURCE LEADS")
for label, path, needle in anchors:
    line_no, line = find_line(path, needle)
    rel = path.relative_to(ROOT)
    print(f"  {label}: {rel}:{line_no}")
    print(f"    {line}")

sep("WHY THE QUESTION IS NARROWER NOW")
print("  The voicepacks are already explained structurally as low-rank compiled tables.")
print("  So the remaining question is not whether there was an exporter, but which upstream")
print("  StyleTTS2-style mechanism produced the per-voice coefficients before packing.")

sep("RANKING")
for rank, h in enumerate(sorted(hypotheses, key=lambda x: x["score"], reverse=True), start=1):
    print(f"  {rank}. {h['name']}  score={h['score']:.2f}")
    print(f"     mechanism: {h['mechanism']}")
    print("     pros:")
    for item in h["pros"]:
        print(f"       - {item}")
    print("     cons:")
    for item in h["cons"]:
        print(f"       - {item}")

sep("WORKING CONCLUSION")
print("  Most likely origin: a hybrid exporter built from StyleTTS2 reference style")
print("  extraction plus a smoothing/compilation step, possibly borrowing the same")
print("  alpha/beta mixing and long-form s_prev smoothing ideas used in the demos.")
print("  Second most likely: a fully deterministic teacher-forced export using the")
print("  utterance-level 256-d style targets and predictor path over controlled lengths.")
print("  Least likely: raw per-length diffusion sampling saved directly without extra smoothing.")

sep("MINIMAL CANDIDATE EXPORTER")
print("  1. Compute base reference style ref_s = [ref_ss || ref_sp] from reference mel.")
print("  2. Sweep a canonical ordered set of phoneme lengths 1..510.")
print("  3. For each length, generate or update a style proposal using:")
print("     - text-conditioned diffusion, or")
print("     - deterministic teacher-forced predictor features.")
print("  4. Mix proposal with ref_s using fixed alpha/beta-like weights.")
print("  5. Smooth across successive lengths using an s_prev-like recurrence.")
print("  6. Save the resulting 510 x 256 table.")

sep()
print("Done.")
