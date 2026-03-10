"""
Trace the StyleTTS2/Kokoro style graph and emit candidate Kokoro export pseudocode.

This script is source-driven: it locates the local code paths around
  - s_trg construction
  - diffusion sampling
  - predictor usage
  - Kokoro voicepack lookup

and then prints a consolidated graph plus one candidate export algorithm.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "models": ROOT / "StyleTTS2/models.py",
    "train_second": ROOT / "StyleTTS2/train_second.py",
    "kokoro_model": ROOT / "kokoro/kokoro/model.py",
    "kokoro_pipeline": ROOT / "kokoro/kokoro/pipeline.py",
}


def sep(title="", width=88, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


def read_lines(path):
    return path.read_text().splitlines()


def find_line(path, needle):
    for idx, line in enumerate(read_lines(path), start=1):
        if needle in line:
            return idx, line.strip()
    return None, None


def ref(label, key, needle):
    path = FILES[key]
    line_no, line = find_line(path, needle)
    return {
        "label": label,
        "path": path,
        "line_no": line_no,
        "line": line,
    }


refs = [
    ref("StyleEncoder definition", "models", "class StyleEncoder(nn.Module):"),
    ref("StyleEncoder global pool", "models", "nn.AdaptiveAvgPool2d(1)"),
    ref("Predictor duration head", "models", "self.duration_proj = LinearNorm(d_hid, max_dur)"),
    ref("Predictor F0/N head", "models", "def F0Ntrain(self, x, s):"),
    ref("Training s_trg build", "train_second", "s_trg = torch.cat([gs, s_dur], dim=-1).detach()"),
    ref("Training ref_s build", "train_second", "ref_s = torch.cat([ref_ss, ref_sp], dim=1)"),
    ref("Diffusion sample", "train_second", "s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(texts.device),"),
    ref("Predictor text encoder usage", "train_second", "d = model.predictor.text_encoder(d_en[bib, :, :input_lengths[bib]].unsqueeze(0),"),
    ref("Duration projection usage", "train_second", "duration = model.predictor.duration_proj(x)"),
    ref("Predictor F0/N inference usage", "train_second", "F0_pred, N_pred = model.predictor.F0Ntrain(en, s)"),
    ref("Kokoro pack lookup", "kokoro_pipeline", "return model(ps, pack[len(ps)-1], speed, return_output=True)"),
    ref("Kokoro style split", "kokoro_model", "s = ref_s[:, 128:]"),
    ref("Kokoro decoder style usage", "kokoro_model", "audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()"),
]


print("=" * 88)
print("  STYLE GRAPH TRACE: StyleTTS2 -> Candidate Kokoro Export")
print("=" * 88)

sep("SOURCE ANCHORS")
for item in refs:
    path = item["path"]
    line_no = item["line_no"]
    line = item["line"]
    print(f"  {item['label']}: {path.relative_to(ROOT)}:{line_no}")
    print(f"    {line}")

sep("TRAINING GRAPH")
print("  1. Utterance mel -> predictor_encoder(mel) -> s_dur (128-d prosodic style)")
print("  2. Utterance mel -> style_encoder(mel) -> gs (128-d acoustic style)")
print("  3. Concatenate -> s_trg = [gs || s_dur] (256-d style target)")
print("  4. Train diffusion model to denoise/reconstruct s_trg from text embedding")
print("  5. Predictor consumes the prosodic half for duration/F0/N generation")
print("  6. Decoder consumes the acoustic half for waveform generation")

sep("STYLETTS2 INFERENCE GRAPH")
print("  1. Text -> BERT -> bert_encoder -> d_en")
print("  2. sampler(...) -> s_pred (256-d)")
print("  3. Split s_pred into:")
print("     acoustic ref = s_pred[:, :128]")
print("     prosodic s   = s_pred[:, 128:]")
print("  4. predictor.text_encoder(d_en, s, ...) -> duration features")
print("  5. duration_proj -> predicted durations -> alignment")
print("  6. predictor.F0Ntrain(en, s) -> F0_pred, N_pred")
print("  7. decoder(asr/aligned text, F0_pred, N_pred, ref) -> waveform")

sep("KOKORO INFERENCE GRAPH")
print("  1. Text -> phonemes ps")
print("  2. Voicepack lookup -> ref_s = pack[len(ps)-1]")
print("  3. Split ref_s into:")
print("     acoustic ref = ref_s[:, :128]")
print("     prosodic s   = ref_s[:, 128:]")
print("  4. Reuse StyleTTS2-like predictor path with fixed s")
print("  5. Reuse StyleTTS2-like decoder path with fixed ref")
print("  6. Result: no runtime style_encoder and no runtime diffusion")

sep("DELTA TO EXPLAIN")
print("  StyleTTS2 ships a function that produces one 256-d style vector per inference call.")
print("  Kokoro ships a precomputed table of 510 such vectors per voice and indexes it by phoneme length.")
print("  Therefore Kokoro likely added an offline export/compilation step between StyleTTS2-style")
print("  training and final inference packaging.")

sep("CANDIDATE KOKORO EXPORT ALGORITHM")
print("  Pseudocode:")
print("  ")
print("  for each voice/reference identity:")
print("      ref_ss = style_encoder(reference_mel)         # acoustic half, 128-d")
print("      ref_sp = predictor_encoder(reference_mel)     # prosodic half, 128-d")
print("      ref_s  = concat(ref_ss, ref_sp)               # 256-d base style")
print("  ")
print("      for phoneme_len in 1..510:")
print("          text_prompt = canonical_prompt_with_length(phoneme_len)")
print("          bert_dur = bert(text_prompt)")
print("  ")
print("          # candidate export choice:")
print("          # use diffusion or a deterministic mapper to adapt the base style")
print("          # into a length-conditioned style vector while preserving identity")
print("          s_len = style_sampler_or_mapper(")
print("              embedding=bert_dur,")
print("              features=ref_s,")
print("              base_style=ref_s,")
print("              target_length=phoneme_len,")
print("          )")
print("  ")
print("          voicepack[phoneme_len - 1] = s_len        # 256-d")
print("  ")
print("      save voicepack as (510, 1, 256)")

sep("WHY THIS CANDIDATE FITS")
print("  - It preserves the 128+128 split used by both StyleTTS2 and Kokoro.")
print("  - It explains why Kokoro can bypass style encoders and diffusion at runtime.")
print("  - It explains the smooth length axis observed in shipped voicepacks.")
print("  - It matches multispeaker StyleTTS2's use of reference-conditioned diffusion features.")

sep("MAIN UNCERTAINTY")
print("  The unresolved part is the exporter itself: whether it used diffusion sampling,")
print("  deterministic interpolation, or another learned mapper to turn one reference")
print("  style identity into 510 length-conditioned slots.")

sep()
print("Done.")
