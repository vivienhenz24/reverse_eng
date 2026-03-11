"""
Audit the Turkish front-end path for Kokoro using local espeak-ng.

Goal:
  determine whether Turkish can fit into the current Kokoro IPA vocab with only
  light normalization, or whether vocab expansion is required immediately.
"""

from pathlib import Path
import json
import subprocess
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "kokoro/weights/config.json"

SAMPLES = [
    "Merhaba dunya. Bugun hava cok guzel.",
    "Turkce ses sentezi icin yeni bir model egitiyoruz.",
    "Istanbul, Ankara, Izmir ve Bursa buyuk sehirlerdir.",
    "Soguk ruzgar denizden esiyor.",
    "Cocuklar okuldan sonra bahcede oyun oynuyor.",
    "Gunes dogarken kuslar otmeye basladi.",
    "Yagmur yaginca sokaklar sessizlesti.",
    "Doktor hastaya ilac kullanmasini soyledi.",
    "Kirmizi, yesil, mavi ve sari en bilinen renklerdir.",
]

NORMALIZE = {
    "\u200d": "",   # zero-width joiner inserted around some affricates
    "ɫ": "l",       # dark l -> plain l to fit current Kokoro vocab
}


def sep(title="", width=88, char="─"):
    if title:
        pad = width - len(title) - 2
        print(f"\n{char} {title} {char * (pad - 1)}")
    else:
        print(char * width)


def g2p(text: str) -> str:
    return subprocess.check_output(
        ["espeak-ng", "-q", "--ipa=3", "-v", "tr", text],
        text=True,
    ).strip()


def normalize(ipa: str) -> str:
    for src, dst in NORMALIZE.items():
        ipa = ipa.replace(src, dst)
    return ipa


cfg = json.loads(CONFIG_PATH.read_text())
vocab = set(cfg["vocab"])
raw = "\n".join(g2p(s) for s in SAMPLES)
norm = normalize(raw)

raw_chars = sorted(set(raw) - {"\n", " "})
norm_chars = sorted(set(norm) - {"\n", " "})
raw_missing = sorted(set(raw_chars) - vocab)
norm_missing = sorted(set(norm_chars) - vocab)

print("=" * 88)
print("  TURKISH FRONT-END AUDIT")
print("=" * 88)

sep("RAW ESPEAK IPA")
for s in SAMPLES:
    out = g2p(s)
    print(f"  text: {s}")
    print(f"  ipa:  {out}")

sep("CHARACTER COVERAGE")
print(f"  raw unique chars:        {''.join(raw_chars)}")
print(f"  raw missing from vocab:  {raw_missing}")
print(f"  normalized missing:      {norm_missing}")

sep("FREQUENCY OF RAW MISSING CHARS")
ctr = Counter(ch for ch in raw if ch not in {" ", "\n"} and ch not in vocab)
for ch, n in ctr.most_common():
    print(f"  {repr(ch)} count={n} mapped_to={repr(NORMALIZE.get(ch, None))}")

sep("CONCLUSION")
if not norm_missing:
    print("  Turkish looks feasible without immediate vocab expansion on this audit set.")
    print("  Minimal front-end normalization is:")
    print("    - strip zero-width joiner")
    print("    - map ɫ -> l")
else:
    print("  Additional vocab entries would be needed:")
    print(f"    {norm_missing}")

sep()
print("Done.")
