from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import torch


ROOT = Path(__file__).resolve().parents[2]
KOKORO_ROOT = ROOT / "kokoro"
PACKAGE_ROOT = KOKORO_ROOT / "kokoro"
CONFIG_PATH = KOKORO_ROOT / "weights/config.json"
CHECKPOINT_PATH = KOKORO_ROOT / "weights/kokoro-v1_0.pth"
VOICES_DIR = KOKORO_ROOT / "weights/voices"
DEFAULT_TURKISH_MANIFEST = ROOT / "combined_dataset/kokoro_turkish_manifest.csv"
DEFAULT_CANONICAL_ALIGNMENTS = ROOT / "alignments_kokoro_tr"
DEFAULT_F0_PATH = ROOT / "StyleTTS2/Utils/JDC/bst.t7"


def load_local_kmodel():
    pkg_name = "_local_kokoro_training_pkg"

    if "kokoro" not in sys.modules:
        public_pkg = types.ModuleType("kokoro")
        public_pkg.__path__ = [str(PACKAGE_ROOT)]
        sys.modules["kokoro"] = public_pkg

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(PACKAGE_ROOT)]
        sys.modules[pkg_name] = pkg

    for name in ["custom_stft", "istftnet", "modules", "model"]:
        full_name = f"{pkg_name}.{name}"
        if full_name in sys.modules:
            continue
        path = PACKAGE_ROOT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        sys.modules[f"kokoro.{name}"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules[f"{pkg_name}.model"].KModel


def load_styletts2_symbols_and_losses():
    styletts_root = ROOT / "StyleTTS2"
    losses_spec = importlib.util.spec_from_file_location("_styletts2_losses", styletts_root / "losses.py")
    losses_mod = importlib.util.module_from_spec(losses_spec)
    assert losses_spec is not None and losses_spec.loader is not None
    losses_spec.loader.exec_module(losses_mod)

    jdc_spec = importlib.util.spec_from_file_location("_styletts2_jdc", styletts_root / "Utils/JDC/model.py")
    jdc_mod = importlib.util.module_from_spec(jdc_spec)
    assert jdc_spec is not None and jdc_spec.loader is not None
    jdc_spec.loader.exec_module(jdc_mod)

    def load_f0_models(path):
        f0_model = jdc_mod.JDCNet(num_class=1, seq_len=192)
        params = torch.load(path, map_location="cpu")["net"]
        f0_model.load_state_dict(params)
        _ = f0_model.train()
        return f0_model

    def log_norm(x, mean=-4, std=4, dim=2):
        return torch.log(torch.exp(x * std + mean).norm(dim=dim))

    return losses_mod.MultiResolutionSTFTLoss, load_f0_models, log_norm


def mean_released_pack() -> torch.Tensor:
    packs = []
    for path in sorted(VOICES_DIR.glob("*.pt")):
        packs.append(torch.load(path, map_location="cpu", weights_only=True).squeeze(1).float())
    return torch.stack(packs).mean(0)


def named_pack(name: str) -> torch.Tensor:
    return torch.load(VOICES_DIR / f"{name}.pt", map_location="cpu", weights_only=True).squeeze(1).float()


def init_voicepacks(trainer, mode: str):
    with torch.no_grad():
        if mode == "random":
            return
        if mode == "mean":
            base = mean_released_pack()
        elif mode.startswith("voice:"):
            base = named_pack(mode.split(":", 1)[1])
        else:
            raise ValueError(mode)
        trainer.voicepacks.table[0].copy_(base)
        trainer.voicepacks.table[1].copy_(base)
