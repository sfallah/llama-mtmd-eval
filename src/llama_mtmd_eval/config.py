"""Runtime paths + device, resolved with precedence CLI > env > config.toml > defaults.

External resources (the llama-mtmd-cli binary, the GGUF dir, the HF model dir) live
outside this repo and differ per machine, so they are configured here. The bundled
test data (images + ground truths) lives in the repo and is found relative to the
project root.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

# src/llama_mtmd_eval/config.py -> project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CASES_DIR = PROJECT_ROOT / "cases"

_ENV = {
    "llama_bin": "MTMD_LLAMA_BIN",
    "gguf_dir": "MTMD_GGUF_DIR",
    "hf_dir": "MTMD_HF_DIR",
    "device": "MTMD_DEVICE",
}


@dataclass
class Config:
    llama_bin: Path
    gguf_dir: Path
    hf_dir: Path
    device: str = "auto"

    def gguf(self, rel: str) -> Path:
        return self.gguf_dir / rel

    def hf_model(self, rel: str) -> Path:
        return self.hf_dir / rel

    def image(self, name: str) -> Path:
        return DATA_DIR / "images" / name

    def ground_truth(self, name: str) -> Path:
        return DATA_DIR / "ground_truth" / name


def load_config(config_path: Path | None = None, overrides: dict | None = None) -> Config:
    """Merge defaults < config.toml < env vars < explicit overrides (CLI)."""
    values: dict[str, str] = {}

    path = config_path or (PROJECT_ROOT / "config.toml")
    if path.exists():
        with open(path, "rb") as f:
            values.update({k: v for k, v in tomllib.load(f).items() if k in _ENV})

    for key, env in _ENV.items():
        if os.environ.get(env):
            values[key] = os.environ[env]

    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})

    missing = [k for k in ("llama_bin", "gguf_dir", "hf_dir") if k not in values]
    if missing:
        raise SystemExit(
            f"missing config: {', '.join(missing)}. Copy config.example.toml -> "
            f"config.toml (or set {', '.join(_ENV[k] for k in missing)})."
        )

    return Config(
        llama_bin=Path(values["llama_bin"]).expanduser(),
        gguf_dir=Path(values["gguf_dir"]).expanduser(),
        hf_dir=Path(values["hf_dir"]).expanduser(),
        device=values.get("device", "auto"),
    )
