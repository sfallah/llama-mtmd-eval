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


def repo_data_file(path: Path) -> Path:
    """The bundled cases/ + data/ ship with the git checkout, not the wheel."""
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run from a repo checkout (`uv run mtmd-eval`); "
            "installed wheels do not bundle cases/ and data/."
        )
    return path

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
    hf_dir: Path | None = None   # only needed by the hf/compare commands
    device: str = "auto"

    def gguf(self, rel: str) -> Path:
        return self.gguf_dir / rel

    def hf_model(self, rel: str) -> Path:
        if self.hf_dir is None:
            raise SystemExit(f"hf_dir is not configured (set it in config.toml or {_ENV['hf_dir']})")
        return self.hf_dir / rel

    def image(self, name: str) -> Path:
        return DATA_DIR / "images" / name

    def ground_truth(self, name: str) -> Path:
        return DATA_DIR / "ground_truth" / name


def load_config(config_path: Path | None = None, overrides: dict | None = None,
                *, need_hf: bool = False) -> Config:
    """Merge defaults < config.toml < env vars < explicit overrides (CLI)."""
    values: dict[str, str] = {}

    if config_path and not config_path.exists():
        raise SystemExit(f"config file not found: {config_path}")
    path = config_path or (PROJECT_ROOT / "config.toml")
    if path.exists():
        with open(path, "rb") as f:
            values.update({k: v for k, v in tomllib.load(f).items() if k in _ENV})

    for key, env in _ENV.items():
        if os.environ.get(env):
            values[key] = os.environ[env]

    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})

    required = ("llama_bin", "gguf_dir") + (("hf_dir",) if need_hf else ())
    missing = [k for k in required if k not in values]
    if missing:
        raise SystemExit(
            f"missing config: {', '.join(missing)}. Copy config.example.toml -> "
            f"config.toml (or set {', '.join(_ENV[k] for k in missing)})."
        )

    return Config(
        llama_bin=Path(values["llama_bin"]).expanduser(),
        gguf_dir=Path(values["gguf_dir"]).expanduser(),
        hf_dir=Path(values["hf_dir"]).expanduser() if "hf_dir" in values else None,
        device=values.get("device", "auto"),
    )
