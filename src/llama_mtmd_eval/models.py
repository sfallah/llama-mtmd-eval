"""Model registry — one entry per family carrying BOTH the llama.cpp side and the
HF reference side, loaded from cases/models.toml.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .config import CASES_DIR, repo_data_file


@dataclass(frozen=True)
class LlamaSpec:
    gguf: str
    mmproj: str
    prompt: str = "Free OCR."
    n_predict: int = 2048
    n_ctx: int | None = None
    dry: bool = False
    strip_grounding: bool = False


@dataclass(frozen=True)
class HFSpec:
    dir: str
    env: str              # "hf-deepseek" | "hf-unlimited" (which conflict-extra it needs)
    prompt: str
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    eval_mode: bool = True
    strip_grounding: bool = False
    infer_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    llama: LlamaSpec
    hf: HFSpec


def load_models(path: Path | None = None) -> dict[str, ModelSpec]:
    path = repo_data_file(path or (CASES_DIR / "models.toml"))
    with open(path, "rb") as f:
        raw = tomllib.load(f)["models"]
    out: dict[str, ModelSpec] = {}
    for key, m in raw.items():
        out[key] = ModelSpec(
            key=key,
            label=m["label"],
            family=m["family"],
            llama=LlamaSpec(**m["llama"]),
            hf=HFSpec(**m["hf"]),
        )
    return out
