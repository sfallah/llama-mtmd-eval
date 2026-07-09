"""Case registry — the evaluation cases (model + image + ground truth + baseline +
tolerances), loaded from cases/cases.toml.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import CASES_DIR, repo_data_file


@dataclass(frozen=True)
class Case:
    model: str
    label: str
    image: str
    ground_truth: str
    cer: float          # baseline CER (HF reference, or provisional)
    chrf: float         # baseline chrF
    cer_tol: float
    chrf_tol: float
    flash_attn: bool = False
    provisional: bool = False   # baseline is a llama self-measurement, not an HF ref

    @property
    def cer_max(self) -> float:
        return self.cer + self.cer_tol

    @property
    def chrf_min(self) -> float:
        return self.chrf - self.chrf_tol


def load_cases(path: Path | None = None, known_models: set[str] | None = None) -> list[Case]:
    path = repo_data_file(path or (CASES_DIR / "cases.toml"))
    with open(path, "rb") as f:
        raw = tomllib.load(f)["cases"]
    cases = []
    for c in raw:
        base = c["baseline"]
        tol = c["tol"]
        cases.append(Case(
            model=c["model"],
            label=c["label"],
            image=c["image"],
            ground_truth=c["ground_truth"],
            cer=base["cer"],
            chrf=base["chrf"],
            cer_tol=tol["cer"],
            chrf_tol=tol["chrf"],
            flash_attn=c.get("flash_attn", False),
            provisional=c.get("provisional", False),
        ))
    if known_models is not None:
        unknown = {c.model for c in cases} - known_models
        if unknown:
            raise ValueError(f"cases reference unknown model keys: {sorted(unknown)}")
    return cases
