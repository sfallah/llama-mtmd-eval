"""Run a case (llama or HF), score it, and grade against the baseline tolerances."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .cases import Case
from .config import Config
from .models import ModelSpec
from .scoring import Score, score


@dataclass
class Result:
    model: str
    label: str
    image: str
    source: str          # "llama" | "hf"
    cer: float
    chrf: float
    baseline_cer: float
    baseline_chrf: float
    cer_max: float
    chrf_min: float
    passed: bool
    provisional: bool = False
    approximate: bool = False   # HF run not on CUDA
    device: str = ""
    error: str = ""             # run/preflight failure; scores are meaningless
    ocr_text: str = ""          # raw model output (kept for --out json diagnosis)
    aligned_text: str = ""      # the normalized OCR span that was scored

    def as_dict(self) -> dict:
        return asdict(self)


def _grade(case: Case, s: Score) -> bool:
    return s.cer <= case.cer_max and s.chrf >= case.chrf_min


def _result(case: Case, spec: ModelSpec, s: Score, source: str, raw: str, **extra) -> Result:
    return Result(
        model=spec.key, label=case.label, image=case.image, source=source,
        cer=round(s.cer, 4), chrf=round(s.chrf, 2),
        baseline_cer=case.cer, baseline_chrf=case.chrf,
        cer_max=round(case.cer_max, 4), chrf_min=round(case.chrf_min, 2),
        passed=_grade(case, s), provisional=case.provisional,
        ocr_text=raw, aligned_text=s.aligned, **extra,
    )


def error_result(spec: ModelSpec, case: Case, source: str, message: str) -> Result:
    return Result(
        model=spec.key, label=case.label, image=case.image, source=source,
        cer=0.0, chrf=0.0, baseline_cer=case.cer, baseline_chrf=case.chrf,
        cer_max=round(case.cer_max, 4), chrf_min=round(case.chrf_min, 2),
        passed=False, provisional=case.provisional, error=message,
    )


def evaluate_llama(spec: ModelSpec, case: Case, cfg: Config) -> Result:
    from .llama_runner import run_mtmd_cli
    raw = run_mtmd_cli(spec, cfg.image(case.image), cfg, flash_attn=case.flash_attn)
    s = score(raw, cfg.ground_truth(case.ground_truth).read_text(encoding="utf-8"),
              strip_grounding_markup=spec.llama.strip_grounding)
    return _result(case, spec, s, "llama", raw)


def evaluate_hf(spec: ModelSpec, case: Case, cfg: Config, device: str = "auto") -> Result:
    from .hf_runner import run_hf
    hf = run_hf(spec, cfg.hf_model(spec.hf.dir), cfg.image(case.image), device=device)
    s = score(hf.text, cfg.ground_truth(case.ground_truth).read_text(encoding="utf-8"),
              strip_grounding_markup=spec.hf.strip_grounding)
    return _result(case, spec, s, "hf", hf.text, approximate=hf.approximate, device=hf.device)
