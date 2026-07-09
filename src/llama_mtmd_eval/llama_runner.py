"""Run llama-mtmd-cli for a case and return the raw OCR text.

Ports the exact command the llama.cpp in-repo test builds, so results match:
greedy (--temp 0), eager attention (--flash-attn off) to mirror the HF reference,
DRY sampling for the loop-prone models (v2/unlimited), and a larger context for
Unlimited-OCR's grounding output.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import Config
from .models import ModelSpec
from .scoring import strip_grounding

RUN_TIMEOUT = 300  # seconds


def build_cmd(spec: ModelSpec, image: Path, cfg: Config, flash_attn: bool = False) -> list[str]:
    ll = spec.llama
    cmd = [
        str(cfg.llama_bin),
        "-m", str(cfg.gguf(ll.gguf)),
        "--mmproj", str(cfg.gguf(ll.mmproj)),
        "--image", str(image),
        "-p", ll.prompt,
        "--chat-template", "deepseek-ocr",
        "--temp", "0",
        "--flash-attn", "on" if flash_attn else "off",
        "--no-warmup",
        "-n", str(ll.n_predict),
    ]
    if ll.dry:
        # HF decodes with no_repeat_ngram_size; llama.cpp's analog is DRY. Default DRY
        # breakers include "\n", so clear them with --dry-sequence-breaker none.
        cmd += [
            "--dry-multiplier", "0.8",
            "--dry-base", "1.75",
            "--dry-allowed-length", "2",
            "--dry-penalty-last-n", "-1",
            "--dry-sequence-breaker", "none",
        ]
    if ll.n_ctx is not None:
        cmd += ["-c", str(ll.n_ctx)]
    return cmd


def run_mtmd_cli(spec: ModelSpec, image: Path, cfg: Config, flash_attn: bool = False) -> str:
    cmd = build_cmd(spec, image, cfg, flash_attn)
    try:
        result = subprocess.run(cmd, capture_output=True, text=False, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        raise RuntimeError(f"llama-mtmd-cli timed out after {RUN_TIMEOUT}s\n{stderr}")
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-mtmd-cli failed ({result.returncode})\n{stderr}")
    output = result.stdout.decode("utf-8", errors="replace").strip()
    visible = strip_grounding(output) if spec.llama.strip_grounding else output
    if not visible.strip():
        raise RuntimeError("llama-mtmd-cli produced no output on stdout"
                           + (" (after grounding strip)" if output else ""))
    return output
