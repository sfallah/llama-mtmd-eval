"""Torch-free unit tests for the shared scoring pipeline."""
from llama_mtmd_eval.scoring import (
    compute_cer,
    locally_align,
    normalize_text,
    score,
    strip_grounding,
)


def test_normalize_collapses_whitespace():
    assert normalize_text("a\n  b\t c\n") == "a b c"


def test_strip_grounding_removes_ref_and_det():
    t = "keep <|ref|>x<|/ref|> and <|det|>[[1,2]]<|/det|> tail"
    assert strip_grounding(t) == "keep  and  tail"


def test_score_identical_is_perfect():
    s = score("The quick brown fox", "The quick brown fox")
    assert s.cer == 0.0
    assert s.chrf == 100.0


def test_locally_align_extracts_body_span():
    gt = "the moon landing was historic"
    ocr = "NAV MENU HOME the moon landing was historic FOOTER COPYRIGHT"
    assert locally_align(gt, normalize_text(ocr)) == gt


def test_compute_cer_one_substitution():
    # one char substituted out of 4 -> CER 0.25
    assert compute_cer("abcd", "abxd") == 0.25


def test_score_strips_grounding_before_scoring():
    gt = "hello world"
    ocr = "<|det|>[[0,0,1,1]]<|/det|>hello world"
    assert score(ocr, gt, strip_grounding_markup=True).cer == 0.0
