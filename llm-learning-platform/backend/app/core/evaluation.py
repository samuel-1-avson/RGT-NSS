"""
Model Evaluation & Benchmarking Engine

Implements perplexity, BLEU, ROUGE, and custom evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import Counter
import math


def compute_perplexity(losses: List[float]) -> float:
    """Compute perplexity from a list of cross-entropy losses."""
    if not losses:
        return float("inf")
    avg_loss = sum(losses) / len(losses)
    return float(np.exp(min(avg_loss, 100)))


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
    """Compute BLEU score (1-4 gram) between reference and hypothesis."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return {"bleu": 0.0, "brevity_penalty": 0.0, "precisions": [0.0] * max_n}

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1))

        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)
        precisions.append(clipped / total)

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))

    # Geometric mean of precisions
    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / max_n
    bleu = bp * math.exp(log_avg)

    return {
        "bleu": round(float(bleu), 4),
        "brevity_penalty": round(float(bp), 4),
        "precisions": [round(p, 4) for p in precisions],
    }


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    def ngram_overlap(ref: List[str], hyp: List[str], n: int) -> Dict[str, float]:
        ref_ng = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        hyp_ng = Counter(tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1))
        overlap = sum(min(ref_ng[ng], hyp_ng[ng]) for ng in ref_ng if ng in hyp_ng)
        precision = overlap / max(sum(hyp_ng.values()), 1)
        recall = overlap / max(sum(ref_ng.values()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    # ROUGE-L (longest common subsequence)
    def lcs_length(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs = lcs_length(ref_tokens, hyp_tokens)
    r_prec = lcs / max(len(hyp_tokens), 1)
    r_recall = lcs / max(len(ref_tokens), 1)
    r_f1 = 2 * r_prec * r_recall / max(r_prec + r_recall, 1e-10)

    return {
        "rouge1": ngram_overlap(ref_tokens, hyp_tokens, 1),
        "rouge2": ngram_overlap(ref_tokens, hyp_tokens, 2),
        "rougeL": {"precision": round(r_prec, 4), "recall": round(r_recall, 4), "f1": round(r_f1, 4)},
    }


def run_benchmark_suite(model_name: str = "MicroGPT") -> Dict:
    """Run real benchmark evaluation using an actual MicroGPT model."""
    from app.core.model import MicroGPT, PRESET_CONFIGS

    config = PRESET_CONFIGS.get("nano", PRESET_CONFIGS["nano"])
    model = MicroGPT(config)
    model.set_training(False)

    eval_texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Language models learn statistical patterns from large text corpora.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
    ]
    reference_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn patterns from data.",
    ]

    # Real perplexity computation on each eval text
    perplexity_results = {}
    all_losses = []
    for idx, text in enumerate(eval_texts):
        token_ids = np.array([[ord(c) % config.vocab_size for c in text]])
        inputs = token_ids[:, :-1]
        targets = token_ids[:, 1:]
        result = model.forward(inputs, targets)
        loss = result["loss"]
        all_losses.append(loss)
        perplexity_results[f"eval_text_{idx+1}"] = round(float(np.exp(min(loss, 20))), 2)

    # Real text generation + BLEU/ROUGE evaluation
    prompt_ids = np.array([[ord(c) % config.vocab_size for c in "The "]])
    generated_ids, _ = model.generate(prompt_ids, max_new_tokens=30, temperature=0.8)
    generated_text = "".join(chr(max(0, min(t, 127))) for t in generated_ids[0])

    bleu_result = compute_bleu(reference_texts[0], generated_text)
    rouge_result = compute_rouge(reference_texts[0], generated_text)

    benchmarks = {
        "perplexity": perplexity_results,
        "aggregate_perplexity": round(float(compute_perplexity(all_losses)), 2),
        "generation_quality": {
            "avg_bleu": bleu_result["bleu"],
            "bleu_precisions": bleu_result["precisions"],
            "avg_rouge1_f1": rouge_result["rouge1"]["f1"],
            "avg_rouge2_f1": rouge_result["rouge2"]["f1"],
            "avg_rougeL_f1": rouge_result["rougeL"]["f1"],
        },
        "model_info": {
            "name": model_name,
            "parameters": config.num_parameters,
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "vocab_size": config.vocab_size,
        },
        "generated_sample": generated_text[:200],
    }

    return {"model_name": model_name, "benchmarks": benchmarks}


def compare_models(model_names: List[str]) -> Dict:
    """Compare multiple models on real benchmarks."""
    results = {}
    for name in model_names:
        results[name] = run_benchmark_suite(name)["benchmarks"]
    return {"models": results}
