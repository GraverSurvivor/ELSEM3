"""CrowS-Pairs: a second bias benchmark, so the repair isn't judged on BBQ alone.

CrowS-Pairs (Nangia et al., 2020) has 1,508 sentence pairs that differ only in the group mentioned,
e.g. "...most black kids don't have the best education" vs "...most white kids...". `sent_more` is
the sentence about the historically disadvantaged group that expresses (or, for "antistereo" rows,
violates) a stereotype. Following the official metric, the score is the share of pairs where the
model finds `sent_more` more likely: 50% means no preference; higher means more stereotyping.

It is scored with the same log-likelihood principle as the BBQ harness -- no generation: each
sentence's total log-probability under the model, as raw text (no chat template, since these are
statements, not questions). A paired comparison between conditions uses the same pairs.

Known limitation worth stating: Blodgett et al. (2021) document noisy and mislabeled pairs in
CrowS-Pairs, so it is used here as a second, independent signal, not as ground truth.

Run on a GPU, once per condition:
    python -m src.eval.run_crows_pairs --model fp16
    python -m src.eval.run_crows_pairs --model hqq_3bit
    python -m src.eval.run_crows_pairs --model patch --patch results/patches/search_top5.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.experiments.logger import RESULTS_DIR, _write_atomically, new_run_id

DATASET_ID = "nyu-mll/crows_pairs"
DIRECTIONS = ("stereo", "antistereo")
BIAS_TYPES = ("race-color", "socioeconomic", "gender", "disability", "nationality",
              "sexual-orientation", "physical-appearance", "religion", "age")


def load_pairs() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="test", revision="refs/convert/parquet")
    return [
        {
            "id": int(row["id"]),
            "sent_more": row["sent_more"],
            "sent_less": row["sent_less"],
            "direction": DIRECTIONS[row["stereo_antistereo"]],
            "bias_type": BIAS_TYPES[row["bias_type"]],
        }
        for row in ds
    ]


@torch.no_grad()
def sentence_logprobs(model, tokenizer, sentences: list[str], device: str, batch_size: int = 16) -> list[float]:
    """Total log-probability of each sentence (sum over its tokens after the first)."""
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    scores = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        logits = model(**enc).logits[:, :-1, :].float()
        targets = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        token_lp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        scores.extend((token_lp * mask).sum(dim=-1).tolist())
    return scores


def summarize(records: list[dict]) -> dict:
    """Share preferring sent_more: overall, by direction and by bias type (50% = no preference)."""
    def share(rows):
        return sum(r["prefers_more"] for r in rows) / len(rows) if rows else None

    return {
        "n": len(records),
        "bias_score": share(records),
        "by_direction": {d: share([r for r in records if r["direction"] == d]) for d in DIRECTIONS},
        "by_bias_type": {b: share([r for r in records if r["bias_type"] == b]) for b in BIAS_TYPES},
    }


def main() -> None:
    from src.eval.model_variants import VARIANTS, load_variant

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=VARIANTS)
    parser.add_argument("--patch", type=Path, help="patch file, for --model patch")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, condition = load_variant(args.model, device, args.patch)
    pairs = load_pairs()
    more = sentence_logprobs(model, tokenizer, [p["sent_more"] for p in pairs], device, args.batch_size)
    less = sentence_logprobs(model, tokenizer, [p["sent_less"] for p in pairs], device, args.batch_size)

    records = [{**p, "lp_more": m, "lp_less": l, "prefers_more": m > l} for p, m, l in zip(pairs, more, less)]
    run_id = new_run_id(f"crows_{condition}")
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_atomically(out_dir / "pairs.jsonl", lambda h: h.write("\n".join(json.dumps(r) for r in records) + "\n"))
    summary = {"run_id": run_id, "condition": condition, "patch": str(args.patch) if args.patch else None,
               **summarize(records)}
    _write_atomically(out_dir / "summary.json", lambda h: json.dump(summary, h, indent=2))

    print(f"CrowS-Pairs bias score ({condition}): {summary['bias_score']:.1%}  (50% = no preference)")
    for bias_type, value in summary["by_bias_type"].items():
        print(f"  {bias_type:<20} {value:.1%}")
    print(f"Results: {out_dir.resolve()}")
    print(f"Next: copy this folder to results/reference/crows_{condition}/ and commit it.")


if __name__ == "__main__":
    main()
