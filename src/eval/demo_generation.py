"""Open-ended text generation for the live demo — NOT used by the benchmark loop.

The benchmark's option-scoring evaluator (option_scorer.py) deliberately never generates text
(CLAUDE.md: "No free-text generation for the benchmark loop") — logits over A/B/C are fast and
deterministic, which matters when the same benchmark gets re-run dozens of times (Stage 4's
search) or in full (Stage 5's verification). But a live demo showing a biased answer flip to a
correct one after repair needs the model to actually SAY something in a sentence, not just lean
toward a letter internally — a bare "B" on screen doesn't land with an audience.

Kept in its own module, separate from option_scorer.py, so this never gets accidentally called
from the real evaluation path.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.data.bbq_loader import BBQItem


def format_open_ended_prompt(item: BBQItem, tokenizer: PreTrainedTokenizerBase) -> str:
    """Builds a natural-language prompt with no A/B/C letters — demo generation only."""
    content = (
        f"{item.context}\n\n{item.question}\n\n"
        "Answer in a single sentence. If the context doesn't give enough information to "
        "answer, say so clearly instead of guessing."
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    item: BBQItem,
    device: str,
    max_new_tokens: int = 60,
) -> str:
    """Generates a natural-language answer to `item`, for the live demo.

    Greedy decoding (`do_sample=False`), deliberately — a live demo needs to say the same thing
    every time it's rehearsed and every time it's actually presented, not risk a different
    answer live in front of an audience.
    """
    prompt_text = format_open_ended_prompt(item, tokenizer)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
