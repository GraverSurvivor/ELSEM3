"""Reopening a saved model (tab 6): the repaired model rebuilt from what the database holds, and checked to
be the model that was measured.

Where it comes from, in order:
1. its saved weights on Hugging Face, when the run has a link (a private repo needs this machine to have
   access: the token is never stored);
2. its recipe: the compressed model built layer by layer from the public checkpoint at the saved commit,
   with the recipe's parts at their bits (the same code the Ship screen uses).

Then it answers the run's 50 fingerprint questions. HQQ's output can change with library versions, so a copy
that answers fewer than 48 the same is shown with a warning naming the versions that differ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.db.importer import library_versions
from src.db.records import patch_bits
from src.db.scrub import safe_error

WARN_BELOW = 48 / 50
_HUB_URL = re.compile(r"^https://huggingface\.co/([A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95})/?$")


class CannotReopen(RuntimeError):
    """Shown on the screen as it is."""


@dataclass
class Reopened:
    model: object
    tokenizer: object
    device: str
    source: str           # in words: where it came from
    checkpoint: str       # the original's files: a folder, or a Hub id
    notes: list[str] = field(default_factory=list)


def recipe_of(run: dict) -> dict:
    """The run's recipe, or CannotReopen saying why there is no model to open."""
    artifact = run.get("artifact") or {}
    report = run.get("report") or {}
    if report.get("synthetic") or report.get("replayed_from") or run.get("pipeline") in ("fake", "replay"):
        raise CannotReopen("This run is a stand-in or a replay: there is no real model to open.")
    recipe = artifact.get("recipe")
    if not recipe:
        raise CannotReopen("This run ended without a repair, so there is no repaired model to open.")
    return recipe


def saved_weights_repo(run: dict) -> str | None:
    match = _HUB_URL.match((run.get("artifact") or {}).get("weights_url") or "")
    return match.group(1) if match else None


def checkpoint_for(model_id: str, revision: str | None) -> str:
    """The original model's files: a folder, at the saved commit when it is known. The local copy if there
    is one (no network, no token), else downloaded; a Hub id only when no commit was saved."""
    from src.product.studio import cached_copy

    if Path(model_id).is_dir() or not revision:
        return cached_copy(model_id)
    from huggingface_hub import snapshot_download

    files = ["*.json", "*.model", "*.txt", "tokenizer*", "*.tiktoken", "*.safetensors"]
    try:
        folder = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
        if any(folder.rglob("*.safetensors")) or any(folder.rglob("*.bin")):
            return str(folder)
    except Exception:  # noqa: BLE001 -- not cached: download it
        pass
    folder = Path(snapshot_download(model_id, revision=revision, allow_patterns=files))
    if not any(folder.rglob("*.safetensors")):  # an older checkpoint: weights only as .bin
        folder = Path(snapshot_download(model_id, revision=revision, allow_patterns=files + ["*.bin"]))
    return str(folder)


def load_weights(repo: str, device: str):
    """A published model, in HQQ's own format (torch.load(weights_only=True): data, never code)."""
    import torch
    from hqq.models.hf.base import AutoHQQHFModel
    from transformers import AutoTokenizer

    from src.eval.option_scorer import prepare_tokenizer

    model = AutoHQQHFModel.from_quantized(repo, cache_dir=None, compute_dtype=torch.float16, device=device)
    model.eval()
    return model, prepare_tokenizer(AutoTokenizer.from_pretrained(repo))


def reopen(run: dict, device: str) -> Reopened:
    """The run's repaired model on `device`."""
    recipe = recipe_of(run)
    model_id, base = recipe["model_id"], recipe["base"]["nbits"]
    revision = recipe.get("revision") or run.get("revision")
    notes = []
    repo = saved_weights_repo(run)
    if repo:
        try:
            model, tokenizer = load_weights(repo, device)
            # The original's files are only needed to show the compressed answers: looked up then.
            return Reopened(model, tokenizer, device, f"saved weights, {repo}", model_id, notes)
        except Exception as error:  # noqa: BLE001 -- gone, or private without access here: rebuild it
            notes.append(f"Its saved weights ({repo}) couldn't be opened here ({type(error).__name__}), so it was "
                         "rebuilt from its recipe.")
    from src.quant.streaming import load_streaming

    try:
        checkpoint = checkpoint_for(model_id, revision)
        model, tokenizer = load_streaming(checkpoint, base, device=device, keep=patch_bits(recipe))
    except Exception as error:
        raise CannotReopen(f"{model_id} couldn't be rebuilt: {safe_error(error)}") from None
    where = f", commit {revision[:7]}" if revision else ""
    if not revision:
        notes.append("Its commit wasn't saved (a run from before November 2026), so this is the model as it is today.")
    return Reopened(model, tokenizer, device, f"rebuilt from its recipe ({model_id}{where})", checkpoint, notes)


# --- is it the model that was measured? ------------------------------------------------------------

@dataclass
class Fingerprint:
    asked: int
    matched: int
    missing: int            # fingerprint questions this copy of the benchmark doesn't have
    differing: list[str]    # libraries whose version isn't the one it was measured with

    @property
    def ok(self) -> bool:
        return self.asked > 0 and not self.missing and self.matched >= WARN_BELOW * self.asked

    def text(self) -> str:
        if self.asked == 0:
            return ("No fingerprint was saved with this run, so this copy can't be compared with the model that was "
                    "measured." + (f" {self.versions_text()}" if self.differing else ""))
        line = f"matches what was measured, {self.matched}/{self.asked}"
        if self.ok:
            return "✅ " + line[0].upper() + line[1:]
        lost = f" ({self.missing} of its questions aren't in this copy of the benchmark)" if self.missing else ""
        return (f"⚠ Only {line}{lost}: this copy is not exactly the model the numbers describe. "
                + self.versions_text())

    def versions_text(self) -> str:
        return ("Library versions that differ: " + "; ".join(self.differing) + "." if self.differing
                else "The library versions are the same, or weren't saved.")


def version_differences(saved: dict | None, now: dict | None = None) -> list[str]:
    now = library_versions() if now is None else now
    return [f"{name} {now.get(name) or 'not installed'} (measured with {was})" for name, was in (saved or {}).items()
            if was and now.get(name) != was]


def benchmark_items() -> dict[str, dict]:
    """set name (as answers.json has it) -> question id -> question."""
    from src.data.bbq_loader import load_all_categories

    return {name: {it.question_id: it for items in load_all_categories(context).values() for it in items}
            for name, context in (("ambig", "ambig"), ("stated", "disambig"))}


def check_fingerprint(run: dict, model, tokenizer, device: str, items: dict[str, dict] | None = None,
                      batch_size: int = 16) -> Fingerprint:
    """Asks the model the run's fingerprint questions, scored as the diagnosis scored them."""
    from src.eval.option_scorer import score_items

    artifact = run.get("artifact") or {}
    saved = artifact.get("fingerprint") or []
    differing = version_differences(artifact.get("versions") or run.get("versions"))
    if not saved:
        return Fingerprint(0, 0, 0, differing)
    items = benchmark_items() if items is None else items
    known = [(entry, items.get(entry["set"], {}).get(entry["question_id"])) for entry in saved]
    asked = [(entry, item) for entry, item in known if item is not None]
    picks = score_items(model, tokenizer, [item for _, item in asked], device, batch_size=batch_size) if asked else []
    matched = sum(int(pick) == int(entry["pick"]) for pick, (entry, _) in zip(picks, asked))
    return Fingerprint(len(saved), matched, len(saved) - len(asked), differing)
