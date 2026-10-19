"""Ship screen, "Get the fixed model": publish the repaired model to the user's Hugging Face account.

The repo holds the repaired model itself, saved in HQQ's own format (`AutoHQQHFModel.save_quantized`:
config.json + qmodel.pt, loaded with `weights_only=True`), the tokenizer, patch.json and a model card.
Anyone with access then loads it in two lines, no FairGate code and no patching:

    from hqq.models.hf.base import AutoHQQHFModel
    model = AutoHQQHFModel.from_quantized("you/Model-fairgate")

Saved and loaded back, the model gives bit-identical outputs, its 8-bit parts staying 8-bit (tested).
`transformers`' own `from_pretrained` no longer loads saved HQQ models (5.x), so HQQ's loader it is.
Models whose layers are GPT-2 style `Conv1D` can't be loaded back by HQQ: they are refused, and the
API and the repair kit still serve them.

The token is used for this upload only: passed to `HfApi`, never written, logged or returned. Repos are
private unless the user unticks "Private".
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from src.product.repair_kit import headline, results_lines

REPO_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class PublishError(Exception):
    """Shown on the screen as it is: written for the user, never containing the token."""


def default_repo_name(model_id: str) -> str:
    name = re.split(r"[/\\]", model_id.rstrip("/\\"))[-1]  # a Hub id or a local folder
    return re.sub(r"[^A-Za-z0-9._-]", "-", name)[:80] + "-fairgate"


def full_repo_id(user: str, name: str, model_id: str) -> str:
    """`user/name`; a bare name goes under the token's account, an empty one gets the default."""
    name = (name or "").strip() or default_repo_name(model_id)
    owner, _, repo = name.rpartition("/")
    owner = owner or user
    if not REPO_NAME.match(repo) or not REPO_NAME.match(owner) or repo.endswith((".", "-")) or ".." in repo:
        raise PublishError(f"\"{name}\" isn't a valid repo name: use letters, digits, '-', '_' and '.'.")
    return f"{owner}/{repo}"


def load_code(repo_id: str, device: str = "cuda") -> str:
    """Ready-to-run Python (run as it is by the tests, on a local copy)."""
    return (f"# pip install transformers hqq accelerate\n"
            f"from hqq.models.hf.base import AutoHQQHFModel\n"
            f"from transformers import AutoTokenizer\n\n"
            f"model = AutoHQQHFModel.from_quantized(\"{repo_id}\", cache_dir=None, device=\"{device}\")\n"
            f"tokenizer = AutoTokenizer.from_pretrained(\"{repo_id}\")\n\n"
            f"question = \"Who was late, the nurse or the mechanic?\"\n"
            f"if tokenizer.chat_template:\n"
            f"    inputs = tokenizer.apply_chat_template([{{\"role\": \"user\", \"content\": question}}],\n"
            f"                                           add_generation_prompt=True, return_tensors=\"pt\", return_dict=True)\n"
            f"else:\n"
            f"    inputs = tokenizer(question, return_tensors=\"pt\")\n"
            f"inputs = inputs.to(\"{device}\")\n"
            f"output = model.generate(**inputs, max_new_tokens=80, do_sample=False,\n"
            f"                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)\n"
            f"print(tokenizer.decode(output[0, inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True))\n")


def model_card(patch: dict, summary: dict, repo_id: str) -> str:
    model_id = patch["model_id"]
    measured = results_lines(summary)
    return "\n".join([
        "---",
        f"base_model: {model_id}",
        "library_name: hqq",
        "tags: [fairgate, hqq, quantized, fairness, bias-repair]",
        "---", "",
        f"# {repo_id.split('/')[-1]}", "",
        f"**{model_id}, compressed with HQQ and repaired for fairness by FairGate.**", "",
        headline(patch, summary), "",
        *(measured or ["Its fairness numbers weren't measured on the full benchmark (a repair put together by "
                       "hand in FairGate Studio).", ""]),
        "## Use it", "",
        "```python", load_code(repo_id).rstrip(), "```", "",
        "A private repo needs you to be logged in (`huggingface-cli login`) with access to it.", "",
        "## What's inside", "",
        f"- Every linear layer (except the output head) HQQ-quantized to {patch['base']['nbits']}-bit, group "
        f"size 64; the {len(patch['modules'])} parts below kept at {patch['nbits']}-bit.",
        "- `patch.json`: the repair, for FairGate's own tools.", "",
        "Parts kept at 8-bit:", "",
        *[f"- `{path}`" for path in patch["modules"]], "",
        "The fairness check is BBQ's ambiguous questions (a fair model says \"can't tell\" when the story "
        "doesn't say who did it), on questions the repair never saw.", "",
    ])


def check_publishable(model) -> None:
    """HQQ's loader rebuilds the model from its config, so a model with GPT-2 style Conv1D layers
    (turned into Linear before quantizing) can't be loaded back: refuse before uploading anything."""
    if getattr(model.config, "model_type", "") in ("gpt2", "gpt_bigcode", "imagegpt") or any(
            type(module).__name__ == "Conv1D" for module in model.modules()):
        raise PublishError("This model's layers are GPT-2 style, which HQQ's loader can't rebuild, so it can't be "
                           "published as a ready model. Use it now (the API) or download the repair kit instead.")


def save_repaired(model, tokenizer, patch: dict, summary: dict, folder: Path, repo_id: str) -> Path:
    """Everything the repo holds, written to `folder`."""
    import json

    from hqq.models.hf.base import AutoHQQHFModel

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    AutoHQQHFModel.save_quantized(model, str(folder))
    tokenizer.save_pretrained(str(folder))
    (folder / "patch.json").write_text(json.dumps(patch, indent=2) + "\n", encoding="utf-8")
    (folder / "README.md").write_text(model_card(patch, summary, repo_id), encoding="utf-8")
    return folder


def _explain(error: Exception) -> str:
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status == 401:
        return ("Hugging Face didn't accept this token. Make a token with **Write** access at "
                "huggingface.co/settings/tokens and paste it again.")
    if status == 403:
        return ("This token can't create or write that repo. It needs **Write** access (a read-only or "
                "fine-grained token without repo write won't do), and the account must own the name you chose.")
    return f"Hugging Face refused the upload ({type(error).__name__}{f', HTTP {status}' if status else ''})."


def publish(model, tokenizer, patch: dict, summary: dict, token: str, repo_name: str = "",
            private: bool = True, api=None, workdir: Path | None = None) -> dict:
    """Creates (or updates) the repo and uploads the repaired model. Returns {"repo_id", "url", "private"}.
    `api` is for tests; by default `huggingface_hub.HfApi(token=token)`."""
    token = (token or "").strip()
    if not token:
        raise PublishError("Paste a Hugging Face token with Write access first (huggingface.co/settings/tokens).")
    check_publishable(model)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except Exception as error:  # noqa: BLE001 -- explained in plain words, without the token
        raise PublishError(_explain(error)) from None
    repo_id = full_repo_id(user, repo_name, patch["model_id"])
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = save_repaired(model, tokenizer, patch, summary, Path(tmp) / "repo", repo_id)
        try:
            api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
            api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                              commit_message=f"FairGate repair of {patch['model_id']} "
                                             f"({len(patch['modules'])} parts at {patch['nbits']}-bit)")
        except Exception as error:  # noqa: BLE001
            raise PublishError(_explain(error)) from None
    return {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}", "private": private}
