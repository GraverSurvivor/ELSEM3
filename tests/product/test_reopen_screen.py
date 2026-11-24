"""Tab 6 on a real tiny model: reopened from its recipe or its saved weights, asked, served, published, kept to
one model at a time, and deleted."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hqq")
pytest.importorskip("gradio")
pytest.importorskip("sqlalchemy")

from src.data.bbq_loader import BBQItem  # noqa: E402
from src.db.store import Store, sqlite_engine  # noqa: E402
from tests.product.test_recording import data  # noqa: E402, F401 -- the Studio's data, loaded once
from tests.product.test_recording import STORY  # noqa: E402

PART = "model.layers.0.self_attn.q_proj"
OTHER = "model.layers.1.mlp.down_proj"


def _item(i, context="ambig"):
    return BBQItem(example_id=i, category="Age", question_polarity="neg", context=f"{context} story {i}", question="Who?",
                   options=("The old one", "The young one", "Unknown"), label=2 if context == "ambig" else 0,
                   unknown_index=2, stereotype_index=0)


ITEMS = {"ambig": {f"Age:{i}": _item(i) for i in range(30)}, "stated": {f"Age:{i}": _item(i, "stated") for i in range(30)}}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """(folder of a tiny Qwen2 with Qwen's tokenizer, its repair's recipe, the repaired model)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from src.quant.streaming import load_streaming

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    folder = tmp_path_factory.mktemp("models") / "tiny"
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(AutoConfig.for_model(
        "qwen2", vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, initializer_range=0.5)).save_pretrained(folder)
    tok.save_pretrained(folder)
    recipe = {"name": "t", "model_id": str(folder), "modules": [PART, OTHER], "nbits": 8, "module_bits": {OTHER: 4},
              "base": {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"}}
    return folder, recipe, load_streaming(str(folder), 3, device="cpu", keep={PART: 8, OTHER: 4})


@pytest.fixture
def screen(tmp_path, data, tiny, monkeypatch, request):  # noqa: F811
    from src.eval.option_scorer import score_items
    from src.product import reopen, studio
    from src.product.api_server import ApiService
    from src.product.byom import JobRunner
    from src.product.studio_core import SavedBackend

    folder, recipe, (model, tok) = tiny
    store = Store(sqlite_engine(tmp_path / "x.sqlite"), ran_by="Asha")
    request.addfinalizer(store.close)
    entries = [(s, q, it) for s, table in ITEMS.items() for q, it in list(table.items())[:25]]
    picks = score_items(model, tok, [it for _, _, it in entries], "cpu", batch_size=16)
    report = {"model_id": str(folder), "arch": "qwen2", "mode": "fast", "original": {"usr": 0.8}, "compressed": {"usr": 0.4},
              "repaired": {"usr": 0.7}, "recovery": {"point": 0.75, "lower": 0.6, "upper": 0.9}, "kept": [PART, OTHER],
              "gate": "PASS", "gate_failures": [], "outcome": "repaired", "memory_mb": 0.1,
              "fingerprint": [{"question_id": q, "set": s, "pick": int(p)} for (s, q, _), p in zip(entries, picks)]}
    store.start_run("tiny", str(folder), mode="fast", job="20261125T100000000000Z_tiny")
    store.finish_run("tiny", "done", report=report, patch=recipe)
    store.save_answers("tiny", {"ambig": {"Age:0": {"split": "final", "original": 1, "compressed": 0, "repaired": 2}}})
    monkeypatch.setattr(reopen, "benchmark_items", lambda: ITEMS)

    uploads = []

    def fake_publish(model, tokenizer, patch, summary, token, repo_name="", private=True):
        uploads.append({"model": model, "patch": patch, "private": private, "token": token})
        return {"repo_id": "asha/tiny-fairgate", "url": "https://huggingface.co/asha/tiny-fairgate", "private": private}

    monkeypatch.setattr(studio, "publish", fake_publish)
    monkeypatch.setattr(studio, "OUT_DIR", tmp_path / "out")
    backend = SavedBackend(data.runs, list(data.patch.modules), repairs=data.saved_repairs)
    api = ApiService()
    runner = JobRunner(tmp_path / "jobs", pipeline="real", store=store)
    app, _ = studio.build_app(backend, data, studio.ByomSetup(runner), api)
    handlers = {f.fn.__name__: f.fn for f in app.fns.values() if getattr(f, "fn", None) is not None}
    return type("Screen", (), dict(h=handlers, store=store, api=api, uploads=uploads, runner=runner, report=report,
                                   tmp_path=tmp_path))


def logits(model, tok, text="Who was late?"):
    with torch.no_grad():
        return model(**tok(text, return_tensors="pt")).logits


# --- the saved model is exactly the one that was measured -------------------------------------------

def test_a_saved_models_three_views(tiny):
    """Repaired = the recipe, bit for bit; compressed = the plain base model, bit for bit; original = saved picks."""
    from src.product.studio import LiveBackend
    from src.quant.streaming import load_streaming

    folder, recipe, (repaired, tok) = tiny
    plain, _ = load_streaming(str(folder), 3, device="cpu")
    backend = LiveBackend(device="cpu", model_id=str(folder), patch=recipe, saved_original={"Age:0": 1}, extra_mb=1.5)
    assert torch.equal(logits(backend.model, tok), logits(repaired, tok))
    with backend._compressed() as model:
        assert torch.equal(logits(model, tok), logits(plain, tok))
    assert torch.equal(logits(backend.model, tok), logits(repaired, tok))  # put back exactly
    assert backend.restored() == [PART, OTHER] and backend.overhead_mb() == 1.5
    assert backend.model.model.layers[1].mlp.down_proj.meta["nbits"] == 4  # each part at its own bits (C7)
    answers = backend.answers(ITEMS["ambig"]["Age:0"], False)
    assert answers["original"].pick == 1 and answers["compressed"].probs and answers["repaired"].probs
    assert backend.answers(ITEMS["ambig"]["Age:1"], False)["original"] is None
    with pytest.raises(RuntimeError, match="can't be changed"):
        backend.set_restored([])
    assert not backend.release_original()


def test_re_check_matches_and_says_so(screen):
    note = screen.h["mm_check_model"]("tiny")
    assert "✅ Matches what was measured, 50/50" in note and "rebuilt from its recipe" in note
    assert "commit wasn't saved" in note  # said, not hidden


def test_a_copy_that_answers_differently_is_shown_with_the_versions_that_differ(screen):
    wrong = [{**e, "pick": (e["pick"] + 1) % 3} for e in screen.report["fingerprint"][:5]]
    screen.store.finish_run("tiny", "done", report={**screen.report, "fingerprint": wrong + screen.report["fingerprint"][5:]},
                            patch=screen.store.run("tiny")["artifact"]["recipe"], versions={"hqq": "0.1.0"})
    note = screen.h["mm_check_model"]("tiny")
    assert "⚠ Only matches what was measured, 45/50" in note and "(measured with 0.1.0)" in note and "hqq" in note


# --- asking, serving, publishing ---------------------------------------------------------------------

def test_the_ask_screen_asks_the_reopened_model_until_told_to_go_back(screen):
    note, _, row, text = screen.h["mm_ask_model"]("tiny")
    assert "50/50" in note and row["visible"] is True and "tiny" in text
    out = screen.h["ask_custom"](*STORY, False)
    cards = out[4:]
    assert "original model isn&#x27;t loaded" in cards[0] and "2 parts restored" in cards[2]
    assert not any("GB" in card for card in cards) and ">3-bit<" in cards[1]  # not Phi-3.5's sizes
    assert "answer orders" in cards[1] and "answer orders" in cards[2]
    [asked] = screen.store.recent_questions()
    assert asked["run_id"] == "tiny" and set(asked["answers"]) == {"compressed", "repaired"}
    assert screen.h["ask_back"]()[0]["visible"] is False
    assert "needs the live app" in screen.h["ask_custom"](*STORY, False)[5]  # the preview backend again


def test_serving_then_publishing_uses_the_one_held_model(screen):
    note, _, *shown = screen.h["mm_serve_model"]("tiny")
    assert screen.api.running and shown[2] == screen.api.key and "tiny-fairgate" in shown[0]
    served = screen.api.model
    assert screen.api.reply([{"role": "user", "content": "Hi"}], 3, 0.0)["choices"]
    message, box = screen.h["mm_publish_model"]("tiny", "hf_secret1", "", False)
    assert message.startswith("✅ **Published:**") and box == "" and screen.uploads[0]["model"] is served
    assert screen.uploads[0]["private"] is False and screen.api.running
    artifact = screen.store.run("tiny")["artifact"]
    assert artifact["weights_url"].endswith("asha/tiny-fairgate") and artifact["weights_private"] is False
    screen.h["api_stop"]()
    assert not screen.api.running


def test_keep_weights_is_always_private_and_the_link_is_used_to_reopen(screen, monkeypatch, tiny):
    from src.product import reopen

    message, box = screen.h["mm_keep_weights"]("tiny", "hf_secret1", "")
    assert "Weights kept" in message and box == "" and screen.uploads[0]["private"] is True
    screen.h["api_stop"]()  # frees it
    opened = []
    monkeypatch.setattr(reopen, "load_weights", lambda repo, device: (opened.append(repo) or tiny[2]))
    note = screen.h["mm_check_model"]("tiny")
    assert opened == ["asha/tiny-fairgate"] and "saved weights, asha/tiny-fairgate" in note and "50/50" in note
    screen.h["api_stop"]()

    def gone(repo, device):
        raise OSError("404 with token hf_secret1")

    monkeypatch.setattr(reopen, "load_weights", gone)
    note = screen.h["mm_check_model"]("tiny")
    assert "couldn't be opened here (OSError)" in note and "rebuilt from its recipe" in note and "hf_" not in note


def test_weights_published_and_loaded_back_are_the_same_model(tiny, tmp_path):
    from src.product.hub_publish import save_repaired
    from src.product.reopen import load_weights

    folder, recipe, (repaired, tok) = tiny
    saved = save_repaired(repaired, tok, recipe, {"memory_mb": 1}, tmp_path / "repo", "asha/tiny-fairgate")
    model, loaded_tok = load_weights(str(saved), "cpu")
    assert torch.equal(logits(model, loaded_tok), logits(repaired, tok))


# --- one model at a time -----------------------------------------------------------------------------

def test_nothing_is_opened_while_a_diagnosis_runs(screen, monkeypatch):
    job = screen.tmp_path / "jobs" / "20261125T110000000000Z_x"
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"model_id": "a/b", "mode": "fast", "ended": None}))
    monkeypatch.setattr(screen.runner, "is_running", lambda j: True)
    for handler in ("mm_check_model", "mm_ask_model", "mm_serve_model"):
        out = screen.h[handler]("tiny")
        assert "A diagnosis is running" in (out if isinstance(out, str) else out[0])
    assert "A diagnosis is running" in screen.h["mm_publish_model"]("tiny", "hf_x", "", True)[0]
    assert not screen.api.running and not screen.uploads


def test_a_diagnosis_frees_the_reopened_model_first(screen, monkeypatch):
    screen.h["mm_serve_model"]("tiny")
    started = []
    monkeypatch.setattr(screen.runner, "start", lambda *a, **k: started.append(screen.api.running))
    screen.h["byom_start"]("Qwen/Qwen2.5-1.5B-Instruct", "fast", 0, 0.10, "")
    assert started == [False]  # freed before the pipeline started
    from sqlalchemy import select

    from src.db.schema import api_sessions

    with screen.store.read() as connection:
        assert connection.execute(select(api_sessions.c.stopped_at)).scalar() is not None


def test_the_demos_16_bit_model_stays_away_while_a_model_is_held(screen, monkeypatch):
    """The screen's timer used to load it back 1.5 s later, next to the held model."""
    from src.product import studio

    calls = []
    monkeypatch.setattr(studio, "after_job", lambda backend, runner: calls.append("back"))
    screen.h["mm_check_model"]("tiny")
    assert calls == []
    screen.h["byom_refresh"]()
    assert calls == []
    screen.h["api_stop"]()
    assert calls == ["back"]


def test_a_model_that_cannot_be_rebuilt_is_explained_and_leaves_nothing_held(screen):
    recipe = {**screen.store.run("tiny")["artifact"]["recipe"], "model_id": str(screen.tmp_path / "gone")}
    (screen.tmp_path / "gone").mkdir()
    screen.store.start_run("lost", "x/lost")
    screen.store.finish_run("lost", "done", report=screen.report, patch=recipe)
    note = screen.h["mm_check_model"]("lost")
    assert note.startswith("**Not opened.**") and "couldn't be rebuilt" in note and "Traceback" not in note
    assert "50/50" in screen.h["mm_check_model"]("tiny")  # the next one opens as usual
    assert "Pick a run first" in screen.h["mm_check_model"](None)


# --- list, files, delete -----------------------------------------------------------------------------

def test_the_list_filters_and_keeps_the_picked_run(screen):
    status, rows, picker, other, methods = screen.h["mm_list"]("", "All", "All", "tiny", "missing")
    assert len(rows) == 1 and picker["value"] == "tiny" and other["value"] is None
    assert methods["choices"] == ["All", "hqq"]
    assert screen.h["mm_list"]("nothing like it")[1] == [] and screen.h["mm_list"]("", "Rejected")[1] == []
    assert len(screen.h["mm_list"]("asha", "Repaired", "hqq")[1]) == 1


def test_the_kit_and_the_report_are_files(screen):
    note, file = screen.h["mm_make_kit"]("tiny")
    assert file["visible"] and file["value"].endswith(".zip")
    screen.h["mm_check_model"]("tiny")
    note, file = screen.h["mm_make_report"]("tiny")
    assert "50/50" in open(file["value"], encoding="utf-8").read()
    assert screen.h["mm_make_kit"](None)[1]["visible"] is False


def test_delete_asks_first_and_never_touches_hugging_face(screen):
    screen.store.set_weights("tiny", "https://huggingface.co/asha/tiny-fairgate")
    job = screen.tmp_path / "jobs" / "20261125T100000000000Z_tiny"
    job.mkdir(parents=True)
    (job / "repair_kit.zip").write_bytes(b"kit")
    (job / "report.json").write_text("{}")
    screen.h["mm_serve_model"]("tiny")
    note, sure, text = screen.h["mm_ask_delete"]("tiny")
    assert sure["visible"] is True and "Nothing on Hugging Face is touched" in text and "asha/tiny-fairgate" in text
    assert screen.store.run("tiny") is not None  # only asked
    out = screen.h["mm_do_delete"]("tiny", "", "All", "All")
    assert out[0].startswith("Deleted") and screen.store.run("tiny") is None and not screen.api.running
    assert not (job / "repair_kit.zip").exists() and (job / "report.json").exists() and not screen.uploads
    assert screen.h["mm_do_delete"]("tiny", "", "All", "All")[0] == "Already gone."


def test_a_running_run_is_not_deleted(screen):
    screen.store.start_run("busy", "a/b")
    note, sure, _ = screen.h["mm_ask_delete"]("busy")
    assert "still going" in note and sure["visible"] is False


def test_the_tab_says_when_there_is_no_database(data):  # noqa: F811
    from src.product import studio
    from src.product.studio_core import SavedBackend

    app, _ = studio.build_app(SavedBackend(data.runs, list(data.patch.modules), repairs=data.saved_repairs), data)
    handlers = {f.fn.__name__: f.fn for f in app.fns.values() if getattr(f, "fn", None) is not None}
    assert "database: off" in handlers["mm_list"]()[0]
    assert "database is off" in handlers["mm_check_model"]("x") and handlers["mm_show"]("x", None)[0]
    assert "Pick a run" in handlers["mm_ask_delete"]("x")[0]
