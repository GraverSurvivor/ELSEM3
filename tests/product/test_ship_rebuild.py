"""The Ship screen rebuilds step 5's repair with the patch's own compression method (an RTN repair is never
rebuilt as HQQ), on a tiny model made on the spot."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hqq")

from src.product.studio import rebuild_repair  # noqa: E402
from src.quant.repaired import load_repaired  # noqa: E402
from tests.product.test_methods_chain import logits, patch_for  # noqa: E402
from tests.quant.tiny_models import save_tiny  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    folder = tmp_path_factory.mktemp("qwen2")
    return folder, save_tiny(folder, "qwen2")


@pytest.mark.parametrize("method", ["hqq", "rtn"])
def test_the_ship_screen_rebuilds_the_repair_that_was_measured(tiny, method):
    folder, _ = tiny
    patch = patch_for(folder, method=method)
    model, tok = rebuild_repair(patch, "cpu")
    same, _ = load_repaired(patch, "cpu")
    assert torch.equal(logits(model, tok), logits(same, tok))
    other, _ = load_repaired(patch_for(folder, method="hqq" if method == "rtn" else "rtn"), "cpu")
    assert not torch.equal(logits(model, tok), logits(other, tok))  # the method matters: it isn't ignored
