from src.quant.hqq_quantizer import DEFAULT_GROUP_SIZE, DEFAULT_NBITS, build_3bit_quant_config


def _weight_params(config):
    # BaseQuantizeConfig is a plain dict subclass; nbits/group_size live under
    # ["weight_quant_params"] (verified against the real object, not just docs).
    return config["weight_quant_params"]


def test_default_config_is_3bit():
    params = _weight_params(build_3bit_quant_config())
    assert params["nbits"] == DEFAULT_NBITS
    assert params["group_size"] == DEFAULT_GROUP_SIZE


def test_config_respects_overrides():
    params = _weight_params(build_3bit_quant_config(nbits=4, group_size=32))
    assert params["nbits"] == 4
    assert params["group_size"] == 32
