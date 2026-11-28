# SPDX-License-Identifier: Apache-2.0
"""The vendored ERNIE 5.5 Flash model and the loader hooks it needs on the pinned mlx-vlm."""

import importlib
import json
from pathlib import Path

import mlx.core as mx
import mlx_vlm.utils as vlm_utils
import pytest
from mlx.utils import tree_flatten
from test_ernie55_dspark import TINY_CONFIG

from omlx.patches import mlx_vlm_ernie5_5_compat as compat

compat.apply_mlx_vlm_ernie5_5_compat_patch()
ernie = importlib.import_module("mlx_vlm.models.ernie5_5_moe")
prepare_preview_weights = importlib.import_module(
    "mlx_vlm.models.ernie5_5_moe.ernie5_5_moe"
).prepare_preview_weights

VENDOR = Path(compat.__file__).parent / "vendor"


@pytest.mark.parametrize(
    "module",
    [
        "mlx_vlm.models.ernie5_5_moe.language",
        "mlx_vlm.speculative.drafters.ernie5_5_dspark.ernie5_5_dspark",
        "mlx_vlm.speculative.drafters.ernie5_5_mtp.split",
    ],
)
def test_vendored_modules_take_the_place_of_the_pinned_ones(module):
    path = Path(importlib.import_module(module).__file__)
    assert VENDOR in path.parents


def test_ernie55_preview_keeps_native_mxfp4_and_block_fp8_values():
    expert = "model.layers.1.block_sparse_moe.experts.0.w1"
    output = "model.layers.0.self_attn.linear_o_group_proj"
    expert_weight = mx.arange(64, dtype=mx.uint8).reshape(2, 32)
    expert_scales = mx.full((2, 2), 127, dtype=mx.uint8)
    output_weight = mx.full((130, 160), 56, dtype=mx.uint8)
    output_scales = mx.array([[1.0, 0.5], [2.0, 4.0]])
    raw = {
        expert + ".weight": expert_weight.view(mx.int8),
        expert + ".weight_scale_inv": expert_scales,
        output: output_weight,
        output + "_scale_inv": output_scales,
        "model.mtp.feature_projection.weight": mx.zeros((1,)),
    }
    weights, quantization = prepare_preview_weights(
        raw, {"quantization_config": {"expert_dtype": "mxfp4"}}
    )
    assert mx.array_equal(weights[expert + ".weight"].view(mx.uint8), expert_weight)
    assert mx.array_equal(weights[expert + ".scales"], expert_scales)
    assert mx.array_equal(weights[output + ".weight"].view(mx.uint8), output_weight)
    scales = weights[output + ".scales"]
    assert scales.shape == (130, 5)
    assert scales[0].tolist() == [127, 127, 127, 127, 126]
    assert scales[-1].tolist() == [128, 128, 128, 128, 129]
    assert quantization["mode"] == "mxfp8"
    path = "language_model.model.layers.1.ffn.switch_mlp.gate_proj"
    assert quantization[path] == {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    assert not any(key.startswith("model.mtp.") for key in weights)


def _write_preview_checkpoint(path):
    config = dict(TINY_CONFIG)
    config["quantization_config"] = {
        "quant_method": "fp8",
        "expert_dtype": "mxfp4",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    }
    model = ernie.Model(ernie.ModelConfig.from_dict(config))
    weights = dict(tree_flatten(model.language_model.parameters()))
    prefix = "model.layers.1.ffn"
    for projection, name in (
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
    ):
        original = weights.pop(f"{prefix}.switch_mlp.{projection}.weight")
        packed, scales = mx.quantize(original, mode="mxfp4")
        for i in range(config["n_routed_experts"]):
            expert = f"{prefix}.experts.{i}.{name}.weight"
            weights[expert] = packed[i].view(mx.int8)
            weights[expert + "_scale_inv"] = scales[i]
    weights["model.layers.0.mlp.up_proj.weight"] = mx.full(
        (config["intermediate_size"], config["hidden_size"]), 56, dtype=mx.uint8
    )
    weights["model.layers.0.mlp.up_proj.weight_scale_inv"] = mx.ones((1, 1))
    weights.pop("model.layers.0.ffn.up_proj.weight")
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "model.safetensors"), weights)


def test_ernie55_preview_loads_mixed_weights_strictly(tmp_path):
    _write_preview_checkpoint(tmp_path)
    loaded = vlm_utils.load_model(tmp_path)
    layers = loaded.language_model.layers
    assert layers[0].ffn.up_proj.mode == "mxfp8"
    assert layers[1].ffn.switch_mlp.gate_proj.mode == "mxfp4"
    result = loaded(mx.array([[3, 4]]), cache=loaded.make_cache())
    assert mx.all(mx.isfinite(result.logits)).item()


def test_an_unobserved_load_evaluates_one_layer_at_a_time(tmp_path, monkeypatch):
    _write_preview_checkpoint(tmp_path)
    evaluated = []
    stock = mx.eval
    monkeypatch.setattr(
        mx,
        "eval",
        lambda *trees: evaluated.append(len(tree_flatten(trees))) or stock(*trees),
    )
    loaded = vlm_utils.load_model(tmp_path)
    layers = [
        len(tree_flatten(layer.parameters())) for layer in loaded.language_model.layers
    ]
    assert evaluated == [*layers, len(tree_flatten(loaded.parameters()))]


def test_the_preview_tokenizer_needs_the_callers_trust_remote_code(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "ernie5_5_moe"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "ErnieChatTokenizer"})
    )
    with pytest.raises(ValueError, match="requires trust_remote_code=True"):
        vlm_utils.load_processor(tmp_path)
