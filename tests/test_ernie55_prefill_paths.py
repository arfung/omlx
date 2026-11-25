# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 prefill attention behind the caches oMLX hands the model.

A cold request prefills on the model's own caches. After a prefix-cache hit
the scheduler rebuilds them from its store with mlx-lm classes, a rotating
subclass of its own and its pooling cache. The banded window and gathered
long-range attention paths must engage, and agree with the dense paths, on
both.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

ernie = pytest.importorskip("mlx_vlm.models.ernie5_5_moe")
language = pytest.importorskip("mlx_vlm.models.ernie5_5_moe.language")

from mlx_lm.models.cache import CacheList, KVCache  # noqa: E402

from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache  # noqa: E402
from omlx.patches.deepseek_v4 import apply_pooling_cache_support  # noqa: E402

WINDOW = 4
RATIO = 8


@pytest.fixture(scope="module")
def model():
    config = ernie.ModelConfig.from_dict(
        {
            "model_type": "ernie5_5_moe",
            "hidden_size": 64,
            "vocab_size": 128,
            "num_hidden_layers": 3,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "head_dim": 16,
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "o_groups": 2,
            "o_lora_rank": 16,
            "sliding_window": WINDOW,
            "compress_ratios": [RATIO, RATIO, -2],
            "hybrid_mla_num_attention_heads": 4,
            "hybrid_mla_q_lora_rank": 32,
            "hybrid_mla_kv_lora_rank": 16,
            "hybrid_mla_qk_nope_head_dim": 8,
            "hybrid_mla_qk_rope_head_dim": 8,
            "hybrid_mla_v_head_dim": 16,
            "index_n_heads": 8,
            "index_head_dim": 16,
            "index_topk": 4,
            "moe_latent_size": 32,
            "moe_intermediate_size": 32,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    mx.random.seed(0)
    model = ernie.Model(config)
    model.update(
        tree_unflatten(
            [
                (
                    name,
                    (
                        1 + 0.1 * mx.random.normal(value.shape)
                        if "norm" in name and name.endswith("weight")
                        else 0.2 * mx.random.normal(value.shape)
                    ),
                )
                for name, value in tree_flatten(model.parameters())
            ]
        )
    )
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.fixture(autouse=True)
def _switches():
    hca, mla = language.HCAAttention, language.MLAAttention
    saved = (hca.banded_window, mla.gathered_keys, mla.gathered_min_keys)
    mla.gathered_min_keys = 0
    yield
    hca.banded_window, mla.gathered_keys, mla.gathered_min_keys = saved


def _restored_caches():
    """The cache classes the scheduler rebuilds a stored prefix with."""
    # Returns False when an earlier import already installed it.
    apply_pooling_cache_support()
    try:
        from mlx_lm.models.cache import PoolingCache
    except ImportError:
        pytest.skip("pooling cache support is unavailable")

    return [
        CacheList(PrefillReadyRotatingKVCache(max_size=WINDOW), PoolingCache(RATIO)),
        CacheList(PrefillReadyRotatingKVCache(max_size=WINDOW), PoolingCache(RATIO)),
        CacheList(KVCache(), KVCache()),
    ]


def _run(model, cache, ids, chunks, new_paths):
    language.HCAAttention.banded_window = new_paths
    language.MLAAttention.gathered_keys = new_paths
    outs, at = [], 0
    for rows in chunks:
        outs.append(model.language_model(ids[:, at : at + rows], cache=cache).logits)
        mx.eval(outs[-1])
        at += rows
    return mx.concatenate(outs, axis=1)


def _spy(monkeypatch):
    taken = {"banded": 0, "gathered": 0}
    banded = language.HCAAttention._banded_attention
    gathered = language.MLAAttention._gathered_attention

    def spy_banded(self, *args):
        taken["banded"] += 1
        return banded(self, *args)

    def spy_gathered(self, *args):
        taken["gathered"] += 1
        return gathered(self, *args)

    monkeypatch.setattr(language.HCAAttention, "_banded_attention", spy_banded)
    monkeypatch.setattr(language.MLAAttention, "_gathered_attention", spy_gathered)
    return taken


@pytest.mark.parametrize("restored", [False, True])
def test_prefill_takes_the_new_paths_and_matches_dense(model, monkeypatch, restored):
    make = _restored_caches if restored else model.language_model.make_cache
    mx.random.seed(1)
    chunks = [6, 30, 1, 27]
    ids = mx.random.randint(0, 128, (1, sum(chunks)))
    want = _run(model, make(), ids, chunks, new_paths=False)
    taken = _spy(monkeypatch)
    got = _run(model, make(), ids, chunks, new_paths=True)
    # Two window layers and one long-range layer, on the two long chunks; the
    # 6-token and 1-token steps keep the decode paths.
    assert taken == {"banded": 4, "gathered": 2}
    scale = mx.abs(want).max().item()
    assert mx.abs(got - want).max().item() <= 5e-3 * scale


def test_switches_turn_the_new_paths_off(model, monkeypatch):
    mx.random.seed(2)
    ids = mx.random.randint(0, 128, (1, 40))
    taken = _spy(monkeypatch)
    _run(model, model.language_model.make_cache(), ids, [40], new_paths=False)
    assert taken == {"banded": 0, "gathered": 0}
