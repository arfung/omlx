# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 prefill attention: banded window and gathered long-range paths."""

import copy
import importlib

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.cache import (
    CacheList,
    KVCache,
    PoolingCache,
    RotatingKVCache,
)

from omlx.patches.mlx_vlm_ernie5_5_compat import apply_mlx_vlm_ernie5_5_compat_patch

apply_mlx_vlm_ernie5_5_compat_patch()
ernie = importlib.import_module("mlx_vlm.models.ernie5_5_moe")
language = importlib.import_module("mlx_vlm.models.ernie5_5_moe.language")
HCAAttention, Indexer, MLAAttention = (
    language.HCAAttention,
    language.Indexer,
    language.MLAAttention,
)

WINDOW = 4
TOPK = 4
HIDDEN = 64


def _model(seed=0):
    config = ernie.ModelConfig.from_dict(
        {
            "model_type": "ernie5_5_moe",
            "hidden_size": HIDDEN,
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
            "compress_ratios": [8, 8, -2],
            "hybrid_mla_num_attention_heads": 4,
            "hybrid_mla_q_lora_rank": 32,
            "hybrid_mla_kv_lora_rank": 16,
            "hybrid_mla_qk_nope_head_dim": 8,
            "hybrid_mla_qk_rope_head_dim": 8,
            "hybrid_mla_v_head_dim": 16,
            "index_n_heads": 8,
            "index_head_dim": 16,
            "index_topk": TOPK,
            "moe_latent_size": 32,
            "moe_intermediate_size": 32,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    mx.random.seed(seed)
    model = ernie.Model(config)
    # Modest weights keep the scores small enough that the two forms of each
    # attention agree to fp32 rounding instead of amplifying it.
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


@pytest.fixture(scope="module")
def model():
    return _model()


@pytest.fixture(autouse=True)
def _paths():
    """Every test starts from the shipped switches and leaves them that way."""
    saved = (
        HCAAttention.banded_window,
        MLAAttention.gathered_keys,
        MLAAttention.gathered_min_keys,
        MLAAttention.gathered_min_keys_short,
        MLAAttention.gathered_rows,
        Indexer.score_elements,
        language._index_kernel,
    )
    MLAAttention.gathered_min_keys = 0
    MLAAttention.gathered_min_keys_short = 0
    yield
    (
        HCAAttention.banded_window,
        MLAAttention.gathered_keys,
        MLAAttention.gathered_min_keys,
        MLAAttention.gathered_min_keys_short,
        MLAAttention.gathered_rows,
        Indexer.score_elements,
        language._index_kernel,
    ) = saved


def _inputs(rows, seed=1, batch=1):
    mx.random.seed(seed)
    return mx.random.normal((batch, rows, HIDDEN))


def _chunked(attn, cache, x, chunks):
    outs, at = [], 0
    for rows in chunks:
        outs.append(attn(x[:, at : at + rows], cache))
        mx.eval(outs[-1])
        at += rows
    return mx.concatenate(outs, axis=1)


def _close(got, want, tolerance=2e-4):
    scale = mx.abs(want).max().item()
    assert mx.abs(got - want).max().item() <= tolerance * scale


CHUNKS = (
    [64],
    [1, 63],
    [3, 16, 45],
    [37, 27],
    [9, 1, 1, 30, 23],
    [16, 16, 16, 16],
    [64, 1, 9, 30],
    [2, 17, 19, 21],
)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_banded_window_matches_dense(model, chunks):
    attn = model.language_model.layers[0].attn
    x = _inputs(sum(chunks))
    results = []
    for banded in (False, True):
        HCAAttention.banded_window = banded
        results.append(_chunked(attn, model.language_model.make_cache()[0], x, chunks))
    _close(results[1], results[0])


def test_banded_window_matches_dense_without_a_cache(model):
    attn = model.language_model.layers[1].attn
    for batch, rows in ((1, 50), (2, 40), (1, 16)):
        x = _inputs(rows, batch=batch)
        HCAAttention.banded_window = False
        want = attn(x)
        HCAAttention.banded_window = True
        _close(attn(x), want)


def test_banded_window_is_taken_only_behind_a_plain_windowed_mask(model):
    attn = model.language_model.layers[0].attn
    rows, earlier = 4 * WINDOW, WINDOW - 1
    local = RotatingKVCache(max_size=WINDOW)
    mask = mx.ones((rows, earlier + rows), dtype=mx.bool_)
    pooled = mx.zeros((1, 3, 16))
    pool_mask = mx.ones((rows, 3), dtype=mx.bool_)

    def banded(rows=rows, earlier=earlier, local=local, mask=mask, pool_mask=pool_mask):
        return attn._banded(rows, earlier, local, mask, pooled, pool_mask)

    assert banded()
    assert banded(local=None, pool_mask=None)
    # Decode and verify blocks stay dense.
    assert not banded(rows=rows - 1, mask=mask[1:])
    # So do batched caches, whose masks carry per-row padding ...
    assert not banded(mask=mask[None, None])
    assert not banded(pool_mask=pool_mask[None])
    # ... a cache that keeps a prefix or a wider buffer than the window ...
    assert not banded(local=RotatingKVCache(max_size=WINDOW, keep=1))
    assert not banded(local=RotatingKVCache(max_size=2 * WINDOW))
    # ... and a cache that returned more history than a window holds.
    assert not banded(earlier=WINDOW, mask=mx.ones((rows, WINDOW + rows)))
    assert not banded(mask=None)
    HCAAttention.banded_window = False
    assert not banded()


def test_banded_window_runs_behind_a_rotating_cache_of_another_class(model):
    """A server may rebuild the window cache as its own subclass."""

    class Restored(RotatingKVCache):
        pass

    attn = model.language_model.layers[0].attn
    x = _inputs(48, seed=8)
    results = []
    for banded in (False, True):
        HCAAttention.banded_window = banded
        cache = CacheList(Restored(max_size=WINDOW), PoolingCache(8))
        results.append(_chunked(attn, cache, x, [5, 20, 23]))
    _close(results[1], results[0])
    taken = []
    original = HCAAttention._banded_attention

    def spy(self, *args):
        taken.append(args[0].shape[2])
        return original(self, *args)

    HCAAttention._banded_attention = spy
    try:
        cache = CacheList(Restored(max_size=WINDOW), PoolingCache(8))
        _chunked(attn, cache, x, [5, 20, 23])
    finally:
        HCAAttention._banded_attention = original
    assert taken == [20, 23]


def test_window_bands_show_each_query_its_own_window():
    window, earlier, rows = 4, 3, 13
    blocks = -(-rows // window)
    # Keys carry their own position so a block's contents can be read back.
    kv = mx.arange(earlier + rows, dtype=mx.float32).reshape(1, 1, -1, 1) + 1
    keys, mask = language._window_bands(kv, earlier, window, blocks)
    assert keys.shape == (1, blocks, 2 * window, 1)
    assert mask.shape == (blocks, window, 2 * window)
    keys = keys[0, :, :, 0].tolist()
    mask = mask.tolist()
    for row in range(rows):
        block, slot = divmod(row, window)
        seen = sorted(
            int(key) - 1
            for key, visible in zip(keys[block], mask[block][slot])
            if visible
        )
        position = earlier + row
        assert seen == list(range(max(0, position - window + 1), position + 1))


@pytest.mark.parametrize("chunks", CHUNKS)
def test_gathered_keys_match_dense(model, chunks):
    attn = model.language_model.layers[2].attn
    x = _inputs(sum(chunks), seed=2)
    results = []
    for gathered in (False, True):
        MLAAttention.gathered_keys = gathered
        results.append(_chunked(attn, model.language_model.make_cache()[2], x, chunks))
    _close(results[1], results[0])


def test_gathered_keys_match_dense_in_row_blocks(model):
    attn = model.language_model.layers[2].attn
    x = _inputs(70, seed=3)
    MLAAttention.gathered_keys = False
    want = attn(x)
    MLAAttention.gathered_keys = True
    for rows in (7, 64, 512):
        MLAAttention.gathered_rows = rows
        _close(attn(x), want)


def test_gathered_keys_are_taken_only_for_long_single_row_chunks(model):
    attn = model.language_model.layers[2].attn
    few = TOPK + WINDOW
    assert attn._gathered(1, 16, few + 1, KVCache())
    assert attn._gathered(1, 16, few + 1, None)
    # A context no longer than what a query keeps is attended densely.
    assert not attn._gathered(1, 16, few, KVCache())
    assert not attn._gathered(2, 16, 64, KVCache())
    assert not attn._gathered(1, 16, 64, RotatingKVCache(max_size=8))
    # A verify block gathers from its own context length on; one decoded
    # token takes its keys from the indexer's list instead.
    assert attn._gathered(1, language.DECODE_BLOCK_SIZE, 64, KVCache())
    assert attn._gathered(1, 2, 64, KVCache())
    assert not attn._gathered(1, 1, 64, KVCache())
    MLAAttention.gathered_min_keys_short = 65
    assert not attn._gathered(1, language.DECODE_BLOCK_SIZE, 64, KVCache())
    assert attn._gathered(1, language.DECODE_BLOCK_SIZE + 1, 64, KVCache())
    MLAAttention.gathered_min_keys_short = 0

    class KVCacheSubclass(KVCache):
        pass

    assert not attn._gathered(1, 16, 64, KVCacheSubclass())
    MLAAttention.gathered_min_keys = 65
    assert not attn._gathered(1, 16, 64, KVCache())
    MLAAttention.gathered_min_keys = 0
    MLAAttention.gathered_keys = False
    assert not attn._gathered(1, 16, 64, KVCache())


def _indexer_inputs(model, first, rows, seed=4):
    attn = model.language_model.layers[2].attn
    cache = model.language_model.make_cache()[2]
    x = _inputs(first + rows, seed=seed)
    if first:
        mx.eval(attn(x[:, :first], cache))
    x = x[:, first:]
    qr = attn.q_a_layernorm(attn.q_a_proj(x))
    mask = create_attention_mask(x, cache[0], return_array=True)
    return attn.indexer, x, qr, mask, cache[1]


@pytest.mark.parametrize("first,rows", [(0, 64), (5, 59), (1, 20), (30, 34)])
def test_gathered_selection_names_the_keys_of_the_dense_mask(model, first, rows):
    indexer, x, qr, mask, cache = _indexer_inputs(model, first, rows)
    dense, none = indexer(x, qr, mask, copy.deepcopy(cache))
    assert none is None
    nothing, (ids, usable) = indexer(x, qr, mask, copy.deepcopy(cache), gather=True)
    assert nothing is None
    assert ids.shape == usable.shape == (rows, TOPK + WINDOW)
    dense = dense[0, 0].tolist()
    for row, (keys, kept) in enumerate(zip(ids.tolist(), usable.tolist())):
        chosen = [key for key, keep in zip(keys, kept) if keep]
        # No key twice, or it would be counted twice in the softmax.
        assert len(chosen) == len(set(chosen))
        assert sorted(chosen) == [k for k, seen in enumerate(dense[row]) if seen]


@pytest.mark.parametrize("first,rows", [(0, 64), (30, 34), (41, 23)])
def test_gathered_selection_is_the_same_ranked_in_row_blocks(model, first, rows):
    indexer, x, qr, mask, cache = _indexer_inputs(model, first, rows)
    _, (ids, usable) = indexer(x, qr, mask, copy.deepcopy(cache), gather=True)
    # A budget this small ranks the chunk a decode block of rows at a time.
    Indexer.score_elements = 1
    _, (block_ids, block_usable) = indexer(
        x, qr, mask, copy.deepcopy(cache), gather=True
    )
    assert block_ids.shape == ids.shape

    def chosen(ids, usable):
        return [
            sorted(key for key, keep in zip(keys, kept) if keep)
            for keys, kept in zip(ids.tolist(), usable.tolist())
        ]

    assert chosen(block_ids, block_usable) == chosen(ids, usable)


def test_index_kernel_matches_the_plain_head_sum(model):
    if language._index_kernel is None:
        pytest.skip("needs the Metal backend")
    indexer = model.language_model.layers[2].attn.indexer
    mx.random.seed(5)
    # 77 keys is not a multiple of the kernel's thread group.
    for batch, rows, keys in ((1, 20, 77), (2, 9, 64), (1, 600, 33)):
        q = mx.random.normal((batch, indexer.heads, rows, indexer.dim))
        k = mx.random.normal((batch, 1, keys, indexer.dim))
        weights = mx.random.normal((batch, rows, indexer.heads))
        fast = indexer._scores(q, k, weights)
        kernel, language._index_kernel = language._index_kernel, None
        plain = indexer._scores(q, k, weights)
        language._index_kernel = kernel
        assert fast.shape == plain.shape == (batch, 1, rows, keys)
        assert fast.dtype == mx.float32
        _close(fast, plain, tolerance=1e-5)


def test_index_kernel_reads_half_precision_scores(model):
    if language._index_kernel is None:
        pytest.skip("needs the Metal backend")
    indexer = model.language_model.layers[2].attn.indexer
    mx.random.seed(6)
    q = mx.random.normal((1, indexer.heads, 24, indexer.dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, 1, 50, indexer.dim)).astype(mx.bfloat16)
    weights = mx.random.normal((1, 24, indexer.heads))
    fast = indexer._scores(q, k, weights)
    kernel, language._index_kernel = language._index_kernel, None
    plain = indexer._scores(q, k, weights)
    language._index_kernel = kernel
    _close(fast, plain, tolerance=1e-5)


def test_whole_model_prefill_agrees_across_paths(model):
    lm = model.language_model
    mx.random.seed(7)
    ids = mx.random.randint(0, 128, (1, 80))

    def run(new_paths, chunks):
        HCAAttention.banded_window = MLAAttention.gathered_keys = new_paths
        cache, outs, at = lm.make_cache(), [], 0
        for rows in chunks:
            outs.append(lm(ids[:, at : at + rows], cache=cache).logits)
            mx.eval(outs[-1])
            at += rows
        return mx.concatenate(outs, axis=1), cache

    want, old_cache = run(False, [40, 40])
    got, new_cache = run(True, [40, 40])
    _close(got, want, tolerance=5e-3)
    # Each layer writes its cache before it attends, so the first layer's is
    # identical and the rest differ only by what rounding the layers below
    # passed up; decode continues from either prefill alike.
    for index, (old, new) in enumerate(zip(old_cache, new_cache)):
        for (_, a), (_, b) in zip(tree_flatten(old.state), tree_flatten(new.state)):
            if a is None or b is None:
                assert a is b
            elif a.size == 0:
                assert a.shape == b.shape
            elif index == 0:
                assert mx.array_equal(a, b)
            else:
                _close(b, a, tolerance=5e-3)
