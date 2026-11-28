# SPDX-License-Identifier: Apache-2.0
"""INT8-activation (A8) prefill for ERNIE 5.5 Flash routed experts.

ERNIE 5.5 Flash Preview ships its routed experts in MXFP4 and was trained with
8-bit activations (W4A8). On GPUs with tensor units a prefill chunk's routed
gate, up and down projections run through the sorted INT8 gather: the
activations are quantized to INT8 once per token row and multiplied with the
MXFP4 codes as INT8 x INT8 -> INT32.

Only the routed experts take this path. The dense MXFP8 projections stay on
MLX's own quantized matmul, which measured faster than the one-expert INT8
gather on an M5 Ultra (6.6 ms against 11.0 ms per layer at 2048 rows), while
the routed experts gained (10.2 ms down to 7.1 ms).

ERNIE's SiTU activation is not the SwiGLU the gather can fuse, so gate and up
are two gathers and the activation runs between them and the down projection.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import mlx.core as mx

from ..m5_gather_qmm_a8 import (
    _FMT_MXFP4,
    _MIN_TOKENS_FLOOR,
    mx_shape_supported,
    sorted_gather_qmm_a8_mx,
)

logger = logging.getLogger(__name__)

# Decode and verify blocks stay on the A16 path; the setting can only raise it.
MIN_TOKENS_FLOOR = _MIN_TOKENS_FLOOR
# Attribute of mlx-vlm's ERNIE ``LatentMoE`` that offers routed outputs.
BACKEND = "routed_backend"


def _mxfp4(projection: Any) -> bool:
    """Stacked MXFP4 expert codes with one E8M0 scale per 32 and no bias."""
    weight, scales = projection.get("weight"), projection.get("scales")
    return (
        getattr(projection, "mode", None) == "mxfp4"
        and getattr(projection, "group_size", None) == 32
        and getattr(projection, "bits", None) == 4
        and "bias" not in projection
        and projection.get("biases") is None
        and weight is not None
        and weight.ndim == 3
        and weight.dtype == mx.uint32
        and scales is not None
        and scales.dtype == mx.uint8
        and scales.shape[:-1] == weight.shape[:-1]
        and weight.shape[-1] == scales.shape[-1] * 4
        and mx_shape_supported(weight.shape[1], scales.shape[-1] * 32)
    )


def supported(switch: Any) -> bool:
    """Whether every routed projection of ``switch`` has the layout read here."""
    projections = [
        getattr(switch, name, None) for name in ("gate_proj", "up_proj", "down_proj")
    ]
    return all(p is not None and _mxfp4(p) for p in projections)


def routed(switch: Any, x: mx.array, indices: mx.array, *, min_tokens: int):
    """Routed outputs ``[..., k, dim]`` of ``switch`` for a prefill chunk, or
    None to keep the A16 path.

    ``x`` holds the token rows ``[..., K]`` and ``indices`` each token's
    experts ``[..., k]``.
    """
    width = x.shape[-1]
    tokens = x.size // width
    if tokens < max(int(min_tokens), MIN_TOKENS_FLOOR):
        return None
    if x.dtype not in (mx.float16, mx.bfloat16) or not supported(switch):
        return None
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    flat = indices.reshape(-1)
    order = mx.argsort(flat)
    sorted_ids = flat[order].astype(mx.uint32)
    row_map = (order // indices.shape[-1]).astype(mx.uint32)
    rows = x.reshape(tokens, 1, width)
    projected = [
        sorted_gather_qmm_a8_mx(
            rows, p.weight, p.scales, sorted_ids, row_map, fmt=_FMT_MXFP4
        )
        for p in (gate, up)
    ]
    if any(y is None for y in projected):
        return None
    hidden = switch.activation(projected[1], projected[0])
    y = sorted_gather_qmm_a8_mx(
        hidden,
        down.weight,
        down.scales,
        sorted_ids,
        mx.arange(sorted_ids.size, dtype=mx.uint32),
        fmt=_FMT_MXFP4,
        # The activation has outliers that a row-wide scale would spread
        # over the whole row.
        group_scales=True,
    )
    if y is None:
        return None
    return y[mx.argsort(order)].reshape(*indices.shape, -1)


def tag(model: Any, min_tokens: int) -> int:
    """Offer A8 routed outputs on every backbone MoE layer of ``model``.

    Returns how many layers were tagged. Never raises: on an error nothing
    stays tagged and the model keeps the A16 path. The draft stages are not
    visited; they only ever see decode-sized blocks.
    """
    tagged = []
    try:
        language_model = getattr(model, "language_model", model)
        for layer in language_model.layers:
            ffn = getattr(layer, "ffn", None)
            switch = getattr(ffn, "switch_mlp", None)
            if switch is None or not supported(switch):
                continue
            object.__setattr__(
                ffn, BACKEND, partial(routed, min_tokens=int(min_tokens))
            )
            tagged.append(ffn)
    except Exception:
        logger.warning("ERNIE A8 prefill not applied", exc_info=True)
        for ffn in tagged:
            object.__setattr__(ffn, BACKEND, None)
        return 0
    return len(tagged)


def untag(model: Any) -> None:
    """Remove the A8 backend from every layer of ``model``."""
    language_model = getattr(model, "language_model", model)
    for layer in language_model.layers:
        ffn = getattr(layer, "ffn", None)
        if ffn is not None and getattr(ffn, BACKEND, None) is not None:
            object.__setattr__(ffn, BACKEND, None)
