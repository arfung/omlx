# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 block drafting for several requests that share a verify forward.

Each request keeps its own cache and drafter state; a round verifies all the
requests' blocks in one forward. Whatever the other requests do, a request
must decode what it decodes alone.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import test_ernie55_dspark as helpers
from mlx_vlm.speculative.dflash import _dflash_rounds

from omlx.speculative.block_rows import SharedBlockRounds

PROMPTS = (
    [5, 9, 2, 44, 7, 81],
    [11, 3, 60, 61, 62, 8, 19, 100, 4],
    [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
)


def _greedy(logits):
    return mx.argmax(logits, axis=-1)


def _sample(logprobs):
    return mx.random.categorical(logprobs)


def _admit(rounds, model, draft, prompt, count, sampler=_greedy, greedy=True):
    lm = model.language_model
    cache = model.make_cache()
    output = lm(
        mx.array([prompt]),
        cache=cache,
        capture_layer_ids=list(draft.config.target_layer_ids),
    )
    hidden = mx.concatenate(output.hidden_states, axis=-1)
    first = int(sampler(output.logits[:, -1]).item())
    stream = rounds.add(
        cache, hidden, first, count, sampler, greedy=greedy, history=prompt
    )
    return stream, cache


def _drain(streams, outputs, steps=None):
    """Step every live stream, as the scheduler does, until all have ended."""
    live = dict(streams)
    done = 0
    while live and (steps is None or done < steps):
        for key in list(live):
            try:
                outputs[key].extend(next(live[key]))
            except StopIteration:
                del live[key]
        done += 1
    return live


@pytest.fixture(scope="module")
def target():
    return helpers._target()


def test_requests_drafting_together_decode_what_they_decode_alone(target):
    config, model = target
    count = 30
    reference = [helpers._decode_one_by_one(model, p, count) for p in PROMPTS]
    draft = helpers._drafter(config, 4)
    rounds = SharedBlockRounds(model, draft, block_size=4)
    streams, outputs = {}, {}
    for index, prompt in enumerate(PROMPTS):
        streams[index], _ = _admit(rounds, model, draft, prompt, count)
        outputs[index] = []
    _drain(streams, outputs)
    assert [outputs[i] for i in range(len(PROMPTS))] == reference
    # Every round verified all three, and the forward held their rows.
    assert rounds.rounds < count
    assert not rounds.rows


def test_four_requests_share_the_forward_with_an_anchor_and_a_proposal_each(target):
    config, model = target
    prompts = (*PROMPTS, [9, 8, 7, 6, 5, 4, 3])
    count = 24
    reference = [helpers._decode_one_by_one(model, p, count) for p in prompts]
    draft = helpers._drafter(config, 4)
    rounds = SharedBlockRounds(model, draft, block_size=4, max_requests=4)
    assert rounds.max_requests == 4
    streams, outputs = {}, {}
    for index, prompt in enumerate(prompts):
        streams[index], _ = _admit(rounds, model, draft, prompt, count)
        outputs[index] = []
    assert not rounds.has_room()
    # Eight rows hold four requests only at the shortest block.
    assert rounds._block_sizes(list(rounds.rows)) == [2, 2, 2, 2]
    _drain(streams, outputs)
    assert [outputs[i] for i in range(len(prompts))] == reference
    assert not rounds.rows


def test_requests_may_join_and_leave_between_rounds(target):
    config, model = target
    counts = (30, 12, 22)
    reference = [
        helpers._decode_one_by_one(model, p, n) for p, n in zip(PROMPTS, counts)
    ]
    draft = helpers._drafter(config, 4)
    rounds = SharedBlockRounds(model, draft, block_size=4)
    streams, outputs = {}, {0: [], 1: [], 2: []}
    streams[0], _ = _admit(rounds, model, draft, PROMPTS[0], counts[0])
    live = _drain(streams, outputs, steps=3)
    live[1], _ = _admit(rounds, model, draft, PROMPTS[1], counts[1])
    live = _drain(live, outputs, steps=2)
    live[2], _ = _admit(rounds, model, draft, PROMPTS[2], counts[2])
    _drain(live, outputs)
    assert [outputs[i] for i in range(3)] == reference
    assert not rounds.rows


def test_a_closed_request_leaves_the_others_undisturbed(target):
    config, model = target
    count = 24
    reference = helpers._decode_one_by_one(model, PROMPTS[0], count)
    draft = helpers._drafter(config, 4)
    rounds = SharedBlockRounds(model, draft, block_size=4)
    outputs = {0: [], 1: []}
    streams = {}
    streams[0], _ = _admit(rounds, model, draft, PROMPTS[0], count)
    streams[1], _ = _admit(rounds, model, draft, PROMPTS[1], count)
    live = _drain(streams, outputs, steps=3)
    live.pop(1).close()
    assert len(rounds.rows) == 1
    _drain(live, outputs)
    assert outputs[0] == reference


def test_the_caches_hold_exactly_the_emitted_tokens(target):
    config, model = target
    count = 17
    draft = helpers._drafter(config, 4)
    rounds = SharedBlockRounds(model, draft, block_size=4)
    streams, caches, outputs = {}, {}, {}
    for index, prompt in enumerate(PROMPTS[:2]):
        streams[index], caches[index] = _admit(rounds, model, draft, prompt, count)
        outputs[index] = []
    _drain(streams, outputs)
    for index, prompt in enumerate(PROMPTS[:2]):
        # The last token is sampled but not yet fed, as in plain decoding.
        held = caches[index][-1].caches[0].offset
        assert held == len(prompt) + count - 1


def test_one_sampled_request_matches_the_stock_round_loop(target):
    config, model = target
    count, prompt = 26, PROMPTS[1]
    lm = model.language_model

    def stock():
        draft = helpers._drafter(config, 4)
        cache = model.make_cache()
        output = lm(
            mx.array([prompt]),
            cache=cache,
            capture_layer_ids=list(draft.config.target_layer_ids),
        )
        hidden = mx.concatenate(output.hidden_states, axis=-1)
        tokens = [int(_sample(output.logits[:, -1]).item())]
        draft.begin_request(prompt)
        draft.observe(tokens[0])
        for token, _ in _dflash_rounds(
            model,
            draft,
            cache,
            hidden,
            first_bonus=tokens[0],
            max_tokens=count,
            sampler=_sample,
            draft_block_size=4,
            greedy_sampling=False,
            use_model_initial_block_size=False,
        ):
            tokens.append(int(token))
            draft.observe(int(token))
        return tokens

    def shared():
        draft = helpers._drafter(config, 4)
        rounds = SharedBlockRounds(model, draft, block_size=4)
        outputs = {0: []}
        stream, _ = _admit(
            rounds, model, draft, prompt, count, sampler=_sample, greedy=False
        )
        _drain({0: stream}, outputs)
        return outputs[0]

    mx.random.seed(7)
    expected = stock()
    mx.random.seed(7)
    assert shared() == expected
    assert len(expected) == count


def test_sampled_requests_keep_their_own_random_streams(target):
    config, model = target
    count = 20

    def run(prompts):
        draft = helpers._drafter(config, 4)
        rounds = SharedBlockRounds(model, draft, block_size=4)
        streams, outputs = {}, {}
        mx.random.seed(11)
        for index, prompt in enumerate(prompts):
            streams[index], _ = _admit(
                rounds, model, draft, prompt, count, sampler=_sample, greedy=False
            )
            outputs[index] = []
        _drain(streams, outputs)
        return outputs

    together = run(PROMPTS[:2])
    assert all(len(tokens) == count for tokens in together.values())
    # The same two requests again draw the same numbers.
    assert run(PROMPTS[:2]) == together


@pytest.mark.parametrize("sampler,greedy", [(_greedy, True), (_sample, False)])
def test_one_drafter_forward_serves_the_requests_of_a_round(target, sampler, greedy):
    config, model = target
    count = 24

    def run(share):
        draft = helpers._drafter(config, 8)
        if not share:
            draft.draft_block_bases = None
        rounds = SharedBlockRounds(model, draft, block_size=8)
        drafted = []
        stock = rounds._draft

        def record(row, size, base=None):
            stock(row, size, base)
            drafted.append((row, row.draft_tokens.tolist()))

        rounds._draft = record
        mx.random.seed(11)
        streams, outputs = {}, {}
        for index, prompt in enumerate(PROMPTS):
            streams[index], _ = _admit(
                rounds, model, draft, prompt, count, sampler=sampler, greedy=greedy
            )
            outputs[index] = []
        order = {id(stream.row): index for index, stream in streams.items()}
        _drain(streams, outputs)
        return (
            outputs,
            [(order[id(row)], tokens) for row, tokens in drafted],
            rounds,
        )

    outputs, drafted, rounds = run(share=True)
    apart_outputs, apart_drafted, apart_rounds = run(share=False)
    # All but the first round, which reads each prompt into the drafter, and
    # the rounds after the other requests have finished.
    assert rounds.shared_drafts > rounds.rounds // 2
    assert apart_rounds.shared_drafts == 0
    # The requests are drafted the same tokens either way.
    assert drafted == apart_drafted
    assert outputs == apart_outputs


def test_block_sizes_fit_the_verify_forward(target):
    config, model = target
    draft = helpers._drafter(config, 8)
    rounds = SharedBlockRounds(model, draft, block_size=8)
    for prompt in PROMPTS:
        _admit(rounds, model, draft, prompt, 40)
    sizes = rounds._block_sizes(list(rounds.rows))
    assert sum(sizes) <= rounds.limit
    assert all(size >= 2 for size in sizes)
    assert rounds.max_requests == 3
    assert not rounds.has_room()


@pytest.mark.parametrize(
    "drafting,plain,peers,contended",
    [
        (0, 0, 0, False),  # alone
        (0, 0, 1, False),  # two starting together draft together
        (0, 0, 2, True),  # three starting together batch without drafting
        (1, 0, 0, False),  # joins a request that is drafting
        (2, 0, 1, False),  # and a third may join two
        (3, 0, 0, True),  # the rounds are full
        (1, 1, 0, True),  # a peer already decodes without drafting
        (0, 1, 0, True),
    ],
)
def test_route_lets_a_few_requests_draft_together(
    monkeypatch, caplog, drafting, plain, peers, contended
):
    import logging
    from types import SimpleNamespace

    import omlx.scheduler as scheduler_mod
    from omlx.scheduler import Scheduler

    monkeypatch.setattr(
        scheduler_mod,
        "shared_block_rounds",
        lambda *args, **kwargs: SimpleNamespace(max_requests=3, start_together=2),
    )
    request = SimpleNamespace(
        request_id="req",
        sampling_params=SimpleNamespace(
            max_tokens=64, stop_token_ids=None, logprobs=False
        ),
        rope_deltas=0.0,
    )
    sched = SimpleNamespace(
        _vlm_mtp_drafter=object(),
        _vlm_mtp_active={-i: object() for i in range(1, drafting + 1)},
        waiting=[object()] * peers,
        running={f"r{i}": object() for i in range(drafting + plain)},
        prefilling=[request],
        model=SimpleNamespace(),
    )
    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        uid = Scheduler._route_to_vlm_mtp(
            sched, request, [object()], [42], lambda logits: logits, object()
        )
    # The stub model has no language model, so a request that passes the
    # gate stops at the next check.
    assert uid is None
    assert ("scheduler contention" in caplog.text) is contended
    assert ("rollback_speculative_cache" in caplog.text) is not contended
    assert "drafter is busy" not in caplog.text


class _Adapter:
    """The model as the scheduler holds it: bare logits for the generation
    batch, the whole output for the forward that seeds the drafter."""

    def __init__(self, model):
        self._language_model = model.language_model

    def __call__(self, tokens, cache=None, **kwargs):
        output = self._language_model(tokens, cache=cache, **kwargs)
        return output if kwargs.get("capture_layer_ids") else output.logits


def _scheduler(model, draft, block_size=4):
    """A scheduler's decode side: the real generation batch and the real
    routing, promotion and drafter steps, around a small model."""
    from collections import deque
    from types import SimpleNamespace

    from mlx_lm.generate import BatchGenerator

    from omlx.scheduler import Scheduler
    from omlx.speculative.vlm_mtp import VLMMTPDrafter

    adapter = _Adapter(model)
    sched = SimpleNamespace(
        model=adapter,
        batch_generator=BatchGenerator(adapter, sampler=_greedy),
        _vlm_mtp_drafter=VLMMTPDrafter(draft, "dflash", "/p"),
        _vlm_mtp_draft_block_size=block_size,
        _vlm_mtp_active={},
        _vlm_mtp_next_uid=-1,
        _model_suppress_tokens=(),
        _pending_abort_ids=set(),
        _stream=mx.default_stream(mx.default_device()),
        _PROMOTE_MIN_TOKENS=Scheduler._PROMOTE_MIN_TOKENS,
        _get_stop_tokens=lambda: set(),
        request_id_to_uid={},
        uid_to_request_id={},
        running={},
        waiting=deque(),
        prefilling=deque(),
        tokens={},
    )
    for name in (
        "_memory_requeue_row",
        "_plain_rows_to_promote",
        "_promote_plain_requests",
        "_route_to_vlm_mtp",
        "_step_vlm_mtp",
        "_log_vlm_mtp_stats",
    ):
        setattr(sched, name, getattr(Scheduler, name).__get__(sched))
    return sched


def _insert_plain(sched, model, name, prompt, count, sampler=_greedy):
    """Prefill a request and put it in the generation batch, not drafting."""
    from types import SimpleNamespace

    cache = model.make_cache()
    mx.eval(sched.model(mx.array([prompt[:-1]]), cache=cache))
    (uid,) = sched.batch_generator.insert(
        [prompt[-1:]],
        max_tokens=[count],
        caches=[cache],
        all_tokens=[list(prompt[:-1])],
        samplers=[sampler],
        logits_processors=[[]],
    )
    request = SimpleNamespace(
        request_id=name,
        prompt_token_ids=list(prompt),
        rope_deltas=0.0,
        sampling_params=SimpleNamespace(
            max_tokens=count,
            temperature=0.0 if sampler is _greedy else 1.0,
            logprobs=False,
            stop_token_ids=None,
            ignore_eos=True,
        ),
    )
    sched.running[name] = request
    sched.request_id_to_uid[name] = uid
    sched.uid_to_request_id[uid] = name
    sched.tokens[name] = []
    return request


def _step(sched):
    """One decode step, as Scheduler.step runs it; drops requests that end."""
    responses = sched._promote_plain_requests()
    responses.extend(sched.batch_generator.next_generated())
    responses.extend(sched._step_vlm_mtp())
    for response in responses:
        name = sched.uid_to_request_id.get(response.uid)
        if name not in sched.running:
            continue
        assert getattr(response, "error", None) is None
        sched.tokens[name].append(int(response.token))
        if response.finish_reason is not None:
            del sched.running[name]
            del sched.request_id_to_uid[name]
    return responses


def _run(sched, limit=400):
    for _ in range(limit):
        if not sched.running:
            return
        _step(sched)
    raise AssertionError("requests did not finish")


def test_a_plain_request_starts_drafting_when_its_peer_finishes(target):
    config, model = target
    counts = (12, 60)
    reference = [
        helpers._decode_one_by_one(model, p, n) for p, n in zip(PROMPTS, counts)
    ]
    sched = _scheduler(model, helpers._drafter(config, 4))
    requests = [
        _insert_plain(sched, model, f"r{i}", PROMPTS[i], counts[i]) for i in range(2)
    ]
    # Two plain requests could draft together; hold them back as a third
    # request waiting to start would.
    sched.waiting.append(object())
    while "r0" in sched.running:
        _step(sched)
    assert not sched._vlm_mtp_active
    assert sched.request_id_to_uid["r1"] >= 0
    done = len(sched.tokens["r1"])
    sched.waiting.clear()
    _step(sched)
    # The request left the batch for the drafter and lost no token on the way.
    assert sched.request_id_to_uid["r1"] < 0
    assert requests[1].batch_uid == sched.request_id_to_uid["r1"]
    assert not len(sched.batch_generator._generation_batch)
    assert len(sched.tokens["r1"]) > done + 1
    _run(sched)
    assert [sched.tokens["r0"], sched.tokens["r1"]] == reference
    rounds = sched._vlm_mtp_drafter.shared_rounds
    assert 0 < rounds.rounds < counts[1] - done
    assert not rounds.rows


def test_two_plain_requests_start_drafting_together(target):
    config, model = target
    counts = (8, 44, 52)
    reference = [
        helpers._decode_one_by_one(model, p, n) for p, n in zip(PROMPTS, counts)
    ]
    sched = _scheduler(model, helpers._drafter(config, 4))
    for i in range(3):
        _insert_plain(sched, model, f"r{i}", PROMPTS[i], counts[i])
    while "r0" in sched.running:
        _step(sched)
        # Three do not draft together, so none moves while all three run.
        assert "r0" not in sched.running or not sched._vlm_mtp_active
    _step(sched)
    assert len(sched._vlm_mtp_active) == 2
    _run(sched)
    assert [sched.tokens[f"r{i}"] for i in range(3)] == reference
    assert sched._vlm_mtp_drafter.shared_rounds.shared_drafts > 0


def test_a_plain_request_joins_one_that_drafts(target):
    config, model = target
    counts = (40, 40)
    reference = [
        helpers._decode_one_by_one(model, p, n) for p, n in zip(PROMPTS, counts)
    ]
    sched = _scheduler(model, helpers._drafter(config, 4))
    _insert_plain(sched, model, "r0", PROMPTS[0], counts[0])
    _step(sched)
    _step(sched)
    assert sched.request_id_to_uid["r0"] < 0
    # A request that reached the batch while the first one drafts.
    _insert_plain(sched, model, "r1", PROMPTS[1], counts[1])
    for _ in range(3):
        _step(sched)
    assert sched.request_id_to_uid["r1"] < 0
    _run(sched)
    assert [sched.tokens["r0"], sched.tokens["r1"]] == reference


@pytest.mark.parametrize(
    "change,moves",
    [
        ({}, True),
        ({"logprobs": True}, False),  # the drafter's loop reports none
        ({"left": 8}, False),  # too little left to be worth the move
        ({"waiting": 1}, False),  # a request about to start would not draft
        ({"aborting": True}, False),
    ],
)
def test_which_plain_requests_move_to_the_drafter(target, change, moves):
    config, model = target
    count = change.get("left", 60) + 3
    sched = _scheduler(model, helpers._drafter(config, 4))
    request = _insert_plain(sched, model, "r0", PROMPTS[0], count)
    request.sampling_params.logprobs = change.get("logprobs", False)
    sched.waiting.extend([object()] * change.get("waiting", 0))
    if change.get("aborting"):
        sched._pending_abort_ids.add("r0")
    for _ in range(4):
        _step(sched)
    assert (sched.request_id_to_uid.get("r0", 0) < 0) is moves


def test_a_sampled_request_moves_with_its_budget_intact(target):
    config, model = target
    count = 50
    sched = _scheduler(model, helpers._drafter(config, 4))
    mx.random.seed(3)
    _insert_plain(sched, model, "r0", PROMPTS[0], count, sampler=_sample)
    _step(sched)
    _step(sched)
    assert sched.request_id_to_uid["r0"] < 0
    _run(sched)
    # Every token the request was allowed, and no more.
    assert len(sched.tokens["r0"]) == count


def test_a_request_that_cannot_start_drafting_fails_cleanly(target):
    config, model = target
    sched = _scheduler(model, helpers._drafter(config, 4))
    _insert_plain(sched, model, "r0", PROMPTS[0], 40)
    sched.batch_generator.next_generated()
    sched._route_to_vlm_mtp = lambda *args, **kwargs: None
    (response,) = sched._promote_plain_requests()
    # Reported against the uid the scheduler still maps to the request.
    assert response.finish_reason == "error" and response.error
    assert sched.uid_to_request_id[response.uid] == "r0"
    assert not len(sched.batch_generator._generation_batch)
