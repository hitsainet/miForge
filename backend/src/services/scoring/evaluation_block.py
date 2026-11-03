"""The run log's ``evaluation`` block - **F4 writes it**  (FR-004.28; foundation 3.3a).

``step-record-v1.json`` is a discriminated union on ``block`` with five members, and this is
F4's. Its own description states the reason it exists, and it is the reason this module is
careful:

    "THIS BLOCK EXISTS BECAUSE FOUR OF ITS FIELDS CHANGE WHAT A SCORE MEANS AND THE
    MANIFESTS DO NOT CARRY THEM. ... Still absent from both manifests, and therefore
    recorded here: the aggregation method, the decode parameters, the subsample seed and the
    split markers. Two checkpoints can both report '0.82' and not be comparable without
    them."

``ScoreRecord`` gained ``provider`` on 2026-10-13, so a score's provenance now travels on the
artifact - which matters because an adapter is routinely moved without its run log. The other
four values did **not** come with it (DEF-2, which BRD-02 §9 records as *"still owed and
deliberately not bundled"*), so this block is their only home, and it is the only home of
**any** score at a step with no save.

## One block per pass, carrying both sets

``scores`` is an array with ``minItems: 1`` and no ``kind`` on its entries, so a paired test
-and-retention measurement is **one** block. That is the contract agreeing with FR-004.24:
the two scores share the step, the provider and the decode parameters by construction rather
than by two calls being scheduled to coincide, and two calls will eventually be configured,
or retried, differently.

## Which entry is the retention set

``ScoreEntry`` has ``set``, ``score``, ``n``, ``floor`` and ``provider`` - and **no kind**.
The discriminator is ``floor``: a retention entry always has one (FR-004.27 refuses a
retention set without a floor, and the table constrains it) and a test entry's is null
(*"Set for retention (BRD-02 R-02.25); normally null for test"*, the manifest's own words).
:func:`kind_of_entry` is the single place that rule is written, because ``reindex`` rebuilds
``mf_scores.kind`` from the log alone and two derivations would be two answers.

## ``retention_floor_breached`` is a FACT on the pass

The schema says so: *"Recorded as a FACT on the pass rather than derived later by comparing
two numbers, so the decision the loop took is reconstructible from the log alone (ADR-004:
reindex rebuilds from the log)."* Recomputing ``score < floor`` at reindex time against a
floor that has since been edited gives a different answer to the one the run acted on.

## F4 does not write this file; it hands blocks to the writer F1 settled

:func:`training.run_log.append` validates every line against the schema and refuses to write
one that does not conform - *a record a reader cannot trust is worse than a gap*. This module
assembles and nothing more, so there is exactly one writer and one validator for the log.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from core.errors import refuse
from providers.base import GenerationParams
from services.scoring.tables import KIND_RETENTION, KIND_TEST

#: The discriminator. Required on every line, and the reason ``reindex`` can classify a line
#: it is holding without knowing which file it came from.
BLOCK = "evaluation"


def decode_params_payload(params: GenerationParams) -> dict[str, Any]:
    """``EvaluationBlock.decode_params``, from the gate interface's own type.

    The schema requires ``temperature``, ``top_p`` and ``max_tokens`` and permits ``stop``.
    Built from :class:`providers.base.GenerationParams` rather than from loose keyword
    arguments so the values recorded are **the values that were sent** - a second assembly
    of the same four numbers is how a log claims a pass ran at temperature 0 that did not.

    ``seed`` is deliberately **not** here: the schema puts the subsample seed at the block
    level as ``subsample_seed``, which is what it is - *which ``n`` prompts*, not a decode
    parameter.
    """
    payload: dict[str, Any] = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
    }
    if params.stop:
        payload["stop"] = list(params.stop)
    return payload


def score_entry(
    *,
    set_ref: str,
    score: float,
    n: int,
    floor: float | None,
    provider: str | None,
) -> dict[str, Any]:
    """One ``ScoreEntry``. **All five fields are required by the schema**, including nulls.

    ``provider`` is in ``ScoreEntry.required`` here even though it is *not* required on the
    manifest's ``ScoreRecord`` - the run log is stricter, deliberately, because it is the
    authoritative copy and a score whose provenance is missing from the authority cannot be
    recovered from anywhere.
    """
    return {
        "set": set_ref,
        "score": score,
        "n": n,
        "floor": floor,
        "provider": provider,
    }


def kind_of_entry(entry: Mapping[str, Any]) -> str:
    """``test`` or ``retention`` for one ``ScoreEntry``. **The single derivation.**

    ``ScoreEntry`` carries no kind, so the rule is: *a floor means retention*. It holds
    because FR-004.27 refuses a retention set with no floor and the manifest documents the
    test set's floor as *"normally null for test"*. Written once because ``reindex`` rebuilds
    ``mf_scores.kind`` from the log alone and a second copy of this rule would drift.
    """
    return KIND_RETENTION if entry.get("floor") is not None else KIND_TEST


def evaluation_block(
    *,
    run_id: str,
    step: int,
    scores: Sequence[Mapping[str, Any]],
    provider: str,
    aggregation: str,
    decode_params: Mapping[str, Any],
    subsample_seed: int | None,
    split: Sequence[str],
    retention_floor_breached: bool,
    ts: datetime | None = None,
) -> dict[str, Any]:
    """Assemble one ``evaluation`` block. **Every field here is the schema's.**

    Raises:
        Refusal: ``evaluation_block_invalid`` for an empty ``scores`` array. The schema's
            ``minItems: 1`` would catch it at the writer, but a block with no scores means a
            pass produced nothing and recorded it anyway, and the message is clearer here.

    The block is **not** written by this function - :func:`training.run_log.append` validates
    and writes, which keeps one writer and one validator for the log.
    """
    if not scores:
        raise refuse(
            "evaluation_block_invalid",
            "An evaluation block must carry at least one score (the schema's minItems is "
            "1). A pass that produced no score records a refusal, not an empty block: an "
            "empty block reads as 'this pass ran and found nothing', which is a different "
            "and false claim.",
            run_id=run_id,
            step=step,
        )
    _assert_finite(scores)
    return {
        "block": BLOCK,
        "run_id": run_id,
        "step": step,
        "ts": (ts or datetime.now(UTC)).isoformat(),
        "scores": [dict(entry) for entry in scores],
        "provider": provider,
        "aggregation": aggregation,
        "decode_params": dict(decode_params),
        # Null is an assertion that no seed was set, which makes the number irreproducible -
        # worth recording as a fact rather than hiding. The schema says so in terms.
        "subsample_seed": subsample_seed,
        "split": list(split),
        "retention_floor_breached": retention_floor_breached,
    }


def score_rows_from_block(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One ``evaluation`` block as ``mf_scores`` rows. **The single projection.**

    The live writer and the ``reindex`` rebuild both call this, which is what makes the
    reindex-equivalence test meaningful: two projections would agree until one of them was
    changed, and the test compares API responses, so the disagreement would surface as a
    reindex that quietly returns different numbers.

    Only the fields the log carries are produced. ``floor_layer``, the per-prompt detail and
    the selector/grader hashes have no field in ``EvaluationBlock`` - the block's
    ``additionalProperties: false`` closes it - so a rebuilt row reports them as unknown
    rather than as absent, and **that gap is reported** rather than smuggled into another
    field.
    """
    decode = dict(record.get("decode_params") or {})
    split = list(record.get("split") or [])
    rows: list[dict[str, Any]] = []
    for entry in record.get("scores") or []:
        kind = kind_of_entry(entry)
        rows.append(
            {
                "run_id": record["run_id"],
                "step": record["step"],
                "kind": kind,
                "set_rendered": entry["set"],
                "score": float(entry["score"]),
                "n": int(entry["n"]),
                "floor": entry.get("floor"),
                # A breach is a fact of the PASS, and the pass carries at most one retention
                # entry, so it attaches to the retention row. A test row is never breached:
                # it has no floor to breach, and the check constraint refuses the
                # combination.
                "breached": bool(record.get("retention_floor_breached")) and kind == KIND_RETENTION,
                "provider": entry.get("provider") or record["provider"],
                "aggregation": record["aggregation"],
                "decode": decode,
                "subsample_seed": record.get("subsample_seed"),
                "split_markers": split,
                "ts": record.get("ts"),
            }
        )
    return rows


def _assert_finite(scores: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a non-finite score or floor **inside** an entry, naming the field.

    The writer's own finite check (:func:`training.run_log._assert_finite_numbers`) walks the
    record's **top-level** values only, so a NaN nested inside ``scores[i].score`` passes it:
    JSON Schema's ``number`` admits a NaN float, the block validates, and ``canonical_bytes``
    then refuses it as a bare ``ValueError`` - which reaches a client as a 500 with a
    traceback rather than as a named refusal. *Reported as a finding on the shared writer;
    guarded here meanwhile, at the boundary that produces the value.*

    In practice :mod:`services.scoring.aggregate` refuses a non-finite score before it can
    reach a block at all. This is the second line, at the point the value becomes a record.
    """
    for index, entry in enumerate(scores):
        for field_name in ("score", "floor", "n"):
            value = entry.get(field_name)
            if isinstance(value, float) and (
                value != value or value in (float("inf"), float("-inf"))
            ):
                raise refuse(
                    "evaluation_block_invalid",
                    f"Score entry {index} carries a non-finite {field_name} ({value}). It "
                    f"cannot be written as JSON - no parser is required to accept the "
                    f"literal NaN - so the log would validate going out and fail coming "
                    f"back. Refusing at the point the value becomes a record.",
                    entry=index,
                    field=field_name,
                    value=str(value),
                )
