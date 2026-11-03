"""Every edge case in the FPRD's §2 table is produced by a test  (AC-002.20, task 12.8).

*A reason code with no test is a code that exists only in prose.* This file is the audit,
not the coverage: each row names the code and **where it is actually asserted**, and the
test below reads the refusal codes this feature can raise out of the **source tree** and
requires every one of them to be named here. So a code added without a test fails this file
rather than quietly joining the prose.

Why an inventory rather than twenty more assertions: the behaviour of each refusal is
asserted where it happens, with the payload that matters. Repeating them here would be
twenty tests that agree with the table by construction - and the thing worth catching is a
code that **nothing** asserts.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
TESTS = Path(__file__).resolve().parents[1]

#: Edge case -> (refusal code or outcome, where the behaviour is asserted).
#: ``None`` as a code means the row is an **outcome, not an error** - a count on a
#: successful record - which is the distinction FTID §12 keeps in code and in test names.
EDGE_CASES: dict[str, tuple[str | None, str]] = {
    "E-1 unknown source name": ("signal_source_unknown", "test_signal_registry.py"),
    "E-2 part kind mismatch": ("part_kind_mismatch", "test_signal_parts_reader.py"),
    "E-3 held-out / evaluation-only split": (
        "split_not_available_for_training",
        "test_signal_parts_reader.py",
    ),
    "E-4 no usable content semantic": ("corpus_no_content_column", "test_signal_semantics.py"),
    "E-5 masking policy unset": (
        "sft_masking_policy_not_configured",
        "test_signal_settings.py + test_signal_corpus_end_to_end.py",
    ),
    "E-6 generation failed mid-batch": ("generation_failed", "test_signal_self_generation.py"),
    "E-7 everything rejected": (
        "selection_admitted_nothing",
        "test_signal_self_generation.py",
    ),
    "E-8 pending human review": (None, "test_signal_selection.py (the three-value table)"),
    "E-9 empty completion": (None, "test_signal_self_generation.py + test_objective_sft.py"),
    "E-10 malformed pair": (None, "test_signal_pairs.py + test_signal_preference_pairs.py"),
    "E-11 tie with no ordering scalar": (None, "test_signal_pairs.py"),
    "E-12 pairs part without preference columns": (
        "pairs_no_preference_columns",
        "test_signal_semantics.py",
    ),
    "E-13 DPO reference source unset": (
        "dpo_reference_source_not_configured",
        "test_objective_dpo.py + test_signal_preference_pairs.py",
    ),
    "E-14 grader value out of range": (
        "grader_score_out_of_range",
        "F3's own; propagated, re-checked nowhere - test_signal_reward.py",
    ),
    "E-15 zero spread in a group": (None, "test_signal_reward.py + test_signal_reward_source.py"),
    "E-16 group smaller than two": ("reward_group_too_small", "test_signal_reward.py"),
    "E-17 record fails validation": ("batch_record_invalid", "test_batch_record.py"),
    "E-18 miDataworks unconfigured (NORMAL)": (
        None,
        "test_standalone_without_siblings.py",
    ),
    "E-19 external part provenance": (None, "test_signal_parts_reader.py"),
    "E-20 cancellation mid-assembly": (
        "batch_assembly_cancelled",
        "test_signal_self_generation.py",
    ),
}

#: Codes this feature raises that are **not** one of §2's twenty, each with why it exists.
#: Listed so the exhaustiveness check below can tell "new and considered" from "new and
#: nobody looked" - which is the distinction that makes the check bite.
ADDITIONAL_CODES: dict[str, str] = {
    "batch_source_not_in_contract": "the gate contract's `source` enum has no `reward` value",
    "batch_record_hash_mismatch": "a record whose hash covers different content",
    "batch_block_malformed": "a run-log line that is not a batch block",
    "run_log_absent": "a rebuild with no authoritative log to rebuild from",
    "batch_not_found": "a hash no batch block carries",
    "batch_page_too_large": "a page over the configured cap, named",
    "signal_source_already_registered": "two plugins claiming one name",
    "signal_source_capability_mismatch": "a listing that names a different source",
    "signal_source_spec_mismatch": "a spec handed to the wrong source",
    "signal_source_no_part": "a source with no material",
    "signal_source_not_preflighted": "a batch asked for before pre-flight",
    "part_index_unavailable": "no session bound; parts come from F10's index",
    "part_provenance_absent": "a part with neither a reference nor provenance",
    "part_directory_missing": "an indexed part whose directory is gone",
    "part_declares_no_columns": "a content kind with no column declarations",
    "column_semantic_ambiguous": "two usable candidates and no configuration",
    "column_binding_unknown": "a configured column that cannot fill the role",
    "unmapped_column_semantic": "a semantic the vendored schema added",
    "split_format_unsupported": "a split file this release cannot read",
    "split_file_missing": "a declared split that is not on disk",
    "split_corrupt": "a row that is not valid JSON - refused, not skipped",
    "parquet_reader_unavailable": "pyarrow is owed in pyproject.toml",
    "corpus_column_not_text": "a cell that is not text - refused, not coerced",
    "prompt_column_not_text": "as above, for a prompt",
    "pair_column_not_text": "as above, for a preference side",
    "corpus_exhausted": "no rows left; a zero-row batch is not produced",
    "prompt_set_exhausted": "no prompts left; R-02.26's refresh is F6's",
    "no_pairs_produced": "no orderable pair in the round",
    "pair_ordering_not_configured": "nothing produces the scalar that orders two generations",
    "batch_could_not_fill": "the configured shortfall behaviour is refuse_batch",
    "generation_provider_not_configured": "no provider bound; the default is in-process",
    "generation_timeout": "a batch over its generation budget",
    "generation_short_reply": "fewer completions than prompts",
    "generation_cancelled": "cancelled between completions",
    "generating_identity_unresolved": "no candidate, so no model to record",
    "generating_identity_ambiguous": "two models answered one batch",
    "prompt_too_long": "bounded before the call, refused rather than truncated",
    "verdict_count_mismatch": "a positional join over mismatched lists",
    "reward_score_count_mismatch": "scores that do not line up with completions",
    "selector_not_configured": "a selecting source with no selector",
    "grader_not_configured": "a scoring source with no grader",
    "objective_unknown": "no fall-back to cross-entropy",
    "objective_already_registered": "two implementations claiming one name",
    "batch_empty": "a zero-example step",
    "batch_material_mismatch": "material for an objective the batch does not name",
    "sft_masking_policy_absent": "a batch that reached the loss with no policy",
    "sft_masking_policy_unknown": "a policy that is neither of the two",
    "dpo_reference_source_absent": "a batch that reached the loss with no reference",
    "dpo_reference_source_unknown": "a reference source that is neither of the two",
    "dpo_reference_logprobs_missing": "precomputed, and nothing was precomputed",
    "dpo_reference_model_unavailable": "frozen_copy, and no frozen set was supplied",
    "torch_unavailable": "the one path that needs a tensor library",
    "tokenizer_unavailable": "a batch with no tokenizer",
    "tokenizer_unsupported": "a tokenizer with no encode()",
    "tokenizer_has_no_pad_token": "padding with id 0 would train on a real token",
    "chat_template_unavailable": "an invented chat format is text the model never saw",
    "example_empty_completion": "a zero-length training target",
    "pending_shortfall_not_configured": "OQ-1, left to an operator",
    "schema_not_vendored": "the gate contract is not on disk",
    "selection_admitted_nothing": "see E-7",
    "evaluation_set_empty": "F4's, reached through the provider",
}


def raised_codes() -> set[str]:
    """Every string literal this feature passes to ``refuse`` as its code.

    Read from the **source tree**, so a code added without a row here fails this file. The
    first argument of a ``refuse(...)`` call, which is where ``core.errors`` puts the code.
    """
    paths = list((SRC / "training" / "signals").rglob("*.py"))
    paths += list((SRC / "training" / "objectives").rglob("*.py"))
    paths += [
        SRC / "services" / "signal_source_service.py",
        SRC / "api" / "v1" / "endpoints" / "signal_sources.py",
        SRC / "api" / "v1" / "endpoints" / "signal_batches.py",
    ]
    found: set[str] = set()
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in {"refuse", "Refusal"} or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.add(first.value)
    return found


class TestEveryEdgeCaseIsCovered:
    def test_all_twenty_rows_are_present(self) -> None:
        assert len(EDGE_CASES) == 20
        assert sorted(EDGE_CASES)[0].startswith("E-1 ")

    def test_every_row_names_where_its_behaviour_is_asserted(self) -> None:
        for case, (_code, where) in EDGE_CASES.items():
            for name in where.replace("+", " ").split():
                if name.endswith(".py"):
                    matches = list(TESTS.rglob(name))
                    assert matches, f"{case} points at {name}, which does not exist"

    def test_every_refusal_code_this_feature_raises_is_accounted_for(self) -> None:
        """**The check that bites.** A code added without a test and without a row here
        fails, rather than existing only in prose."""
        documented = {code for code, _where in EDGE_CASES.values() if code} | set(ADDITIONAL_CODES)
        undocumented = raised_codes() - documented
        assert not undocumented, f"refusal codes with no row and no test: {sorted(undocumented)}"

    def test_the_scanner_actually_found_the_codes(self) -> None:
        """A positive check that the measurement happened: an empty scan would make the
        assertion above pass vacuously, which is the construction-green failure this
        project keeps finding."""
        codes = raised_codes()
        assert len(codes) > 40
        assert "batch_record_invalid" in codes
        assert "signal_source_unknown" in codes
