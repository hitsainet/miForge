"""The corpus source  (R-02.4; FR-002.5 - FR-002.9).

Plain text or chat examples from a miDataworks version - *"files or a directory are still
accepted, but only as a contract-conformant export of such a version"*, R-02.4's own words,
which is why every part comes through F10's ingest and index and nothing here reads a
directory of its own.

**The source with neither generation nor selection**, which is why it is built first: it
proves the surface - the reader, the admissibility rule, the column binding, the record and
the hash - before anything harder is built on it.

Three things it does **not** do:

* **nothing is normalised on the way in.** No lowercasing, no whitespace squashing, and
  above all **no truncation**: truncation changes the training example *without changing any
  hash*, so the record would describe a batch that was never trained on. Where a bound is
  needed, the refusal names it.
* **no selector and no verdict.** Its verdict counts are zero and its selection health
  reports zero candidates - see :func:`training.signals.batch_record.assemble` for why that
  is not the same as reporting that everything was accepted.
* **no shuffling by default.** The order is the part's, so two runs over one part consume
  the same material in the same order and their losses are comparable step for step.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from core.errors import refuse
from training.signals.base import (
    BatchCounts,
    RunContext,
    SourceCapability,
    SupervisedExample,
    TrainingBatch,
)
from training.signals.common import BaseSource
from training.signals.parts_reader import read_rows
from training.signals.settings import require_masking_policy

NAME = "corpus"

CAPABILITY = SourceCapability(
    name=NAME,
    part_kind="corpus",
    needs_selector=False,
    needs_grader=False,
    needs_generation=False,
    objective="sft",
    summary=(
        "plain text or chat examples from a dataset version, trained with next-token "
        "cross-entropy; no generation, no selection and no other app needed"
    ),
)


class CorpusSource(BaseSource):
    """R-02.4. Reads a corpus part and yields supervised examples."""

    NAME = NAME
    CAPABILITY = CAPABILITY

    def __init__(self, context: RunContext | None = None) -> None:
        super().__init__(context)
        self._rows: Iterator[tuple[str, Mapping[str, Any]]] | None = None

    def resolve_extra(self, resolved: Any) -> None:
        """Refuse a missing masking policy **at pre-flight**, not at the first step (E-5)."""
        require_masking_policy(self.settings)

    def next(self) -> TrainingBatch:
        """One batch of supervised examples, recorded before it is returned."""
        resolved = self.require_resolved()
        self.check_cancelled()
        policy = require_masking_policy(self.settings)
        wanted = resolved.spec.batch_size

        if self._rows is None:
            self._rows = self._row_stream()

        examples: list[SupervisedExample] = []
        exhausted = False
        template: str | None = None
        for prompt_id, row in self._rows:
            example, rendered = self._example(prompt_id, row)
            template = template or rendered
            examples.append(example)
            if len(examples) >= wanted:
                break
        else:
            exhausted = True

        counts = BatchCounts(
            generated=0,
            admitted=len(examples),
            rejected=0,
            pending=0,
            exhausted=exhausted,
            requested=wanted,
        )
        if not examples:
            raise refuse(
                "corpus_exhausted",
                "This corpus has no rows left to read. A zero-row batch is a step that "
                "trains on nothing while every log line looks healthy, so the run stops "
                "here rather than producing one. The run loop decides what happens next.",
                run_id=self.context.run_id,
            )
        self.handle_shortfall(counts, requested=wanted)
        return self.record(
            examples=examples,
            counts=counts,
            masking_policy=policy,
            chat_template=template,
        )

    def _row_stream(self) -> Iterator[tuple[str, Mapping[str, Any]]]:
        resolved = self.require_resolved()
        for part in resolved.parts:
            # The columns the binding resolved, and no others: reading a column nobody
            # bound would materialise material the record does not describe.
            columns = [name for name, _semantic in part.binding.roles.values()]
            yield from read_rows(part, columns=columns)

    def _example(
        self, prompt_id: str, row: Mapping[str, Any]
    ) -> tuple[SupervisedExample, str | None]:
        """One row as an example, by the **bound** role - never by a column's name."""
        resolved = self.require_resolved()
        binding = resolved.parts[0].binding
        roles = binding.roles
        if "messages" in roles:
            from training.objectives.tokenize import render_messages

            raw = row[binding.column("messages")]
            messages = raw if isinstance(raw, list) else [raw]
            text, template = render_messages(self.context.tokenizer, messages)
            return (
                SupervisedExample(
                    prompt_id=prompt_id,
                    prompt="",
                    completion=text,
                    messages=tuple(messages),
                ),
                template,
            )
        if "prompt" in roles and "completion" in roles:
            return (
                SupervisedExample(
                    prompt_id=prompt_id,
                    prompt=_text(row[binding.column("prompt")], "prompt"),
                    completion=_text(row[binding.column("completion")], "completion"),
                ),
                None,
            )
        return (
            SupervisedExample(
                prompt_id=prompt_id,
                prompt="",
                completion=_text(row[binding.column("text")], "text"),
            ),
            None,
        )


def _text(value: Any, role: str) -> str:
    """A cell as text, **refusing a non-string rather than coercing one** (ADR-018).

    ``str(None)`` is ``"None"`` and ``str(['a'])`` is ``"['a']"``: both are training
    examples nobody wrote, and both train as happily as a real one.
    """
    if not isinstance(value, str):
        raise refuse(
            "corpus_column_not_text",
            f"The column bound to {role!r} holds {type(value).__name__}, not text. "
            f"Refusing rather than coercing it: str() of a null or a list is a training "
            f"example nobody wrote.",
            role=role,
            value_type=type(value).__name__,
        )
    return value


def factory(context: RunContext) -> CorpusSource:
    return CorpusSource(context)
