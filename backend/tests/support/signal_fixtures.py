"""Parts, providers and contexts for F2's tests  (ADR-016).

**Every part here goes through F10's real ingest**, from a manifest derived from
miDataworks' own committed fixture. ADR-016: *fixtures whose fields agree by construction
are how a suite stays green over a real bug* - and the fields that must not agree by
construction in this feature are exactly the provenance ones. A hand-built part row would
agree with whatever the reader expected and would never notice that the contract moved.

Split bodies are written as **JSON lines** rather than Parquet, and the manifest's declared
paths say so. ``pyarrow`` is not in ADR-002's dependency list and ``pyproject.toml`` has one
owner, so the Parquet branch of :mod:`training.signals.parts_reader` refuses by name until
that dependency is added - which is reported, not worked around. The reader dispatches on
the declared suffix, so a ``.jsonl`` export exercises the same code path a ``.parquet`` one
would, minus the one lazily-imported call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from providers.fakes import FakeModelHandle
from providers.in_process import InProcessProvider
from services.parts import ingest as parts_ingest
from services.parts.resolve import resolve_one
from tests.support import part_fixtures


#: Column declarations by role, in the contract's own shape. ``semantic`` is what binds a
#: column to a role; ``name`` is deliberately uninformative in some fixtures, because the
#: decisive test is that a column called ``text`` whose semantic is ``other`` is refused.
def column(name: str, semantic: str, role: str = "content") -> dict[str, Any]:
    """One declared column, in the contract's own shape.

    ``label_values`` is a **list** for a ``label`` column and null otherwise, because the
    schema makes that conditional (``if semantic == "label" then label_values is an
    array``). Getting it wrong here would make every fixture refuse at F10's validator -
    which is the fixture builder earning its keep: a hand-built part row would simply have
    agreed with whatever the reader expected.
    """
    return {
        "name": name,
        "arrow_type": "string",
        "role": role,
        "semantic": semantic,
        "label_values": ["a", "b"] if semantic == "label" else None,
        "extensions": {},
    }


def rows_manifest(
    *,
    miforge_set_kind: str,
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    split_name: str = "train",
    held_out: bool = False,
    evaluation_only: bool = False,
    reference: tuple[str, str, str] | None = ("prompt-set", "support-tickets", "3"),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """A one-split ``miforge_set`` version with the columns and rows a test needs.

    Built from miDataworks' real document through ``part_fixtures.miforge_set_manifest``, so
    only the fields a test is about differ from theirs - a fixture that differs in three
    ways cannot say which of the three a refusal was about.
    """
    body = b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows)
    manifest, _bodies = part_fixtures.miforge_set_manifest(
        miforge_set_kind=miforge_set_kind,
        reference=reference,
        split_bodies={split_name: body},
    )
    manifest["content"]["columns"] = columns
    splits = [split for split in manifest["content"]["splits"] if split["name"] == split_name]
    split = splits[0]
    split["path"] = f"{split_name}.jsonl"
    split["rows"] = len(rows)
    split["held_out"] = held_out
    split["evaluation_only"] = evaluation_only
    manifest["content"]["splits"] = [split]
    return manifest, {split_name: body}


def import_part(
    session: Session,
    tmp_path: Path,
    *,
    manifest: dict[str, Any],
    bodies: dict[str, bytes],
    name: str = "export",
    source: str = "midataworks_contract",
    external_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an export directory and take it in **through F10's real ingest**.

    Returns the part payload as F10's index serves it - the same object a source resolves at
    pre-flight, so the test and the production path read one shape.
    """
    staged = part_fixtures.write_export(tmp_path / name, manifest, bodies)
    result = parts_ingest.ingest(
        session,
        staged,
        parts_root=tmp_path / "parts",
        source=source,
        external_provenance=external_provenance,
    )
    session.flush()
    return dict(resolve_one(session, result.part_id).payload)


def corpus_part(
    session: Session, tmp_path: Path, *, texts: list[str], **kwargs: Any
) -> dict[str, Any]:
    """A corpus part whose single content column declares ``text``."""
    manifest, bodies = rows_manifest(
        miforge_set_kind="corpus",
        columns=[column("body", "text"), column("label", "label", role="metadata")],
        rows=[{"body": text, "label": "a"} for text in texts],
        reference=("corpus", "support-notes", "1"),
        **kwargs,
    )
    return import_part(session, tmp_path, manifest=manifest, bodies=bodies, name="corpus")


def prompt_part(
    session: Session, tmp_path: Path, *, prompts: list[str], **kwargs: Any
) -> dict[str, Any]:
    """A prompt-set part whose single content column declares ``prompt``."""
    manifest, bodies = rows_manifest(
        miforge_set_kind="prompt_set",
        columns=[column("q", "prompt")],
        rows=[{"q": prompt} for prompt in prompts],
        **kwargs,
    )
    return import_part(session, tmp_path, manifest=manifest, bodies=bodies, name="prompts")


def pairs_part(
    session: Session, tmp_path: Path, *, pairs: list[tuple[str, str]], **kwargs: Any
) -> dict[str, Any]:
    """A preference-pairs part declaring ``chosen`` and ``rejected``."""
    manifest, bodies = rows_manifest(
        # THREE vocabularies touch this file and they are not aliases for one another:
        # miDataworks' `miforge_set_kind` (["prompt_set","corpus","preference_pairs",
        # "test_set","retention_set"] - theirs, underscored, not ours to respell), miForge's
        # PART kinds (also underscored), and miForge's signal SOURCES (hyphenated, and
        # identical to the checkpoint manifest's `signal_source`). A rename that treated
        # them as one broke this file on 2026-10-14; the comment is here so the next person
        # sees the distinction before reaching for a global replace.
        miforge_set_kind="preference_pairs",
        columns=[column("good", "chosen"), column("bad", "rejected")],
        rows=[{"good": good, "bad": bad} for good, bad in pairs],
        reference=("preference_pairs", "tone", "2"),
        **kwargs,
    )
    return import_part(session, tmp_path, manifest=manifest, bodies=bodies, name="pairs")


def in_process_provider(answers: Any = "an answer") -> InProcessProvider:
    """F4's **real** in-process provider over its fake handle - no card, no sibling.

    The real provider rather than a stand-in, so the identity-capture path that records
    ``resolved_model`` runs for real: a stub provider that returned a hand-built
    ``ProviderIdentity`` would make that assertion agree with itself.
    """
    return InProcessProvider(FakeModelHandle(answers=answers))


def rule_selector(
    *, forbidden: tuple[str, ...] = ("reject-me",), name: str = "no-boilerplate"
) -> dict[str, Any]:
    """A rule selector configuration: pure, in-process, **no sibling** (AC20's path).

    ``expression`` holds JSON because that is what F3's contract says it holds - *"the rule
    body, interpreted by F3. Opaque to this contract"* - and F3 parses it with a closed
    model. A rule with no checks is refused by F3 at ``prepare``, because a selector that
    accepts everything is not selecting; so the fixture always carries one.
    """
    return {
        "kind": "rule",
        "name": name,
        "config_sha256": None,
        "expression": json.dumps({"forbidden_terms": list(forbidden)}),
    }


# ====================================================================== the torch seam


class FakeTensor:
    """Enough tensor semantics to carry real data through the slicing a loss does.

    A **recording implementation, not a mock**: the labels really are sliced and the logits
    really are reshaped, so an assertion about what reached ``cross_entropy`` is about what
    the loss would hand it rather than about what the test arranged. ``torch`` is not
    installed on a machine without a card (ADR-023) and the loss is the only path that needs
    it, so this stands in at the one import seam.
    """

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getitem__(self, key: Any) -> FakeTensor:
        if isinstance(key, tuple):
            rows = self.data[key[0]]
            return FakeTensor([row[key[1]] for row in rows])
        return FakeTensor(self.data[key])

    def size(self, dim: int) -> int:
        node: Any = self.data
        while isinstance(node[0], list):
            node = node[0]
        return len(node) if dim in (-1, 2) else len(self.data)

    def reshape(self, *shape: int) -> FakeTensor:
        flat: list[Any] = []
        for row in self.data:
            flat.extend(row)
        return FakeTensor(flat)

    def gather(self, dim: int, index: Any) -> FakeTensor:
        return FakeTensor([[0.0 for _ in row] for row in self.data])

    def squeeze(self, dim: int) -> FakeTensor:
        return self

    def unsqueeze(self, dim: int) -> FakeTensor:
        return self

    def clamp(self, min: int = 0) -> FakeTensor:
        return self

    def sum(self) -> float:
        return -1.0

    def __ne__(self, other: Any) -> FakeTensor:  # type: ignore[override]
        return FakeTensor([[1 for _ in row] for row in self.data])

    def __mul__(self, other: Any) -> FakeTensor:
        return self


class FakeScalar:
    """A loss value that records ``backward()`` - the shape F1's step consumes."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value
        self.backward_calls = 0

    def backward(self) -> None:
        self.backward_calls += 1

    def detach(self) -> FakeScalar:
        return self

    def item(self) -> float:
        return self.value


class FakeTorch:
    """The one import seam the three objectives reach through, recorded."""

    long = "long"

    def __init__(self) -> None:
        self.cross_entropy_calls: list[tuple[Any, Any, dict[str, Any]]] = []
        self.stacked: list[list[Any]] = []
        outer = self

        class _Functional:
            @staticmethod
            def cross_entropy(logits: Any, labels: Any, **kwargs: Any) -> Any:
                outer.cross_entropy_calls.append((logits.data, labels.data, kwargs))
                return FakeScalar()

            @staticmethod
            def log_softmax(tensor: Any, dim: int = -1) -> Any:
                return tensor

            @staticmethod
            def logsigmoid(value: Any) -> Any:
                return value

        class _Nn:
            functional = _Functional()

        self.nn = _Nn()

    def tensor(self, data: Any, dtype: Any = None, device: Any = None) -> FakeTensor:
        return FakeTensor(data)

    def ones_like(self, tensor: Any) -> FakeTensor:
        return tensor

    def stack(self, items: list[Any]) -> Any:
        self.stacked.append(list(items))
        return FakeScalar()

    def as_tensor(self, value: Any) -> Any:
        return value

    def no_grad(self) -> Any:
        from contextlib import nullcontext

        return nullcontext()


class FakeTokenizer:
    """One token per character, so a label position maps to a character a test can read."""

    pad_token_id = 0
    chat_template = "{% for m in messages %}{{m.content}}{% endfor %}"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        return "|".join(str(message["content"]) for message in messages)


class FakeModel:
    """A model that really holds parameters and returns logits shaped like a real one."""

    def __init__(self) -> None:
        self.parameter_list = [object(), object()]
        self.calls = 0

    def parameters(self) -> list[Any]:
        return self.parameter_list

    def __call__(self, *, input_ids: Any, attention_mask: Any) -> Any:
        self.calls += 1
        return FakeTensor([[[0.0] * 128 for _ in row] for row in input_ids.data])


class FakeOptimizer:
    """A recording optimizer that really holds parameter groups (AC-002.11's subject)."""

    def __init__(self, params: list[Any]) -> None:
        self.param_groups = [{"params": list(params)}]
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self, set_to_none: bool = False) -> None:
        return None
