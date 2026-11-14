"""BRD criterion 14's ``(§5.10 only)`` clause: the attached base is **read-only**.

**AC-011.18, FR-011.60; MUT-19.** R-02.43 is **F8's requirement and is not implemented
here**: F11 supplies a base the adapter run reads, F8 supplies the adapter, its
configuration and its artifacts. *A requirement lives in exactly one feature; an acceptance
clause may be owned by the feature that implements the mode it describes.*

**Three assertions, and the first is the weakest.** The guarantee criterion 14 states is
that the resident base is byte-unchanged, so the honest check is a hash before and after,
through the real rebuild path. But **a before/after hash passes whenever no gradient step
happens to land on the base** - which is most of the time, and certainly in a short test. So
the ones that bite are:

* the client's ``quiesce`` / ``commit-begin`` / ``resume`` call counts, asserted **zero**
  (*"an adapter run never interrupts serving"*: a bracket here holds serving for a write
  that never happens, and the symptom is latency rather than an error); and
* ``requires_grad`` set on the adapter's parameters and **not** on any rebuilt base tensor,
  so the base *cannot* take a gradient even if one were computed.

The hash halves that need a card are ``gpu``-marked; the call-count and ``requires_grad``
halves run with no GPU, which is where the real protection is.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import adapter_read


class Param(bytes):
    """A parameter whose bytes are readable without torch, so the digest is a real digest.

    F8's ``_parameter_bytes`` falls back to ``bytes(memoryview(p))``, and this exercises
    that route: the digest path here is the production one rather than a stub that agrees
    with itself.

    **A `bytes` subclass rather than `__buffer__`**, for the reason recorded on
    ``tests.support.test_adapter_doubles.FakeParameter``: ``__buffer__`` is PEP 688 and
    needs Python 3.12, so on the project's declared 3.11 floor this double was unreadable
    and all three tests below failed with ``base_digest_unreadable`` in CI while passing
    locally. The instance is rebuilt from its ``bytearray`` on every ``named_parameters()``
    call, so being immutable costs nothing - ``Module`` owns the mutable payload.
    """

    requires_grad: bool

    def __new__(cls, data: bytearray | bytes, requires_grad: bool = False) -> Param:
        parameter = super().__new__(cls, bytes(data))
        parameter.requires_grad = requires_grad
        return parameter


@dataclass
class Module:
    """A module with base parameters and, after wrapping, an adapter's."""

    params: dict[str, bytearray] = field(
        default_factory=lambda: {
            "model.embed_tokens.weight": bytearray(b"\x01\x02\x03\x04"),
            "model.layers.0.self_attn.q_proj.weight": bytearray(b"\x05\x06\x07\x08"),
        }
    )
    adapter: dict[str, bytearray] = field(default_factory=dict)
    grads: dict[str, bool] = field(default_factory=dict)

    def named_parameters(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        for name, data in {**self.params, **self.adapter}.items():
            out.append((name, Param(data, requires_grad=self.grads.get(name, False))))
        return out

    def wrap_with_an_adapter(self) -> None:
        """What F8's ``build_adapter`` does: add ``lora_`` parameters and make them
        trainable. The base's ``requires_grad`` is left alone, which is the property."""
        self.adapter["model.layers.0.self_attn.q_proj.lora_A.weight"] = bytearray(b"\x00" * 4)
        self.grads["model.layers.0.self_attn.q_proj.lora_A.weight"] = True


class TestNoBracketIsEverOpened:
    def test_the_bracket_call_counts_are_zero_across_a_whole_adapter_run(
        self, harness: Harness
    ) -> None:
        """**AC-011.18's load-bearing assertion, and MUT-19's first half.**"""
        response = harness.attach()
        harness.fake.calls.clear()
        # The rebuild itself needs torch, so the no-GPU half exercises everything around
        # it: the device check, the read-only contract, and - most of all - the absence of
        # any barrier call on this path.
        # ⚠ THE CODE, NOT `Exception`. This read `pytest.raises(Exception)` with the
        # comment *"torch is absent in CI by design"* — and that premise is FALSE: torch is
        # declared (`torch>=2.4,<3`) and installed. So the honest path and the mutated path
        # both raise an `Exception` and the assertion could not tell them apart, which is
        # why **MUT-19b survived**: flipping `requires_grad=False` to `True` in
        # `attach_base` left this file at 14 passed. `rebuild.build` *does* refuse the
        # conflict, and a direct test pins that function — but nothing pinned THE CALL
        # SITE, and the call site is what FR-011.60's read-only contract lives in.
        # ⚠ `client=` IS REQUIRED HERE, and its absence made this test vacuous.
        # This is **AC-011.18's load-bearing assertion**, and it built `_Acquirer(response)`
        # with no client - so the four zero-call-count assertions below were guaranteed BY
        # THE DOUBLE rather than by the behaviour: with no client the code cannot open a
        # bracket even if it tries. The class has carried a `client` since the first sweep
        # (MUT-19a needed it) and this one call site never passed it, so the fix reached the
        # sibling test and not the criterion's own. A double that cannot express the defect
        # makes its assertion vacuous, however many assertions there are.
        with pytest.raises(Refusal) as caught:
            adapter_read.attach_base(_Acquirer(response, client=harness.client), response=response)
        assert caught.value.code == "weights_rebuild_failed", (
            f"expected the honest rebuild failure, got {caught.value.code!r}. "
            f"`weights_rebuild_arguments_conflict` here means the call site asked for "
            f"gradients on a read-only rebuild (MUT-19b)."
        )
        assert harness.fake.call_count("quiesce") == 0
        assert harness.fake.call_count("commit-begin") == 0
        assert harness.fake.call_count("commit-end") == 0
        assert harness.fake.call_count("resume") == 0

    def test_the_module_does_not_import_the_bracket(self) -> None:
        """Task 10a.13. *An import is the cheapest thing to add and the hardest to notice*,
        and a ``quiesce`` on this path holds serving for a write that never happens - the
        symptom is **latency, not an error**, so no test would otherwise see it."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(adapter_read))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
                imported += [f"{node.module}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
        assert not any(
            "barrier" in name for name in imported
        ), f"adapter_read imports the bracket: {imported}"

    def test_the_serving_fake_was_never_held(self, harness: Harness) -> None:
        response = harness.attach()
        with contextlib.suppress(Exception):
            adapter_read.attach_base(_Acquirer(response, client=harness.client), response=response)
        assert harness.fake.quiesced is False
        assert harness.fake.serve_completion().startswith("served by")
        assert harness.fake.served_while_not_clean == []


class TestOnlyTheAdapterRequiresGrad:
    def test_the_base_carries_no_gradient_and_F8s_own_assertion_is_the_one_used(self) -> None:
        """**MUT-19's second half.** One definition of the rule, in the feature that owns
        the adapter: F11 calls F8's assertion rather than restating it."""
        module = Module()
        module.wrap_with_an_adapter()
        trainable = adapter_read.assert_only_the_adapter_requires_grad(module)
        assert len(trainable) == 1

    def test_a_base_tensor_with_requires_grad_is_refused_by_name(self) -> None:
        """The mutation, applied: *in this form the base **can** receive a gradient, which
        is the one thing R-02.43 guarantees it cannot.*"""
        module = Module()
        module.wrap_with_an_adapter()
        module.grads["model.embed_tokens.weight"] = True
        with pytest.raises(Refusal) as caught:
            adapter_read.assert_only_the_adapter_requires_grad(module)
        assert caught.value.code == "adapter_base_is_trainable"

    def test_a_read_only_rebuild_refuses_to_set_requires_grad_at_all(self) -> None:
        """The two arguments disagree about whether the base may take a gradient, so the
        conflict is refused rather than resolved by precedence - *picking one silently is
        how MUT-19's ``requires_grad`` half would ship.*"""
        from training.hotupdate import rebuild

        with pytest.raises(Refusal) as caught:
            rebuild.build({"parameters": [{}]}, read_only=True, requires_grad=True)
        assert caught.value.code == "weights_rebuild_arguments_conflict"


class TestTheBaseIsHashedThroughTheRealPath:
    def test_the_digest_is_F8s_and_refuses_an_empty_module(self) -> None:
        """*A digest over nothing compares equal to another digest over nothing, and
        criterion 14 would pass against a run that rewrote the whole base.* F8's function
        already refuses that, and F11 reuses it rather than writing a second one."""
        assert adapter_read.base_digest(Module()) == adapter_read.base_digest(Module())
        with pytest.raises(Refusal) as caught:
            adapter_read.base_digest(Module(params={}))
        assert caught.value.code == "base_digest_empty"

    def test_a_changed_base_byte_is_detected(self) -> None:
        """The guarantee, as a check. Not a copy: the same module object is hashed before
        and after, because *a fixture that hashes its own copy agrees with itself.*"""
        module = Module()
        base = adapter_read.ReadOnlyBase(
            module=module,
            tensors={},
            device_uuid=None,
            digest_before=adapter_read.base_digest(module),
        )
        assert adapter_read.assert_unchanged(base) == base.digest_before
        module.params["model.embed_tokens.weight"][0] = 0xFF
        with pytest.raises(Refusal) as caught:
            adapter_read.assert_unchanged(base)
        assert caught.value.code == "resident_base_changed"

    def test_the_adapters_own_tensors_may_change_freely(self) -> None:
        """They are *supposed* to - including them would make the comparison fail on every
        successful run, which is why F8 excludes them by name."""
        module = Module()
        module.wrap_with_an_adapter()
        before = adapter_read.base_digest(module)
        module.adapter["model.layers.0.self_attn.q_proj.lora_A.weight"][0] = 0x7F
        assert adapter_read.base_digest(module) == before


class TestCrossDeviceIsImpossibleRatherThanInadvisable:
    def test_it_refuses_naming_both_devices_and_the_alternative(self, harness: Harness) -> None:
        """E-17. *A CUDA IPC handle is valid only on the device that owns the tensor*, so
        this is a refusal rather than a fallback - and ADR-018 forbids taking F8's copy path
        silently, because that would train a copy while miLLM serves the original."""
        response = harness.attach()
        with pytest.raises(Refusal) as caught:
            adapter_read.attach_base(
                _Acquirer(response), compute_device_uuid="GPU-other-card", response=response
            )
        assert caught.value.code == "attachment_cross_device"
        assert caught.value.details["compute_device"] == "GPU-other-card"
        assert caught.value.details["serving_device"] == "GPU-fake-0000"
        assert "default" in caught.value.details["alternative"]

    def test_the_same_device_is_accepted(self, harness: Harness) -> None:
        response = harness.attach()
        # Same correction as above: the code, not `Exception`. "It got past the gate" is
        # the actual claim, and only the code can state which side of the gate it died on.
        with pytest.raises(Refusal) as caught:
            adapter_read.attach_base(
                _Acquirer(response), compute_device_uuid="GPU-fake-0000", response=response
            )
        assert (
            not isinstance(caught.value, Refusal) or caught.value.code != "attachment_cross_device"
        )

    def test_no_attachment_at_all_refuses_rather_than_falling_back(self) -> None:
        with pytest.raises(Refusal) as caught:
            adapter_read.attach_base(_Acquirer(None))
        assert caught.value.code == "attachment_missing"
        assert "refusing rather than silently taking it" in caught.value.message.lower()


def _fenced_import_sites(tree: object) -> list[int]:
    """Line numbers where a module reaches ``hotupdate`` - statically OR dynamically.

    *Widened 2026-10-16 (suite tracker 4.29).* This walked only ``ast.Import`` and
    ``ast.ImportFrom``, so ``importlib.import_module("training.hotupdate")`` on F8's adapter
    or merge path left it green (verified 2026-10-16: only the global importer-count guard
    caught it). It now also sees ``importlib.import_module`` / ``import_module`` /
    ``__import__`` with a literal argument, and - because a dynamic import can be spelled in
    ways no list of call names anticipates - **any non-docstring string constant naming the
    package**, which is what a dynamic import of it has to carry somewhere.
    """
    import ast

    assert isinstance(tree, ast.AST)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    sites: list[int] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and "hotupdate" in (node.module or "")) or (
            isinstance(node, ast.Import) and any("hotupdate" in a.name for a in node.names)
        ):
            sites.append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in {"import_module", "__import__"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and "hotupdate" in str(first.value):
                    sites.append(node.lineno)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "training.hotupdate" in node.value
            and id(node) not in docstrings
        ):
            sites.append(node.lineno)
    return sites


class TestF8IsNotRoutedIntoThisPackage:
    def test_no_F8_module_imports_the_fenced_package(self) -> None:
        """FR-011.62, and the reason it is written as a prohibition on F8 rather than as a
        capability of F11: *routing F8 into ``hotupdate/`` would make a **core-path**
        feature the package's second importer*, which the counted fence test fails on. The
        mechanism instead is the one F1 established - the acquirer hands the run a
        read-only base and a bracketed writer, and F8 uses whichever it is handed."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        paths = list((src / "training" / "runtypes").rglob("*.py"))
        assert len(paths) >= 3, "the scan found no F8 modules - it would prove nothing"
        offenders = [
            f"{path.relative_to(src)}:{line}"
            for path in paths
            for line in _fenced_import_sites(ast.parse(path.read_text("utf-8")))
        ]
        assert offenders == []

    @pytest.mark.parametrize(
        "planted",
        [
            "from training.hotupdate import barrier",
            "import training.hotupdate",
            "import importlib\nm = importlib.import_module('training.hotupdate')",
            "from importlib import import_module\nm = import_module('training.hotupdate.save')",
            "m = __import__('training.hotupdate')",
            "NAME = 'training.hotupdate'\nimport importlib\nm = importlib.import_module(NAME)",
        ],
    )
    def test_the_scanner_sees_every_import_form(self, planted: str) -> None:
        """The guard's self-test: each form, planted, is seen. A scanner never seen to
        fire is a scanner trusted rather than tested."""
        import ast

        assert _fenced_import_sites(ast.parse(planted)) != []

    def test_a_docstring_naming_the_package_is_not_an_import(self) -> None:
        import ast

        source = '"""Explains why training.hotupdate is never imported here."""\nX = 1\n'
        assert _fenced_import_sites(ast.parse(source)) == []

    def test_no_F8_module_branches_on_the_attach_mode(self) -> None:
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        offenders: list[str] = []
        for path in (src / "training" / "runtypes").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text("utf-8"))):
                if isinstance(node, ast.Compare):
                    for operand in [node.left, *node.comparators]:
                        if isinstance(operand, ast.Constant) and operand.value == "attach":
                            offenders.append(f"{path.name}:{node.lineno}")
                        if isinstance(operand, ast.Attribute) and operand.attr == "ATTACH":
                            offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == []


class _Acquirer:
    """The acquirer's shape: a raw attach response **and its client**.

    **The client is here because the first mutation sweep needed it.** MUT-19a opens a
    bracket on this path through ``acquirer.client``, and with no ``client`` attribute on
    the double the mutated code took a ``None`` branch and quietly did nothing - so the
    mutation **survived**, not because the zero-call-count assertion was wrong but because
    the double could not carry a call. A test double that cannot express the defect makes
    its assertion vacuous, which is the fixture-agrees-by-construction failure CLAUDE.md
    names, arriving through the double rather than through the data.

    The rebuild is **not** mocked: it is the real one, which is why the no-torch cases raise.
    """

    def __init__(self, response: dict[str, Any] | None, client: Any = None) -> None:
        self.attachment = None if response is None else _Attachment(response)
        self.client = client


class _Attachment:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
