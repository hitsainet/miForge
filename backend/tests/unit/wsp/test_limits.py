"""FR-016.7: each violation kind is named, an empty limit set passes, patterns are fnmatch."""

from __future__ import annotations

from wsp.weight_ops.limits import KINDS, Violation, check_limits


def _patch(*tensors: tuple[str, int]) -> dict:
    return {
        "changed_entries_total": sum(n for _, n in tensors),
        "tensors": [{"name": name, "changed_entries": n, "append": None} for name, n in tensors],
    }


def test_empty_limits_pass() -> None:
    assert check_limits(_patch(("weight", 3)), {}) == []
    assert (
        check_limits(
            _patch(("weight", 3)),
            {"max_changed_entries": None, "allowed_tensors": None, "frozen_tensors": []},
        )
        == []
    )


def test_max_changed_entries() -> None:
    assert check_limits(_patch(("weight", 3)), {"max_changed_entries": 3}) == []
    (v,) = check_limits(_patch(("weight", 3), ("bias", 1)), {"max_changed_entries": 3})
    assert v.kind == "max_changed_entries" and v.tensor is None and "4" in v.detail


def test_frozen_tensor_by_pattern() -> None:
    (v,) = check_limits(_patch(("std", 1), ("weight", 1)), {"frozen_tensors": ["s*"]})
    assert v == Violation("frozen_tensor", "std", "tensor 'std' is frozen")


def test_tensor_not_allowed() -> None:
    (v,) = check_limits(_patch(("weight", 1), ("bias", 1)), {"allowed_tensors": ["weig?t"]})
    assert v.kind == "tensor_not_allowed" and v.tensor == "bias"


def test_matching_is_case_sensitive() -> None:
    assert check_limits(_patch(("Weight", 1)), {"frozen_tensors": ["weight"]}) == []


def test_all_three_kinds_together_in_a_stable_order() -> None:
    found = check_limits(
        _patch(("std", 5), ("bias", 1)),
        {"max_changed_entries": 2, "frozen_tensors": ["std"], "allowed_tensors": ["std"]},
    )
    assert [v.kind for v in found] == ["max_changed_entries", "frozen_tensor", "tensor_not_allowed"]
    assert set(KINDS) == {v.kind for v in found}


def test_an_unchanged_tensor_is_not_a_violation() -> None:
    assert check_limits(_patch(("std", 0)), {"frozen_tensors": ["std"]}) == []


def test_limits_module_imports_only_the_standard_library() -> None:
    import ast
    import sys
    from pathlib import Path

    import wsp.weight_ops.limits as module

    tree = ast.parse(Path(module.__file__).read_text())
    tops = {
        (n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert tops and tops <= set(sys.stdlib_module_names) | {"__future__"}, tops
