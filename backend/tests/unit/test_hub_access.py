"""Every Hugging Face Hub call states whose credential it may use (017 FR-017.28; A28, A36).

An AST inventory of ``backend/src``:

- every call of a ``huggingface_hub`` fetch (``repo_info``, ``snapshot_download``,
  ``get_safetensors_metadata``, ``hf_hub_download``, ``list_repo_files``, any ``HfApi`` method)
  and of miForge's own Hub helpers (``resolve_revision``, ``download_snapshot``,
  ``remote_config``, ``remote_weights``, ``remote_files``) passes a ``token`` keyword; without
  one, ``huggingface_hub`` sends the pod's ``HF_TOKEN`` or token file;
- every ``from_pretrained`` call is listed below, by its exact text, with why it reads a local
  directory. ``AutoTokenizer.from_pretrained`` on a Hub name sends the pod's token even with
  ``token=False`` (A36), so no such call may exist at all: a new or changed one fails here.

A scan is paired with behaviour tests on the wire (``test_agent_hub_anonymous.py``), because a
scan only proves the text (memory ``source-scrape-guards-fail-open``); this one asserts how many
sites it found before it asserts anything about them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

LIBRARY_CALLS = frozenset(
    {
        "repo_info",
        "snapshot_download",
        "get_safetensors_metadata",
        "hf_hub_download",
        "list_repo_files",
        "list_repo_tree",
        "model_info",
        "file_exists",
    }
)
HELPER_CALLS = frozenset(
    {"resolve_revision", "download_snapshot", "remote_config", "remote_weights", "remote_files"}
)

#: (file relative to backend/src, the call's exact text) -> why its first argument is local.
LOCAL_FROM_PRETRAINED: dict[tuple[str, str], str] = {
    (
        "training/acquisition/local_dir.py",
        "auto_model.from_pretrained(directory, dtype=dtype, trust_remote_code=False)",
    ): "a local directory: confined to the model roots (or a fetched snapshot) by the caller",
    (
        "training/acquisition/local_dir.py",
        "auto_tokenizer.from_pretrained(directory, trust_remote_code=False)",
    ): "the same local directory",
    (
        "training/acquisition/scratch.py",
        "AutoTokenizer.from_pretrained(str(tokenizer_directory), trust_remote_code=False)",
    ): "resolve_tokenizer_directory's answer: a confined path or a fetched tokenizer snapshot",
    (
        "training/runtypes/adapters/merge.py",
        "PeftModel.from_pretrained(base_module, str(adapter_directory))",
    ): "an adapter checkpoint directory miForge wrote",
    (
        "training/hotupdate/attach.py",
        "AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)",
    ): "a directory or a path-spelled source, loaded with local_files_only: no Hub request",
    (
        "training/hotupdate/attach.py",
        "AutoTokenizer.from_pretrained(str(local), local_files_only=True, trust_remote_code=False)",
    ): "the tokenizer files download_snapshot just fetched with the run's token (A36)",
}


def _calls() -> list[tuple[str, int, ast.Call, str]]:
    out: list[tuple[str, int, ast.Call, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name is None:
                continue
            out.append((str(path.relative_to(SRC)), node.lineno, node, name))
    return out


def _has_token(call: ast.Call) -> bool:
    return any(k.arg == "token" for k in call.keywords)


def test_every_hub_call_passes_a_token() -> None:
    calls = _calls()
    library = [(f, n, c) for f, n, c, name in calls if name in LIBRARY_CALLS]
    helpers = [
        (f, n, c)
        for f, n, c, name in calls
        if name in HELPER_CALLS and isinstance(c.func, ast.Attribute | ast.Name)
    ]
    # repo_info, two snapshot_download, get_safetensors_metadata, hf_hub_download,
    # list_repo_files: six library sites, all in training/acquisition/hub.py.
    assert len(library) >= 6, library
    assert len(library) + len(helpers) >= 7
    missing = [f"{f}:{n} {ast.unparse(c)}" for f, n, c in library + helpers if not _has_token(c)]
    assert not missing, "a Hub call without a token keyword:\n" + "\n".join(missing)


def test_no_hfapi_is_constructed_outside_the_hub_module() -> None:
    """A bare ``HfApi()`` would carry its own implicit-token lookup past this inventory."""
    sites = [
        f"{f}:{n}"
        for f, n, _c, name in _calls()
        if name == "HfApi" and f != "training/acquisition/hub.py"
    ]
    assert not sites, sites


def test_every_from_pretrained_is_a_listed_local_load() -> None:
    found = {(f, ast.unparse(c)): n for f, n, c, name in _calls() if name == "from_pretrained"}
    assert len(found) >= 5  # the scan read the tree
    unlisted = sorted(
        f"{f}:{n} {text}"
        for (f, text), n in found.items()
        if (f, text) not in LOCAL_FROM_PRETRAINED
    )
    assert not unlisted, (
        "from_pretrained on a source not listed as local. A Hub name must be fetched with "
        "hub.download_snapshot(..., token=...) and loaded locally (A36):\n" + "\n".join(unlisted)
    )
    stale = sorted(set(LOCAL_FROM_PRETRAINED) - set(found))
    assert not stale, f"listed but no longer in the code: {stale}"


@pytest.mark.parametrize(
    ("actor_origin", "expected"), [("agent", False), ("operator", None), (None, False)]
)
def test_hub_token_by_actor(actor_origin: str | None, expected: bool | None) -> None:
    from api.agent_origin import Actor
    from services import hub_access

    actor = None if actor_origin is None else Actor(origin=actor_origin)  # type: ignore[arg-type]
    assert hub_access.hub_token(actor) is expected


def test_hub_token_reads_the_requesting_and_executing_scopes() -> None:
    from api.agent_origin import Actor, executing_as
    from services import hub_access

    assert hub_access.hub_token() is False  # nobody named: fail closed
    with hub_access.requesting(Actor(origin="operator")):
        assert hub_access.hub_token() is None
        # The approval executor's requester wins: an approved agent request is the agent's.
        with executing_as(Actor(origin="agent", agent="agent:forge-mcp")):
            assert hub_access.hub_token() is False
    with hub_access.requesting(Actor(origin="agent", agent="agent:forge-mcp")):
        assert hub_access.hub_token() is False
    assert hub_access.hub_token() is False


def test_fetch_is_metadata_for_an_agent() -> None:
    from services import hub_access

    assert hub_access.fetch_for(False) == "metadata"
    assert hub_access.fetch_for(None) == "snapshot"


def test_the_worker_scope_and_its_fail_closed_default() -> None:
    from types import SimpleNamespace

    from services import hub_access

    assert hub_access.worker_hub_token() is False
    for access, token in ((None, None), ("anonymous", False)):
        with hub_access.run_scope(hub_access.hub_token_for_run(SimpleNamespace(hub_access=access))):
            assert hub_access.worker_hub_token() is token
    assert hub_access.worker_hub_token() is False


@pytest.mark.parametrize("value", ["operator", "ANONYMOUS", "", 0, False, "none"])
def test_a_run_with_an_unexpected_hub_access_fails_closed(value: object) -> None:
    """Only an explicit null is the operator's run; anything else is anonymous (OR2-017-4)."""
    from types import SimpleNamespace

    from services import hub_access

    assert hub_access.hub_token_for_run(SimpleNamespace(hub_access=value)) is False


def test_a_run_object_with_no_hub_access_fails_closed() -> None:
    from types import SimpleNamespace

    from services import hub_access

    assert hub_access.hub_token_for_run(SimpleNamespace()) is False
    assert hub_access.hub_token_for_run(SimpleNamespace(hub_access=None)) is None


def test_the_helpers_take_token_as_a_required_keyword() -> None:
    """017-D24: a forgotten token is a TypeError, never a silent None."""
    import inspect

    from training.acquisition import hub

    for name in HELPER_CALLS:
        param = inspect.signature(getattr(hub, name)).parameters["token"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert param.default is inspect.Parameter.empty, name
    preflight = inspect.signature(hub.HubAcquirer.preflight).parameters
    for name in ("token", "fetch"):
        assert preflight[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert preflight[name].default is inspect.Parameter.empty
    from services import acquisition as acquisition_service

    service = inspect.signature(acquisition_service.preflight).parameters
    for name in ("token", "fetch"):
        assert service[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert service[name].default is inspect.Parameter.empty
