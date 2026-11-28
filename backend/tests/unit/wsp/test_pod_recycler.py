"""``wsp.pod_recycler.KubernetesPodRecycler`` (operator 2026-11-26: a fresh WSP worker pod per
session), over an ``httpx`` mock transport standing in for the API server:

- one ``GET`` and one ``DELETE`` of exactly
  ``/api/v1/namespaces/miforge/pods/miforge-wsp-worker-<i>``, the delete carrying the read pod's
  ``uid`` as its precondition, with the projected token;
- safe to repeat: a pod that is gone, or newer than the session's boot, is left alone and makes no
  delete; a 404 or 409 answer to the delete is done, not an error;
- every refusal (no token, 401, 403, an unreachable API server, an odd answer) is logged
  ``wsp_worker_pod_delete_failed`` and returned as ``refused:<reason>``; nothing raises.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from wsp import pod_recycler
from wsp.pod_recycler import KubernetesPodRecycler

BOOT_SEEN = dt.datetime(2026, 11, 26, 6, 0, 0, tzinfo=dt.UTC)
PATH = "/api/v1/namespaces/miforge/pods/miforge-wsp-worker-1"
UID = "0f1e2d3c-aaaa-bbbb-cccc-000000000001"


def _pod(created: str = "2026-11-26T05:59:00Z", uid: str = UID) -> dict[str, object]:
    return {
        "kind": "Pod",
        "metadata": {"name": "miforge-wsp-worker-1", "uid": uid, "creationTimestamp": created},
    }


def _recycler(
    tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response], token: str | None = "tok"
) -> tuple[KubernetesPodRecycler, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    account = tmp_path / "sa"
    account.mkdir()
    if token is not None:
        (account / "token").write_text(token + "\n")
    recycler = KubernetesPodRecycler(
        "miforge-wsp-worker-{index}",
        "miforge",
        account_dir=account,
        base_url="https://api.test:443",
        transport=httpx.MockTransport(record),
    )
    return recycler, seen


def _answers(get: httpx.Response, delete: httpx.Response | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return get
        assert delete is not None, "a DELETE was not expected"
        return delete

    return handler


def test_it_reads_then_deletes_the_one_pod_with_its_uid_as_the_precondition(
    tmp_path: Path,
) -> None:
    recycler, seen = _recycler(
        tmp_path, _answers(httpx.Response(200, json=_pod()), httpx.Response(200, json={}))
    )
    assert recycler.recycle(1, BOOT_SEEN) == "deleted"
    assert [(r.method, r.url.path) for r in seen] == [("GET", PATH), ("DELETE", PATH)]
    assert all(r.headers["Authorization"] == "Bearer tok" for r in seen)
    assert json.loads(seen[1].content) == {
        "kind": "DeleteOptions",
        "apiVersion": "v1",
        "preconditions": {"uid": UID},
    }
    assert seen[0].content == b""


def test_a_202_is_a_delete_too(tmp_path: Path) -> None:
    recycler, _ = _recycler(
        tmp_path, _answers(httpx.Response(200, json=_pod()), httpx.Response(202, json={}))
    )
    assert recycler.recycle(1, BOOT_SEEN) == "deleted"


def test_a_pod_that_is_gone_is_done_with_no_delete(tmp_path: Path) -> None:
    recycler, seen = _recycler(tmp_path, _answers(httpx.Response(404, json={})))
    assert recycler.recycle(1, BOOT_SEEN) == "already_gone"
    assert [r.method for r in seen] == ["GET"]


def test_a_pod_already_being_deleted_is_done_with_no_second_delete(tmp_path: Path) -> None:
    pod = _pod()
    pod["metadata"]["deletionTimestamp"] = "2026-11-26T06:00:05Z"  # type: ignore[index]
    recycler, seen = _recycler(tmp_path, _answers(httpx.Response(200, json=pod)))
    assert recycler.recycle(1, BOOT_SEEN) == "already_deleting"
    assert [r.method for r in seen] == ["GET"]


def test_a_pod_newer_than_the_boot_is_the_replacement_and_is_left_alone(tmp_path: Path) -> None:
    recycler, seen = _recycler(
        tmp_path, _answers(httpx.Response(200, json=_pod(created="2026-11-26T06:00:07Z")))
    )
    assert recycler.recycle(1, BOOT_SEEN) == "already_replaced"
    assert [r.method for r in seen] == ["GET"]


@pytest.mark.parametrize(("status", "word"), [(404, "already_gone"), (409, "already_replaced")])
def test_a_delete_that_finds_the_pod_gone_or_replaced_is_done(
    tmp_path: Path, status: int, word: str
) -> None:
    recycler, seen = _recycler(
        tmp_path, _answers(httpx.Response(200, json=_pod()), httpx.Response(status, json={}))
    )
    assert recycler.recycle(1, BOOT_SEEN) == word
    assert len(seen) == 2


def test_repeating_it_after_the_replacement_deletes_nothing_more(tmp_path: Path) -> None:
    """The first call deletes the old pod; the StatefulSet's new pod is newer than the boot, so a
    repeat reads it and stops: two calls, one delete."""
    pods = [_pod(), _pod(created="2026-11-26T06:00:09Z", uid="new-uid")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=pods.pop(0))
        return httpx.Response(200, json={})

    recycler, seen = _recycler(tmp_path, handler)
    assert recycler.recycle(1, BOOT_SEEN) == "deleted"
    assert recycler.recycle(1, BOOT_SEEN) == "already_replaced"
    assert [r.method for r in seen] == ["GET", "DELETE", "GET"]


@pytest.mark.parametrize(
    ("get", "delete", "reason"),
    [
        (httpx.Response(403, json={}), None, "get_403"),
        (httpx.Response(401, json={}), None, "get_401"),
        (httpx.Response(200, json=_pod()), httpx.Response(403, json={}), "delete_403"),
        (httpx.Response(200, json=_pod()), httpx.Response(500, json={}), "delete_500"),
        (httpx.Response(200, json=_pod(uid="")), None, "no_uid"),
        (httpx.Response(200, content=b"not json"), None, "bad_answer"),
    ],
)
def test_a_refusal_is_logged_and_returned_never_raised(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    get: httpx.Response,
    delete: httpx.Response | None,
    reason: str,
) -> None:
    recycler, _ = _recycler(tmp_path, _answers(get, delete))
    with caplog.at_level(logging.WARNING, logger="wsp.pod_recycler"):
        assert recycler.recycle(1, BOOT_SEEN) == f"refused:{reason}"
    (line,) = caplog.records
    assert line.code == pod_recycler.FAILED_CODE  # type: ignore[attr-defined]
    assert line.reason == reason and line.worker_index == 1  # type: ignore[attr-defined]


def test_an_unreachable_api_server_is_a_refusal(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    recycler, _ = _recycler(tmp_path, handler)
    assert recycler.recycle(1, BOOT_SEEN) == "refused:unreachable_ConnectError"


@pytest.mark.parametrize("token", [None, ""])
def test_no_token_makes_no_request(tmp_path: Path, token: str | None) -> None:
    recycler, seen = _recycler(tmp_path, _answers(httpx.Response(200, json=_pod())), token=token)
    assert recycler.recycle(1, BOOT_SEEN) == "refused:no_token"
    assert seen == []


def test_settings_without_a_name_or_its_index_replace_nothing() -> None:
    assert pod_recycler.from_settings("", "miforge") is None
    assert pod_recycler.from_settings("miforge-wsp-worker", "miforge") is None
    made = pod_recycler.from_settings("miforge-wsp-worker-{index}", "miforge")
    assert isinstance(made, KubernetesPodRecycler) and made.pod_name(3) == "miforge-wsp-worker-3"
