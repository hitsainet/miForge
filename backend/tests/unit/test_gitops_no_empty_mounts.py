"""No host mount may be created empty: every ``hostPath`` is ``type: Directory`` (FR-015.11).

``DirectoryOrCreate`` makes a missing folder on the node silently into an EMPTY one, and the pod
comes up healthy against it. That is how a moved folder (suite P2: miLLM's checkpoint folder
becomes miServer's) or a mistyped path loses a database or a checkpoint tree without a single
error. ``Directory`` makes the pod wait in Pending instead, which someone notices. The
checkpoint mount was fixed first (D47); this guard covers every hostPath in ``gitops/``,
including the ones inside kustomize patch strings.
"""

from __future__ import annotations

from typing import Any

from tests.support import gitops_docs

PVS = {
    "miforge-checkpoints-pv",
    "miforge-trainer-state-pv",
    "miforge-runs-pv",
    "miforge-parts-pv",
    "miforge-packs-pv",
    "miforge-hf-cache-pv",
    "miforge-pgdata-pv",
    "miforge-miserver-checkpoints-pv",
}


def _all_host_paths() -> list[tuple[str, dict[str, Any]]]:
    found = []
    for doc in gitops_docs.walk():
        where = f"{doc.path.relative_to(gitops_docs.REPO_ROOT)}#{doc.index}"
        if doc.embedded_in:
            where += f" ({doc.embedded_in})"
        name = (doc.data.get("metadata") or {}).get("name") if isinstance(doc.data, dict) else None
        for trail, mapping in gitops_docs.host_paths(doc.data):
            found.append((f"{where} {name}: {trail}", mapping))
    return found


def test_every_hostpath_is_directory() -> None:
    found = _all_host_paths()
    assert len(found) >= len(PVS), f"only {len(found)} hostPath mappings found"
    wrong = [
        f"{where} -> type {m.get('type')!r}" for where, m in found if m.get("type") != "Directory"
    ]
    assert not wrong, (
        "every hostPath must be `type: Directory` (a missing folder must keep the pod Pending, "
        "never become an empty mount):\n" + "\n".join(wrong)
    )


def test_discovers_eight_pvs() -> None:
    names = {
        d.data["metadata"]["name"]
        for d in gitops_docs.walk()
        if isinstance(d.data, dict) and d.data.get("kind") == "PersistentVolume"
    }
    assert PVS <= names, f"PVs not found by the walker: {PVS - names}"


def test_walker_reads_embedded_patches() -> None:
    embedded = [d for d in gitops_docs.walk() if d.embedded_in]
    assert embedded, "no kustomize patch string was parsed; the hostIPC component would be missed"
    assert any("#patch" in (d.embedded_in or "") for d in embedded)
