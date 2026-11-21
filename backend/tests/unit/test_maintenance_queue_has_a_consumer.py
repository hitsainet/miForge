"""The ``maintenance`` queue has a running consumer on both deploy paths  (task 10.7).

**A declared queue with no worker is an unreachable capability** - the thing ADR-016 #1
exists to catch - and ``maintenance`` is the sharpest case in the product because it
carries the chain sweep, which is **the only advance path a chain has** (FR-006.45). With
no consumer, ``POST /chains`` returns 202, stage one runs, and the curriculum sits at stage
one forever: green tests, a healthy stack, and nothing anywhere saying why.

## Why this reads BOTH deploy paths

``docker-compose.yml``'s own comment claims *"the compose path and the GitOps path stay the
same story rather than diverging"*, and on 2026-10-14 two of compose's four fixes had never
been applied to the cluster path - *"nothing connected the two deploy paths."* So this file
asserts the same property against both, in one place, rather than trusting either file's
comment about the other.

## What it deliberately does NOT assert

``default`` and ``generate`` have no consumer in ``docker-compose.yml``. That is the
foundation's gap (task 8.1 names five services) and is **reported, not closed here** -
nothing F6 enqueues goes to either queue, and silently extending a file this task does not
own would hide the gap rather than fix it. The gitops path does carry all four, and the
general form below asserts that, so a fifth queue added to ``QUEUES`` fails in a test
rather than in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
COMPOSE = REPO / "docker-compose.yml"
GITOPS_BACKEND = REPO / "gitops" / "pilot" / "backend.yaml"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def _gitops_containers() -> list[dict]:
    documents = [
        document
        for document in yaml.safe_load_all(GITOPS_BACKEND.read_text(encoding="utf-8"))
        if document and document.get("kind") == "Deployment"
    ]
    assert documents, "no Deployment in gitops/pilot/backend.yaml"
    spec = documents[0]["spec"]["template"]["spec"]
    return list(spec.get("containers") or [])


def _queues_consumed(commands: list[list[str]]) -> set[str]:
    """Every queue named by a ``-Q`` flag in a worker command.

    Read from the command rather than from the container's name: a container called
    ``worker-maintenance`` that passes ``-Q default`` consumes ``default``, and the name
    would say otherwise. **The flag is what the broker sees.**
    """
    consumed: set[str] = set()
    for command in commands:
        if "worker" not in command:
            continue
        for index, token in enumerate(command):
            if token == "-Q" and index + 1 < len(command):
                consumed.update(part.strip() for part in str(command[index + 1]).split(","))
    return consumed


class TestTheScannerActuallyReadsSomething:
    """The positive check that the measurement happened.

    A YAML read that returned an empty mapping would make every assertion below pass
    vacuously - *a measurement that did not happen reported as a result*, which is the
    family of false measurements this project has now hit four times.
    """

    def test_both_files_parse_and_contain_workers(self) -> None:
        assert _compose()["services"], "compose declares no services"
        assert _gitops_containers(), "the gitops Deployment declares no containers"
        assert _queues_consumed([["celery", "-A", "x", "worker", "-Q", "train"]]) == {"train"}
        assert _queues_consumed([["celery", "-A", "x", "beat"]]) == set()


class TestCompose:
    def test_the_maintenance_queue_has_a_consumer(self) -> None:
        """Task 10.7. Deleting the ``celery-maintenance`` service must make this red."""
        commands = [
            list(service.get("command") or []) for service in _compose()["services"].values()
        ]
        assert "maintenance" in _queues_consumed(commands), (
            "no compose service consumes `maintenance`, so the chain sweep and the janitor "
            "are both enqueued onto a queue nothing reads"
        )

    def test_every_worker_is_solo_with_concurrency_one(self) -> None:
        """ADR-006, and the consequences are the design: ``revoke(terminate=True)`` does not
        work and a busy solo worker never reads the control queue, so everything that must
        reach a running job goes through the database."""
        for name, service in _compose()["services"].items():
            command = list(service.get("command") or [])
            if "worker" not in command:
                continue
            assert "--pool=solo" in command, name
            assert "-c" in command and command[command.index("-c") + 1] == "1", name


class TestGitOps:
    @pytest.mark.parametrize("queue", ["default", "train", "generate", "maintenance"])
    def test_every_declared_queue_has_a_consumer(self, queue: str) -> None:
        """The general form, so **a fifth queue added to ``QUEUES`` fails here.**

        Parametrised over the literal four rather than over the imported tuple on purpose:
        importing ``QUEUES`` would make a queue *removed* from the code silently drop its
        own assertion, and the manifest would keep a container for a queue that no longer
        exists with nothing saying so. The cross-check below is what ties the two together.
        """
        commands = [list(container.get("command") or []) for container in _gitops_containers()]
        assert queue in _queues_consumed(commands), f"no cluster container consumes {queue!r}"

    def test_the_manifest_and_the_code_declare_the_same_queues(self) -> None:
        """Both directions. A queue in the code with no container is an unreachable
        capability; a container for a queue the code does not declare is a worker reading a
        queue nothing routes to."""
        from workers.celery_app import QUEUES

        commands = [list(container.get("command") or []) for container in _gitops_containers()]
        assert _queues_consumed(commands) == set(QUEUES)

    def test_beat_runs_as_its_own_container(self) -> None:
        """**Without Beat the sweep never fires**, and a chain never advances.

        One Beat, ever - two would each schedule the sweep - which ``replicas: 1`` plus
        ``strategy: Recreate`` guarantees. It is deliberately not embedded with
        ``worker --beat``: embedding ties the one scheduler's lifetime to one worker's, so a
        maintenance-worker restart silently skips a window.
        """
        beats = [
            container
            for container in _gitops_containers()
            if "beat" in list(container.get("command") or [])
        ]
        assert len(beats) == 1, "exactly one Beat container, and it must exist"
        assert "--beat" not in " ".join(
            " ".join(str(part) for part in container.get("command") or [])
            for container in _gitops_containers()
        ), "an embedded beat ties the scheduler's lifetime to one worker's"
