"""The live miDataworks round trip - **F10 task 8.6, measured on a real LAN service.**

``upstream``-marked, so **CI never runs it** (``pyproject.toml``'s ``addopts`` is
``-m 'not gpu and not upstream'``), and additionally skipped when ``MIDATAWORKS_BASE_URL`` is
unset - which is the *production* setting, so an unset one means miForge is standalone and
there is genuinely nothing to measure. The pair is deliberate: a test that silently never
runs is worth nothing, and a test that makes CI depend on a sibling being up is worth less
than nothing.

**Every request below is a GET.** Nothing here creates, changes or deletes anything in
miDataworks, and the one probe that uses a non-GET method aims it at an id that does not
exist, so it cannot remove anything. That is not fastidiousness - see
``TestWhyThePostLegIsNotMeasuredHere``, which is the measured reason the write half of task
8.6 is **not** taken.

## Identity: three surfaces, three credentials, none shared

``X-Dataworks-Agent: agent:miforge`` on every request, matching
``^agent:[a-z0-9][a-z0-9-]{0,47}$``. **No bearer token on this surface** - the shared MCP
token belongs to miDataworks' MCP server, and the ``trainer`` token is miLLM's. Asserted
below rather than assumed, because the one thing worse than the wrong credential here is
sending miLLM's.

## THE RECORDED FIGURE  (2026-10-16)

Taken from inside the ``miforge-backend`` pod on the pilot cluster through the **real**
``clients.midataworks.MiDataworksClient`` - the single outbound call site (ADR-010), with no
injected transport and no fake clock.

* **Service**: miDataworks, ``http://midataworks-backend.midataworks.svc.cluster.local:8000``,
  health ``{"status":"ok"}`` with postgres, redis, data_volume and millm all ``ok``.
* **Topology**: pod-to-pod across namespaces on one microk8s cluster, resolved through
  ``*.svc.cluster.local``. **Same-cluster east-west; this is NOT a WAN figure** and not a
  figure for an operator reaching miDataworks across a site link. ADR-024: a number without
  its topology is not a measurement.
* **Sample**: 20 measured calls per leg after 2 discarded warm-up calls per leg; the whole
  file run **twice**, minutes apart, and both runs are quoted below.
* **Live data, not a fixture**: the two exports that happened to exist, the larger carrying
  ``seed.jsonl`` at **909,841 bytes** and ``seed.row_keys.parquet`` at **94,321 bytes** -
  **1,004,162 bytes moved per full round trip.**
* **No human waiting is in any figure**, as task 8.6 requires. There is no decision latency
  here at all, because no human is asked for anything: the legs measured are transport,
  miDataworks' own query and its file service.

| leg (all read-only) | median | p95 | min / max |
|---|---|---|---|
| ``ping()`` -> ``GET /api/health`` | **14.0 ms** | 15.4 ms | 12.8 / 17.2 ms |
| ``list_exports(limit=50)`` | **2.0 ms** | 18.2 ms | 1.7 / 18.3 ms |
| ``get_export(id)`` | **1.4 ms** | 17.4 ms | 1.1 / 17.8 ms |
| ``fetch_export_file`` 94,321 B (parquet) | **1.8 ms** | 2.6 ms | 1.5 / 14.4 ms |
| ``fetch_export_file`` 909,841 B (jsonl) | **2.7 ms** | 2.8 ms | 2.5 / 3.4 ms |
| read-back leg, absent queue -> ``404`` | **1.6 ms** | 19.4 ms | 1.4 / 19.8 ms |
| **full round trip** (list, get, both files, hash) | **9.1 ms** | 26.2 ms | 7.7 / 26.5 ms |

Medians from a second run minutes later, in the same order: **15.4, 2.0, 2.1, 1.9, 3.3,
1.9, 9.6 ms** - every leg within ~0.7 ms of the first run. A third run, taken after the
file was finalised, agreed again: 15.6, 2.2, 1.5, 1.9, 2.9, 1.6, 8.8 ms.

**Three things in that table are worth more than the headline number.**

1. **The medians are stable and the p95s are not.** Every leg's median reproduces within
   ~0.7 ms across two runs, while several p95s sit at ten times their own median - the
   distribution is a tight body with an occasional ~20-30 ms outlier. At n=20 the p95 *is*
   effectively the maximum, so **it is a worst-of-twenty, not a percentile**, and quoting it
   as one would be the number-without-its-conditions failure ADR-024 names. The figure to
   plan against is the median; the outlier is real and unexplained and is recorded rather
   than smoothed away.
2. **The health probe is the slowest leg, by seven times, and that is correct.**
   ``/api/health`` is 14.0 ms against ``get_export``'s 1.4 ms because it is doing work -
   miDataworks checks postgres, redis, the data volume **and** miLLM behind it, returning
   ``{"status":"ok","dependencies":{...}}``. So this is **not** a transport figure, and
   nothing here should be read as "the network costs 14 ms". The transport floor on this
   link is the ~1.4 ms of a trivial GET.
3. **Per-request overhead dominates below about 100 kB.** The 94,321-byte file moves at
   **53.4 MB/s** observed and the 909,841-byte one at **341.9 MB/s** - a 6.4x throughput
   difference on the same link in the same run, entirely because the small transfer is
   mostly round trip. A part-import figure quoted without the file size is therefore
   meaningless, and the full-round-trip median of 9.1 ms for ~1 MB is **four sequential
   requests**, not a bandwidth measurement.

**The hashes verify on live data.** Each file's bytes were digested **as received**
(FR-010.14) and compared with the ``sha256`` miDataworks declares for it in the export
record. Both files matched, in both runs. That is the part of this measurement worth more
than the timings: it proves the live service and miForge agree about the bytes, which no
fake can establish - a fake's fixture agrees with itself by construction, which ADR-016
names as the usual reason a suite stays green over a real bug.

## The write half of task 8.6 is NOT measured, and that is a finding

8.6 asks for a **post + read-back** figure. The post leg was **not taken**, because on a
live miDataworks it cannot be cleaned up: ``DELETE /api/v1/review-queues/{id}`` answers
**405 METHOD_NOT_ALLOWED** and there is no PATCH or PUT either, while the queue it would
create lands in a human reviewer's work surface in state ``open``. Measured, not inferred,
and pinned as a tripwire - see ``TestWhyThePostLegIsNotMeasuredHere``, which says what it
would take to close the half and goes **red the day miDataworks adds a delete.**
"""

from __future__ import annotations

import hashlib
import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import pytest

BASE_URL = os.environ.get("MIDATAWORKS_BASE_URL", "").strip()

WARMUP_CALLS = 2
MEASURED_CALLS = 20

#: The agent name shape miDataworks requires on its REST surface. Its own pattern, copied
#: here only so this test can assert the header miForge actually sends conforms.
AGENT_PATTERN = r"^agent:[a-z0-9][a-z0-9-]{0,47}$"

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(
        not BASE_URL,
        reason=(
            "MIDATAWORKS_BASE_URL is unset, so miForge is standalone and there is no LAN "
            "miDataworks to measure. F10 task 8.6's figure cannot be taken against the "
            "in-process fake: that would be a number without its conditions, which ADR-024 "
            "names as the failure."
        ),
    ),
]


def _timed(call: Callable[[], Any]) -> tuple[list[float], Any]:
    """Warm up, then time :data:`MEASURED_CALLS` calls. Returns sorted ms and the last result."""
    result = None
    for _ in range(WARMUP_CALLS):
        result = call()
    samples: list[float] = []
    for _ in range(MEASURED_CALLS):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return samples, result


def _report(label: str, samples: list[float]) -> None:
    p95 = samples[min(len(samples) - 1, round(0.95 * (len(samples) - 1)))]
    print(
        f"  {label:<46} median {statistics.median(samples):7.1f}  p95 {p95:7.1f}  "
        f"min/max {samples[0]:7.1f} / {samples[-1]:7.1f}  ms  (n={len(samples)})"
    )


@pytest.fixture
def live_client():  # type: ignore[no-untyped-def]
    """The real client, no injected transport, no fake clock.

    Constructed exactly as a service constructs it, so the availability guard, the agent
    header and the retry policy are the production ones. ``capability`` is the real registry
    key for the read path this file exercises.
    """
    from clients.midataworks import MiDataworksClient

    client = MiDataworksClient(
        action="measuring the live round trip", capability="part_import_http"
    )
    try:
        yield client
    finally:
        client.close()


class TestTheIdentityOnThisSurface:
    """Three surfaces, three credentials, none shared - asserted, not assumed."""

    def test_the_agent_header_is_sent_and_no_bearer_token_is(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """``X-Dataworks-Agent: agent:miforge``, and **no** ``Authorization``.

        The failure this guards against is not a missing header - that refuses loudly with
        ``ORIGIN_APP_REQUIRED``. It is sending miLLM's ``trainer`` bearer token, or
        miDataworks' MCP token, to a surface that wants neither: a credential on the wire
        where none belongs is a disclosure, and it would never be noticed because the
        request would still succeed.
        """
        import re

        from clients.midataworks import AGENT_HEADER

        headers = live_client._client.headers
        assert headers[AGENT_HEADER] == "agent:miforge"
        assert re.match(
            AGENT_PATTERN, headers[AGENT_HEADER]
        ), "the agent name must match miDataworks' own pattern or every request is refused"
        assert "authorization" not in {key.lower() for key in headers}, (
            "this surface takes no bearer token: the shared MCP token is the MCP server's "
            "and the trainer token is miLLM's"
        )


class TestTheServiceAnswers:
    """The positive check. A latency figure from a service that is down is not a figure."""

    def test_health_is_ok_and_the_probe_is_timed(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """``ping()`` -> ``GET /api/health``, which is **outside** the version prefix.

        Mounted at ``/api``, not ``/api/v1``, for the reason miForge mounts its own there: a
        liveness probe that moves when the API version moves breaks a cluster on a day
        nobody was changing the cluster.
        """
        samples, reachable = _timed(live_client.ping)
        print("\n=== F10 task 8.6 - the LIVE miDataworks round trip (read-only) ===")
        print(f"  service  : {BASE_URL}")
        print("  topology : pod-to-pod across namespaces, one microk8s cluster")
        print("             SAME-CLUSTER EAST-WEST. This is NOT a WAN figure.")
        print(f"  sample   : {MEASURED_CALLS} measured, {WARMUP_CALLS} warm-up discarded, per leg")
        print("  human    : NO human waiting is in any figure - nothing here asks a person")
        _report("ping() -> GET /api/health", samples)
        assert reachable is True, "miDataworks did not answer its health probe"


class TestThePartImportRoundTrip:
    """The read half of task 8.6, measured leg by leg **and** end to end.

    This is F10's primary live path (FR-010.19, FR-010.14): list what miDataworks offers,
    read one export's record, fetch its files as bytes, and digest them **as received**.
    """

    def test_measure_each_leg_and_verify_the_hashes(self, live_client) -> None:  # type: ignore[no-untyped-def]
        exports_samples, exports = _timed(lambda: live_client.list_exports(limit=50))
        _report("list_exports(limit=50) -> GET /api/v1/exports", exports_samples)
        if not exports:
            pytest.skip(
                "the live miDataworks offers no exports, so there is no part to fetch. "
                "Stated rather than passed vacuously: a round-trip figure over zero bytes "
                "is not a figure."
            )

        # The largest export, so the byte leg is measured over something worth measuring.
        def total_bytes(record: dict[str, Any]) -> int:
            return sum(int(f.get("bytes") or 0) for f in (record.get("files") or []))

        export_id = max(exports, key=total_bytes)["id"]

        record_samples, record = _timed(lambda: live_client.get_export(export_id))
        _report("get_export(id) -> GET /api/v1/exports/{id}", record_samples)

        files = record.get("files") or []
        assert files, f"export {export_id} declares no files, so there is nothing to fetch"

        verified = 0
        for entry in sorted(files, key=lambda f: int(f.get("bytes") or 0)):
            # miDataworks names a file by `path`, which is what the route's `{name}`
            # segment takes. Read off the live record rather than spelled a second time:
            # every restated name in this project has eventually drifted.
            path = entry["path"]
            size = int(entry.get("bytes") or 0)
            samples, payload = _timed(lambda p=path: live_client.fetch_export_file(export_id, p))
            _report(f"fetch_export_file {size:,} B  {path}", samples)
            assert (
                len(payload) == size
            ), f"{path} arrived as {len(payload)} bytes against the {size} declared"
            declared = entry.get("sha256")
            if declared:
                digest = hashlib.sha256(payload).hexdigest()
                assert digest == declared, (
                    f"{path} hashes to {digest} as received, against the {declared} "
                    f"miDataworks declares. FR-010.14 requires the digest be taken on the "
                    f"bytes as received; a mismatch here is the live service and miForge "
                    f"disagreeing about the bytes, which no fake could have shown."
                )
                verified += 1
                if size:
                    median = statistics.median(samples)
                    print(
                        f"      hash verified as received; {size / median * 1000 / 1e6:.1f} "
                        f"MB/s observed at the application layer"
                    )
        assert verified, (
            "no file carried a declared sha256, so nothing was verified. The timings above "
            "are then transport only, and this test's correctness half did not happen."
        )

    def test_measure_the_round_trip_end_to_end(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """One figure for the whole thing, because an operator waits on the sum, not a leg."""

        def round_trip() -> int:
            exports = live_client.list_exports(limit=50)
            if not exports:
                return 0
            record = live_client.get_export(exports[0]["id"])
            read = 0
            for entry in record.get("files") or []:
                payload = live_client.fetch_export_file(record["id"], entry["path"])
                hashlib.sha256(payload).hexdigest()
                read += len(payload)
            return read

        samples, read = _timed(round_trip)
        _report("FULL part-import round trip (list->get->files->hash)", samples)
        print(f"      bytes moved per round trip: {read:,}")
        print("==================================================================")
        if read == 0:
            pytest.skip("no exports on the live service; nothing was moved")


class TestTheReviewReadBackLeg:
    """The read-back half of task 8.6's round trip, on the live service.

    The decisions route is miForge's read-back endpoint and takes **no** agent header -
    verified in miDataworks' own ``review.py``, whose docstring says the candidates POST
    needs the origin header and *"``GET /review-queues/{id}/decisions`` (read-back) does
    not."* Measured here against a queue id that does not exist, which is the only form of
    this leg that writes nothing: the figure is therefore miDataworks' lookup and refusal,
    not a decision read. **Stated plainly rather than quoted as a read-back number.**
    """

    def test_an_absent_queue_refuses_by_name_and_the_leg_is_timed(self, live_client) -> None:  # type: ignore[no-untyped-def]
        from clients.midataworks_errors import MiDataworksRefused

        def call() -> str:
            try:
                live_client.read_decisions("rq_measurement_absent_by_design")
            except Exception as exc:  # the type is asserted below
                return type(exc).__name__
            return "no-refusal"

        samples, outcome = _timed(call)
        _report("read-back leg, absent queue -> 404 QUEUE_NOT_FOUND", samples)
        assert outcome == MiDataworksRefused.__name__, (
            f"reading decisions for a queue that does not exist returned {outcome!r}. The "
            f"live service answers 404 QUEUE_NOT_FOUND; anything else means the client is "
            f"not surfacing miDataworks' own code."
        )


class TestThePostAndReadBackAgainstTheFake:
    """Task 8.6's **other** half: the same round trip against the in-process fake.

    8.6 asks for the figure *"against the fake and against a LAN miDataworks if
    available"*, and its own warning is that a fake figure alone *"would be a number without
    its conditions."* So it is taken here, **beside** the live read figures and never
    instead of them, with its conditions spelled out - which is what makes the pair
    informative rather than misleading.

    **What the fake's number is and is not.** It is an ``httpx.MockTransport``: no socket,
    no TCP, no server process, no database, no disk. So it measures miForge's own request
    building, serialisation and response parsing **and nothing else** - which is exactly why
    it is useful next to the live numbers. The difference between the two is the cost of
    everything the fake removes, and neither is quoted for the other.

    It lives in this ``upstream``-marked file rather than in CI because its whole value is
    the comparison, and the comparison needs the live service configured. A timing test in
    CI would be a flake; the round trip's *correctness* against the fake is already covered
    by ``tests/integration/test_review_queue_round_trip.py`` (task 7.13).

    **Measured 2026-10-16**, 20 reps after 2 warm-ups, batch of 100 candidates:

    | leg, against the fake | median | p95 |
    |---|---|---|
    | ``create_external_queue`` | **0.1 ms** | 0.2 ms |
    | ``post_candidates``, batch of 100 | **0.7 ms** | 0.7 ms |
    | ``read_decisions`` read-back | **0.1 ms** | 0.1 ms |
    | create + post + read-back | **0.9 ms** | 1.0 ms |

    The live service's floor for a *trivial* GET is ~1.4 ms, so the fake's whole three-leg
    round trip costs **less than one live request** - and the per-leg gap is roughly
    **fourteen times** on the two small legs. All of that difference is the conditions the
    fake removes: a socket, a server process, a PostgreSQL insert and a commit. **This is
    why 8.6 refused to accept the fake's figure as the answer.** Read together, the two
    tables are the honest statement of what the write half would cost: miForge's own share
    of a post is a few hundred microseconds even at a batch of 100, and everything else
    belongs to the service - which is the part that could not be measured here.
    """

    def test_measure_the_post_and_read_back_against_the_fake(self) -> None:
        """A write round trip that is safe to take, because nothing real is written."""
        from clients.midataworks import MiDataworksClient
        from tests.support.fake_midataworks import FakeMiDataworks

        batch = 100
        candidates = [
            {
                "external_id": f"measure-{index:04d}",
                "prompt": "What is the capital of France, and why is it the capital?",
                "completion": f"candidate {index}",
                "provenance": {"part": "prompt-set/measurement@1"},
            }
            for index in range(batch)
        ]

        fake = FakeMiDataworks()
        with MiDataworksClient(
            action="measuring the fake round trip",
            capability="human_review_selector",
            transport=fake.transport(),
        ) as client:

            def create() -> str:
                created = client.create_external_queue(
                    question="Is this completion good enough to train on?",
                    label_set=["good", "needs-rework"],
                    external_ref={"run_id": "measurement"},
                )
                return str(created["id"])

            create_samples, queue_id = _timed(create)
            post_samples, accepted = _timed(lambda: client.post_candidates(queue_id, candidates))
            # A decision a person would have made, recorded directly on the fake so **no
            # human waiting is in the figure** - task 8.6's explicit requirement.
            fake.decide(queue_id, "measure-0000", "accept", label="good")
            read_samples, decisions = _timed(lambda: client.read_decisions(queue_id))

            def whole() -> int:
                qid = create()
                client.post_candidates(qid, candidates)
                return len(client.read_decisions(qid))

            whole_samples, _ = _timed(whole)

        print("\n=== F10 task 8.6 - the SAME round trip against the in-process FAKE ===")
        print("  conditions: httpx.MockTransport - no socket, no server, no database, no disk")
        print(f"  batch     : {batch} candidates; no human waiting in any figure")
        _report("fake: create_external_queue", create_samples)
        _report(f"fake: post_candidates (batch of {batch})", post_samples)
        _report("fake: read_decisions (read-back)", read_samples)
        _report("fake: create + post + read-back", whole_samples)
        print("  NOTE: not comparable with the live figures above - it carries none of the")
        print("        conditions. The DIFFERENCE between the two is what the fake removes.")
        print("==================================================================")

        assert len(accepted) == batch, "the fake must acknowledge every candidate posted"
        assert len(decisions) == 1, "the read-back must return the one recorded decision"
        assert decisions[0].external_id == "measure-0000"


class TestWhyThePostLegIsNotMeasuredHere:
    """**Task 8.6's write half is NOT taken, and this is the measured reason.**

    8.6 asks for a *post + read-back* figure. The post leg is
    ``POST /api/v1/review-queues`` followed by
    ``POST /api/v1/review-queues/{id}/candidates``, and on a live miDataworks that is not a
    measurement with a cleanup - it is permanent, human-facing work:

    * miDataworks creates an external queue in state ``open`` with ``origin_app: miforge``
      (its ``queue_service.create_external``);
    * its **UI lists every queue unfiltered** - ``frontend/src/api/review.ts`` fetches
      ``/api/v1/review-queues?limit=200`` with no ``kind`` filter, and its ``DecisionBar``
      renders external queues for a human to decide;
    * **and there is no way to remove it.** Measured, not inferred: ``DELETE
      /api/v1/review-queues/{id}`` answers **405 METHOD_NOT_ALLOWED** and ``DELETE
      /api/v1/review-items/{id}`` answers **404**. No PATCH or PUT exists on the queue
      either, so it cannot even be closed.

    So the only honest post measurement would leave a reviewer a permanent queue of
    synthetic rows asking to be labelled, with no API to delete it. That is operator data a
    measurement has no business creating, so **it was not created.** A measured *"cannot be
    done and here is why"* closes this half honestly; an invented number does not.

    **What it would take to close it**, so the next session does not re-investigate: a
    delete or a close on miDataworks' review-queue surface - which is **miDataworks'
    decision, not miForge's** - or an operator explicitly accepting one permanent queue and
    saying so. The test below is the tripwire: **the day a DELETE appears, it goes red and
    says the post figure is now takeable.**
    """

    def test_midataworks_still_offers_no_way_to_delete_a_review_queue(self) -> None:
        """Aimed at an id that does not exist, so it cannot remove anything.

        405 means the path exists for other methods and has no DELETE handler, which is the
        statement being made; a 404 would mean no route matched at all.
        """
        import httpx

        response = httpx.request(
            "DELETE",
            f"{BASE_URL.rstrip('/')}/api/v1/review-queues/rq_absent_by_design",
            headers={"X-Dataworks-Agent": "agent:miforge"},
            timeout=15.0,
        )
        print(
            f"\n[cleanup] DELETE /api/v1/review-queues/{{id}} -> {response.status_code} "
            f"{response.json().get('error', {}).get('code')}"
        )
        assert response.status_code == 405, (
            f"DELETE on a review queue now answers {response.status_code}, not 405. If "
            f"miDataworks has added a delete, the post half of F10 task 8.6 is now "
            f"measurable with a cleanup: take the post + read-back figure and replace this "
            f"test with it."
        )

    def test_nothing_in_this_file_writes_to_the_live_service(self) -> None:
        """A static guard on this file's own honesty, asserted rather than promised.

        Every HTTP verb this module names must be a GET, except the DELETE above, which is
        aimed at an absent id. Written as a source check because the claim being made is
        about the whole file, and a reviewer reading one test cannot see it.
        """
        import pathlib
        import re

        source = pathlib.Path(__file__).read_text()
        # Strip docstrings and comments: the prose above names POST deliberately, and a
        # guard that cannot tell prose from code is the false measurement this project
        # keeps paying for. The positive check that the stripping did not strip
        # everything is the assertion on `markers` below.
        code = re.sub(r'""".*?"""', "", source, flags=re.S)
        code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
        # Assembled from fragments so this guard's own verb list is not what it finds -
        # the first run of it failed against itself, which is the self-matching shape of
        # the false-measurement family: the signal was indistinguishable from the defect.
        verbs = [v + "(" for v in ("po" "st", "p" "ut", "pa" "tch")]
        verbs += ['"' + v + '"' for v in ("PO" "ST", "P" "UT", "PA" "TCH")]
        offenders = [verb for verb in verbs if verb in code]
        assert not offenders, (
            f"this file issues {offenders} against a live miDataworks. Every leg here is "
            f"read-only by design; a write needs a cleanup, and miDataworks offers none."
        )
        # The needle is assembled at runtime, so this line's own source does not contain
        # it and cannot inflate the count.
        markers = code.count('"' + "DELE" "TE" + '"')
        assert markers == 1, (
            f"expected exactly one DELETE (the absent-id cleanup probe), found {markers}. "
            f"If this is 0 the docstring stripping removed the code it was meant to search, "
            f"and this guard proved nothing."
        )
