"""The ``attach`` acquisition mode  (R-02.1, R-02.47; FR-011.1 - FR-011.8, FR-011.47).

**A registration, not a branch.** F1's registry gains one entry and F1's code gains no
``if mode == "attach"`` - a test asserts the absence of that comparison anywhere under
``training/`` outside this package. §5.10 changes the *acquisition mode*, not the run type:
there is no fourth run type and F8's three are untouched.

## The attachment id is a credential and it is run state

``attach`` returns it **once** and no read path returns it again - not status, not health,
not a log line (miLLM Feature 031 FR-31.1.6). Losing it loses the ability to commit, and it
never appears in a log line, a step record, a run log, a manifest or an API response
(FR-011.6, ADR-020) - only a digest reference does.

**It must live on the run row, and today it cannot.** ``mf_runs`` has no ``attachment_id``
column: that DDL is owed and this feature does not write Alembic revisions, because the
migration graph is serialised through one owner (two revisions off one parent break
``alembic upgrade head`` for everyone). So :meth:`AttachAcquirer.acquire` **calls**
``run_state.remember`` when it is given a session - the seam is wired rather than declared,
which is the difference this project has paid for three times - and that call **refuses by
name** with the exact DDL until the columns exist. Without a session the id lives in the
client for the attachment's lifetime, which survives a step and not a worker restart. *That
is stated here rather than implied, because the previous wording of this paragraph said the
id "is persisted on the run row" and the code did not do it.*

## Pre-flight order, and why it is that order

availability -> reachability -> **build identity** -> already-attached / not-attachable /
stale -> ``weights_state``. Cheapest and most diagnostic first, and **all of it before a
lease and before any allocation** (ADR-008). The build check comes before the rebuild for a
reason stated in ``rebuild.py``: the handle format is a private contract between matching
builds.

## The base URL comes from configuration, never from a request body

Otherwise a run submission can point miForge's ``trainer`` credential at an arbitrary host.
``AcquisitionSpec`` carries no URL for the ``attach`` mode and this module does not read one
from anywhere else.

## Unavailable is reported by name, and never satisfied by a load

With the base URL unset the mode reports unavailable with the capabilities it would have
provided, and pre-flight refuses by name. **It never falls back to loading the same weights
from disk** (R-02.48): attaching and loading train *different tensors*, and both report
success.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.errors import refuse
from training.hotupdate import barrier, rebuild, state
from training.hotupdate import heartbeat as heartbeat_module
from training.hotupdate import identity as identity_module
from training.hotupdate import settings as settings_module
from training.hotupdate.client import WeightsClient
from training.hotupdate.timings import StepTimings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from training.acquisition.base import (
        AcquiredModel,
        AcquisitionMode,
        AcquisitionRecord,
        AcquisitionSpec,
        Availability,
    )

# NOTHING IN THIS PACKAGE IMPORTS `training.acquisition` AT MODULE SCOPE, and that is a
# property of the fence rather than a style choice. The dependency runs **one way at import
# time** - the registry reaches into this package, never the reverse - so this package stays
# importable on its own and the registry's import graph does not depend on it. A module-level
# import here would also be a genuine cycle: importing `training.acquisition.base` executes
# `training/acquisition/__init__.py`, which is the module that registers this one.

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Attachment:
    """What an attach produced. **The id is here and nowhere else in plaintext.**"""

    attachment_id: str
    resident_model: str
    manifest_sha256: str | None
    resident_identity: state.ResidentIdentity
    base: rebuild.RebuiltBase
    raw: dict[str, Any]

    @property
    def digest_ref(self) -> str | None:
        """The only form of the id that may be logged or correlated against."""
        return identity_module.digest_ref(self.attachment_id)


class AttachAcquirer:
    """F1's ``Acquirer`` protocol, for BRD-02 §5.10's mode."""

    #: F1's ``Acquirer`` protocol declares ``mode`` as a **property**, which is what lets
    #: this one resolve the enum lazily and keep the import one-way (see the note above).
    #: ``AcquisitionMode`` is a ``StrEnum``, so the registry keyed by it and a lookup by
    #: ``"attach"`` are the same key either way.
    @property
    def mode(self) -> AcquisitionMode:
        from training.acquisition.base import AcquisitionMode

        return AcquisitionMode.ATTACH

    def __init__(self, transport: Any = None) -> None:
        #: For tests: the fake weights surface is reached by pointing the **configured base
        #: URL** at a transport, which exercises the real client - its headers, its refusal
        #: mapping and its no-retry rule. Patching ``httpx`` would bypass all three.
        self._transport = transport
        #: The live attachment, once acquired. Held so the heartbeat, the bracket and the
        #: recovery path all speak to one client with one connection.
        self.attachment: Attachment | None = None
        self.client: WeightsClient | None = None
        #: The live heartbeat, started on attach and stopped on detach. Held here because
        #: the attachment's liveness is the acquirer's responsibility for as long as it
        #: holds the attachment - see `acquire`.
        self.heartbeat: Any = None
        #: The step the loop takes for this acquisition (R-02.52), built on attach.
        self.stepper: Any = None
        #: R-02.2's check, run by the loop before step 1, built on attach.
        self.verifier: Any = None
        #: Writes the attachment's observed state into the run record (set by bind_run).
        self.publish_status: Any = None

    # ------------------------------------------------------------------ availability

    def available(self) -> Availability:
        """**No network, no GPU.** Module present, base URL set - that is the whole answer.

        F1's ``GET /api/v1/models/sources`` calls this with no run in hand and must answer
        instantly. See ``settings.py`` on why the reason stays
        ``hot_update_mode_not_available`` rather than becoming ``millm_not_configured``.
        """
        from training.acquisition.base import Availability, ml_runtime_availability

        cfg = settings_module.load()
        if not cfg.configured:
            return Availability(
                available=False,
                reason=settings_module.NOT_AVAILABLE_REASON,
                message=settings_module.unavailable_message(settings_module.CAUSE_NOT_CONFIGURED),
                unavailable_capabilities=settings_module.UNAVAILABLE_CAPABILITIES,
            )
        if not cfg.trainer_token or not cfg.audit_label:
            return Availability(
                available=False,
                reason=settings_module.NOT_AVAILABLE_REASON,
                message=settings_module.unavailable_message(
                    settings_module.CAUSE_IDENTITY_INCOMPLETE
                ),
                unavailable_capabilities=settings_module.UNAVAILABLE_CAPABILITIES,
            )
        # A mode is usable only if it can also acquire, and acquiring rebuilds tensors.
        # F1's own helper answers that, so this does not re-decide what "the ML runtime is
        # present" means.
        return ml_runtime_availability(*settings_module.UNAVAILABLE_CAPABILITIES)

    # ------------------------------------------------------------------ pre-flight

    def preflight(self, spec: AcquisitionSpec) -> AcquisitionSpec:
        """Every refusal, before the lease and before any allocation.

        Raises:
            Refusal: ``acquisition_mode_unavailable`` when the mode is not usable - **the
                same code F1's unavailable stub raises**, because ``GET /models/sources``
                and ``POST /runs`` must answer the same way whether this package is present
                or absent, with the *cause* in the details. Then whatever
                ``refusals.interpret`` names for miLLM's own refusals, and
                ``weights_not_clean`` from the state read.
        """
        self._refuse_if_unavailable()
        client = self._open_client()
        # Reachability, then the state read. Both are `status` (BRD-01 R-01.20), so one call
        # answers "is there a weights surface at all" and "what state is it in" - and a
        # surface that answers /v1 but has no /api/weights/* refuses here by name, which is
        # the "specced, not implemented" case this feature spends its life in.
        body = state.read_status(client)
        weights_state, resident = state.interpret_status(body)

        # ── THE BARRIER MUST BE ABLE TO PAUSE, AND THAT IS KNOWABLE BEFORE ANY CALL ──────
        # miLLM reports `pause_unavailable` on the same status body (no token, no slot),
        # computed by the function its `409 BARRIER_UNAVAILABLE` refusal raises from. Read
        # here rather than discovered on a refused `quiesce`, because the condition is a
        # DEPLOYMENT SETTING - continuous batching, or more than one request slot - so a
        # retry cannot help and a mid-run stop is the wrong place to learn it. Refusing at
        # pre-flight means the operator is told before a lease and before any allocation,
        # which is the same argument the `weights_not_clean` refusal below already makes.
        #
        # Taken from the body already fetched: a second status call could straddle a
        # deployment change and describe a different moment (see `read_status`).
        cannot_pause = state.pause_unavailable(body)
        if cannot_pause is not None:
            raise refuse(
                "weights_barrier_unavailable",
                f"miLLM cannot pause serving for a weight barrier ({cannot_pause!r}), so a "
                f"quiesce, a weights load and an adapter change would all be refused. This "
                f"is a deployment setting on the serving side - continuous batching, or "
                f"more than one request slot - so RETRYING WILL NOT HELP: the serving "
                f"deployment must be changed, or this run must use an acquisition mode that "
                f"does not attach. Refused at pre-flight, before a lease and before any "
                f"allocation.",
                reason=cannot_pause,
                weights_state=weights_state,
            )

        if weights_state != state.CLEAN:
            raise refuse(
                "weights_not_clean",
                f"miLLM's parameter set is {weights_state!r}, so this run will not attach "
                f"and step against it. If a commit was interrupted, recovery is R-02.30's: "
                f"the process that opened it completes it, or a checkpoint load replaces "
                f"the whole set. Refusing at pre-flight, before a lease and before any "
                f"allocation.",
                weights_state=weights_state,
                resident_model=resident.model,
            )
        log.info(
            "attach pre-flight passed",
            extra={
                "code": "attach_preflight_ok",
                "resident_model": resident.model,
                "manifest_sha256": resident.manifest_sha256,
                "weights_state": weights_state,
            },
        )
        return spec

    # ------------------------------------------------------------------ acquire

    def acquire(
        self,
        spec: AcquisitionSpec,
        *,
        session: Any = None,
        run_id: str | None = None,
        on_identity_rejected: Any = None,
    ) -> AcquiredModel:
        """Attach, rebuild, and fill F1's acquisition record. **Adds no field to it.**

        The forward pass that verifies the reconstruction (R-02.2) is **not** here: it needs
        the serving process's own answer, so it is the caller's next step and is run before
        the first training step, before any test and before any save.

        Args:
            spec: F1's acquisition spec. Carries no base URL for this mode - that comes from
                configuration only, so a run submission cannot point miForge's credential at
                an arbitrary host.
            session: a database session, for persisting the attachment id as run state
                (FR-011.5). **Optional only because the columns it needs do not exist yet**:
                given one, ``run_state.remember`` is called and refuses by name with the owed
                DDL. Not given one, the id lives in the client for the attachment's lifetime,
                which is enough for a step and not enough for a worker restart.
            run_id: the run the id belongs to. Required with ``session``.
        """
        # THE SAME REFUSAL AS PRE-FLIGHT, and `acquire` makes it again rather than trusting
        # that pre-flight ran. R-02.48's prohibition is on *this* method: "never quietly
        # satisfied by loading the same model from disk". F1's own test calls `acquire`
        # directly on an unavailable mode and requires `acquisition_mode_unavailable` - and
        # it is right to, because the client's own `millm_not_configured` would be a
        # different code for the same situation depending on which method was called.
        from training.acquisition.base import AcquiredModel

        self._refuse_if_unavailable()
        client = self._open_client()
        response = client.call("attach")
        attachment_id = response.get("attachment_id")
        if not isinstance(attachment_id, str) or not attachment_id:
            raise refuse(
                "attach_response_incomplete",
                "miLLM's attach response carries no attachment id. It is returned once and "
                "no read path returns it again, so without it this run could never commit - "
                "refusing now rather than discovering it one step later as a refused "
                "commit.",
                op="attach",
            )
        # BEFORE any other call: the id is the capability, and the caller persists it as run
        # state. A request-scoped variable loses it on the first worker restart and the
        # symptom appears a step later.
        client.remember_attachment(attachment_id)

        if session is not None:
            # **BEFORE ANY OTHER CALL** (FR-011.5), and that ordering is the requirement
            # rather than tidiness: the rebuild below is the expensive part and the one that
            # needs a card, so persisting afterwards would leave a live attachment whose id
            # exists only in this process for the whole of it. It is also WIRED rather than
            # declared - with the columns absent this refuses, naming them and the DDL,
            # which is louder than quietly holding a capability that cannot survive the
            # restart it exists for. *Found by the test for this call: placed after the
            # rebuild, it was unreachable on a machine without the ML wheels, which is
            # every machine in CI.*
            from training.hotupdate import run_state

            run_state.remember(
                session,
                run_id or "",
                attachment_id=attachment_id,
                base_url=settings_module.load().base_url,
            )

        # ── THE HEARTBEAT STARTS HERE, AND THE PLACEMENT IS THE REQUIREMENT ──────────
        # **Nothing started it before this, and that was not a gap in reachability but a
        # mode that could not work.** `interval_from` had no production caller, this class
        # was never constructed outside its own tests, and
        # `hotupdate_heartbeat_interval_seconds` was configured, plumbed into F11's settings
        # and read by nobody. So a §5.10 run attached and never beat: miLLM's as-built
        # timeout is 30 s, R-01.4 marks the attachment stale, and the run dies **looking
        # like a crash** - the exact wrong diagnosis `beat_once` already warns about for the
        # credential case.
        #
        # **Before `rebuild.build`, not after.** The rebuild is the expensive step - a
        # multi-gigabyte parameter set reconstructed from CUDA IPC handles - and it can
        # plausibly take longer than the staleness window on its own. Starting the heartbeat
        # after it would mean the attachment could go stale *during the very operation the
        # attachment exists for*, and the first symptom would be a refused commit much
        # later. `run_state.remember` above is first for its own reason (FR-011.5), so the
        # order is: persist the id, prove the attachment live, then do the slow work.
        interval = heartbeat_module.interval_from(
            response, override=settings_module.load().heartbeat_interval_seconds
        )
        self.heartbeat = heartbeat_module.Heartbeat(
            client=client,
            interval_seconds=interval,
            # Passed through rather than defaulted: stopping the RUN is the caller's
            # decision (this class's docstring says so), and the acquirer does not own the
            # run. With nothing supplied a rejected credential still stops the heartbeat and
            # logs `heartbeat_identity_rejected` loudly; it does not stop the run, and
            # wiring that is owed by whatever drives a §5.10 run.
            on_identity_rejected=on_identity_rejected,
        )
        self.heartbeat.start()

        rebuild.check_build_identity(response)
        # requires_grad on miForge's side, always, for the mode F1 asks for: a `change` run
        # updates every parameter. The READ-ONLY base an adapter run needs comes through
        # `adapter_read.attach_base`, which calls `rebuild.build` itself with the opposite
        # pair - NOT through a flag read off the spec. `AcquisitionSpec` is frozen and
        # forbids extra fields, so a `getattr(spec, "read_only", False)` here would be a
        # default that always fires, which is the fabricated value ADR-018 forbids.
        base = rebuild.build(response, read_only=False, requires_grad=True)
        module = build_module(response, base)

        # THE ATTACH RESPONSE'S IDENTITY IS `model: {model_id, name, manifest_hash}` (miLLM
        # FTDD 031 §5.3), read from their schema and confirmed against their implementation's
        # `AttachResponse`/`AttachedModel`. This read `resident_model` with a fallback to
        # `model` — and `model` IS the object, so `str(...)` produced a **stringified Python
        # dict**, while `manifest_sha256` is not a top-level key at all and was always None.
        #
        # **This was the worse half of the field-name defect, and it is not merely cosmetic.**
        # `state.read` takes its identity from the STATUS response, where the same block is
        # spelled `resident_model` and is also an object. Once that side was corrected, the
        # two sides produced different strings for the same model — so `differs_from` would
        # report `model` as changed and the first `require_clean` of every run would refuse
        # `weights_identity_changed`, diagnosing "the serving process restarted" for a
        # process that had not. A §5.10 run could not have taken one step.
        # ONE derivation, shared with `state.read` (which reads the same block under its
        # status-response name). Not re-derived here: two readings of one value is what
        # produced the disagreement described above.
        identity_of_set = state.identity_from_attach(response)
        resident_model = identity_of_set.model
        self.attachment = Attachment(
            attachment_id=attachment_id,
            resident_model=resident_model,
            manifest_sha256=identity_of_set.manifest_sha256,
            resident_identity=identity_of_set,
            base=base,
            raw=response,
        )
        record = acquisition_record(
            resident_model=resident_model,
            manifest_sha256=identity_of_set.manifest_sha256,
            module=module,
            base_url=settings_module.load().base_url,
        )
        log_attachment(self.attachment, base_url=settings_module.load().base_url)
        # THE SAVE PATH (OQ-5 / AMB-M, settled 2026-10-14: F11 owns the attached-mode save).
        # Supplied as a value on the acquisition, so F1's loop dispatches on its presence
        # and holds no `mode == "attach"` test - ADR-022's fence by construction. It carries
        # its own mutable baseline because `Attachment` is frozen and a successful save can
        # rename the resident model; see `save.py` on why that is re-read, not assumed.
        #
        # Imported HERE rather than at module scope, for the same reason `AcquisitionMode`
        # is resolved lazily above: `save` reaches F5's `services.checkpoints`, whose
        # `manifest` module imports `training.acquisition.base`, whose package __init__
        # installs THIS acquirer. A module-scope import closes that cycle and the error is
        # `cannot import name 'AttachAcquirer' from partially initialized module` - i.e. the
        # fence's own registration hook is what the cycle runs through. One-way imports are
        # the property being kept, and this keeps them.
        from training.hotupdate import save as save_module

        self.saver = save_module.AttachedSaver(client, expected=identity_of_set)
        # THE STEP (R-02.52), supplied the same way as the save: a value the loop dispatches
        # on by presence. Without it the loop steps through the run type's plain `run_one`
        # and writes the resident tensors with no bracket - which is what it did until
        # 2026-10-16. The baseline is read from the saver at call time (see `AttachedStepper`).
        live = settings_module.load()
        self.stepper = barrier.AttachedStepper(
            client,
            StepTimings(
                watchdog_ms=live.commit_watchdog_ms,
                creep_fraction=live.commit_creep_fraction,
                run_id=run_id,
            ),
            expected=identity_of_set,
            saver=self.saver,
            precommit_retries=live.precommit_retries,
        )
        return AcquiredModel(
            module=module,
            tokenizer=None,
            record=record,
            device_uuid=base.device_uuid,
            saver=self.saver,
            stepper=self.stepper,
            verifier=self._verifier(module, resident_model, live),
            bind_run=self.bind_run,
            # Detach at run end (heartbeat stopped first, refused while not `clean`). Until
            # 2026-10-16 nothing in the loop called `detach`, so the heartbeat kept the
            # attachment alive in the worker after the run - holding miLLM's single slot.
            release=self.detach,
        )

    # ------------------------------------------------------------------ the run record

    def bind_run(self, *, run_id: str, session_scope: Any = None) -> None:
        """Tie the attachment to its run: persist it, and publish its live state.

        ``acquire`` is reached through ``acquire_for_run(spec)``, which carries neither a
        run id nor a session - so ``run_state.remember`` (FR-011.5) had **no production
        caller** until this hook, and neither did any state the UI could read. Each write is
        its own short transaction through the run's ``session_scope``; without one (a test
        driving the loop directly) nothing is persisted and the chip reads "not reported".
        """
        from training.hotupdate import run_state

        if self.stepper is not None:
            self.stepper.timings.run_id = run_id
        if session_scope is None or self.client is None:
            return

        def publish(body: Any) -> None:
            record = run_state.status_record(
                body, holds_attachment=self.client is not None and bool(self.client.attachment_id)
            )
            with session_scope() as session:
                run_state.record_status(session, run_id, record)

        self.publish_status = publish
        if self.stepper is not None:
            self.stepper.publish_status = publish
        with session_scope() as session:
            run_state.remember(
                session,
                run_id,
                attachment_id=str(self.client.attachment_id),
                base_url=settings_module.load().base_url,
            )
        publish(state.read_status(self.client))

    # ------------------------------------------------------------------ verification

    def _verify_provider(self, resident_model: str | None, live: Any) -> Any:
        """R-02.49's provider pointed at the serving process (FR-011.10: no private path).

        The OpenAI-compatible ``/v1`` surface at the configured base URL, naming the
        resident model. A seam so a test can supply a provider over a stub transport.
        """
        from providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            base_url=str(live.base_url),
            model=str(resident_model or ""),
            connect_timeout_s=float(live.request_timeout_seconds),
            read_timeout_s=float(live.request_timeout_seconds),
        )

    def _tokenizer_loader(self, resident_model: str | None, live: Any) -> Any:
        """The SERVING model's tokenizer, loaded when the check runs (a seam for tests)."""
        source = live.verify_tokenizer or resident_model

        def load() -> Any:
            if not source:
                raise refuse(
                    "verification_tokenizer_unknown",
                    "R-02.2's check needs the serving model's tokenizer, and neither "
                    "HOTUPDATE_VERIFY_TOKENIZER nor the attach response names one.",
                )
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                str(source), trust_remote_code=False
            )

        return load

    def _verifier(self, module: Any, resident_model: str | None, live: Any) -> Any:
        from training.hotupdate import verify

        self.verifier = verify.Verifier(
            module=module,
            tokenizer_loader=self._tokenizer_loader(resident_model, live),
            provider=self._verify_provider(resident_model, live),
            prompt=live.verify_prompt,
            token_set_size=live.verify_token_set_size,
            tolerance=live.verification_tolerance,
            tolerance_is_provisional=live.tolerance_is_provisional,
        )
        return self.verifier

    # ------------------------------------------------------------------ detach

    def detach(self, heartbeat: Any = None) -> None:
        """Clean exit: stop the heartbeat, then ``detach``. **Never while ``writing``.**

        The ordering of a *step* is ``commit-end`` -> ``resume`` -> ``detach``, and the
        bracket guarantees the first two on every path out of itself, including an
        exception. This refuses to detach while the set is not ``clean`` because **an
        unclean exit converts an orderly shutdown into a serving outage**: a detach during a
        write marks the set ``torn`` at once on the other side, and ``torn`` survives detach.
        """
        if self.client is None:
            return
        # Defaults to the one `acquire` started. It stayed a bare parameter while nothing
        # constructed a heartbeat at all, so `detach()` with no argument silently stopped
        # nothing - the other half of the same defect.
        beating = heartbeat if heartbeat is not None else self.heartbeat
        if beating is not None:
            beating.stop()
            if beating is self.heartbeat:
                self.heartbeat = None
        weights_state, _ = state.read(self.client)
        if weights_state != state.CLEAN:
            raise refuse(
                "detach_refused_set_not_clean",
                f"miForge will not detach while miLLM's parameter set is {weights_state!r}: "
                f"a detach during a write leaves the set torn, and torn survives detach - "
                f"so an orderly shutdown would become a serving outage inherited by the "
                f"next process to attach. Complete or roll back the commit first (R-02.30).",
                weights_state=weights_state,
            )
        self.client.call("detach")
        if self.publish_status is not None:
            # The record says what is true now: nothing is attached for this run.
            try:
                self.publish_status(state.read_status(self.client))
            except Exception as exc:
                log.warning(
                    "could not record the detached state",
                    extra={"code": "hotupdate_status_write_failed", "error": str(exc)},
                )
        self.client.close()
        self.client = None
        self.attachment = None

    # ------------------------------------------------------------------ internals

    def _refuse_if_unavailable(self) -> None:
        """Refuse an unusable mode with **F1's code**, carrying the cause in the details."""
        state_of_mode = self.available()
        if state_of_mode.available:
            return
        cause = (
            settings_module.CAUSE_NOT_CONFIGURED
            if not settings_module.load().configured
            else settings_module.CAUSE_IDENTITY_INCOMPLETE
        )
        raise refuse(
            "acquisition_mode_unavailable",
            state_of_mode.message
            or "The 'attach' acquisition mode is not available in this build.",
            mode=self.mode.value,
            reason=state_of_mode.reason,
            # The machine-readable cause, which the mode-level `reason` cannot carry: F1's
            # surface pins one `reason` per mode and forbids a `*_not_configured` one there.
            cause=cause,
            capabilities_unavailable=list(state_of_mode.unavailable_capabilities),
        )

    def _open_client(self) -> WeightsClient:
        """One client, one connection, for the attachment's lifetime."""
        if self.client is None:
            self.client = WeightsClient(settings=settings_module.load(), transport=self._transport)
        return self.client


def build_module(response: Any, base: rebuild.RebuiltBase) -> Any:
    """The model graph in miForge's process, with the rebuilt tensors as its parameters.

    **Instantiated on the meta device and then assigned**, so nothing allocates a second
    copy of the parameter set on a card that is already holding one: ``from_config`` under
    ``torch.device("meta")`` builds the structure with no storage, and ``load_state_dict(…,
    assign=True)`` makes the *attached* tensors the module's parameters rather than copying
    into fresh ones. A copy here would double 2.4 GB on the one card the serving process is
    also using, and - worse - would train a copy while miLLM served the original.

    ``trust_remote_code`` is off, as everywhere in this project: an architecture that ships
    its own Python would otherwise run arbitrary code inside the worker that holds the card.

    Raises:
        Refusal: ``attach_response_incomplete`` without an architecture config;
            ``weights_rebuild_failed`` when the structure and the handles disagree - naming
            the parameters that did not line up, never proceeding with a partial graph.
    """
    config = response.get("config") if isinstance(response, dict) else None
    if not isinstance(config, dict):
        raise refuse(
            "attach_response_incomplete",
            "miLLM's attach response carries no architecture config, so the model graph "
            "cannot be rebuilt in this process. Refusing rather than guessing an "
            "architecture from the parameter names: a guessed graph is a different model "
            "that trains successfully.",
            op="attach",
        )
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise refuse(
            "attach_response_incomplete",
            "miLLM's architecture config names no model_type, so the graph cannot be built "
            "from it. Refusing rather than inferring one from the parameter names.",
            op="attach",
        )
    parsed = AutoConfig.for_model(
        model_type, **{key: value for key, value in config.items() if key != "model_type"}
    )
    with torch.device("meta"):
        module = AutoModelForCausalLM.from_config(  # type: ignore[no-untyped-call]
            parsed, trust_remote_code=False
        )
    missing, unexpected = module.load_state_dict(base.tensors, strict=False, assign=True)
    if missing or unexpected:
        raise refuse(
            "weights_rebuild_failed",
            f"The architecture miLLM named and the handles it returned do not line up: "
            f"{len(missing)} parameters of the graph have no handle and {len(unexpected)} "
            f"handles belong to no parameter. Refusing rather than training the "
            f"intersection - a partial graph differs from the served model in exactly the "
            f"tensors that are missing, and reports success.",
            missing=[str(name) for name in missing][:20],
            unexpected=[str(name) for name in unexpected][:20],
        )
    return module


def acquisition_record(
    *,
    resident_model: str,
    manifest_sha256: str | None,
    module: Any,
    base_url: str | None,
) -> AcquisitionRecord:
    """Fill **F1's** record for this mode. No new field, and no new record (FR-011.8).

    ``starting_weights_sha256`` (R-02.47) is the resident checkpoint's manifest hash when
    miLLM states one. When it states ``null`` - *"not loaded from a checkpoint manifest"*,
    which R-01.1 returns for an instantiated resident model - the content hash is computed
    over the attached parameters through **F8's existing digest function**, because that is
    the one definition of "a hash over this module's base parameters" in the codebase and a
    second one would be a second answer. The distinction is carried, never flattened into
    "unknown".
    """
    from training.acquisition.base import (
        AcquisitionMode,
        AcquisitionRecord,
        SourceKind,
        lineage_for_first_load,
        lineage_for_loaded_checkpoint,
    )

    if manifest_sha256 is not None:
        starting = manifest_sha256
    else:
        from training.runtypes.adapter import base_parameter_digest

        starting = base_parameter_digest(module)
    parsed = _parse_checkpoint_name(resident_model)
    if parsed is not None:
        base_model, parent = lineage_for_loaded_checkpoint(
            checkpoint_name=resident_model, checkpoint_base_model=parsed
        )
    else:
        base_model, parent = lineage_for_first_load(model_identity=resident_model)
    return AcquisitionRecord(
        mode=AcquisitionMode.ATTACH,
        source_kind=SourceKind.ATTACHED,
        # The serving process this run attached to. A location, not a credential - and the
        # one field of this record that is §5.10-specific, which is why it reuses
        # `source_path` rather than adding a field F1 would then have to carry.
        source_path=base_url,
        starting_weights_sha256=starting,
        base_model=base_model,
        parent=parent,
    )


def _parse_checkpoint_name(name: str) -> str | None:
    """The base-model part when ``name`` is a miForge checkpoint's qualified name.

    **This is the lineage carry-forward, and getting it wrong is invisible.** If the resident
    model *is* a checkpoint, then it is not the root: setting ``base_model`` to it would
    terminate every lineage walk one hop short of the real root, on a tree that validates,
    saves and looks plausible. F1's ``lineage_for_loaded_checkpoint`` is the carry-forward
    and F5's path parser is the one place that knows the name's shape.
    """
    from services.checkpoints.paths import parse_qualified_name

    parsed = parse_qualified_name(name)
    return None if parsed is None else parsed[0]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def log_attachment(attachment: Attachment, *, base_url: str | None) -> None:
    """The one log line that mentions an attachment. **A named function on purpose.**

    It is extracted from :meth:`AttachAcquirer.acquire` so that the secret-hygiene test can
    drive *this line* without a GPU: ``acquire`` rebuilds tensors, so on a machine without
    the ML wheels the log call is unreachable - and a mutation that logged the id itself
    therefore **survived** the first sweep. The test found the gap in the test, not in the
    code, which is the usual way round for this class.

    **MUT-13 is the control**: replace ``attachment.digest_ref`` with the id and
    ``test_hotupdate_secret_hygiene.py`` must go red.
    """
    log.info(
        "attached to a serving process's resident tensors",
        extra={
            "code": "weights_attached",
            "resident_model": attachment.resident_model,
            # The manifest hash, or an explicit null meaning "not loaded from a checkpoint
            # manifest" - never "unknown".
            "manifest_sha256": attachment.manifest_sha256,
            "base_url": base_url,
            # NEVER the id itself (ADR-020, FR-011.6).
            "attachment_ref": attachment.digest_ref,
            "parameters": len(attachment.base.tensors),
        },
    )
