"""The attached-mode save: F11 decides and brackets, miLLM writes  (FR-011.26, FR-011.56).

**OQ-5 / AMB-M, SETTLED 2026-10-14 (operator): F11 owns the attached-mode save.** The
question was genuine - R-02.20's closing sentence puts the save on the serving side
(*"In §5.10 the resident tensors are miLLM's, so the save is miLLM's too and R-01.10-13
govern as before"*), R-02.20 is **F5's** requirement, and F5's PRD states it models nothing
of §5.10. So the *artifact* is miLLM's and the *requirement* is F5's, and nothing owned the
decision to take one.

**ADR-022 decides it.** An attached-mode branch inside F5's save service would put a miLLM
endpoint reference on the core path and turn the standalone control red - the fence's whole
purpose. So the decision and the protocol live here, and **this module writes no checkpoint
bytes, no manifest and no `latest` pointer.** What it owns is: refuse unless the set is
`clean`, send the trainer's fields through the one client with the holder header, re-read
the identity afterwards, and index what was published through **F5's own** indexer.

## What it owns, in order, and why that order

1. **`require_clean` locally** (FR-011.56, R-01.11). *A save is refused locally rather than
   relying on the other side's refusal* - because the other side's refusal is a `409` that
   arrives after the request, and because this is the **third of `require_clean`'s three
   call sites** (FR-011.36). Until this module existed, two of the three were wired and the
   test for the third performed the sequence itself, so the gate function was protected
   while *the wiring to a save was not*.
2. **The caller's manifest fields only** - derived from the schema, see below.
3. **One `save` call through `client.call`** (FR-011.26): same client, same credential, same
   header discipline. `save` is holder-only (R-01.31), and the client is the only module
   that knows that.
4. **Re-read the identity** (see "the rename", below).
5. **Index through F5's `upsert_from_manifest`** when the published directory is readable
   from this process, and say so by name when it is not.

## The caller's fields are DERIVED from the schema, never listed here

R-01.12: *"Each field is marked as written by the caller or written by miLLM."* The marking
is in `checkpoint-manifest-v1.json`'s own property descriptions - **the normative file** -
so :func:`caller_manifest` reads them rather than restating a list. A list written here
would be the twelfth instance of this project's signature defect: miLLM rejects a save
carrying a field it writes itself, so a stale list is a refused save on every step.

The four miLLM writes in this mode are `base_model`, `created_at`, `name` and
`weights_sha256`. That set is *computed*, not typed, and ``test_hotupdate_save.py`` asserts
the computation against the schema - so a schema change moves the body with it.

## The rename, and why the identity is RE-READ rather than assumed

A save that succeeds **can change the resident model's name**: miLLM renames the resident
model to the checkpoint it just wrote, so `status.resident_model.name` - which is exactly
:class:`state.ResidentIdentity`'s `model` - is a different string afterwards. The run's
baseline identity is therefore stale the moment a save succeeds, and the **next**
`require_clean` with the old baseline refuses `weights_identity_changed`, blaming a serving
restart that never happened. On a run saving every N steps that is: first checkpoint fine,
second save refused, run dead with a diagnosis pointing at the wrong system. *The same
shape as the four field names F11 read that the upstream does not send, and as the attach/
status identity disagreement - both found by reading the other side rather than our own.*

**This module does not encode the rename either way.** It re-reads the identity after the
save and adopts what it reads, which is correct whether or not the other side renames, and
needs no claim about an upstream behaviour that **published BRD-01 does not state** (it is
in miLLM's own manual, on a branch). `require_clean`'s docstring already gives the rule:
*"a cached answer about another process's memory is a guess with a timestamp."*

**Still owed, with the owner named:** ``barrier.hot_update_step`` takes ``expected_identity``
as an argument and nothing in ``src/`` passes it yet, so the step's baseline has the same
staleness and no production path to fix. When the §5.10 loop is wired, it must take its
baseline from :attr:`AttachedSaver.expected` rather than from the frozen
``Attachment.resident_identity``, which is a snapshot of the attach.

## What it deliberately does NOT do

* **No Socket.IO event.** F5 declares ``CHECKPOINT_SAVED_EVENT`` and builds its payload
  from a ``SaveOutcome`` this module cannot construct - there is no local model, and with
  the directory unreadable there is no manifest either. Emitting a *second shape* under one
  event name would leave F9's store reading fields that are sometimes absent, which is worse
  than a missing event. **Owed by F6**, whose loop owns the run's event vocabulary.
* **No `latest` pointer, no atomic rename, no hash.** R-01.13 is miLLM's.
* **No retry.** `save` is in ``refusals.RETRY_UNSAFE_OPS``, and a second attempt against a
  published checkpoint is ``CHECKPOINT_EXISTS``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.checkpoints import index as checkpoint_index
from services.checkpoints import paths as checkpoint_paths
from services.checkpoints.manifest import MANIFEST_NAME, ManifestFields
from training.hotupdate import state

log = logging.getLogger("miforge.hotupdate.save")

__all__ = ["AttachedSaveOutcome", "AttachedSaver", "caller_manifest", "serving_written_fields"]

#: The marker the schema uses on a field whose writer depends on the mode. Read from the
#: property descriptions rather than hard-coded as a field list - see the module docstring.
#: It is a *substring* of the schema's own sentence, and ``test_hotupdate_save.py`` asserts
#: that it still matches something, so a reworded schema fails loudly instead of silently
#: classifying every field as the caller's and sending four miLLM rejects.
_SERVING_WRITER_MARKER = "Written by whichever process writes the checkpoint"

#: Fields with no "Written by" sentence at all, because they are schema constants. miLLM
#: accepts them **only** with their constant values, so sending them buys nothing and risks
#: a mismatch; they are the schema's, not the trainer's.
_CONSTANTS = frozenset({"kind", "schema_version"})


def serving_written_fields() -> frozenset[str]:
    """The manifest fields **miLLM** writes in attached mode, read out of the schema.

    Raises:
        Refusal: ``manifest_writer_marking_unreadable`` when the marker matches no property
            at all. An empty result would classify every field as the caller's and produce a
            save refused upstream on every step for a reason naming none of this - so the
            unreadable case is refused here rather than returned as an empty set.
    """
    from core.errors import refuse

    properties = _root_properties()
    marked = frozenset(
        name
        for name, prop in properties.items()
        if _SERVING_WRITER_MARKER in str(prop.get("description") or "")
    )
    if not marked:
        raise refuse(
            "manifest_writer_marking_unreadable",
            f"checkpoint-manifest-v1.json no longer marks any field with "
            f"{_SERVING_WRITER_MARKER!r}, so which fields miLLM writes in attached mode "
            f"cannot be derived from the normative schema. Refusing rather than sending "
            f"every field: miLLM rejects a save carrying a field it writes itself "
            f"(MANIFEST_FIELD_NOT_CALLERS), so the guess would fail every save with a "
            f"message naming none of this.",
            marker=_SERVING_WRITER_MARKER,
            properties=sorted(properties),
        )
    return marked


def _root_properties() -> dict[str, Any]:
    """The manifest's top-level properties, from F5's loader. **One schema reader.**"""
    from services.checkpoints import manifest as manifest_module

    schema = manifest_module._schema()
    root = manifest_module._root_def(schema)
    return dict(root.get("properties") or {})


def caller_manifest(fields: ManifestFields) -> dict[str, Any]:
    """The trainer's fields, and only those, from F5's ``ManifestFields``.

    ``ManifestFields`` carries ``base_model`` and ``created_at`` because F5 writes the whole
    manifest standalone. In this mode miLLM writes both, so both are **dropped here** rather
    than F5's dataclass being narrowed - narrowing it would change the standalone writer for
    the sake of the optional mode, which is the fence pointing the wrong way.

    A value of ``None`` is kept where the schema admits it: *null is a statement and an
    absent key is an omission*, and a dropped ``domain`` is a general-pretraining run that
    forgot to say so.
    """
    serving = serving_written_fields()
    body = {
        key: value
        for key, value in fields.as_kwargs().items()
        if key not in serving and key not in _CONSTANTS
    }
    extra = dict(getattr(fields, "extra", {}) or {})
    if extra:
        # Refused by name rather than smuggled: `extra` exists in F5's dataclass so a value
        # whose home is not yet agreed is refused at F5's clause (a) instead of silently
        # written. The same must hold here, and more strongly - the receiving end is another
        # repository's validator, and an unknown key is `INVALID_MANIFEST` listing a JSON
        # path, which reads as a miForge bug in miLLM's logs.
        from core.errors import refuse

        raise refuse(
            "attached_save_extra_fields",
            f"These manifest values have no agreed home and cannot be sent to a serving "
            f"process: {sorted(extra)}. In attached mode miLLM validates the body against "
            f"the vendored schema and refuses an unknown key. Agree the field in "
            f"checkpoint-manifest-v1.json first.",
            extra_fields=sorted(extra),
        )
    return body


@dataclass
class AttachedSaveOutcome:
    """What one attached-mode save did.

    ``directory`` is a **Path for the caller's record**, resolved the same way F5 resolves
    one (R-01.10's layout, R-01.24's padding), not read from the response: published BRD-01
    defines the layout and does **not** define the response's fields, and reading a field
    the contract does not define is the defect this feature was corrected for once already.
    """

    #: The reported location, as this process would address it. See the note above.
    directory: Path
    #: miLLM's response, **opaque and unparsed**. Carried so the run log and the operator
    #: can see what the other side said, without miForge asserting a schema for it.
    response: dict[str, Any]
    #: The identity **re-read after the save**. The caller's new baseline.
    identity: state.ResidentIdentity
    #: The checkpoint name indexed, or ``None`` when the directory was not readable here.
    indexed: str | None = None
    status: str = "written"
    effects: list[str] = field(default_factory=list)


class AttachedSaver:
    """One attachment's save path. **Bound to the attachment, not to the mode.**

    Handed to F1's loop as :attr:`training.acquisition.base.AcquiredModel.saver`, which is a
    *registration* in the same sense as the acquirer itself (FR-011.1, ADR-022): the loop
    dispatches on *"did this acquisition supply a saver"* and contains no ``mode ==
    "attach"`` test, so the fence holds by construction.

    **Why the loop cannot simply call F5's `save`:** an attached acquisition returns
    ``AcquiredModel(module=<rebuilt module>, tokenizer=None)`` - there is no tokenizer,
    because the tokenizer files are the serving process's. F5's writer calls
    ``tokenizer.save_pretrained``, so the first save point of every attached run raised
    ``AttributeError: 'NoneType' object has no attribute 'save_pretrained'`` - a traceback,
    not a refusal, at the moment a run first tries to checkpoint. *That is what OQ-5 being
    unanswered actually cost, and it was "written twice or not at all" in its worst form:
    not at all, and crashing.*
    """

    def __init__(self, client: Any, *, expected: state.ResidentIdentity) -> None:
        self.client = client
        #: The run's live baseline. **Mutable, and updated by every save**, because
        #: ``Attachment`` is frozen and its ``resident_identity`` is a snapshot of the
        #: attach. See the module docstring on the rename.
        self.expected = expected
        #: ``(step, checkpoint name)`` of the last save miLLM accepted, or ``None``. What
        #: R-02.30(b)'s rollback loads (``POST /api/weights/load {"name": ...}``).
        self.last_saved: tuple[int, str] | None = None

    def __call__(
        self,
        *,
        fields: ManifestFields,
        session: Any,
        root: Path | None = None,
        emit: Any = None,
        heartbeat: Any = None,
    ) -> AttachedSaveOutcome:
        """Save through the serving process. Signature-compatible with the loop's call.

        ``emit`` is accepted and **not used** - see the module docstring; taking it and
        ignoring it is wrong, so it is taken, named in the log, and the omission is recorded
        rather than hidden behind a parameter that looks wired.

        Raises:
            Refusal: ``weights_not_clean`` or ``weights_identity_changed`` from the
                precondition; whatever ``refusals`` maps miLLM's answer to; and
                ``attached_save_extra_fields`` from :func:`caller_manifest`.
        """
        outcome_effects: list[str] = []

        # 1. THE PRECONDITION, locally. The third of require_clean's three call sites.
        self.expected = state.require_clean(self.client, self.expected)
        outcome_effects.append("require_clean")

        # 2 and 3. The trainer's fields, through the one client.
        body = {"manifest": caller_manifest(fields)}
        response = self.client.call("save", body)
        outcome_effects.append("save")
        saved_name = response.get("name") if isinstance(response, dict) else None
        if isinstance(saved_name, str) and saved_name:
            self.last_saved = (int(fields.step), saved_name)

        # 4. Re-read, never assume. A successful save can rename the resident model.
        _, identity = state.read(self.client)
        self.expected = identity
        outcome_effects.append("identity_rebaselined")

        directory = checkpoint_paths.checkpoint_dir(fields.run_id, fields.step, root=root)
        outcome = AttachedSaveOutcome(
            directory=directory,
            response=dict(response) if isinstance(response, dict) else {},
            identity=identity,
            effects=outcome_effects,
        )

        # 5. Index through F5's own indexer, if what miLLM published is readable here.
        if (directory / MANIFEST_NAME).is_file():
            row = checkpoint_index.upsert_from_manifest(session, directory, root=root)
            outcome.indexed = str(row.name)
            outcome.effects.append("index")
        else:
            # NOT a refusal: the checkpoint exists and is valid upstream, and `save` is
            # retry-unsafe, so failing here would report a failure for a save that
            # succeeded and leave a retry to hit CHECKPOINT_EXISTS. Logged by name so the
            # gap is visible rather than silent - miLLM writes under its own
            # WEIGHTS_CHECKPOINT_DIR, and an operator who has not mounted it here gets a
            # checkpoint this installation's tree cannot show.
            outcome.status = "written_upstream_only"
            log.warning(
                "attached save published a checkpoint this process cannot read",
                extra={
                    "code": "attached_checkpoint_not_readable_here",
                    "run_id": fields.run_id,
                    "step": fields.step,
                    "expected_path": str(directory),
                    "reason": (
                        "The serving process writes under its own checkpoint root. Mount "
                        "that root here to index it; the checkpoint itself is published "
                        "and valid, so this is an index gap and not a lost save."
                    ),
                },
            )
            outcome.effects.append("index_skipped")

        log.info(
            "attached save",
            extra={
                "code": "attached_checkpoint_saved",
                "run_id": fields.run_id,
                "step": fields.step,
                "status": outcome.status,
                "resident_model": identity.model,
                "socket_event": "not emitted here - F5 owns the payload, F6 owns the room",
            },
        )
        outcome.effects.append("log")

        if heartbeat is not None:
            heartbeat.touch(session)
            outcome.effects.append("heartbeat")

        return outcome
