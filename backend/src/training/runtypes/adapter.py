"""``adapter`` - train a low-rank adapter beside the base  (R-02.43; FR-008.19 - FR-008.25).

## The cheapest run type, not the dearest

People get this backwards and then budget it as the expensive one. R-02.43 writes **no
resident tensors**, so it needs no commit bracket and holds **no barrier at all** - not even
in section 5.10 - and its gradients and optimizer state cover the adapter rather than the
full parameter set, which is why an adapter fits where full-parameter does not (R-02.42's
last sentence). The consequence worth having is in the requirement's own words: *an adapter
run never interrupts serving.*

## The base is miForge's own, and nothing attaches

In the default path miForge **holds** the base (R-02.46), so there is nothing to attach to
and the adapter run reads the base from its own parameters. R-02.43's *"may attach"* case -
reading a serving process's resident tensors through CUDA IPC handles to avoid duplicating
2.4 GB on one card - is section 5.10's and F11's. Neither mode writes back, so neither needs
the barrier. **No module on this path references a miLLM endpoint** (ADR-022), asserted as an
absence by ``test_runtype_no_hotupdate.py`` rather than left as the author's intention.

## AC14 is checked by hashing, not asserted

:func:`base_parameter_digest` is taken in :meth:`AdapterRunType.prepare` - the earliest point
at which the base is in hand and nothing has had a chance to touch it - and compared after
the run. It **refuses a parameter whose bytes it cannot read** rather than skipping it: a
digest that silently hashed nothing would make two empty hashes equal and AC14 would pass
against a run that rewrote the whole base.

### The digest is taken over CANONICAL names, and that is load-bearing

**AC14's comparison could never pass until 2026-10-16**, whatever happened to the weights.
The digest covers each parameter's *name* as well as its bytes, and ``peft`` **renames every
base tensor on wrap**: ``model.layers.0.self_attn.q_proj.weight`` becomes
``base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight``. Measured with no
training at all - ``prepare()`` and nothing else - the "before" and the digest of the
returned wrapped module disagreed while a **bytes-only** digest over the same two modules
was identical. The weights were correct; the digest was not, and
``test_the_base_digest_is_identical_after_a_real_step`` failed for a reason that had nothing
to do with the base.

So every name is reduced to :func:`canonical_base_name` - ``peft``'s wrapper prefix stripped
and its ``base_layer`` indirection removed - which is **the unwrapped model's own naming
convention**, and therefore the convention of the artifact the base actually is. The
alternative considered and rejected was to take the digest *after* the wrap over base
tensors only: that hashes names which are a function of the adapter configuration (an
untargeted ``down_proj`` keeps its name, a targeted ``q_proj`` gains ``base_layer``) **and**
of the installed ``peft`` version, so the same base would hash differently under two
adapter configurations and nobody holding only the base could reproduce it. The whole point
of the hash is that someone else can check the base was untouched; a digest reproducible
only inside this process under one ``LoraConfig`` is not that.

A rename that made two distinct tensors share a canonical name would silently drop one, so
that is a refusal (``base_digest_name_collision``) rather than a ``dict`` overwrite.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from core.canonical import sha256_of
from core.errors import refuse
from training.acquisition.base import AcquiredModel, AcquisitionRecord, AcquisitionSpec
from training.run_config import RunConfig
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters import save as adapter_save
from training.runtypes.adapters.build import assert_only_adapter_requires_grad
from training.runtypes.adapters.config import validate as validate_adapter_config
from training.runtypes.base import STEP, SaveContext, TrainablePlan
from training.runtypes.change import joined_scores

log = logging.getLogger(__name__)

#: The segment PEFT puts in every adapter tensor's name. A parameter without it is a base
#: tensor, which is what makes "hash the base, not the adapter" expressible.
PEFT_SEGMENT = "lora_"

#: What ``get_peft_model`` prepends to every parameter name. ``PeftModel.base_model`` is the
#: ``LoraModel`` and ``LoraModel.model`` is the module handed in, so the module's own names
#: sit two attributes down. Verified against ``peft`` on this host rather than recalled.
PEFT_WRAPPER_PREFIX = "base_model.model."

#: What ``peft`` inserts for a **targeted** module: the original ``nn.Linear`` is moved to
#: ``<module>.base_layer`` and the LoRA matrices hang beside it. An *untargeted* module keeps
#: its name, which is why the post-wrap naming is a function of the adapter configuration and
#: why the canonical form is the unwrapped one.
PEFT_BASE_LAYER_SEGMENT = "base_layer"


def canonical_base_name(name: str) -> str:
    """One base tensor's name in **the unwrapped model's** convention.

    The identity on an unwrapped module, so a pre-wrap digest and a post-wrap digest of the
    same untouched base agree. The two reductions are structural - a wrapper attribute path
    and an indirection - and neither can change which tensor is named:

    >>> canonical_base_name("model.layers.0.self_attn.q_proj.weight")
    'model.layers.0.self_attn.q_proj.weight'
    >>> canonical_base_name(
    ...     "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    ... )
    'model.layers.0.self_attn.q_proj.weight'
    """
    if name.startswith(PEFT_WRAPPER_PREFIX):
        # Once, not repeatedly: a base model whose own first parameter path happened to be
        # `model.…` must not have a second prefix eaten off it.
        name = name[len(PEFT_WRAPPER_PREFIX) :]
    inner = f".{PEFT_BASE_LAYER_SEGMENT}."
    if inner in name:
        name = name.replace(inner, ".")
    suffix = f".{PEFT_BASE_LAYER_SEGMENT}"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def _reinterpret_as_uint8(parameter: Any) -> bytes:
    """One parameter's bytes, read **through a ``uint8`` view** rather than through numpy's
    understanding of its dtype.

    **This route exists because ``bfloat16`` has no numpy scalar type**, which is the whole
    of the defect recorded below. ``.view(torch.uint8)`` reinterprets the same storage as
    bytes, so the element type never has to cross into numpy at all - and because it is a
    reinterpretation and not a conversion, it returns ``element_size() * numel()`` bytes,
    the identical bytes the ``.numpy().tobytes()`` route returns for every dtype numpy *can*
    represent. Verified for ``float32``, ``float16``, ``bfloat16`` and ``int64``, and
    byte-for-byte against the old route for ``float32``, so **no digest already recorded
    changes value.**

    Separate from :func:`_parameter_bytes`'s lambda list so the reasoning has somewhere to
    live and so the mutation control has a named target.
    """
    import torch

    raw: bytes = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return raw


def _parameter_bytes(parameter: Any) -> bytes:
    """The raw bytes of one parameter, or a refusal.

    Duck-typed so a stub can stand in for a tensor on a machine with no ML stack, and
    **refusing** when none of the routes works rather than returning ``b""``. That refusal
    is the whole point: a digest that skipped what it could not read would compare two empty
    hashes as equal, and AC14 would pass against a run that rewrote every byte of the base.

    ## ``bfloat16`` was REFUSED here until 2026-10-16, on the main path

    *Found by re-scaling the adapter overhead benchmark onto a realistic fixture*, which
    instantiates in ``bfloat16`` - and ``prepare`` refused ``base_digest_unreadable`` before
    a single step ran. Probed on the card rather than reasoned about:

    * ``float32`` and ``float16`` read fine through ``.numpy()``;
    * ``bfloat16`` raises ``TypeError: Got unsupported ScalarType BFloat16`` on **both**
      numpy routes, because numpy has no bfloat16 scalar type;
    * ``bytes(memoryview(p))`` raises ``TypeError`` for a torch ``Tensor`` or ``Parameter``
      of *any* dtype - it is the route for a **non-torch** stub (a numpy array, ``bytes``,
      a ``bytearray``), which is what its docstring line means, and it was never going to
      rescue a torch tensor.

    So all three routes failed for exactly one dtype - **and it is the dtype every real base
    is in.** ``write_bench_model`` and ``write_tiny_model`` both emit ``bfloat16``, the
    checkpoint writer emits ``bfloat16``, and R-02.43's base-unchanged guarantee (AC14) was
    therefore *uncomputable for every real adapter run*. It survived review because the
    adapter fixtures instantiated at **default fp32**: a fixture that agreed with the reader
    by construction, which is this project's named reason a suite stays green over a real
    bug. :func:`_reinterpret_as_uint8` closes it, and
    ``tests/unit/test_base_digest_refuses_bfloat16.py`` is the regression.

    **The blast radius is this function's one caller and nothing else.**
    :func:`base_parameter_digest` is the only call site in ``backend/src/``;
    ``starting_weights_sha256`` is computed by ``weights_sha256`` over safetensors shards
    (``local_dir``, ``hub``) or by ``created_weights_sha256(config, seed)`` (``scratch``) and
    never reaches here, so no published R-02.47 hash moves.

    Raises:
        Refusal: ``base_digest_unreadable``, now carrying **why each route failed** in its
            details - the old version swallowed every exception and refused with only the
            parameter's type name, which is what made a one-dtype numpy gap read as a
            generic unreadable parameter and cost a probe script to diagnose.
    """
    attempts: dict[str, str] = {}
    for name, route in (
        ("uint8_view", _reinterpret_as_uint8),
        ("numpy_contiguous", lambda p: p.detach().cpu().contiguous().numpy().tobytes()),
        ("numpy", lambda p: p.detach().cpu().numpy().tobytes()),
        ("memoryview", lambda p: bytes(memoryview(p))),
    ):
        try:
            value = route(parameter)
        except Exception as exc:
            attempts[name] = f"{type(exc).__name__}: {exc}"
            continue
        if isinstance(value, bytes | bytearray):
            return bytes(value)
        attempts[name] = f"returned {type(value).__name__}, not bytes"
    raise refuse(
        "base_digest_unreadable",
        "A base parameter's bytes could not be read, so the hash that proves the base is "
        "unchanged cannot be computed. Refusing rather than skipping it: a digest over "
        "nothing compares equal to another digest over nothing, and acceptance criterion "
        "14 would pass against a run that rewrote the whole base.",
        parameter_type=type(parameter).__name__,
        routes_tried=attempts,
    )


def base_parameter_digest(module: Any) -> str:
    """SHA-256 over the **base** parameters' canonical names and bytes, in a stable order.

    The same untouched base hashes the same whether or not it has been wrapped by ``peft``,
    because every name is reduced by :func:`canonical_base_name` first. That is what makes
    AC14's before-and-after comparison expressible at all, and it is what makes the value
    reproducible by someone holding only the base - see the module docstring.

    The adapter's own tensors are excluded by name: they are *supposed* to change, and
    including them would make the comparison fail on every successful run.

    Raises:
        Refusal: ``base_digest_empty`` when the module exposes no base parameter at all -
            the same vacuity guard as :func:`_parameter_bytes`, one level up;
            ``base_digest_name_collision`` when two distinct tensors reduce to one canonical
            name, which a ``dict`` would resolve by dropping one of them silently.
    """
    # Names and references only - no bytes are read in this pass, so a 1.2B base is still
    # hashed one tensor at a time rather than buffered whole.
    by_canonical: dict[str, tuple[str, Any]] = {}
    for name, parameter in module.named_parameters():
        if PEFT_SEGMENT in name:
            continue
        key = canonical_base_name(name)
        if key in by_canonical:
            raise refuse(
                "base_digest_name_collision",
                f"Two base parameters of this module reduce to the canonical name {key!r}, "
                f"so one of them would be dropped from the hash that proves the base is "
                f"unchanged. Refusing rather than resolving it silently: a dropped tensor "
                f"is a tensor acceptance criterion 14 stops covering, and nothing would "
                f"report it.",
                canonical_name=key,
                parameters=[by_canonical[key][0], name],
            )
        by_canonical[key] = (name, parameter)

    if not by_canonical:
        raise refuse(
            "base_digest_empty",
            "This module exposes no base parameter, so there is nothing for acceptance "
            "criterion 14's before-and-after comparison to cover. Refusing rather than "
            "returning the hash of an empty input, which would compare equal to itself and "
            "report the guarantee as held.",
        )

    hasher = hashlib.sha256()
    for key in sorted(by_canonical):
        hasher.update(key.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(_parameter_bytes(by_canonical[key][1]))
    return hasher.hexdigest()


def base_manifest_sha256_for(record: AcquisitionRecord, *, root: Path | None = None) -> str | None:
    """``base.manifest_sha256`` when the base **was** a saved miForge checkpoint.

    ``None`` when it was not - a Hub repo or a plain local directory - which is the value
    the schema wants there and is what distinguishes the two cases
    (:func:`training.runtypes.adapters.manifest.base_block`).

    ``None`` **also** when the base is a checkpoint whose manifest cannot be read, and that
    is deliberate: :func:`~training.runtypes.adapters.manifest.base_block` then refuses
    ``adapter_base_unresolved`` naming the base, which is the honest outcome and a better
    message than anything this function could raise. *What must never happen is a null
    written for a base that IS a checkpoint* - the schema's null asserts the base was never
    saved - and that is the refusal, not this return.

    The three steps are F8's existing ones, read from
    :func:`training.runtypes.adapters.merge.resolve_base` rather than invented here: F1
    records the base checkpoint's qualified name as ``record.parent``, F5's
    ``parse_qualified_name`` splits it, and the hash is ``sha256_of`` the manifest document
    - the same canonical hash ``MergedAdapter.manifest_sha256`` and the index's column point
    at. F8 only ever **reads** a checkpoint manifest (R-02.20 is F5's).
    """
    if not record.parent:
        return None

    from services.checkpoints import manifest as checkpoint_manifest
    from services.checkpoints.manifest import MANIFEST_NAME as CHECKPOINT_MANIFEST_NAME
    from services.checkpoints.paths import checkpoint_dir, parse_qualified_name

    parsed = parse_qualified_name(record.parent)
    if parsed is None:
        return None
    _, base_run_id, base_step = parsed
    try:
        manifest_path = checkpoint_dir(base_run_id, base_step, root=root) / CHECKPOINT_MANIFEST_NAME
        if not manifest_path.is_file():
            return None
        return sha256_of(checkpoint_manifest.read(manifest_path))
    except Exception:
        # Any failure here is "unresolved", and `base_block` is what refuses with the base
        # named. Swallowing it to return a *null* would be the false assertion; swallowing it
        # to reach that refusal is not.
        return None


class AdapterRunType:
    """R-02.43's adapter run type. **No barrier, no bracket, no sibling.**"""

    name = "adapter"
    #: The same object every run type carries. An adapter run introduces no new step path.
    step = staticmethod(STEP)

    def preflight(
        self,
        spec: AcquisitionSpec,
        cfg: RunConfig,
        *,
        signal_source: str | None = None,
        adapter: Any = None,
    ) -> None:
        """Validate the adapter configuration against **the schema's** bounds (E-7).

        Raises:
            Refusal: ``adapter_config_required`` when none was supplied - an adapter run
                without a rank is not a run with a default rank; ``adapter_config_invalid``
                from the edge model, naming the field and the bound.
        """
        if adapter is None:
            raise refuse(
                "adapter_config_required",
                "An 'adapter' run needs a rank, an alpha and at least one target module. "
                "Refusing rather than defaulting them: a defaulted rank trains an adapter "
                "the operator did not ask for and records it in a manifest as though they "
                "had.",
                run_type=self.name,
            )
        validate_adapter_config(adapter)

    def prepare(self, acquired: AcquiredModel, *, adapter: Any = None) -> TrainablePlan:
        """Wrap the base with ``peft`` and return **only the adapter's** parameters.

        The base digest is taken here, before anything has had a chance to touch the base
        (TID section 7), and it is **comparable against the wrapped module this returns** -
        which it was not until 2026-10-16, because ``peft``'s rename changed every base
        tensor's name and the digest covers names. Taking it before the wrap is still right
        on the merits; what was wrong was that the two sides were not the same hash.
        ``trainable`` is the adapter's parameters alone, so F1's optimizer
        builder - which collects ``requires_grad`` parameters - constructs an optimizer over
        the adapter and nothing else, with gradients and state tens of megabytes rather than
        gigabytes.
        """
        from training.runtypes.adapters.build import build as build_adapter

        if adapter is None:
            # THE SAME NAMED REFUSAL `preflight` AND `save` CARRY, and it was the one call
            # site of the three without it: `build_adapter(module, None)` validated a null
            # and raised `adapter_config_invalid`, which reads as *"your rank is wrong"* for
            # a run that supplied no adapter block at all. Reachable from 2026-10-16, when
            # `prepare` gained a production caller.
            raise refuse(
                "adapter_config_required",
                "An 'adapter' run needs a rank, an alpha and at least one target module. "
                "Refusing rather than defaulting them: a defaulted rank trains an adapter "
                "the operator did not ask for and records it in a manifest as though they "
                "had.",
                run_type=self.name,
            )

        digest = base_parameter_digest(acquired.module)
        wrapped, _ = build_adapter(acquired.module, adapter)
        trainable = assert_only_adapter_requires_grad(wrapped)
        log.info(
            "adapter run prepared",
            extra={
                "code": "adapter_run_prepared",
                "base_digest": digest,
                "trainable_parameters": len(trainable),
            },
        )
        return TrainablePlan(trainable=trainable, module=wrapped, base_digest=digest)

    def save(self, ctx: SaveContext) -> dict[str, Any]:
        """Publish one adapter directory on the loop's save cadence (FR-008.26).

        **The cadence itself is F6's** ``save_every``: this consumes the save-point signal
        and defines no second cadence. The artifact is a PEFT directory plus miForge's
        additive manifest, written atomically through ``core.atomic_dir``.
        """
        if ctx.adapter is None:
            raise refuse(
                "adapter_config_required",
                "An adapter save needs the run's adapter configuration: the manifest's "
                "rank and alpha and the numbers handed to save_pretrained come from one "
                "object, because miLLM's loader cross-checks them and refuses a mismatch.",
                run_id=ctx.run_id,
                step=ctx.step,
            )
        config = validate_adapter_config(ctx.adapter)
        # RESOLVED WHEN THE CONTEXT DOES NOT CARRY IT. `SaveContext.base_manifest_sha256` is
        # an override for a caller that already holds the hash; the loop does not, and before
        # 2026-10-16 nothing called this method at all, so an adapter run over a miForge
        # checkpoint base would have refused `adapter_base_unresolved` at its first save
        # point - having trained.
        base_manifest_sha256 = ctx.base_manifest_sha256
        if base_manifest_sha256 is None:
            base_manifest_sha256 = base_manifest_sha256_for(ctx.record, root=ctx.root)
        base = adapter_manifest.base_block(ctx.record, manifest_sha256=base_manifest_sha256)
        # THE SCORES ARE JOINED FROM F4'S INDEX, not read off the context. `ctx.test` and
        # `ctx.retention` have no production filler, so this manifest recorded
        # `test: null, retention: null` on every save - and an adapter manifest is the one
        # artifact built to **travel without its run log**, so for an adapter the loss is
        # total rather than merely inconvenient. See `change.joined_scores`.
        test, retention = joined_scores(ctx)
        doc = adapter_manifest.build(
            run_id=ctx.run_id,
            step=ctx.step,
            base=base,
            adapter=config,
            optimizer=ctx.optimizer,
            parts=ctx.parts,
            test=test,
            retention=retention,
        )
        dest = adapter_paths.adapter_dir(ctx.run_id, ctx.step, root=ctx.root)
        published = adapter_save.save(ctx.module, doc, dest)
        ctx.effects.append("adapter_write")
        # WHAT WAS PUBLISHED - see `SaveContext.artifacts`. The adapter writer returns the
        # manifest as published and the checkpoint writer returns a `SaveOutcome`, so the
        # loop reads neither; both writers say where they wrote.
        ctx.artifacts.append(str(dest))
        return published
