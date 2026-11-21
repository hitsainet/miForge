"""The ``weightspace`` 0.3.0 adapter, in the untrusted worker's child (feature 016; FR-016.17;
FTDD §6.5; FTID §3.3; D4, D6, P50, R1B-A3, R1B-A4, R1B-A6, R1B-A14, R3A-14).

Nothing of the library crosses this module: the child hands it the parsed child line and gets back
files in ``/out`` (``result.json``, ``transcript.jsonl``, ``patch/``), which the backend's intake
checks as untrusted data. No library type appears in any contract (WR-3).

- **The space** is ``WeightSpace.from_numpy`` over the bundle's EDITABLE tensors only, as the editor
  sees them (bf16 upcast to float32); non-editable tensors never reach the library.
- **The mode:** ``inspect`` reads (``read``), ``repair`` fixes (``fix``, or ``improve`` with a
  ``goal``), ``create`` zeroes (unless ``keep_weights``) and
commits
  by hand and then WRITES through :class:`MiStudioSession`, never ``tasks.write`` (which builds a
  plain ``Session``; R1B-A3).
- **``MiStudioSession._submit``** builds the v1 patch from the LIVE store against the bundle's own
  bits (not ``ws.patch()``, which sees only tensors in ``_original``; R1B-A4), checks the change
  limits, and answers ``refused: …`` without accepting the answer while any limit is broken; the
  library's loop then continues (``_answer`` stays ``None``). ``confidence`` passes unchanged.
- **The patch** written to ``/out/patch/`` is taken from the live store AFTER the session (tool
calls
  after ``submit`` in the same turn still run).
- **The LLM** is built with a pre-built SDK client pointed at the gateway, the session token as its
  key and ``max_retries=0`` (the gateway's refusals are final); ``get_llm`` is never used, since it
  reads provider keys from the environment. Anthropic is non-streaming.
- **The output cap** (fix 2026-11-20): every request carries ``max_tokens`` =
  ``miwsp_engine.context.editor_output_cap(editor)``, the engine's own rule over the same window
  (the endpoint's ``context_window_tokens``, else miServer's served window, else 8,192), so the
  bake-off compares both providers at one cap. ``OpenAIChat`` sends no cap of its own, and the
  gateway's old fallback (32,000) passed a 32,768-token miServer window at the first call.
- **Budgets** (R1B-A6): the library's ``token_budget`` and ``time_budget`` at 90% of the run's caps
  (the time also never past the 10% share before the supervisor's kill: ``library_time_budget``);
  ``pricing`` and ``cost_budget`` only when the endpoint has prices. A ``SessionError`` is caught
  and
  its ``.result`` saved.
- **Observation tools** (FR-016.18): the request's named set, as the library's evaluation
  callbacks at their declared costs, drawing from ``forward_passes``.
- **Program mode** (FR-016.23): the session READS; ``test_program`` (an observation tool) and
  ``submit_program`` (once, after two identical runs) are added; the stored program, run once
  more in a fresh namespace, gives the candidate's patch, and ``/out/program.py`` holds it.
- **Replay** (FR-016.21): the same session over ``ScriptedLLM.from_transcript`` of the pushed
  original transcript; no gateway, no client.
- **Saving** (R3A-14): ``/out/result.json`` is ``res.to_dict()``; ``res.save`` is never called (its
  ``ws/`` files are names the intake refuses).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from weightspace import Pricing, Probe, Session, SessionError, Tool, WeightSpace
from weightspace.llm.base import LLM

from wsp_worker import bundle as bundle_mod
from wsp_worker import limits as limits_mod
from wsp_worker import observation_tools, patch_writer, program, program_tools
from wsp_worker.bundle import Bundle, WorkerTensor
from wsp_worker.miwsp_engine.context import capped, editor_output_cap
from wsp_worker.program_tools import ProgramTools
from wsp_worker.scoring import Scorer

MODE_FOR = {"inspect": "read", "repair": "fix", "create": "write"}


def session_mode(request: dict[str, Any], style: str) -> str:
    """FR-016.31 (16.2): ``inspect`` reads, ``create`` writes, ``repair`` fixes, or IMPROVES when
    the request carries a ``goal``; program mode always reads (its program writes; P19)."""
    if style == "program":
        return "read"
    kind = request["kind"]
    if kind == "repair" and request.get("goal"):
        return "improve"
    return MODE_FOR[kind]


def session_task(request: dict[str, Any]) -> str:
    """The text the session runs: the task, and the goal after it when the run improves."""
    task = str(request.get("task") or "")
    goal = request.get("goal")
    if request["kind"] == "repair" and goal:
        return f"{task}\n\nGoal: {goal}" if task else str(goal)
    return task


BUDGET_SHARE = 0.9
PROVIDER = {"provider": "weightspace", "provider_version": "0.3.0"}


def edited_tensors(ws: WeightSpace, bundle: Bundle) -> dict[str, WorkerTensor]:
    """The live store, cast back to each tensor's stored dtype (bf16 round to nearest even)."""
    ws.sync()
    live = ws.to_numpy()
    return {
        name: patch_writer.from_values(bundle.tensors[name].dtype, np.asarray(values))
        for name, values in live.items()
    }


def build_patch(
    ws: WeightSpace, bundle: Bundle, request: dict[str, Any], run_id: str
) -> tuple[dict[str, Any], bytes] | None:
    parent = request.get("parent") or {"kind": "bundle", "ref": bundle.manifest["bundle_sha256"]}
    return patch_writer.build_patch(
        bundle.tensors,
        edited_tensors(ws, bundle),
        bundle_base_sha256=bundle.manifest["bundle_sha256"],
        parent={"kind": "bundle", "ref": parent.get("ref"), "patch_sha256": None},
        parent_patch_sha256=None,
        created_by={**PROVIDER, "run_id": run_id},
        created_at=dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
    )


class MiStudioSession(Session):  # type: ignore[misc]  # weightspace ships no stubs in this venv
    """``Session`` with one override: ``_submit`` refuses while the live patch breaks a limit."""

    def __init__(self, *args: Any, check: Callable[[], list[str]], **kwargs: Any) -> None:
        self._mistudio_check = check
        super().__init__(*args, **kwargs)

    def _submit(self, answer: Any, confidence: Any = None) -> str:
        violations = self._mistudio_check()
        if violations:
            return "refused: " + "; ".join(violations) + ". Change the edit, then submit again."
        result: str = super()._submit(answer, confidence)
        return result


def build_llm(editor: dict[str, Any], gateway: dict[str, Any], *, output_cap: bool = True) -> LLM:
    """The editor LLM over the gateway: a pre-built client, the session token, no retries, and
    the engine's per-turn output cap (``max_tokens``; the gateway moves it under the endpoint's
    ``output_cap_field``). With the run's output cap off (2026-11-21), the OpenAI client names no
    cap, so the gateway's window-aware default applies; the Anthropic SDK requires one, so it
    keeps the library's own default (16,000), which the gateway still clamps to the budget."""
    base = f"http://{gateway['ip']}:{int(gateway['port'])}"
    token = str(gateway["token"])
    cap = editor_output_cap(editor)
    if editor["kind"] == "anthropic":
        import anthropic
        from weightspace.llm.anthropic import AnthropicChat

        client = anthropic.Anthropic(base_url=base, api_key=token, max_retries=0)
        if not output_cap:
            return AnthropicChat(editor["model"], client=client, stream=False)
        return AnthropicChat(editor["model"], client=client, stream=False, max_tokens=cap)
    import openai
    from weightspace.llm.openai import OpenAIChat

    client_o = openai.OpenAI(base_url=base + "/v1", api_key=token, max_retries=0)
    # ``OpenAIChat`` passes every extra keyword into ``chat.completions.create``.
    if not output_cap:
        return OpenAIChat(editor["model"], client=client_o, base_url=base + "/v1")
    return OpenAIChat(editor["model"], client=client_o, base_url=base + "/v1", max_tokens=cap)


def _pricing(editor: dict[str, Any]) -> Pricing | None:
    prices = editor.get("prices") or {}
    if prices.get("input") is None or prices.get("output") is None:
        return None
    return Pricing(
        input=float(prices["input"]),
        output=float(prices["output"]),
        cached_input=None if prices.get("cache_read") is None else float(prices["cache_read"]),
        cache_write=None if prices.get("cache_write") is None else float(prices["cache_write"]),
    )


def context_text(bundle: Bundle) -> str:
    """The function's description and the tensors' roles. NEVER the practice data."""
    lines = [bundle.manifest["function"]["description"], "", "Tensors:"]
    for t in bundle.manifest["tensors"]:
        role = "editable" if t["editable"] else "read-only (not given to you)"
        lines.append(f"- {t['name']} {t['dtype']} {t['shape']}: {role}")
    return "\n".join(lines)


def observation_context(bundle: Bundle, ws: WeightSpace) -> observation_tools.Context:
    """The practice view and the live head tensors (editable from the store, the rest fixed)."""
    arrays = bundle.practice.arrays
    fixed = {n: t.values().astype(np.float64) for n, t in bundle.tensors.items()}

    def current() -> dict[str, np.ndarray]:
        live = {n: np.asarray(v, dtype=np.float64) for n, v in ws.to_numpy().items()}
        return {**fixed, **live}

    view = observation_tools.PracticeView(
        scorer=Scorer(
            bundle.manifest["function"], bundle.practice.activations, arrays["practice_offsets"]
        ),
        labels=np.asarray(arrays["practice_labels"]),
        validation=np.asarray(arrays["practice_validation_mask"], dtype=bool),
        truncated=np.asarray(arrays["practice_truncated_mask"], dtype=bool),
        target_fpr=float((bundle.manifest.get("operating_point") or {}).get("target_fpr") or 0.01),
        shipped_bar=observation_tools.shipped_bar_of(bundle.manifest),
    )
    return observation_tools.Context(
        view, current, fixed, program.seed_for(bundle.manifest["bundle_sha256"])
    )


def probes_for(specs: list[observation_tools.ToolSpec]) -> list[Probe]:
    """Each observation tool as the library's evaluation callback, at its declared cost."""
    return [
        Probe(s.fn, name=s.name, description=s.description, parameters=s.parameters, cost=s.cost)
        for s in specs
    ]


def program_tools_for(programs: ProgramTools, session: Session) -> list[Tool]:
    """``program_tools.ProgramTools`` as the library's tools: ``test_program`` charges its cost
    through the session's own forward budget (``session._forward``); ``submit_program`` is free."""
    probe = Probe(programs.test_program, name="test_program", cost=programs.COST)
    return [
        Tool(
            "test_program",
            f"[forward pass, costs {programs.COST}] {program_tools.TEST_DESCRIPTION}",
            program_tools.CODE_PARAMETERS,
            lambda code: session._forward(probe, {"code": code}),
        ),
        Tool(
            "submit_program",
            program_tools.SUBMIT_DESCRIPTION,
            program_tools.CODE_PARAMETERS,
            programs.submit_program,
        ),
    ]


def replay_llm(line: dict[str, Any]) -> LLM:
    """R1B-A7: the original transcript, a pushed file checked by the supervisor against the push,
    which the driver checked against the original run's ``transcript_sha256``."""
    from weightspace import ScriptedLLM

    entry = next((f for f in line["files"] if f["role"] == "transcript"), None)
    if entry is None:
        raise bundle_mod.BundleError("a replay's push carries no transcript", "worker_spec_invalid")
    return ScriptedLLM.from_transcript(str(Path(line["paths"]["work"]) / entry["name"]))


def library_time_budget(budgets: dict[str, Any], line: dict[str, Any]) -> float:
    """The library's ``time_budget`` (seconds from its own start): 90% of the run's, and never
    past the 10% share before the SUPERVISOR's kill, which the child line's ``time_left_s``
    measures from the session's start (fix 2026-11-21). ``weightspace`` 0.3.0 has no hook to tell
    the editor that time is nearly up (its callbacks only observe), so this is the one part of the
    engine's graceful end the adapter can take: its own final submit-only turn, in time."""
    time_s = float(budgets["time_s"])
    left = line.get("time_left_s")
    if left is None:
        return time_s * BUDGET_SHARE
    return max(0.0, min(time_s * BUDGET_SHARE, float(left) - (1 - BUDGET_SHARE) * time_s))


def run(line: dict[str, Any], *, llm: LLM | None = None) -> int:
    """One session (or its replay) from the child line; writes ``/out``; the child's exit code."""
    spec, paths = line["spec"], line["paths"]
    request, editor = spec["request"], spec.get("editor") or {}
    style = spec.get("replay_of_mode", "session") if line["mode"] == "replay" else line["mode"]
    out = Path(paths["out"])
    bundle = bundle_mod.load(Path(paths["work"]), spec["manifest"], list(line["files"]))
    editable = {
        name: np.array(t.values(), copy=True)
        for name, t in bundle.tensors.items()
        if bundle.editable(name)
    }
    ws = WeightSpace.from_numpy(editable)
    kind = request["kind"]
    # FR-016.31: create starts from zeroed editable tensors unless keep_weights is set.
    if kind == "create" and style != "program" and not request.get("keep_weights"):
        ws.zero_()
        ws.commit()

    def check() -> list[str]:
        built = build_patch(ws, bundle, request, str(line["run_id"]))
        if built is None:
            return []
        return [v.detail for v in limits_mod.check_limits(built[0], request["change_limits"])]

    budgets = request["budgets"]
    pricing = _pricing(editor)
    cost_cap = budgets.get("cost_usd")
    specs = observation_tools.build(
        list(request.get("observation_tools") or []), observation_context(bundle, ws)
    )
    if llm is None:
        llm = (
            replay_llm(line)
            if line["mode"] == "replay"
            else build_llm(editor, line["gateway"], output_cap=capped(request))
        )
    programs = (
        ProgramTools(bundle, request, str(line["run_id"]), created_by=PROVIDER)
        if style == "program"
        else None
    )
    session = MiStudioSession(
        ws,
        llm,
        check=check,
        # Program mode reads: the editor cannot write weights; its program does (P19).
        mode=session_mode(request, style),
        context=context_text(bundle),
        python=True,
        probes=probes_for(specs),
        forward_budget=int(budgets.get("forward_passes") or 0),
        max_steps=int(budgets["max_steps"]),
        token_budget=int(int(budgets["tokens"]) * BUDGET_SHARE),
        time_budget=library_time_budget(budgets, line),
        pricing=pricing,
        cost_budget=None if pricing is None or cost_cap is None else float(cost_cap) * BUDGET_SHARE,
        thinking=request.get("thinking"),
        log_path=str(out / "transcript.jsonl"),
        parallel_tools=1,
    )
    if programs is not None:
        for t in program_tools_for(programs, session):
            session.tools[t.name] = t
    try:
        result = session.run(session_task(request))
    except SessionError as exc:
        result = exc.result
    (out / "result.json").write_text(json.dumps(result.to_dict(), default=str), encoding="utf-8")
    if programs is not None:
        if programs.submitted is not None:
            (out / program.PROGRAM_NAME).write_text(programs.submitted, encoding="utf-8")
            # The candidate: the submitted program run once more, in a fresh namespace.
            done = programs.candidate()
            if done is not None and done.manifest is not None and done.data is not None:
                patch_writer.write_patch(out / "patch", done.manifest, done.data)
        return 0
    # Every kind, inspect included: an inspect that changed weights (a write through R's
    # writable base, say) yields a patch, and the intake refuses it (FR-016.32; R1B-A4).
    built = build_patch(ws, bundle, request, str(line["run_id"]))
    if built is not None:
        patch_writer.write_patch(out / "patch", built[0], built[1])
    return 0
