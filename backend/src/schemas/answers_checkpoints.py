"""The answer models of the checkpoint and tree routes (feature 021, P3):
``api/v1/endpoints/checkpoints.py`` and ``api/v1/endpoints/tree.py``.

Documentation only (MF9): the routes still return their dicts, and these models change what the
OpenAPI document (and so the published answer contract) says about them, nothing else. Each class
names the function that builds the shape it describes.
"""

from pydantic import BaseModel

from .answers_common import NextStep, may_be_absent
from .answers_documents import CheckpointManifestDocument
from .answers_runs import RunDefineRunAnswer, RunResumeRecord

# ---------------------------------------------------------------------------- manifest pieces


class CheckpointScoreRecord(BaseModel):
    """A ``ScoreRecord`` as a checkpoint manifest recorded it, passed through
    (``checkpoint-manifest-v1.json#/$defs/ScoreRecord``; ``services/reindex.py``,
    ``services/checkpoints/lineage.py:_node``, ``services/tree.py:_test_record``)."""

    set: str
    score: float
    n: int
    floor: float | None = may_be_absent(description="Not required by the manifest schema.")
    provider: str | None = may_be_absent(description="Not required by the manifest schema.")


class CheckpointOptimizerIdentity(BaseModel):
    """A manifest's optimizer identity, passed through
    (``checkpoint-manifest-v1.json#/$defs/OptimizerIdentity``)."""

    name: str
    state: str
    config_hash: str | None = may_be_absent(description="Not required by the manifest schema.")


class CheckpointMergedAdapter(BaseModel):
    """A manifest's merged adapter, passed through
    (``checkpoint-manifest-v1.json#/$defs/MergedAdapter``)."""

    name: str
    step: int
    manifest_sha256: str | None = may_be_absent(description="Not required by the manifest schema.")


class CheckpointIndexRow(BaseModel):
    """One indexed checkpoint (``services/reindex.py:_checkpoint_payload``)."""

    name: str
    run_id: str
    step: int
    domain: str | None
    domain_state: str
    domain_display: str
    signal_source: str
    parent: str | None
    base_model: str
    merged_adapter: CheckpointMergedAdapter | None
    weights_sha256: str
    manifest_sha256: str
    selector_hash: str | None
    grader_hash: str | None
    optimizer: CheckpointOptimizerIdentity | None
    test: CheckpointScoreRecord | None
    retention: CheckpointScoreRecord | None
    path: str
    created_at: str


# ---------------------------------------------------------------------------- checkpoints


class CheckpointPendingSave(BaseModel):
    """The recorded save request (``services/checkpoints/requests.py:request_save``)."""

    run_id: str
    requested_at: str
    request_count: int


class CheckpointRequestCheckpointAnswer(BaseModel):
    """``POST /api/v1/runs/{run_id}/checkpoints``
    (``api/v1/endpoints/checkpoints.py:request_checkpoint``)."""

    run_id: str
    accepted: bool
    pending_save: CheckpointPendingSave


class CheckpointListAllCheckpointsAnswer(BaseModel):
    """``GET /api/v1/checkpoints`` (``api/v1/endpoints/checkpoints.py:list_all_checkpoints``)."""

    checkpoints: list[CheckpointIndexRow]
    count: int
    total: int


class CheckpointListRunCheckpointsAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/checkpoints``
    (``api/v1/endpoints/checkpoints.py:list_run_checkpoints``)."""

    run_id: str
    checkpoints: list[CheckpointIndexRow]
    latest_step: int | None
    steps_on_disk: list[int]


class CheckpointProviderSource(BaseModel):
    """One slot's provider and the source that said so
    (``services/checkpoints/provider_source.py:provider_sources``)."""

    record: CheckpointScoreRecord | None
    provider: str | None
    source: str


class CheckpointProviderSources(BaseModel):
    """The labelled provider join (``services/checkpoints/provider_source.py:provider_sources``)."""

    test: CheckpointProviderSource
    retention: CheckpointProviderSource


class CheckpointGetCheckpointAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/checkpoints/{step}``
    (``api/v1/endpoints/checkpoints.py:get_checkpoint``, ``_manifest_response``)."""

    run_id: str
    step: int
    manifest: CheckpointManifestDocument
    manifest_sha256: str
    provider_sources: CheckpointProviderSources


class CheckpointVerifyCheckpointAnswer(BaseModel):
    """``POST /api/v1/runs/{run_id}/checkpoints/{step}/verify``
    (``api/v1/endpoints/checkpoints.py:verify_checkpoint``)."""

    run_id: str
    step: int
    match: bool
    expected: str
    actual: str


class CheckpointLineageNode(BaseModel):
    """One checkpoint in a chain (``services/checkpoints/lineage.py:LineageNode.as_dict``)."""

    name: str
    run_id: str
    step: int
    domain: str | None
    domain_state: str
    domain_display: str
    parent: str | None
    base_model: str
    test: CheckpointScoreRecord | None
    merged_adapter: CheckpointMergedAdapter | None


class CheckpointGetLineageAnswer(BaseModel):
    """``GET /api/v1/runs/{run_id}/checkpoints/{step}/lineage``
    (``services/checkpoints/lineage.py:Lineage.as_dict``)."""

    run_id: str
    step: int
    nodes: list[CheckpointLineageNode]
    terminated: str
    terminated_at: str | None


# ---------------------------------------------------------------------------- continue, branch


class ContinueOptimizerDecision(BaseModel):
    """The trainer-state decision (``services/checkpoints/resume.py:ResumePlan.as_dict``)."""

    state: str
    reason: str


class ContinueConfigHashChange(BaseModel):
    """The optimizer hash recorded and configured
    (``services/checkpoints/resume.py:ResumePlan.as_dict``)."""

    recorded: str | None
    configured: str | None


class ContinueResumePlan(BaseModel):
    """A continue's or a branch's plan (``services/checkpoints/resume.py:ResumePlan.as_dict``)."""

    run_id: str
    source_run_id: str
    step: int
    directory: str
    name: str
    base_model: str
    parent: str
    optimizer: ContinueOptimizerDecision
    steps_with_trainer_state: list[int]
    config_hash_changed: ContinueConfigHashChange | None
    is_branch: bool
    weights_sha256: str


class ContinuePlanAnswer(BaseModel):
    """``POST /api/v1/runs/{run_id}/continue`` without ``submit``: the plan, 200
    (``api/v1/endpoints/checkpoints.py:continue_run``)."""

    plan: ContinueResumePlan
    run_status: str
    next_steps: list[NextStep]


class ContinueSubmittedAnswer(BaseModel):
    """``POST /api/v1/runs/{run_id}/continue`` with ``submit: true``: the submission, 202
    (``services/run_continue.py:submit_continue``,
    ``api/v1/endpoints/checkpoints.py:continue_run``).
    """

    plan: ContinueResumePlan
    submitted: bool
    resume: RunResumeRecord
    run: RunDefineRunAnswer
    run_status: str


class CheckpointBranchRunAnswer(BaseModel):
    """``POST /api/v1/runs/{run_id}/checkpoints/{step}/branch``
    (``api/v1/endpoints/checkpoints.py:branch_run``)."""

    plan: ContinueResumePlan
    next_steps: list[NextStep]


# ---------------------------------------------------------------------------- tree


class TreeResolvedEdge(BaseModel):
    """One edge's target and resolution (``schemas/tree.py:ResolvedEdge``, dumped as JSON)."""

    target: str | None
    state: str


class TreeForestEdge(BaseModel):
    """One edge of the forest (``schemas/tree.py:TreeEdge``, dumped as JSON)."""

    child: str
    parent: str
    edge: str
    state: str


class TreeForestNode(BaseModel):
    """One artifact in the forest (``schemas/tree.py:TreeNode``, dumped as JSON by
    ``services/reindex.py:checkpoint_tree``)."""

    name: str
    kind: str
    run_id: str
    step: int
    domain: str | None
    domain_state: str
    domain_display: str
    parent: TreeResolvedEdge | None
    base: TreeResolvedEdge | None
    base_manifest_sha256: str | None
    merged: TreeResolvedEdge | None
    merged_step: int | None
    merged_verification: str | None
    test: CheckpointScoreRecord | None
    manifest_sha256: str | None
    path: str
    run_indexed: bool
    index_stale: bool


class TreeForestPage(BaseModel):
    """The node page (``schemas/tree.py:TreePage``, dumped as JSON)."""

    limit: int
    offset: int
    total: int


class TreeGetTreeAnswer(BaseModel):
    """``GET /api/v1/tree`` (``services/tree.py:compose``
    via ``services/reindex.py:checkpoint_tree``)."""

    nodes: list[TreeForestNode]
    edges: list[TreeForestEdge]
    page: TreeForestPage
    index_stale: list[str]
