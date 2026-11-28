"""The answers of the host and acquisition reads (feature 021, P3): ``api/v1/endpoints/host.py``,
``devices.py``, ``models_local.py``, ``tokenizers.py`` and ``attachment.py``.

Documentation only (MF9): the routes keep ``response_model=None`` and serve what they build.
"""

from pydantic import BaseModel

from .answers_common import may_be_absent

# ----------------------------------------------------------------- GET /api/v1/host/resources


class HostCpu(BaseModel):
    """``core/host_metrics.py:PsutilNvidiaCollector._cpu``."""

    percent: float
    logical_cores: int | None


class HostMemory(BaseModel):
    """``core/host_metrics.py:PsutilNvidiaCollector.read``: memory used of total."""

    used_bytes: int
    total_bytes: int


class HostGpu(BaseModel):
    """``core/gpu.py:Telemetry`` (``telemetry_by_uuid``): one card's live readings, by UUID."""

    uuid: str
    name: str
    utilization_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    temperature_c: float | None


class HostUnavailable(BaseModel):
    """``core/host_metrics.py:PsutilNvidiaCollector.read``: a source that could not be read."""

    metric: str
    code: str
    message: str


class HostDiskVolume(BaseModel):
    """``core/host_metrics.py:read_volume``: one volume this pod writes (null figures when it
    cannot be read; the reason is in ``unavailable``)."""

    name: str
    path: str
    free_bytes: int | None
    total_bytes: int | None
    device: int | None


class HostWspWorkerDisk(BaseModel):
    """``wsp/pool.py:worker_disk``: one WSP worker's ``/work`` as its last ``/ready`` reported it
    (null figures until it reports them)."""

    index: int
    boot_id: str | None
    free_bytes: int | None
    total_bytes: int | None
    reported_at: str | None


class HostDisk(BaseModel):
    """``api/v1/endpoints/host.py:host_resources``: the collector's volumes and the WSP workers'
    disk figures."""

    volumes: list[HostDiskVolume]
    wsp_work_limit_bytes: int
    wsp_workers: list[HostWspWorkerDisk]


class HostHostResourcesAnswer(BaseModel):
    """``api/v1/endpoints/host.py:host_resources`` (``PsutilNvidiaCollector.read``)."""

    observed_at: str
    cpu: HostCpu | None
    memory: HostMemory | None
    gpus: list[HostGpu]
    disk: HostDisk
    unavailable: list[HostUnavailable]


# ----------------------------------------------------------------------------- GET /api/v1/gpus


class DeviceRow(BaseModel):
    """``api/v1/endpoints/devices.py:device_row``: one card, by UUID, with no index."""

    uuid: str
    name: str
    free_memory_bytes: int | None


class DeviceUnavailable(BaseModel):
    """``api/v1/endpoints/devices.py:list_gpus``: why the driver could not be read."""

    code: str
    message: str


class DeviceListGpusAnswer(BaseModel):
    """``api/v1/endpoints/devices.py:list_gpus``."""

    devices: list[DeviceRow]
    available: bool
    unavailable: DeviceUnavailable | None
    auto: bool


# ---------------------------------------------------------------------- GET /api/v1/models/local


class LocalModelRow(BaseModel):
    """``api/v1/endpoints/models_local.py:_describe``: one model directory."""

    path: str
    root: str
    relative_path: str
    kind: str
    architecture: str | None
    has_weights: bool
    has_tokenizer: bool
    config_path: str
    modified_at: str


class LocalModelUnreadableRoot(BaseModel):
    """``api/v1/endpoints/models_local.py:list_local_models``: a root that was not read."""

    root: str
    code: str
    reason: str


class LocalModelListLocalModelsAnswer(BaseModel):
    """``api/v1/endpoints/models_local.py:list_local_models``."""

    models: list[LocalModelRow]
    total: int
    limit: int
    offset: int
    roots: list[str]
    unreadable_roots: list[LocalModelUnreadableRoot]
    max_depth: int


# ---------------------------------------------------- GET /api/v1/tokenizers/local, /hub/revision


class TokenizerRow(BaseModel):
    """``api/v1/endpoints/tokenizers.py:_describe``: one directory with tokenizer files."""

    path: str
    root: str
    relative_path: str
    kind: str
    tokenizer_files: list[str]
    modified_at: str


class TokenizerUnreadableRoot(BaseModel):
    """``api/v1/endpoints/tokenizers.py:list_local_tokenizers``: a root that was not read."""

    root: str
    code: str
    reason: str


class TokenizerListLocalTokenizersAnswer(BaseModel):
    """``api/v1/endpoints/tokenizers.py:list_local_tokenizers``."""

    tokenizers: list[TokenizerRow]
    total: int
    limit: int
    offset: int
    roots: list[str]
    unreadable_roots: list[TokenizerUnreadableRoot]
    max_depth: int


class TokenizerCheckHubRevisionAnswer(BaseModel):
    """``api/v1/endpoints/tokenizers.py:check_hub_revision``."""

    repo: str
    revision: str
    commit: str
    is_commit: bool
    tokenizer_files: list[str]
    has_tokenizer: bool
    has_config: bool
    has_weights: bool


# ------------------------------------------------------------------------ GET /api/v1/attachment


class AttachmentAttachmentStatusAnswer(BaseModel):
    """``api/v1/endpoints/attachment.py:attachment_status`` (``training/acquisition``
    ``attachment_status``, ``training/hotupdate/attach.py:live_status``,
    ``training/hotupdate/run_state.py:status_record``)."""

    configured: bool
    serving_base_url: str | None = may_be_absent(
        description="miServer's base URL; sent only when miServer is configured."
    )
    reachable: bool | None = may_be_absent(
        description="Whether the status read answered; sent only when miServer is configured."
    )
    reason: str | None = may_be_absent(
        description="The refusal code of the failed read; sent only when reachable is false."
    )
    observed_at: str | None = may_be_absent(
        description="When the read was made; sent only when miServer is configured."
    )
    attached: bool | None = may_be_absent(
        description="Whether anything holds the attachment; sent only when reachable is true."
    )
    weights_state: str | None = may_be_absent(
        description="miServer's weights_state, passed through (null when it sent none); sent "
        "only when reachable is true."
    )
    resident_model: str | None = may_be_absent(
        description="The resident model's name; sent only when reachable is true."
    )
    holder: str | None = may_be_absent(
        description="role:label of the holder; sent only when reachable is true."
    )
    stale: bool | None = may_be_absent(
        description="Whether the attachment is stale (null when nothing is attached); sent only "
        "when reachable is true."
    )
