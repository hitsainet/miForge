"""The documentation twin of the selector evaluation's answer (feature 021, FT2).

``api/v1/endpoints/selectors.py:EvaluateResponse`` stays the route's enforced
``response_model``; its ``health`` and ``hashes`` fields, and each verdict's ``judge``, are
``dict[str, Any]`` there because the route dumps a model into each. This twin declares those
models, so the published contract types them; what FastAPI validates and serves does not change.
"""

from pydantic import BaseModel

from providers.base import ProviderIdentity
from schemas.answers_packs import PackSelectionHealth
from services.selectors.types import PublishedHashes


class SelectorVerdictDoc(BaseModel):
    """``api/v1/endpoints/selectors.py:VerdictOut``, with ``judge`` typed (``ProviderIdentity``)."""

    value: str
    selector: str
    reason_code: str
    reason: str | None
    score: float | None
    judge: ProviderIdentity | None
    caveats: list[str]
    match_mode: str | None
    match_metric: str | None
    review_item_id: str | None


class EvaluateResponseDoc(BaseModel):
    """``api/v1/endpoints/selectors.py:evaluate_completions``: ``EvaluateResponse``, with
    ``health`` (``SelectionHealth``, as ``PackSelectionHealth`` declares it) and ``hashes``
    (``PublishedHashes``) typed."""

    config_sha256: str
    config_kind: str
    hash_provenance: str
    verdicts: list[SelectorVerdictDoc]
    health: PackSelectionHealth
    hashes: PublishedHashes
