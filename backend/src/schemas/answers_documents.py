"""miForge's own published documents, as the answer contract references them (feature 021,
FR-021.9; the operator's choice (b), 2026-11-26).

Each ``docs/schemas`` document has an absolute ``$id`` (``https://hitsai.net/schemas/<file>``). An
answer that IS one of these documents declares the model built here, and a field that holds one (or
a part of one) declares that model as its type. The contract then carries a ``$ref`` to that URI
(``<$id>#<pointer>`` for a part), and its ``external`` section the file and sha256. Documentation
only: no route serves through these models (MF9).
"""

from __future__ import annotations

from pydantic import RootModel

from .answers_common import PublishedDocument, document_config

BASE = "https://hitsai.net/schemas/"

ADAPTER_MANIFEST = BASE + "adapter-manifest-v1.json"
BATCH_RECORD = BASE + "batch-record-v1.json"
CHECKPOINT_MANIFEST = BASE + "checkpoint-manifest-v1.json"
PACK_DEFINITION = BASE + "pack-definition-v1.json"
STEP_RECORD = BASE + "step-record-v1.json"
WSP_BUNDLE = BASE + "miforge-wsp-bundle-v1.json"
WSP_PATCH = BASE + "miforge-wsp-patch-v1.json"
WSP_RUN = BASE + "miforge-wsp-run-v1.json"


def part(schema_id: str, pointer: str) -> str:
    """A reference to one definition of a document: ``<$id>#<pointer>``."""
    return f"{schema_id}#{pointer}"


# -- whole documents ------------------------------------------------------------------------------
class PackDefinitionDocument(PublishedDocument):
    """``GET /packs/{pack_id}/versions/{version}/export`` (``packs.py:export_pack``): the pack
    document by value, the canonical bytes ``definition_sha256`` was taken over, served as an
    attachment."""

    model_config = document_config(PACK_DEFINITION)


class WspPatchDocument(PublishedDocument):
    """``GET /wsp/candidates/{id}/patch/patch.json`` (``wsp_candidates.py:get_patch``): the stored
    miforge.wsp-patch/v1 manifest, checked against the candidate's ``patch_sha256``."""

    model_config = document_config(WSP_PATCH)


class WspRunDocument(PublishedDocument):
    """A WSP run as miforge.wsp-run/v1 (``wsp/evaluate.py:document_of``): the live document."""

    model_config = document_config(WSP_RUN)


class WspRunSnapshot(PublishedDocument):
    """``GET /wsp/candidates/{id}/run-document?verdict_id=``
    (``wsp_candidates.py:get_run_document``): EXACTLY the bytes stored with that verdict, the run
    document without ``run_sha256`` and ``program_reruns`` (RFC 8785); ``X-Content-SHA256`` is its
    ``run_sha256``."""

    model_config = document_config(WSP_RUN, without=("program_reruns", "run_sha256"))


class CheckpointManifestDocument(PublishedDocument):
    """A checkpoint's manifest as written beside its weights (checkpoint-manifest-v1)."""

    model_config = document_config(CHECKPOINT_MANIFEST)


class AdapterManifestDocument(PublishedDocument):
    """An adapter's manifest as written beside its weights (adapter-manifest-v1)."""

    model_config = document_config(ADAPTER_MANIFEST)


class BatchRecordDocument(PublishedDocument):
    """One signal batch as the run log records it (batch-record-v1)."""

    model_config = document_config(BATCH_RECORD)


class WspBundleManifestDocument(PublishedDocument):
    """A WSP bundle's manifest as imported (miforge.wsp-bundle/v1)."""

    model_config = document_config(WSP_BUNDLE)


# -- parts of documents: a field that holds one definition of a published document ---------------
class OptimizerIdentityPart(PublishedDocument):
    """The adapter manifest's ``optimizer`` object."""

    model_config = document_config(part(ADAPTER_MANIFEST, "/$defs/OptimizerIdentity"))


class ScoreRecordPart(PublishedDocument):
    """A score record of the adapter manifest (``test`` or ``retention``)."""

    model_config = document_config(part(ADAPTER_MANIFEST, "/$defs/ScoreRecord"))


class ExternalProvenancePart(PublishedDocument):
    """An external part's provenance, as the batch record carries it."""

    model_config = document_config(part(BATCH_RECORD, "/$defs/ExternalProvenance"))


class BoundaryBlockPart(PublishedDocument):
    """A step record's boundary block, as the run log carries it."""

    model_config = document_config(part(STEP_RECORD, "/$defs/BoundaryBlock"))


class EvaluationBlockPart(PublishedDocument):
    """A step record's evaluation block, as the run log carries it."""

    model_config = document_config(part(STEP_RECORD, "/$defs/EvaluationBlock"))


class WspRunDocumentOrSnapshot(RootModel[WspRunDocument | WspRunSnapshot]):
    """``GET /wsp/candidates/{id}/run-document``: the live run document, or with ``?verdict_id=``
    the snapshot stored with that verdict."""
