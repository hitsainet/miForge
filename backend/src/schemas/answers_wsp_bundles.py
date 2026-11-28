"""The answers of the WSP bundle routes (feature 021, P3): ``api/v1/endpoints/wsp_bundles.py``.

Documentation only (MF9): the routes still return their dicts; these models describe them.
"""

from pydantic import BaseModel, Field

from .answers_documents import WspBundleManifestDocument


class WspBundleImportFailure(BaseModel):
    """``import_failure`` of a failed import (``wsp/bundles.py:run_import``)."""

    code: str = Field(description="The refusal's code (for example wsp_bundle_invalid).")
    message: str = Field(description="One line, at most 1000 characters.")


class WspBundleDocument(BaseModel):
    """One bundle row (``wsp_bundles.py:document``), as list, get and import show it."""

    bundle_sha256: str = Field(description="The bundle's RFC 8785 sha256 (its id here).")
    owner: str
    bundle_id: str
    bundle_kind: str = Field(description="linear_probe or sae (linear_probe while importing).")
    artifact_kind: str = Field(description="probe while importing, then the manifest's.")
    artifact_id: str = Field(description="Empty while importing.")
    model: str = Field(description="Empty while importing.")
    layer: int = Field(description="0 while importing.")
    sensitivity: str = Field(description="The bundle's sensitivity mark (unknown while importing).")
    bytes: int = Field(description="The total size of the bundle's files (0 while importing).")
    import_status: str = Field(description="importing, imported or failed.")
    import_failure: WspBundleImportFailure | None = Field(
        description="Why the import failed; null unless failed."
    )
    received_at: str = Field(description="When the import was (last) requested, ISO 8601.")
    received_by: str = Field(description="The agent that requested it, or operator.")


class WspBundleImportBundleAnswer(WspBundleDocument):
    """``POST /wsp/bundles/import`` 202 (``wsp_bundles.py:import_bundle``)."""

    queued: bool = Field(description="True when this call queued the pull task.")


class WspBundleListBundlesAnswer(BaseModel):
    """``GET /wsp/bundles`` (``wsp_bundles.py:list_bundles``)."""

    bundles: list[WspBundleDocument] = Field(description="Newest received first.")


class WspBundleGetBundleAnswer(WspBundleDocument):
    """``GET /wsp/bundles/{bundle_sha256}`` (``wsp_bundles.py:get_bundle``)."""

    manifest: WspBundleManifestDocument | None = Field(
        description="The bundle's manifest as imported; null until the import completes."
    )
