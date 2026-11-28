"""``in_container/`` runs only INSIDE the worker image, started by
``test_wsp_worker_image.py::test_in_container_suite``; the host never collects it, so the image
job's output carries no skip at all (its ``MIFORGE_REQUIRE_WSP_IMAGE=1`` makes every other skip a
failure)."""

from __future__ import annotations

collect_ignore = ["in_container"]
