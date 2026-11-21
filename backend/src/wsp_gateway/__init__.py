"""The WSP gateway (feature 016; FR-016.15; FTDD §5.5): the worker's only way out.

Its own Deployment (``miforge-wsp-gateway``), run from the backend image as
``python -m wsp_gateway``. It has no database, Redis or data volume (a fence and a manifest test
pin that): spend lives in the backend, which every request reserves against and settles to through
the internal application (``api/wsp_internal_app.py``). It holds the only private key that opens
an endpoint's sealed API key (D15).
"""
