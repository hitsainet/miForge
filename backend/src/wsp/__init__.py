"""Weight-space programming (WSP), feature 016: the trusted backend half.

**Fence (FTDD §6.1).** This package is imported only by the WSP routes, the WSP Celery tasks and the
internal application. Nothing in it imports ``weightspace`` or any other WSP library (the provider
runs
only in the worker image, ``wsp_worker``), an LLM client, or ``wsp_worker``. ``wsp.weight_ops`` is
narrower still: pure numpy, no HTTP, no LLM, no database (D5). ``test_wsp_fences.py`` enforces both
by
AST, so the fence is a test, not a convention.

No library type crosses a contract (WR-3): the documents this package writes are
``miforge.wsp-bundle/v1``, ``miforge.wsp-patch/v1`` and ``miforge.wsp-run/v1``.
"""
