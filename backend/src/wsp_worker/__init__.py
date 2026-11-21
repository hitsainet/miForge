"""The WSP worker: UNTRUSTED code's home (feature 016; FTDD §6.1, §8.1).

Installed alone in the worker image (``backend/Dockerfile.wsp-worker``), never in the backend's. It
imports nothing from ``wsp``, ``api``, ``db``, ``models``, ``services``, ``training``, ``workers``
or
``clients`` (an AST fence in ``test_wsp_fences.py``), and only ``adapters/`` imports
``weightspace``.
``limits.py`` is a byte-identical copy of ``wsp/weight_ops/limits.py`` (PF-40, pinned).
"""
