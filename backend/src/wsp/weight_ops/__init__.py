"""Deterministic weight operations.

Hashing, the function evaluator, inspect, patches, stacking and limits.

Pure numpy and the standard library (plus ``rfc8785`` for the RFC 8785 digests). **No LLM anywhere
in
this layer (D5)**, no HTTP and no ``wsp_worker`` import: ``test_wsp_fences.py::
test_weight_ops_imports_no_io`` walks every module's imports.
"""
