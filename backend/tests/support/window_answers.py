"""Recorded window answers of the generic servers 019 reads (FTASKS 1.1; FR-019.1; TID §3.2).

**HAND-WRITTEN FROM EACH PROJECT'S PUBLISHED SOURCE, NOT CAPTURED FROM A LIVE SERVER.** No vLLM,
llama.cpp or SGLang server was reachable without touching the cluster, so each body below is the
shape the pinned source serialises, with illustrative values. The sources, read 2026-11-24 (US
Eastern), are recorded with their files and lines in ``0xcc/reviews/019_window_readers.md``:

- vLLM ``v0.34.0``: ``GET /v1/models`` (``vllm/entrypoints/openai/models/api_router.py``) is
  ``ModelList.model_dump()``; each base model's ``ModelCard`` carries ``max_model_len`` =
  ``model_config.max_model_len`` (``models/serving.py``, ``show_available_models``), and its
  ``id`` is the served model name (``--served-model-name``, else ``--model``;
  ``launchers/api_server/app_state.py``). A LoRA adapter's card has ``max_model_len: null``.
- SGLang ``v0.5.23``: ``GET /v1/models`` (``srt/entrypoints/http_server.py``,
  ``available_models``) gives ``max_model_len`` = ``model_config.context_len``; LoRA cards null.
- llama.cpp ``v1.0.0``: ``GET /props`` (``tools/server/server-context.cpp``, ``get_res_props``)
  gives ``default_generation_settings.n_ctx`` = the SLOT's context (``n_ctx_slot()``: the
  per-sequence context, capped by ``--kv-unified-per-slot`` and by the training context), the
  figure one request may use. Its ``/v1/models`` carries no ``max_model_len`` (``meta.n_ctx`` is
  the same slot figure, ``meta.n_ctx_train`` the training length, which is never used). In router
  mode (several models) ``/props`` without ``?model=`` answers ``n_ctx: 0``, and ``/v1/models``
  lists ``context_length`` = the TRAINING length: neither is a window, so both read as nothing.
"""

from __future__ import annotations

from typing import Any

VLLM_MODEL = "Qwen/Qwen2.5-32B-Instruct"
VLLM_WINDOW = 32_768

#: vLLM v0.34.0 ``GET /v1/models``: one base model and one LoRA adapter.
VLLM_MODELS: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "id": VLLM_MODEL,
            "object": "model",
            "created": 1764031200,
            "owned_by": "vllm",
            "root": "/models/Qwen2.5-32B-Instruct",
            "parent": None,
            "max_model_len": VLLM_WINDOW,
            "permission": [
                {
                    "id": "modelperm-1f2e3d4c5b6a",
                    "object": "model_permission",
                    "created": 1764031200,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False,
                }
            ],
        },
        {
            "id": "probe-editor-lora",
            "object": "model",
            "created": 1764031200,
            "owned_by": "vllm",
            "root": "/adapters/probe-editor",
            "parent": VLLM_MODEL,
            "max_model_len": None,
            "permission": [],
        },
    ],
}

SGLANG_MODEL = "qwen3-32b"
SGLANG_WINDOW = 40_960

#: SGLang v0.5.23 ``GET /v1/models`` (no LoRA loaded).
SGLANG_MODELS: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "id": SGLANG_MODEL,
            "object": "model",
            "created": 1764031200,
            "owned_by": "sglang",
            "root": SGLANG_MODEL,
            "parent": None,
            "max_model_len": SGLANG_WINDOW,
        }
    ],
}

LLAMACPP_MODEL = "qwen2.5-32b-instruct-q4_k_m"
#: The slot's context: ``-c 32768 --parallel 4`` without a unified cache gives 8,192 per slot.
LLAMACPP_SLOT_WINDOW = 8_192
LLAMACPP_TRAIN_CONTEXT = 32_768

#: llama.cpp v1.0.0 ``GET /v1/models`` (single-model mode): no ``max_model_len``.
LLAMACPP_MODELS: dict[str, Any] = {
    "models": [
        {
            "name": LLAMACPP_MODEL,
            "model": LLAMACPP_MODEL,
            "modified_at": "",
            "size": "",
            "digest": "",
            "type": "model",
            "description": "",
            "tags": [""],
            "capabilities": ["completion"],
            "parameters": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "",
                "families": [""],
                "parameter_size": "",
                "quantization_level": "",
            },
        }
    ],
    "object": "list",
    "data": [
        {
            "id": LLAMACPP_MODEL,
            "aliases": [],
            "tags": [],
            "object": "model",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "created": 1764031200,
            "owned_by": "llamacpp",
            "meta": {
                "vocab_type": 2,
                "n_vocab": 152064,
                "n_ctx": LLAMACPP_SLOT_WINDOW,
                "n_ctx_train": LLAMACPP_TRAIN_CONTEXT,
                "n_embd": 5120,
                "n_params": 32763876352,
                "size": 19851336704,
                "ftype": "Q4_K - Medium",
            },
        }
    ],
}

#: llama.cpp v1.0.0 ``GET /props`` (single-model mode, four slots).
LLAMACPP_PROPS: dict[str, Any] = {
    "default_generation_settings": {
        "params": {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "n_predict": -1},
        "n_ctx": LLAMACPP_SLOT_WINDOW,
    },
    "total_slots": 4,
    "model_alias": LLAMACPP_MODEL,
    "model_ftype": "Q4_K - Medium",
    "model_path": "/models/qwen2.5-32b-instruct-q4_k_m.gguf",
    "modalities": {"vision": False, "video": False, "audio": False},
    "media_marker": "<__media_x__>",
    "endpoint_slots": True,
    "endpoint_props": False,
    "endpoint_metrics": False,
    "ui": True,
    "ui_settings": {},
    "chat_template": "{{ messages }}",
    "chat_template_caps": {},
    "bos_token": "",
    "eos_token": "<|im_end|>",
    "build_info": "b1000-a6d1a74d",
    "is_sleeping": False,
    "cors_proxy_enabled": False,
}

#: llama.cpp v1.0.0 router mode, ``GET /props`` without ``?model=``: a placeholder ``n_ctx`` of 0.
LLAMACPP_ROUTER_PROPS: dict[str, Any] = {
    "role": "router",
    "max_instances": 4,
    "models_autoload": True,
    "model_alias": "llama-server",
    "model_path": "none",
    "default_generation_settings": {"params": {}, "n_ctx": 0},
    "ui_settings": {},
    "build_info": "b1000-a6d1a74d",
    "cors_proxy_enabled": False,
}

#: llama.cpp v1.0.0 router mode, ``GET /v1/models``: ``context_length`` is the TRAINING length.
LLAMACPP_ROUTER_MODELS: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "id": LLAMACPP_MODEL,
            "aliases": [],
            "tags": [],
            "object": "model",
            "owned_by": "llamacpp",
            "created": 1764031200,
            "status": {"value": "loaded"},
            "context_length": LLAMACPP_TRAIN_CONTEXT,
        }
    ],
}
