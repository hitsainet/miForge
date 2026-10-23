"""``rebuild.build`` consumes miLLM's REAL attach layout  (R-02.1; FR-011.2, FR-011.3).

Found on the GPU node on 2026-10-17, before the first live call (F11 14.3): ``build`` read a
``handle`` off every *parameter* entry, while miLLM puts the CUDA IPC handles on
``storages[]`` (one per storage) and makes each parameter a view naming its storage
(``millm/services/weight_handles.py``). Every caller in this suite stubbed ``build``, and
the fake surface used the invented layout, so nothing ever ran the consumption against the
real shape. Against miLLM, every parameter would have failed.

These tests drive the real ``build`` on the CPU through its two seams - the storage
rebuilder and the UUID-to-index resolution - so the layout logic itself runs: one rebuild
per storage, views with the stated shape/stride/offset/dtype, tied parameters sharing one
storage, ``requires_grad`` on miForge's side only, and a refusal naming every parameter that
could not be rebuilt.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from core.errors import Refusal  # noqa: E402
from training.hotupdate import rebuild  # noqa: E402

UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"


def _storage_entry(index: int, size: int) -> dict[str, Any]:
    """One entry exactly as miLLM's ``weight_handles.py`` emits it (fields and names)."""
    return {
        "index": index,
        "device_index": 0,
        "device_uuid": UUID,
        "dtype": "bfloat16",
        "handle": "aGFuZGxl",
        "storage_size_bytes": size,
        "storage_offset_bytes": 0,
        "ref_counter_handle": "cmVm",
        "ref_counter_offset": 0,
        "event_handle": None,
        "event_sync_required": False,
    }


def _param(name: str, storage: int, shape: list[int], offset: int = 0) -> dict[str, Any]:
    stride = [shape[1], 1] if len(shape) == 2 else [1]
    return {
        "name": name,
        "storage": storage,
        "shape": shape,
        "stride": stride,
        "storage_offset": offset,
        "dtype": "bfloat16",
        "requires_grad": False,
    }


def _response() -> dict[str, Any]:
    """Two storages: a TIED embedding (two parameters, one storage) and a norm weight that
    sits at a non-zero element offset inside the second storage."""
    return {
        "attachment_id": "never-printed",
        "storages": [_storage_entry(0, 16 * 8 * 2), _storage_entry(1, 12 * 2)],
        "parameters": [
            _param("model.embed_tokens.weight", 0, [16, 8]),
            _param("lm_head.weight", 0, [16, 8]),
            _param("model.norm.weight", 1, [8], offset=4),
        ],
    }


class _Rebuilder:
    """Stands in for ``rebuild_cuda_tensor``: a CPU storage per call, counted, with known
    contents so a view's offset is observable."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, entry: dict[str, Any], device_index: int) -> Any:
        self.calls.append((entry["index"], device_index))
        count = entry["storage_size_bytes"] // 2
        return torch.arange(count, dtype=torch.bfloat16).untyped_storage()


def _build(**kwargs: Any) -> tuple[Any, _Rebuilder]:
    rebuilder = _Rebuilder()
    kwargs.setdefault("rebuild_storage", rebuilder)
    kwargs.setdefault("local_device_index", lambda uuid: 0 if uuid == UUID else None)
    return rebuild.build(kwargs.pop("response", _response()), **kwargs), rebuilder


class TestMiLLMsLayout:
    def test_each_storage_is_rebuilt_exactly_once(self) -> None:
        """The tied pair names one storage: ONE rebuild, or the IPC refcounts unbalance and
        the storage is never freed (miLLM's manual, measured on the 3090)."""
        _, rebuilder = _build()
        assert rebuilder.calls == [(0, 0), (1, 0)]

    def test_tied_parameters_share_one_allocation(self) -> None:
        base, _ = _build()
        embed = base.tensors["model.embed_tokens.weight"]
        head = base.tensors["lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr()
        assert base.shared_storages == {"0": ["model.embed_tokens.weight", "lm_head.weight"]}

    def test_views_carry_the_stated_shape_stride_offset_and_dtype(self) -> None:
        base, _ = _build()
        norm = base.tensors["model.norm.weight"]
        assert tuple(norm.shape) == (8,)
        assert norm.dtype == torch.bfloat16
        # Offset 4 elements into a storage holding 0..11: the view starts at 4.
        assert norm.detach().float().tolist() == [4.0, 5, 6, 7, 8, 9, 10, 11]
        embed = base.tensors["model.embed_tokens.weight"]
        assert tuple(embed.shape) == (16, 8) and embed.stride() == (8, 1)

    def test_requires_grad_is_set_on_miforges_side_for_a_change_run(self) -> None:
        base, _ = _build(read_only=False, requires_grad=True)
        assert all(t.requires_grad for t in base.tensors.values())

    def test_a_read_only_base_never_takes_a_gradient(self) -> None:
        base, _ = _build(read_only=True, requires_grad=False)
        assert not any(t.requires_grad for t in base.tensors.values())
        assert base.read_only is True

    def test_the_device_is_named_by_uuid(self) -> None:
        base, _ = _build()
        assert base.device_uuid == UUID


class TestRefusals:
    def test_a_device_this_process_cannot_see_is_refused_by_name(self) -> None:
        with pytest.raises(Refusal) as caught:
            _build(local_device_index=lambda uuid: None)
        assert caught.value.code == "weights_rebuild_failed"
        assert caught.value.details["rebuilt"] == 0
        assert "cannot see" in next(iter(caught.value.details["failed"].values()))

    def test_a_parameter_naming_a_missing_storage_is_refused(self) -> None:
        response = _response()
        response["parameters"].append(_param("model.extra.weight", 7, [2]))
        with pytest.raises(Refusal) as caught:
            _build(response=response)
        assert caught.value.code == "weights_rebuild_failed"
        assert "model.extra.weight" in caught.value.details["failed"]

    def test_a_response_with_no_storages_is_incomplete(self) -> None:
        response = _response()
        del response["storages"]
        with pytest.raises(Refusal) as caught:
            _build(response=response)
        assert caught.value.code == "attach_response_incomplete"

    def test_the_old_per_parameter_handle_layout_is_not_read(self) -> None:
        """The invented layout (a handle on each parameter, no storages) is refused rather
        than half-read."""
        response = {
            "parameters": [{"name": "w", "storage_key": "a", "handle": {"fake": "h"}, "shape": [1]}]
        }
        with pytest.raises(Refusal) as caught:
            _build(response=response)
        assert caught.value.code == "attach_response_incomplete"


class TestTheFakeSurfaceUsesTheSameLayout:
    def test_the_fakes_attach_body_rebuilds_through_the_real_build(self) -> None:
        """The fake and the reader can no longer agree on a private layout: the fake's own
        attach payload goes through ``build`` here."""
        import httpx

        from tests.support.fake_weights_surface import FakeWeightsSurface

        fake = FakeWeightsSurface()
        http = httpx.Client(transport=fake.transport(), base_url="http://fake")
        payload = http.post(
            "/api/weights/attach", headers={"Authorization": f"Bearer {fake.valid_token}"}
        ).json()["data"]
        base = rebuild.build(
            payload,
            rebuild_storage=lambda entry, index: torch.zeros(
                entry["storage_size_bytes"] // 2, dtype=torch.bfloat16
            ).untyped_storage(),
            local_device_index=lambda uuid: 0,
        )
        assert set(base.tensors) == {"model.embed_tokens.weight", "lm_head.weight"}
        assert base.shared_storages == {"0": ["model.embed_tokens.weight", "lm_head.weight"]}
