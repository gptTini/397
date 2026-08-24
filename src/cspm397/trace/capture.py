from __future__ import annotations

import math
from dataclasses import dataclass

from cspm397.adapters import GenerationRequest, ModelAdapter, RouterStep

from .errors import InfTraceError, NaNTraceError, RouterError, TraceError
from .models import RouterMetadata, TraceCapture
from .projection import FixedRandomProjection


def _finite(value: float, *, what: str) -> float:
    converted = float(value)
    if math.isnan(converted):
        raise NaNTraceError(f"NaN in {what}")
    if math.isinf(converted):
        raise InfTraceError(f"Inf in {what}")
    return converted


@dataclass(frozen=True, slots=True)
class TraceCollector:
    adapter: ModelAdapter
    selected_layers: tuple[int, ...]
    signature_dim: int
    projection_seed: int = 397
    capture_router: bool = False
    router_weight_sum_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not self.selected_layers or len(set(self.selected_layers)) != len(
            self.selected_layers
        ):
            raise ValueError("selected_layers must be nonempty and unique")
        if any(
            layer < 0 or layer >= self.adapter.num_layers
            for layer in self.selected_layers
        ):
            raise ValueError("selected layer outside adapter layer range")
        if self.signature_dim < 1:
            raise ValueError("signature_dim must be >= 1")
        if not (0.0 < self.router_weight_sum_tolerance <= 0.1):
            raise ValueError("invalid router weight tolerance")

    def collect(
        self, request: GenerationRequest, *, sequence_id: int = 0
    ) -> TraceCapture:
        if sequence_id < 0:
            raise ValueError("sequence_id must be >= 0")
        projection = FixedRandomProjection(
            self.adapter.hidden_size, self.signature_dim, self.projection_seed
        )
        token_ids: list[int] = []
        sequence_ids: list[int] = []
        positions: list[int] = []
        signatures: list[tuple[tuple[float, ...], ...]] = []
        expert_ids: list[tuple[tuple[int, ...], ...]] = []
        expert_weights: list[tuple[tuple[float, ...], ...]] = []
        router_entropy: list[tuple[float, ...]] = []
        router_meta = self._router_metadata() if self.capture_router else None

        for position, step in enumerate(self.adapter.generate(request)):
            if len(token_ids) >= request.max_new_tokens:
                raise TraceError("adapter emitted more tokens than max_new_tokens")
            if len(step.hidden_states) != self.adapter.num_layers:
                raise TraceError("adapter hidden-state layer count mismatch")
            layer_signatures = tuple(
                projection.project(step.hidden_states[layer])
                for layer in self.selected_layers
            )
            token_ids.append(int(step.token_id))
            sequence_ids.append(sequence_id)
            positions.append(position)
            signatures.append(layer_signatures)
            if self.capture_router:
                if step.router is None:
                    raise RouterError(
                        "router trace requested but adapter emitted no router output"
                    )
                assert router_meta is not None
                ids, weights, entropy = self._validate_router_step(
                    step.router, router_meta
                )
                expert_ids.append(ids)
                expert_weights.append(weights)
                if entropy is not None:
                    router_entropy.append(entropy)
            if request.stop_on_eos and step.is_eos:
                break

        entropy_payload = None
        if self.capture_router and router_entropy:
            if len(router_entropy) != len(token_ids):
                raise RouterError(
                    "router entropy must be present for every token or none"
                )
            entropy_payload = tuple(router_entropy)

        return TraceCapture(
            token_ids=tuple(token_ids),
            sequence_ids=tuple(sequence_ids),
            positions=tuple(positions),
            state_signatures=tuple(signatures),
            selected_layers=self.selected_layers,
            signature_dim=self.signature_dim,
            projection_algorithm=projection.algorithm,
            projection_seed=projection.seed,
            input_dim=projection.input_dim,
            expert_ids=tuple(expert_ids) if self.capture_router else None,
            expert_weights=tuple(expert_weights) if self.capture_router else None,
            router_entropy=entropy_payload,
            router=router_meta,
        )

    def _router_metadata(self) -> RouterMetadata:
        num_experts = self.adapter.num_experts
        top_k = self.adapter.router_top_k
        if num_experts is None or top_k is None:
            raise RouterError("adapter does not declare router metadata")
        return RouterMetadata(
            num_experts,
            top_k,
            self.router_weight_sum_tolerance,
        )

    def _validate_router_step(
        self,
        router: RouterStep,
        metadata: RouterMetadata,
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[float, ...], ...],
        tuple[float, ...] | None,
    ]:
        if (
            len(router.expert_ids) != self.adapter.num_layers
            or len(router.expert_weights) != self.adapter.num_layers
        ):
            raise RouterError("router layer count mismatch")
        if (
            router.entropy is not None
            and len(router.entropy) != self.adapter.num_layers
        ):
            raise RouterError("router entropy layer count mismatch")
        ids_out: list[tuple[int, ...]] = []
        weights_out: list[tuple[float, ...]] = []
        entropy_out: list[float] = []
        for layer in self.selected_layers:
            ids = tuple(int(value) for value in router.expert_ids[layer])
            weights = tuple(
                _finite(value, what="router weights")
                for value in router.expert_weights[layer]
            )
            if len(ids) != metadata.top_k or len(weights) != metadata.top_k:
                raise RouterError("router top_k shape mismatch")
            if any(value < 0 or value >= metadata.num_experts for value in ids):
                raise RouterError("expert id outside declared range")
            if any(value < 0.0 for value in weights):
                raise RouterError("negative router weight")
            if abs(sum(weights) - 1.0) > metadata.weight_sum_tolerance:
                raise RouterError("router weights do not sum to one within tolerance")
            ids_out.append(ids)
            weights_out.append(weights)
            if router.entropy is not None:
                entropy_out.append(
                    _finite(router.entropy[layer], what="router entropy")
                )
        return (
            tuple(ids_out),
            tuple(weights_out),
            tuple(entropy_out) if router.entropy is not None else None,
        )
