from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .base import GenerationRequest, RouterStep, TokenStep


def _u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _unit_float(*parts: object) -> float:
    return (_u64(*parts) / float(2**64 - 1)) * 2.0 - 1.0


@dataclass(slots=True)
class MockModelAdapter:
    model_id: str = "mock/cspm397"
    model_revision: str = "mock-v1"
    num_layers: int = 4
    hidden_size: int = 8
    vocab_size: int = 257
    eos_token_id: int | None = 2
    device: str = "cpu"
    num_experts: int | None = None
    router_top_k: int | None = None
    eos_at_step: int | None = None

    def __post_init__(self) -> None:
        if self.num_layers < 1 or self.hidden_size < 1 or self.vocab_size < 3:
            raise ValueError("invalid mock model dimensions")
        if (self.num_experts is None) != (self.router_top_k is None):
            raise ValueError("num_experts and router_top_k must be configured together")
        if self.num_experts is not None and (
            self.num_experts < 1
            or self.router_top_k is None
            or not (1 <= self.router_top_k <= self.num_experts)
        ):
            raise ValueError("invalid router configuration")

    def generate(self, request: GenerationRequest):
        for step_index in range(request.max_new_tokens):
            forced_eos = self.eos_at_step is not None and step_index == self.eos_at_step
            token_id = (
                self.eos_token_id
                if forced_eos and self.eos_token_id is not None
                else _u64("token", request.prompt, request.seed, step_index)
                % self.vocab_size
            )
            hidden_states = tuple(
                tuple(
                    _unit_float(
                        "hidden", request.prompt, request.seed, step_index, layer, dim
                    )
                    for dim in range(self.hidden_size)
                )
                for layer in range(self.num_layers)
            )
            router = (
                self._router_step(request, step_index)
                if self.num_experts is not None
                else None
            )
            is_eos = self.eos_token_id is not None and token_id == self.eos_token_id
            yield TokenStep(
                token_id=int(token_id),
                hidden_states=hidden_states,
                router=router,
                is_eos=is_eos,
            )
            if request.stop_on_eos and is_eos:
                break

    def _router_step(self, request: GenerationRequest, step_index: int) -> RouterStep:
        assert self.num_experts is not None and self.router_top_k is not None
        ids_by_layer: list[tuple[int, ...]] = []
        weights_by_layer: list[tuple[float, ...]] = []
        entropy_by_layer: list[float] = []
        for layer in range(self.num_layers):
            ranked = sorted(
                range(self.num_experts),
                key=lambda expert: _u64(
                    "expert", request.prompt, request.seed, step_index, layer, expert
                ),
                reverse=True,
            )[: self.router_top_k]
            raw = [
                1.0
                + (
                    _u64(
                        "weight",
                        request.prompt,
                        request.seed,
                        step_index,
                        layer,
                        expert,
                    )
                    % 1000
                )
                for expert in ranked
            ]
            total = sum(raw)
            weights = tuple(value / total for value in raw)
            entropy = -sum(
                weight * __import__("math").log(weight)
                for weight in weights
                if weight > 0.0
            )
            ids_by_layer.append(tuple(ranked))
            weights_by_layer.append(weights)
            entropy_by_layer.append(entropy)
        return RouterStep(
            tuple(ids_by_layer), tuple(weights_by_layer), tuple(entropy_by_layer)
        )
