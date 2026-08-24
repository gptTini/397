from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 16
    seed: int = 397
    stop_on_eos: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be str")
        if isinstance(self.max_new_tokens, bool) or not isinstance(
            self.max_new_tokens, int
        ):
            raise TypeError("max_new_tokens must be int")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be int")


@dataclass(frozen=True, slots=True)
class RouterStep:
    expert_ids: tuple[tuple[int, ...], ...]
    expert_weights: tuple[tuple[float, ...], ...]
    entropy: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class TokenStep:
    token_id: int
    hidden_states: tuple[tuple[float, ...], ...]
    router: RouterStep | None = None
    is_eos: bool = False


@runtime_checkable
class ModelAdapter(Protocol):
    model_id: str
    model_revision: str
    num_layers: int
    hidden_size: int
    device: str
    eos_token_id: int | None
    num_experts: int | None
    router_top_k: int | None

    def generate(self, request: GenerationRequest) -> Iterable[TokenStep]: ...
