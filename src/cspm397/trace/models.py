from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import InfTraceError, NaNTraceError, TraceError


@dataclass(frozen=True, slots=True)
class RouterMetadata:
    num_experts: int
    top_k: int
    weight_sum_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.num_experts < 1 or not (1 <= self.top_k <= self.num_experts):
            raise ValueError("invalid router metadata")
        if not (0.0 < self.weight_sum_tolerance <= 0.1):
            raise ValueError("invalid router weight tolerance")


@dataclass(frozen=True, slots=True)
class TraceCapture:
    token_ids: tuple[int, ...]
    sequence_ids: tuple[int, ...]
    positions: tuple[int, ...]
    state_signatures: tuple[tuple[tuple[float, ...], ...], ...]
    selected_layers: tuple[int, ...]
    signature_dim: int
    projection_algorithm: str
    projection_seed: int
    input_dim: int
    expert_ids: tuple[tuple[tuple[int, ...], ...], ...] | None = None
    expert_weights: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    router_entropy: tuple[tuple[float, ...], ...] | None = None
    router: RouterMetadata | None = None

    def __post_init__(self) -> None:
        token_count = len(self.token_ids)
        if (
            len(self.sequence_ids) != token_count
            or len(self.positions) != token_count
            or len(self.state_signatures) != token_count
        ):
            raise TraceError("T-axis cardinality mismatch")
        if self.signature_dim < 1 or not self.selected_layers:
            raise TraceError("invalid trace dimensions")
        expected_layers = len(self.selected_layers)
        for token in self.state_signatures:
            if len(token) != expected_layers or any(
                len(vector) != self.signature_dim for vector in token
            ):
                raise TraceError("state signature shape mismatch")
            for vector in token:
                for value in vector:
                    if math.isnan(value):
                        raise NaNTraceError("NaN in state signatures")
                    if math.isinf(value):
                        raise InfTraceError("Inf in state signatures")
        router_present = self.router is not None
        arrays_present = (
            self.expert_ids is not None
            or self.expert_weights is not None
            or self.router_entropy is not None
        )
        if router_present != arrays_present:
            raise TraceError("router metadata/payload presence mismatch")
        if self.router is not None:
            if self.expert_ids is None or self.expert_weights is None:
                raise TraceError("router trace requires expert ids and weights")
            if (
                len(self.expert_ids) != token_count
                or len(self.expert_weights) != token_count
            ):
                raise TraceError("router T-axis cardinality mismatch")

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_sequences(self) -> int:
        return len(set(self.sequence_ids))

    def sequence_stats(self) -> dict[str, int | bool | None]:
        if not self.sequence_ids:
            return {
                "unique_count": 0,
                "min_id": None,
                "max_id": None,
                "ids_contiguous_zero_based": True,
                "positions_zero_based_contiguous": True,
                "sequences_interleaved": False,
            }
        unique = sorted(set(self.sequence_ids))
        ids_ok = unique == list(range(len(unique)))
        positions_ok = True
        interleaved = False
        last_sequence: int | None = None
        closed: set[int] = set()
        expected_position: dict[int, int] = {}
        for seq, pos in zip(self.sequence_ids, self.positions):
            if last_sequence is not None and seq != last_sequence:
                closed.add(last_sequence)
                if seq in closed:
                    interleaved = True
            expected = expected_position.get(seq, 0)
            if pos != expected:
                positions_ok = False
            expected_position[seq] = expected + 1
            last_sequence = seq
        return {
            "unique_count": len(unique),
            "min_id": unique[0],
            "max_id": unique[-1],
            "ids_contiguous_zero_based": ids_ok,
            "positions_zero_based_contiguous": positions_ok,
            "sequences_interleaved": interleaved,
        }

    def array_shapes(self) -> dict[str, list[int]]:
        t = self.num_tokens
        l = len(self.selected_layers)
        result = {
            "token_ids": [t],
            "sequence_ids": [t],
            "positions": [t],
            "state_signatures": [t, l, self.signature_dim],
        }
        if self.router is not None:
            result["expert_ids"] = [t, l, self.router.top_k]
            result["expert_weights"] = [t, l, self.router.top_k]
            if self.router_entropy is not None:
                result["router_entropy"] = [t, l]
        return result

    def array_dtypes(
        self, *, float_dtype: str = "float32", int_dtype: str = "int64"
    ) -> dict[str, str]:
        result = {
            "token_ids": int_dtype,
            "sequence_ids": int_dtype,
            "positions": int_dtype,
            "state_signatures": float_dtype,
        }
        if self.router is not None:
            result["expert_ids"] = int_dtype
            result["expert_weights"] = float_dtype
            if self.router_entropy is not None:
                result["router_entropy"] = float_dtype
        return result

    def slice_tokens(self, start: int, end: int) -> TraceCapture:
        if not (0 <= start <= end <= self.num_tokens):
            raise ValueError("invalid token slice")
        return TraceCapture(
            token_ids=self.token_ids[start:end],
            sequence_ids=self.sequence_ids[start:end],
            positions=self.positions[start:end],
            state_signatures=self.state_signatures[start:end],
            selected_layers=self.selected_layers,
            signature_dim=self.signature_dim,
            projection_algorithm=self.projection_algorithm,
            projection_seed=self.projection_seed,
            input_dim=self.input_dim,
            expert_ids=self.expert_ids[start:end]
            if self.expert_ids is not None
            else None,
            expert_weights=self.expert_weights[start:end]
            if self.expert_weights is not None
            else None,
            router_entropy=self.router_entropy[start:end]
            if self.router_entropy is not None
            else None,
            router=self.router,
        )

    def iter_chunks(self, max_tokens: int):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        for start in range(0, self.num_tokens, max_tokens):
            yield self.slice_tokens(start, min(self.num_tokens, start + max_tokens))
