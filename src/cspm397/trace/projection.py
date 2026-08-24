from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .errors import InfTraceError, NaNTraceError, TraceError


def _check_finite(value: float) -> None:
    if math.isnan(value):
        raise NaNTraceError("NaN encountered in hidden state")
    if math.isinf(value):
        raise InfTraceError("Inf encountered in hidden state")


def _sign(seed: int, output_index: int, input_index: int) -> float:
    payload = f"CSPM-RP-v1|{seed}|{output_index}|{input_index}".encode("ascii")
    return 1.0 if hashlib.sha256(payload).digest()[0] & 1 else -1.0


@dataclass(frozen=True, slots=True)
class FixedRandomProjection:
    input_dim: int
    output_dim: int
    seed: int = 397
    algorithm: str = "fixed_seeded_rademacher_v1"

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("projection dimensions must be >= 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("projection seed must be int")

    def project(self, vector: tuple[float, ...] | list[float]) -> tuple[float, ...]:
        if len(vector) != self.input_dim:
            raise TraceError(
                f"hidden state dimension mismatch: expected {self.input_dim}, got {len(vector)}"
            )
        values = tuple(float(value) for value in vector)
        for value in values:
            _check_finite(value)
        scale = 1.0 / math.sqrt(self.output_dim)
        projected = tuple(
            sum(
                _sign(self.seed, out_index, in_index) * value
                for in_index, value in enumerate(values)
            )
            * scale
            for out_index in range(self.output_dim)
        )
        for value in projected:
            _check_finite(value)
        return projected
