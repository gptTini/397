from __future__ import annotations


class TraceError(RuntimeError):
    code = "E220"


class RouterError(TraceError):
    code = "E230"


class NumericTraceError(TraceError):
    code = "E500"


class NaNTraceError(NumericTraceError):
    code = "E510"


class InfTraceError(NumericTraceError):
    code = "E520"
