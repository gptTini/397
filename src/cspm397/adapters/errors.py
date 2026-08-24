from __future__ import annotations


class AdapterError(RuntimeError):
    code = "E200"

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class ModelLoadError(AdapterError):
    code = "E200"


class TokenizerError(AdapterError):
    code = "E210"


class DeviceError(AdapterError):
    code = "E420"
