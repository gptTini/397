from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import GenerationRequest, RouterStep, TokenStep
from .errors import DeviceError, ModelLoadError, TokenizerError

RouterExtractor = Callable[[Any, int], RouterStep | None]


@dataclass(slots=True)
class HuggingFaceAdapter:
    model: Any
    tokenizer: Any
    model_id: str
    model_revision: str
    num_layers: int
    hidden_size: int
    device: str = "cpu"
    eos_token_id: int | None = None
    num_experts: int | None = None
    router_top_k: int | None = None
    router_extractor: RouterExtractor | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str,
        device: str = "cpu",
        router_extractor: RouterExtractor | None = None,
    ) -> HuggingFaceAdapter:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ModelLoadError(
                "transformers/torch are required for HuggingFaceAdapter",
                reason="optional_dependency_unavailable",
            ) from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise DeviceError(
                "requested CUDA device is unavailable", reason="cuda_unavailable"
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        except Exception as exc:
            raise TokenizerError(
                "failed to load tokenizer", reason="tokenizer_load_failed"
            ) from exc
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
            model.to(device)
            model.eval()
        except Exception as exc:
            raise ModelLoadError(
                "failed to load model", reason="model_load_failed"
            ) from exc
        config = model.config
        num_layers = int(config.num_hidden_layers)
        hidden_size = int(config.hidden_size)
        eos = getattr(config, "eos_token_id", getattr(tokenizer, "eos_token_id", None))
        num_experts = getattr(
            config, "num_experts", getattr(config, "num_local_experts", None)
        )
        router_top_k = getattr(
            config, "num_experts_per_tok", getattr(config, "top_k", None)
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            model_revision=revision,
            num_layers=num_layers,
            hidden_size=hidden_size,
            device=device,
            eos_token_id=int(eos) if eos is not None else None,
            num_experts=int(num_experts) if num_experts is not None else None,
            router_top_k=int(router_top_k) if router_top_k is not None else None,
            router_extractor=router_extractor,
        )

    def generate(
        self, request: GenerationRequest
    ):  # pragma: no cover - optional dependency path
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise ModelLoadError(
                "torch is unavailable", reason="optional_dependency_unavailable"
            ) from exc
        try:
            encoded = self.tokenizer(request.prompt, return_tensors="pt")
        except Exception as exc:
            raise TokenizerError(
                "tokenization failed", reason="tokenize_failed"
            ) from exc
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if request.max_new_tokens == 0:
            return

        # The initial forward predicts the first generated token. Each subsequent
        # forward is performed *after* appending that token so the captured hidden
        # state and optional router output are aligned to the emitted token itself,
        # rather than to the previous context token.
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        for step_index in range(request.max_new_tokens):
            next_token = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            next_tensor = torch.tensor(
                [[next_token]], device=input_ids.device, dtype=input_ids.dtype
            )
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            if attention_mask is not None:
                one = torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([attention_mask, one], dim=1)

            with torch.no_grad():
                token_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = token_outputs.hidden_states
            if hidden_states is None or len(hidden_states) < self.num_layers + 1:
                raise ModelLoadError(
                    "model did not return expected hidden states",
                    reason="hidden_states_missing",
                )
            layer_vectors = tuple(
                tuple(
                    float(value)
                    for value in hidden_states[layer + 1][0, -1, :]
                    .detach()
                    .cpu()
                    .tolist()
                )
                for layer in range(self.num_layers)
            )
            router = (
                self.router_extractor(token_outputs, step_index)
                if self.router_extractor
                else None
            )
            is_eos = self.eos_token_id is not None and next_token == self.eos_token_id
            yield TokenStep(next_token, layer_vectors, router, is_eos)
            if request.stop_on_eos and is_eos:
                break
            outputs = token_outputs
