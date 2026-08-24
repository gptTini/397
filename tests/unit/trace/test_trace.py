from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cspm397.adapters import (
    GenerationRequest,
    MockModelAdapter,
    RouterStep,
    TokenStep,
)
from cspm397.trace import (
    FixedRandomProjection,
    InfTraceError,
    NaNTraceError,
    RouterError,
    TraceCollector,
    TraceError,
)


class ProjectionTests(unittest.TestCase):
    def test_projection_is_deterministic(self) -> None:
        vector = (1.0, -2.0, 3.0, 4.0)
        a = FixedRandomProjection(4, 3, 397).project(vector)
        b = FixedRandomProjection(4, 3, 397).project(vector)
        self.assertEqual(a, b)

    def test_projection_rejects_dimension_mismatch(self) -> None:
        with self.assertRaises(TraceError):
            FixedRandomProjection(4, 2).project((1.0, 2.0))

    def test_projection_classifies_nan_and_inf(self) -> None:
        projection = FixedRandomProjection(2, 2)
        with self.assertRaises(NaNTraceError):
            projection.project((math.nan, 1.0))
        with self.assertRaises(InfTraceError):
            projection.project((math.inf, 1.0))


class MockTraceTests(unittest.TestCase):
    def test_zero_token_trace(self) -> None:
        adapter = MockModelAdapter(num_layers=2, hidden_size=4)
        trace = TraceCollector(adapter, (0, 1), 3).collect(
            GenerationRequest("x", max_new_tokens=0)
        )
        self.assertEqual(trace.num_tokens, 0)
        self.assertEqual(trace.num_sequences, 0)
        self.assertEqual(trace.array_shapes()["state_signatures"], [0, 2, 3])

    def test_one_token_trace_and_sequence_stats(self) -> None:
        adapter = MockModelAdapter(num_layers=2, hidden_size=4)
        trace = TraceCollector(adapter, (1,), 2).collect(
            GenerationRequest("x", max_new_tokens=1)
        )
        self.assertEqual(trace.num_tokens, 1)
        self.assertEqual(trace.sequence_ids, (0,))
        self.assertEqual(trace.positions, (0,))
        self.assertEqual(trace.sequence_stats()["unique_count"], 1)
        self.assertTrue(trace.sequence_stats()["positions_zero_based_contiguous"])

    def test_mock_trace_is_reproducible(self) -> None:
        adapter = MockModelAdapter(num_layers=3, hidden_size=5)
        collector = TraceCollector(adapter, (0, 2), 4, projection_seed=123)
        request = GenerationRequest("hello", max_new_tokens=4, seed=999)
        self.assertEqual(collector.collect(request), collector.collect(request))

    def test_eos_stops_generation(self) -> None:
        adapter = MockModelAdapter(num_layers=2, hidden_size=4, eos_at_step=1)
        trace = TraceCollector(adapter, (0,), 2).collect(
            GenerationRequest("x", max_new_tokens=5)
        )
        self.assertEqual(trace.num_tokens, 2)
        self.assertEqual(trace.token_ids[-1], adapter.eos_token_id)

    def test_chunk_slicing_preserves_payload(self) -> None:
        adapter = MockModelAdapter(num_layers=2, hidden_size=4)
        trace = TraceCollector(adapter, (0,), 2).collect(
            GenerationRequest("x", max_new_tokens=5)
        )
        chunks = list(trace.iter_chunks(2))
        self.assertEqual([chunk.num_tokens for chunk in chunks], [2, 2, 1])
        self.assertEqual(
            tuple(token for chunk in chunks for token in chunk.token_ids),
            trace.token_ids,
        )

    def test_router_capture_shapes_and_weights(self) -> None:
        adapter = MockModelAdapter(
            num_layers=3, hidden_size=4, num_experts=8, router_top_k=2
        )
        trace = TraceCollector(adapter, (0, 2), 3, capture_router=True).collect(
            GenerationRequest("router", max_new_tokens=3)
        )
        self.assertIsNotNone(trace.router)
        self.assertEqual(trace.array_shapes()["expert_ids"], [3, 2, 2])
        assert trace.expert_weights is not None
        for token in trace.expert_weights:
            for weights in token:
                self.assertAlmostEqual(sum(weights), 1.0, places=12)

    def test_router_requested_but_absent_is_error(self) -> None:
        adapter = MockModelAdapter(num_layers=2, hidden_size=4)
        with self.assertRaises(RouterError):
            TraceCollector(adapter, (0,), 2, capture_router=True).collect(
                GenerationRequest("x", 1)
            )

    def test_selected_layer_out_of_range_is_error(self) -> None:
        adapter = MockModelAdapter(num_layers=2, hidden_size=4)
        with self.assertRaises(ValueError):
            TraceCollector(adapter, (2,), 2)

    def test_adapter_cannot_emit_more_than_request_limit(self) -> None:
        class BadAdapter:
            model_id = "bad"
            model_revision = "v1"
            num_layers = 1
            hidden_size = 2
            device = "cpu"
            eos_token_id = None
            num_experts = None
            router_top_k = None

            def generate(self, request):
                for i in range(request.max_new_tokens + 1):
                    yield TokenStep(i, ((0.0, 1.0),))

        with self.assertRaises(TraceError):
            TraceCollector(BadAdapter(), (0,), 2).collect(GenerationRequest("x", 1))

    def test_bad_router_shape_is_error(self) -> None:
        class BadRouterAdapter:
            model_id = "bad-router"
            model_revision = "v1"
            num_layers = 1
            hidden_size = 2
            device = "cpu"
            eos_token_id = None
            num_experts = 4
            router_top_k = 2

            def generate(self, request):
                yield TokenStep(1, ((0.0, 1.0),), RouterStep(((0,),), ((1.0,),)))

        with self.assertRaises(RouterError):
            TraceCollector(BadRouterAdapter(), (0,), 2, capture_router=True).collect(
                GenerationRequest("x", 1)
            )


if __name__ == "__main__":
    unittest.main()
