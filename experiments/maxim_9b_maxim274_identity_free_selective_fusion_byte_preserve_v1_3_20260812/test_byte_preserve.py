from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import compose
import protocol
import score_reusing_base251_judge as scorer


class BytePreserveTest(unittest.TestCase):
    def test_threshold(self) -> None:
        self.assertFalse(protocol.threshold_met(249))
        self.assertTrue(protocol.threshold_met(250))
        self.assertFalse(protocol.threshold_met(True))

    def test_source_census_and_no_image_switch(self) -> None:
        sources = compose.validate_sources()
        branches = [protocol.branch(row) for row, _ in sources["selected"]]
        self.assertEqual(sum(value in protocol.BASELINE_BRANCHES for value in branches), 272)
        self.assertEqual(sum(value == protocol.GENERIC_BRANCH for value in branches), 2)

    def test_payload_preserves_exact_source_lines(self) -> None:
        sources = compose.validate_sources()
        payload, census = compose.compose_payload(sources["base"], sources["selected"], sources["image_ids"])
        output = payload.splitlines(keepends=True)
        base = sources["base"]
        selected = sources["selected"]
        for index, ((_, base_raw), (selected_row, selected_raw)) in enumerate(zip(base, selected)):
            expected = base_raw if protocol.branch(selected_row) in protocol.BASELINE_BRANCHES else selected_raw
            self.assertEqual(output[index], expected)
        self.assertEqual(census["baseline_rows_copied_byte_exact"], 272)
        self.assertEqual(census["generic_rows_copied_from_v1_1_byte_exact"], 2)
        self.assertEqual(census["image97_rows_base251_byte_and_object_exact"], 97)

    def test_v1_1_selector_implementation_is_exactly_pinned(self) -> None:
        self.assertEqual(protocol.sha256_file(protocol.V11_IMPLEMENTATION), protocol.PINS["v1_1_implementation"][1])

    def test_hostile_deterministic_mutation_is_rejected(self) -> None:
        sources = compose.validate_sources()
        payload, _ = compose.compose_payload(sources["base"], sources["selected"], sources["image_ids"])
        hostile = bytearray(payload)
        hostile[0] ^= 1
        with self.assertRaises(protocol.ProtocolError):
            scorer.require_exact_mechanical_payload(bytes(hostile))

    def test_implementation_descriptor_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "implementation.py"
            path.write_bytes(b"frozen\n")
            descriptor = protocol.descriptor(path)
            protocol._verify_descriptor(descriptor, path, "test implementation")
            path.write_bytes(b"mutated\n")
            with self.assertRaises(protocol.ProtocolError):
                protocol._verify_descriptor(descriptor, path, "test implementation")


if __name__ == "__main__":
    unittest.main()
