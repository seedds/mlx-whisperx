import unittest

from mlx_whisperx import schema


class SchemaTests(unittest.TestCase):
    def test_public_single_segment_schema_excludes_avg_logprob(self):
        self.assertNotIn("avg_logprob", schema.SingleSegment.__annotations__)
