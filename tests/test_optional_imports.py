import builtins
import unittest
from unittest import mock

import mlx_whisperx
from mlx_whisperx import DiarizationPipeline
from mlx_whisperx.vads.pyannote import Pyannote


def _missing_pyannote_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pyannote" or name.startswith("pyannote."):
        raise ModuleNotFoundError("No module named 'pyannote'")
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


_REAL_IMPORT = builtins.__import__


class OptionalImportTests(unittest.TestCase):
    def test_root_package_exports_diarization_helpers(self):
        self.assertTrue(hasattr(mlx_whisperx, "DiarizationPipeline"))
        self.assertIs(DiarizationPipeline, mlx_whisperx.DiarizationPipeline)

    def test_diarization_pipeline_guides_users_to_extra(self):
        with mock.patch("builtins.__import__", side_effect=_missing_pyannote_import):
            with self.assertRaisesRegex(RuntimeError, r"mlx-whisperx\[diarize\]"):
                DiarizationPipeline()

    def test_pyannote_vad_guides_users_to_extra(self):
        with mock.patch("builtins.__import__", side_effect=_missing_pyannote_import):
            with self.assertRaisesRegex(RuntimeError, r"mlx-whisperx\[diarize\]"):
                Pyannote("cpu", vad_onset=0.5, vad_offset=0.363, chunk_size=30)
