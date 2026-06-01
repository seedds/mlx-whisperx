import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class FakeWeights(dict):
    pass


class FakeModel:
    def __init__(self):
        self.updated = None

    def update(self, weights):
        self.updated = weights

    def parameters(self):
        return []


class LoadModelTests(unittest.TestCase):
    STUBBED_MODULES = [
        "huggingface_hub",
        "mlx_whisperx.backend.mlx_whisper.load_models",
        "numba",
        "scipy",
        "tiktoken",
        "tqdm",
    ]

    def setUp(self):
        self._original_modules = {
            name: sys.modules.get(name)
            for name in self.STUBBED_MODULES
            if name in sys.modules
        }

    def tearDown(self):
        for name in self.STUBBED_MODULES:
            if name in self._original_modules:
                sys.modules[name] = self._original_modules[name]
                parent_name, _, attr = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, attr, self._original_modules[name])
            else:
                sys.modules.pop(name, None)
                parent_name, _, attr = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None and attr in parent.__dict__:
                    delattr(parent, attr)

    def _import_load_models(self):
        sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))
        huggingface_hub = sys.modules.setdefault(
            "huggingface_hub",
            types.ModuleType("huggingface_hub"),
        )
        if not hasattr(huggingface_hub, "snapshot_download"):
            huggingface_hub.snapshot_download = mock.Mock()
        sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
        numba = sys.modules.setdefault("numba", types.ModuleType("numba"))
        if not hasattr(numba, "jit"):
            numba.jit = lambda *args, **kwargs: (lambda func: func)
        scipy = sys.modules.setdefault("scipy", types.ModuleType("scipy"))
        if not hasattr(scipy, "signal"):
            scipy.signal = types.SimpleNamespace(medfilt=lambda x, kernel_size=None: x)
        return importlib.import_module("mlx_whisperx.backend.mlx_whisper.load_models")

    def _write_config(self, directory: Path):
        (directory / "config.json").write_text(
            json.dumps({"n_mels": 80, "n_vocab": 1, "n_audio_ctx": 1, "n_audio_state": 1, "n_audio_head": 1, "n_audio_layer": 1, "n_text_ctx": 1, "n_text_state": 1, "n_text_head": 1, "n_text_layer": 1}),
            encoding="utf-8",
        )

    def test_snapshot_download_receives_cache_settings(self):
        load_models = self._import_load_models()
        fake_model = FakeModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "downloaded"
            model_dir.mkdir()
            self._write_config(model_dir)
            with (
                mock.patch.object(load_models, "snapshot_download", return_value=str(model_dir)) as snapshot_download,
                mock.patch.object(load_models.mx, "load", return_value=FakeWeights({"layer": 1})),
                mock.patch.object(load_models.whisper, "ModelDimensions", side_effect=lambda **kwargs: kwargs),
                mock.patch.object(load_models.whisper, "Whisper", return_value=fake_model),
                mock.patch.object(load_models, "tree_unflatten", side_effect=lambda items: items),
                mock.patch.object(load_models.mx, "eval"),
            ):
                returned = load_models.load_model(
                    "mlx-community/whisper-turbo",
                    model_dir=str(Path(tmpdir) / "cache"),
                    model_cache_only=True,
                )

        self.assertIs(returned, fake_model)
        snapshot_download.assert_called_once_with(
            repo_id="mlx-community/whisper-turbo",
            cache_dir=str(Path(tmpdir) / "cache"),
            local_files_only=True,
        )

    def test_local_paths_skip_snapshot_download(self):
        load_models = self._import_load_models()
        fake_model = FakeModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_config(model_dir)
            with (
                mock.patch.object(load_models, "snapshot_download") as snapshot_download,
                mock.patch.object(load_models.mx, "load", return_value=FakeWeights({"layer": 1})),
                mock.patch.object(load_models.whisper, "ModelDimensions", side_effect=lambda **kwargs: kwargs),
                mock.patch.object(load_models.whisper, "Whisper", return_value=fake_model),
                mock.patch.object(load_models, "tree_unflatten", side_effect=lambda items: items),
                mock.patch.object(load_models.mx, "eval"),
            ):
                returned = load_models.load_model(str(model_dir))

        self.assertIs(returned, fake_model)
        snapshot_download.assert_not_called()
