from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ultrashape_integration import resolve_ultrashape_checkpoint, resolve_ultrashape_root
from ultrashape_source import (
    candidate_ultrashape_roots,
    ensure_ultrashape_source,
    is_ultrashape_source_root,
)


def _make_source_root(root: Path) -> None:
    (root / "ultrashape").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "infer_dit_refine.yaml").write_text("name: test\n", encoding="utf-8")


class UltraShapePathTests(unittest.TestCase):
    def test_checkpoint_resolves_downloaded_model_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            ckpt = models / "UltraShape" / "ultrashape_v1.pt"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"fake")

            self.assertEqual(resolve_ultrashape_checkpoint(models), ckpt)

    def test_root_resolves_standard_source_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp)
            source = app / "UltraShape-1.0"
            _make_source_root(source)

            with patch("ultrashape_source.importlib.util.find_spec", return_value=None):
                self.assertEqual(resolve_ultrashape_root(app), source.resolve())

    def test_root_resolves_nested_comfy_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp)
            source = app / "ComfyUI-UltraShape1" / "UltraShape-1.0"
            _make_source_root(source)

            with patch("ultrashape_source.importlib.util.find_spec", return_value=None):
                self.assertEqual(resolve_ultrashape_root(app), source.resolve())

    def test_relative_env_source_is_resolved_from_app_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp)
            source = app / "local_ultrashape"
            _make_source_root(source)

            with patch.dict(os.environ, {"ULTRASHAPE_ROOT": "local_ultrashape"}):
                with patch("ultrashape_source.importlib.util.find_spec", return_value=None):
                    self.assertEqual(candidate_ultrashape_roots(app)[0], source.resolve())
                    self.assertEqual(resolve_ultrashape_root(app), source.resolve())

    def test_missing_source_message_distinguishes_weights_from_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp)
            with patch.dict(os.environ, {"ULTRASHAPE_AUTO_INSTALL_SOURCE": "0"}, clear=False):
                with patch("ultrashape_source.importlib.util.find_spec", return_value=None):
                    with self.assertRaises(FileNotFoundError) as cm:
                        resolve_ultrashape_root(app)

            self.assertIn("model weights alone are not enough", str(cm.exception))
            self.assertIn("ensure_ultrashape_source.py", str(cm.exception))

    def test_ensure_extracts_source_archive_without_network_in_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "app"
            app.mkdir()

            def fake_urlretrieve(_url: str, filename: str | Path):
                archive = Path(filename)
                with zipfile.ZipFile(archive, "w") as zf:
                    zf.writestr("UltraShape-main/ultrashape/__init__.py", "")
                    zf.writestr("UltraShape-main/configs/infer_dit_refine.yaml", "name: test\n")
                return str(archive), None

            with patch("ultrashape_source.importlib.util.find_spec", return_value=None):
                with patch("ultrashape_source.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                    root = ensure_ultrashape_source(app, allow_download=True)

            self.assertEqual(root, (app / "UltraShape-1.0").resolve())
            self.assertTrue(is_ultrashape_source_root(root))


if __name__ == "__main__":
    unittest.main()
