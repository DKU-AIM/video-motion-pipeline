import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/setup_elice.py"


class BootstrapTests(unittest.TestCase):
    def test_plan_does_not_write_or_require_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "not created"
            p = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "install",
                    "all",
                    "--workspace",
                    str(workspace),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertFalse(workspace.exists())
            for model in ("grounding", "comotion", "multihmr2", "motiongpt", "mgllm"):
                self.assertIn(model, p.stdout)

    def test_missing_setup_is_not_reported_as_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run(
                [sys.executable, str(SCRIPT), "check", "motiongpt", "--workspace", tmp],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("reference_joints.npy", p.stdout)
            self.assertIn("motiongpt_s3_h3d.tar", p.stdout)

    def test_env_file_quotes_workspace(self):
        spec = importlib.util.spec_from_file_location("setup_elice", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "a b'$(false)"
            workspace.mkdir()
            mod.write_env(workspace)
            p = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; printf "%s" "$MOTION_WORKSPACE"',
                    "test",
                    str(workspace / "env.sh"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(p.stdout, str(workspace))

    def test_existing_modified_model_is_not_reset(self):
        spec = importlib.util.spec_from_file_location("setup_elice", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "ml-comotion").mkdir()
            sha = mod.REPOS["comotion"][2]
            with (
                patch.object(
                    mod.subprocess,
                    "check_output",
                    side_effect=[sha + "\n", " M demo.py\n"],
                ),
                patch.object(mod, "run") as runner,
            ):
                with self.assertRaises(RuntimeError):
                    mod.checkout(workspace, "comotion")
                runner.assert_not_called()

    def test_existing_pinned_model_does_not_clone_again(self):
        spec = importlib.util.spec_from_file_location("setup_elice", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "ml-comotion").mkdir()
            sha = mod.REPOS["comotion"][2]
            with (
                patch.object(
                    mod.subprocess, "check_output", side_effect=[sha + "\n", ""]
                ),
                patch.object(mod, "run") as runner,
            ):
                self.assertEqual(
                    mod.checkout(workspace, "comotion"), workspace / "ml-comotion"
                )
                runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
