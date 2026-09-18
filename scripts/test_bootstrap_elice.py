"""Bootstrap orchestration tests; package/GPU commands are isolated fakes."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("bootstrap_elice.sh")
FAKE_PYTHON = r'''#!/usr/bin/env python3
import os, pathlib, sys
a = sys.argv[1:]
with open(os.environ['CALL_LOG'], 'a') as f:
    f.write(repr(a) + '\n')
if a[:2] == ['-m', 'venv']:
    p = pathlib.Path(a[2]); (p/'bin').mkdir(parents=True)
    (p/'pyvenv.cfg').write_text('home = fake\n')
    target = p/'bin/python'; target.write_text(pathlib.Path(__file__).read_text())
    target.chmod(0o755)
    (p/'bin/activate').write_text('export VIRTUAL_ENV=' + repr(str(p)) + '\n')
if '-m' in a and 'pip' in a and 'install' in a and os.getenv('FAIL_INSTALL'):
    sys.exit(7)
if any(x.endswith('check_grounding.py') for x in a) and os.getenv('FAIL_GPU'):
    sys.exit(8)
'''


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        for name, body in {
            "uname": '#!/bin/sh\nif [ "$1" = -m ]; then echo x86_64; else echo Linux; fi\n',
            "nvidia-smi": '#!/bin/sh\necho "fake GPU"\n',
            "bootstrap-python": FAKE_PYTHON,
        }.items():
            p = self.bin / name
            p.write_text(body)
            p.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        VTG_ROOT=str(self.root / "workspace with spaces"),
                        BOOTSTRAP_PYTHON=str(self.bin / "bootstrap-python"),
                        CALL_LOG=str(self.root / "calls"))

    def run_script(self, *args):
        return subprocess.run(["bash", str(SCRIPT), *args], env=self.env,
                              text=True, capture_output=True, input="")

    def test_setup_and_rerun_reuse_venv(self):
        for _ in range(2):
            result = self.run_script("--skip-video", "--skip-model")
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = (self.root / "calls").read_text()
        self.assertEqual(calls.count("'-m', 'venv'"), 1)
        self.assertIn("cu118", calls)
        self.assertIn("check_grounding.py", calls)
        activation = Path(self.env["VTG_ROOT"]) / "activate.sh"
        check = subprocess.run(["bash", "-c", 'source "$1"; test "$FORCE_QWENVL_VIDEO_READER" = torchvision; test -d "$HF_HOME"',
                                "test", str(activation)], capture_output=True)
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_install_failure_does_not_report_success(self):
        self.env["FAIL_INSTALL"] = "1"
        result = self.run_script("--skip-video", "--skip-model")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("READY", result.stdout)
        self.assertNotIn("check_grounding.py", (self.root / "calls").read_text())

    def test_gpu_failure_stops_before_model_download(self):
        self.env["FAIL_GPU"] = "1"
        result = self.run_script("--skip-video")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("READY", result.stdout)
        self.assertNotIn("snapshot_download", (self.root / "calls").read_text())

    def test_rejects_non_venv_directory(self):
        (Path(self.env["VTG_ROOT"]) / "env").mkdir(parents=True)
        result = self.run_script("--skip-video", "--skip-model")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "calls").exists())

    def test_noninteractive_missing_video_fails_before_install(self):
        self.env.pop("VIDEO_URL", None)
        result = self.run_script("--skip-model")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "calls").exists())

    def test_help_has_no_side_effects(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "calls").exists())

    def test_sample_wires_download_and_media_check(self):
        result = self.run_script("--sample", "--skip-model")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = (self.root / "calls").read_text()
        self.assertIn("download_video.py", calls)
        self.assertIn("'--output'", calls)
        self.assertIn("'--video'", calls)
        activation = (Path(self.env["VTG_ROOT"]) / "activate.sh").read_text()
        self.assertIn("VIDEO_PATH=", activation)
        self.assertNotIn("https://", activation)


if __name__ == "__main__":
    unittest.main()
