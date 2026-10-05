"""只使用标准库验证日志监控，无需启动 Isaac Sim。"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class TrainingLoggingTests(unittest.TestCase):
    def run_case(self, mode):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "fake_train.py"
            script.write_text(
                f"import sys\nsys.path.insert(0, {str(SCRIPTS)!r})\n"
                + textwrap.dedent("""
                    import os, signal, json, time
                    from pathlib import Path
                    from training_logging import RUN_DIRECTORY_ENV, supervise, run_child, record_failure
                    mode = sys.argv[1]
                    def main():
                        directory = Path(os.environ[RUN_DIRECTORY_ENV])
                        os.write(1, b'native stdout\\n')
                        os.write(2, b'native stderr\\n')
                        (directory / 'progress.csv').write_text('timesteps\\n524288\\n')
                        if mode == 'failure':
                            raise RuntimeError('original failure')
                        if mode == 'cleanup':
                            record_failure(directory, ValueError('first error'))
                            raise RuntimeError('cleanup failure')
                        if mode == 'parent_interrupt':
                            supervisor = json.loads((directory / 'status.json').read_text())['supervisor_pid']
                            os.kill(supervisor, signal.SIGINT)
                            time.sleep(5)
                        if mode in ('kill', 'interrupt', 'terminate'):
                            signum = {'kill': signal.SIGKILL, 'interrupt': signal.SIGINT,
                                      'terminate': signal.SIGTERM}[mode]
                            os.kill(os.getpid(), signum)
                        if mode == 'early_exit':
                            sys.exit(0)
                    if RUN_DIRECTORY_ENV not in os.environ:
                        sys.exit(supervise(Path(__file__)))
                    if mode == 'startup':
                        raise ImportError('startup dependency missing')
                    sys.exit(run_child(main))
                """),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    mode,
                    "--log-dir",
                    str(root / "runs"),
                    "--run-name",
                    "test",
                    "--recurrent-type",
                    "lstm",
                ],
                check=False,
                capture_output=True,
                timeout=20,
            )
            directory = root / "runs" / "lstm_test"
            status = json.loads((directory / "status.json").read_text())
            log = (directory / "console.log").read_text()
            error_path = directory / "error.json"
            error = json.loads(error_path.read_text()) if error_path.exists() else None
            return result, status, log, error

    def test_success_captures_native_streams_and_progress(self):
        result, status, log, error = self.run_case("success")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["exit_code"], 0)
        self.assertEqual(status["last_training_step"], 524288)
        self.assertIsNotNone(status["ended_at"])
        self.assertIn("native stdout", log)
        self.assertIn("native stderr", log)
        self.assertIsNone(error)

    def test_python_failure(self):
        result, status, log, error = self.run_case("failure")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(error["type"], "RuntimeError")
        self.assertIn("original failure", error["traceback"])
        self.assertIn("Traceback", log)

    def test_startup_failure_before_exception_wrapper(self):
        result, status, log, error = self.run_case("startup")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(status["state"], "failed")
        self.assertIn("startup dependency missing", log)
        self.assertIsNone(error)

    def test_sigkill(self):
        result, status, _log, error = self.run_case("kill")
        self.assertEqual(result.returncode, 137)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["signal"], 9)
        self.assertEqual(status["raw_returncode"], -9)
        self.assertEqual(status["last_training_step"], 524288)
        self.assertIsNone(error)

    def test_interrupt_and_termination(self):
        for mode, code in (
            ("interrupt", 130),
            ("terminate", 143),
            ("parent_interrupt", 130),
        ):
            with self.subTest(mode=mode):
                result, status, _log, error = self.run_case(mode)
                self.assertEqual(result.returncode, code)
                self.assertEqual(status["exit_code"], code)
                self.assertEqual(status["state"], "interrupted")
                self.assertEqual(error["type"], "TrainingInterrupted")

    def test_cleanup_does_not_replace_original_error(self):
        result, _status, log, error = self.run_case("cleanup")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(error["message"], "first error")
        self.assertIn("cleanup failure", log)

    def test_early_zero_exit_is_not_completion(self):
        result, status, _log, _error = self.run_case("early_exit")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(status["state"], "failed")


if __name__ == "__main__":
    unittest.main()
