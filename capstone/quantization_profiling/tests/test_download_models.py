"""Offline behavior checks for model downloads; model weights are never fetched."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_models.sh"
SOURCE = ".modelscope-download-source"
COMPLETE = ".modelscope-download-complete"


def find_bash():
    if os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        return str(candidate) if candidate.is_file() else None
    return shutil.which("bash")


BASH = find_bash()
LAUNCHER = r"""
export PATH="$PWD/mock_bin:/usr/bin:/bin"
if [[ "${MOCK_HIDE_CLI:-0}" == 1 ]]; then
  command() {
    if [[ "$#" == 2 && "$1" == "-v" && "$2" == "modelscope" ]]; then
      return 1
    fi
    builtin command "$@"
  }
  export -f command
fi
exec /bin/bash "$1" "${@:2}"
"""
MOCK_CLI = r"""#!/usr/bin/env bash
set -eu
printf 'CALL\n' >> "$MOCK_LOG"
printf '%s\n' "$@" >> "$MOCK_LOG"
model=''
target=''
while (( $# > 0 )); do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --local_dir) target="$2"; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p -- "$target"
if [[ "$model" == "${MOCK_FAIL_MODEL:-}" ]]; then
  printf 'partial download\n' > "$target/partial.bin"
  exit 7
fi
printf '{"mock": true}\n' > "$target/config.json"
"""


@unittest.skipUnless(BASH, "Bash is required (use Git Bash on Windows)")
class DownloadModelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="quantization-download-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "mock_bin"
        self.bin_dir.mkdir()
        mock = self.bin_dir / "modelscope"
        mock.write_text(MOCK_CLI, encoding="utf-8", newline="\n")
        mock.chmod(0o755)
        self.env = dict(os.environ)
        self.env.update({
            "MODEL_ROOT": "weights",
            "MOCK_LOG": "calls.log",
            "MOCK_FAIL_MODEL": "",
            "MOCK_HIDE_CLI": "0",
        })
        # Exported shell functions from a caller must not alter test commands.
        for key in tuple(self.env):
            if key.startswith("BASH_FUNC_"):
                del self.env[key]

    def run_script(self, *args, hide_cli=False, fail_model="", model_root=None):
        env = dict(self.env)
        env["MOCK_HIDE_CLI"] = "1" if hide_cli else "0"
        env["MOCK_FAIL_MODEL"] = fail_model
        if model_root is not None:
            env["MODEL_ROOT"] = model_root
        return subprocess.run(
            [BASH, "--noprofile", "--norc", "-c", LAUNCHER,
             "download-models-test", SCRIPT.as_posix(), *args],
            cwd=self.root, env=env, text=True, capture_output=True, timeout=30,
        )

    def calls(self):
        log = self.root / "calls.log"
        if not log.exists():
            return []
        return [part.splitlines() for part in log.read_text(encoding="utf-8").split("CALL\n")[1:]]

    def existing(self, name="Qwen3-8B", *, source=None, complete=None):
        target = self.root / "weights" / name
        target.mkdir(parents=True)
        (target / "config.json").write_text('{"existing": true}\n', encoding="utf-8")
        if source is not None:
            (target / SOURCE).write_text(source, encoding="utf-8", newline="\n")
        if complete is not None:
            (target / COMPLETE).write_text(complete, encoding="utf-8", newline="\n")
        return target

    def snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unmarked_existing_model_is_skipped_and_next_model_downloads(self):
        old = self.existing()
        original = (old / "config.json").read_bytes()
        result = self.run_script(
            "--model", "Qwen/Qwen3-8B", "--model", "Qwen/Qwen3-8B-AWQ",
        )
        self.assert_success(result)
        self.assertIn("SKIP", result.stdout)
        self.assertEqual((old / "config.json").read_bytes(), original)
        self.assertFalse((old / SOURCE).exists())
        self.assertEqual(self.calls(), [[
            "download", "--model", "Qwen/Qwen3-8B-AWQ",
            "--local_dir", "weights/Qwen3-8B-AWQ",
        ]])
        self.assertTrue((self.root / "weights/Qwen3-8B-AWQ" / COMPLETE).is_file())

    def test_all_existing_models_succeed_without_modelscope(self):
        self.existing()
        self.existing("Qwen3-8B-AWQ")
        before = self.snapshot()
        result = self.run_script(
            "--model", "Qwen/Qwen3-8B", "--model", "Qwen/Qwen3-8B-AWQ",
            hide_cli=True,
        )
        self.assert_success(result)
        self.assertEqual(result.stdout.count("SKIP"), 2)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.calls(), [])

    def test_matching_completion_marker_skips_already_downloaded_model(self):
        marker = "Qwen/Qwen3-8B\n<default>\n"
        self.existing(source=marker, complete=marker)
        before = self.snapshot()
        result = self.run_script(hide_cli=True)
        self.assert_success(result)
        self.assertIn("SKIP", result.stdout)
        self.assertEqual(self.snapshot(), before)

    def test_empty_directory_is_downloaded(self):
        target = self.root / "weights/Qwen3-8B"
        target.mkdir(parents=True)
        result = self.run_script()
        self.assert_success(result)
        self.assertEqual(len(self.calls()), 1)
        expected = "Qwen/Qwen3-8B\n<default>\n"
        self.assertEqual((target / SOURCE).read_text(encoding="utf-8"), expected)
        self.assertEqual((target / COMPLETE).read_text(encoding="utf-8"), expected)

    def test_matching_source_without_completion_retries_partial_download(self):
        target = self.existing(source="Qwen/Qwen3-8B\n<default>\n")
        result = self.run_script()
        self.assert_success(result)
        self.assertEqual(len(self.calls()), 1)
        self.assertTrue((target / COMPLETE).is_file())

    def test_wrong_completion_marker_does_not_hide_incomplete_download(self):
        target = self.existing(
            source="Qwen/Qwen3-8B\n<default>\n",
            complete="Qwen/Qwen3-8B\nsome-other-revision\n",
        )
        result = self.run_script()
        self.assert_success(result)
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(
            (target / COMPLETE).read_text(encoding="utf-8"),
            "Qwen/Qwen3-8B\n<default>\n",
        )

    def test_dry_run_skips_existing_and_previews_missing_without_writes(self):
        self.existing()
        self.existing("Qwen3-8B-AWQ", source="Qwen/Qwen3-8B-AWQ\n<default>\n")
        before = self.snapshot()
        result = self.run_script(
            "--model", "Qwen/Qwen3-8B", "--model", "Qwen/Qwen3-8B-AWQ",
            "--model", "Qwen/Qwen3-8B-FP8", "--dry-run", hide_cli=True,
        )
        self.assert_success(result)
        self.assertIn("SKIP", result.stdout)
        self.assertEqual(result.stdout.count("Command:"), 2)
        self.assertIn("Qwen/Qwen3-8B-AWQ", result.stdout)
        self.assertIn("Qwen/Qwen3-8B-FP8", result.stdout)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.calls(), [])

    def test_default_dry_run_uses_environment_root_and_creates_nothing(self):
        before = self.snapshot()
        result = self.run_script("--dry-run", model_root="shared weights", hide_cli=True)
        self.assert_success(result)
        self.assertIn("Qwen/Qwen3-8B", result.stdout)
        self.assertIn("shared weights/Qwen3-8B", result.stdout)
        self.assertEqual(self.snapshot(), before)

    def test_download_failure_continues_batch_and_can_be_retried(self):
        args = ("--model", "Qwen/Qwen3-8B", "--model", "Qwen/Qwen3-8B-AWQ")
        failed = self.run_script(*args, fail_model="Qwen/Qwen3-8B")
        self.assertNotEqual(failed.returncode, 0, failed.stdout + failed.stderr)
        self.assertEqual(len(self.calls()), 2)
        first = self.root / "weights/Qwen3-8B"
        second = self.root / "weights/Qwen3-8B-AWQ"
        self.assertTrue((first / SOURCE).is_file())
        self.assertFalse((first / COMPLETE).exists())
        self.assertTrue((second / COMPLETE).is_file())

        retried = self.run_script(*args)
        self.assert_success(retried)
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual(self.calls()[-1][2], "Qwen/Qwen3-8B")
        self.assertTrue((first / COMPLETE).is_file())
        self.assertIn("SKIP", retried.stdout)

    def test_revision_and_model_root_with_spaces_are_forwarded_as_single_arguments(self):
        result = self.run_script(
            "--model", "Qwen/Qwen3-8B-AWQ", "--revision", "release candidate",
            "--model-root", "custom weights",
        )
        self.assert_success(result)
        self.assertEqual(self.calls(), [[
            "download", "--model", "Qwen/Qwen3-8B-AWQ",
            "--local_dir", "custom weights/Qwen3-8B-AWQ",
            "--revision", "release candidate",
        ]])
        target = self.root / "custom weights/Qwen3-8B-AWQ"
        self.assertEqual(
            (target / SOURCE).read_text(encoding="utf-8"),
            "Qwen/Qwen3-8B-AWQ\nrelease candidate\n",
        )
        self.assertFalse((self.root / "weights").exists())

    def test_conflicting_or_broken_source_markers_skip_without_modification(self):
        for marker in (
            "Other/Qwen3-8B\n<default>\n",
            "Qwen/Qwen3-8B\nother-revision\n",
            "broken marker\n",
        ):
            with self.subTest(marker=marker):
                target = self.root / "weights/Qwen3-8B"
                if target.exists():
                    (target / SOURCE).write_text(marker, encoding="utf-8", newline="\n")
                else:
                    self.existing(source=marker)
                before = self.snapshot()
                result = self.run_script(hide_cli=True)
                self.assert_success(result)
                self.assertIn("SKIP", result.stdout)
                self.assertEqual(self.snapshot(), before)

    def test_missing_cli_is_reported_when_a_model_needs_downloading(self):
        result = self.run_script(hide_cli=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("modelscope", result.stderr)
        self.assertFalse((self.root / "weights").exists())
        self.assertEqual(self.calls(), [])

    def test_file_at_target_is_rejected_without_overwriting(self):
        weights = self.root / "weights"
        weights.mkdir()
        target = weights / "Qwen3-8B"
        target.write_text("keep me", encoding="utf-8")
        before = self.snapshot()
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
