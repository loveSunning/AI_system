"""Mock profiler executables verify wrapper wiring without a CUDA runtime."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
GIT_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash required')
class OfflineProfilerCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.requests = self.base / 'requests.jsonl'
        self.requests.write_text('', encoding='utf-8')
        self.args_log = self.base / 'arguments.bin'
        self.builder = self.base / 'make_sqlite.py'
        self.builder.write_text("""import os,sqlite3,sys
db=sqlite3.connect(sys.argv[1])
db.executescript('''
CREATE TABLE StringIds(id INTEGER,value TEXT);
INSERT INTO StringIds VALUES(1,'cudaGraphLaunch_v10000');
CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER);
INSERT INTO NVTX_EVENTS VALUES(1,2,'worker/load_model',NULL);
INSERT INTO NVTX_EVENTS VALUES(40,50,'worker/prefill',NULL);
INSERT INTO NVTX_EVENTS VALUES(60,70,'worker/decode',NULL);
INSERT INTO NVTX_EVENTS VALUES(30,90,'offline/measure',NULL);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(nameId INTEGER,start INTEGER,end INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1,44,45);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,graphNodeId INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(45,48,1);
CREATE TABLE DIAGNOSTIC_EVENT(text TEXT);
''')
if os.environ.get('MOCK_LOSS'):
    db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES('Not all CUDA events were collected')")
db.commit()
db.close()
""", encoding='utf-8')
        self.mock = self.base / 'mock_profiler.sh'
        self.mock.write_text("""#!/usr/bin/env bash
set -e
printf '%s\\0' "$@" >> "$MOCK_ARG_LOG"
printf '\\n' >> "$MOCK_ARG_LOG"
if [[ "$1" == --version ]]; then echo MOCK_ONLY; exit 0; fi
if [[ "$1" == export ]]; then
  for a in "$@"; do
    if [[ "$a" == --output=* ]]; then
      "$PYTHON_BIN" "$MOCK_SQLITE_BUILDER" "$(printf '%s' "$a" | cut -c10-)"
    fi
  done
  exit 0
fi
if [[ "$1" == --import ]]; then echo mock_metrics; exit 0; fi
previous=''
for a in "$@"; do
  if [[ "$previous" == --export ]]; then touch "$a.ncu-rep"; fi
  if [[ "$a" == --output=* ]]; then touch "$(printf '%s' "$a" | cut -c10-).nsys-rep"; fi
  previous="$a"
done
echo MOCK_ONLY_NO_GPU
""", encoding='utf-8', newline='\n')
        self.mock.chmod(0o755)
        self.environment = dict(os.environ, REQUESTS=self.requests.as_posix(),
                                NSYS_OUT=(self.base / 'nsys').as_posix(),
                                NCU_OUT=(self.base / 'ncu').as_posix(),
                                NSYS_BIN=self.mock.as_posix(), NCU_BIN=self.mock.as_posix(),
                                PYTHON_BIN=Path(sys.executable).as_posix(),
                                MOCK_SQLITE_BUILDER=self.builder.as_posix(),
                                MOCK_ARG_LOG=self.args_log.as_posix(),
                                PROFILE_CASE='w128_o128', PROFILE_COUNT='1',
                                PROFILE_BATCH_SIZE='1', PROFILE_BATCH_TOKENS='2048',
                                NCU_GRAPH='1')

    def run_wrapper(self, script, arguments, **environment):
        result = subprocess.run([BASH, str(SCRIPTS / script), *arguments],
                                env=dict(self.environment, **environment),
                                text=True, capture_output=True)
        commands = [[part.decode() for part in row.split(b'\0') if part]
                    for row in (self.args_log.read_bytes().split(b'\n') if self.args_log.exists() else []) if row]
        return result, commands

    def test_nsys_steady_profiles_measured_generate_and_checks_coverage(self):
        result, commands = self.run_wrapper('profile_offline_nsys.sh', ['fp8-local', 'graph', 'steady'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = next(row for row in commands if row[0] == 'profile')
        self.assertIn('--capture-range=cudaProfilerApi', command)
        self.assertIn('--cuda-graph-trace=node', command)
        self.assertIn('--graph', command)
        self.assertIn('--nvtx', command)
        self.assertEqual(command[command.index('--profiler')+1], 'cuda')
        coverage = json.loads(next((self.base / 'nsys').glob('*.coverage.json')).read_text())
        self.assertTrue(coverage['key_data_present'])
        self.assertEqual(coverage['measure_range'], 'offline/measure')

    def test_nsys_all_eight_full_runs_do_not_use_profiler_gate(self):
        result, commands = self.run_wrapper('profile_offline_nsys.sh', ['all', 'all', 'all'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        profiles = [row for row in commands if row[0] == 'profile']
        self.assertEqual(len(profiles), 8)
        self.assertEqual(sum('--graph' in row for row in profiles), 4)
        for command in profiles:
            self.assertFalse(any(arg.startswith('--capture-range=') for arg in command))
            self.assertEqual(command[command.index('--profiler')+1], 'none')
            self.assertIn('--nvtx', command)

    def test_nsys_event_loss_causes_failed_capture_status(self):
        result, _ = self.run_wrapper('profile_offline_nsys.sh', ['bf16', 'eager', 'steady'], MOCK_LOSS='1')
        self.assertNotEqual(result.returncode, 0)
        coverage = json.loads(next((self.base / 'nsys').glob('*.coverage.json')).read_text())
        self.assertFalse(coverage['key_data_present'])

    def test_ncu_uses_identical_driver_worker_annotations_and_decode_gate(self):
        result, commands = self.run_wrapper('profile_offline_ncu.sh', ['fp8-local', 'decode', 'down', 'basic'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = next(row for row in commands if '--export' in row)
        for flag, value in (('--launch-skip','3'),('--launch-count','1'),
                            ('--delay-iters','8'),('--max-iters','2'),
                            ('--target-processes','all'),('--profile-from-start','off')):
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertIn('--nvtx', command)
        self.assertIn('--event-dir', command)
        self.assertIn('--graph', command)
        self.assertTrue(any(arg.endswith('profile_kernels.py') for arg in command))


    def test_general_ncu_wrapper_selects_remaining_model_families(self):
        for model, family in (('bf16', 'internal::gemvx::kernel<'),
                              ('awq', 'marlin::Marlin<'),
                              ('fp8-official', '_w8a8_triton_block_scaled_mm')):
            with self.subTest(model=model):
                result, commands = self.run_wrapper(
                    'profile_offline_ncu.sh', [model, 'decode', 'first4', 'basic'])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                command = next(row for row in reversed(commands) if '--export' in row)
                self.assertIn(family, command[command.index('--kernel-name')+1])
                self.assertEqual(command[command.index('--launch-count')+1], '4')
                self.assertEqual(command[command.index('--profile')+1], model)


    def test_ncu_exact_kernel_name_prefill_custom_workload_and_overrides(self):
        kernel = 'void example::kernel<float>(float *, int)'
        result, commands = self.run_wrapper('profile_offline_ncu.sh', [
            '--model', 'bf16', '--phase', 'prefill', '--level', 'basic',
            '--kernel-name', kernel, '--kernel-name-base', 'demangled',
            '--launch-skip', '5', '--launch-count', '3', '--mode', 'eager',
            '--case', 'w4096_o128', '--batch-size', '2', '--batch-tokens', '1024',
            '--warmup', '7', '--output-dir', str(self.base / 'custom reports')])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = next(row for row in commands if '--export' in row)
        for flag, value in (('--kernel-name', kernel), ('--launch-skip','5'),
                            ('--launch-count','3'), ('--delay-iters','0'), ('--max-iters','1'),
                            ('--case','w4096_o128'), ('--batch-size','2'),
                            ('--batch-tokens','1024'), ('--warmup','7')):
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertNotIn('--graph', command)
        self.assertEqual(command[command.index('--nvtx-include')+1],
                         r'quantization_profiling@worker\/prefill/')
        self.assertEqual(command.count('--section'), 3)

    def test_ncu_regex_detail_and_custom_decode_window(self):
        expression = 'gemv.*|_w8a8_triton_block_scaled_mm'
        result, commands = self.run_wrapper('profile_offline_ncu.sh', [
            '--model', 'fp8-local', '--phase', 'decode', '--level', 'detail',
            '--kernel-regex', expression, '--launch-skip', '3', '--launch-count', '2',
            '--delay-iters', '12', '--max-iters', '4', '--mode', 'graph'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = next(row for row in commands if '--export' in row)
        self.assertEqual(command[command.index('--kernel-name')+1], 'regex:'+expression)
        self.assertEqual(command[command.index('--delay-iters')+1], '12')
        self.assertEqual(command[command.index('--max-iters')+1], '4')
        self.assertEqual(command.count('--section'), 8)
        self.assertIn('SourceCounters', command)
        self.assertEqual(command[command.index('--nvtx-include')+1],
                         r'quantization_profiling@worker\/decode/')

    def test_nsys_named_options_override_environment(self):
        result, commands = self.run_wrapper('profile_offline_nsys.sh', [
            '--model', 'bf16', '--mode', 'eager', '--scope', 'steady',
            '--count', '2', '--batch-size', '2', '--batch-tokens', '1536', '--warmup', '6'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = next(row for row in commands if row[0] == 'profile')
        for flag, value in (('--profile','bf16'), ('--count','2'), ('--batch-size','2'),
                            ('--batch-tokens','1536'), ('--warmup','6')):
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertNotIn('--graph', command)

    def test_help_does_not_require_snapshot_or_profiler(self):
        for script in ('profile_offline_nsys.sh','profile_offline_ncu.sh'):
            with self.subTest(script=script):
                result, commands = self.run_wrapper(script, ['--help'], REQUESTS='/missing/requests.jsonl')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('--model', result.stdout)
                self.assertEqual(commands, [])

    def test_invalid_filters_and_counts_are_rejected_before_profiler(self):
        cases = [
            ['--kernel-name','A','--kernel-regex','B'],
            ['--kernel-name','A','--launch-count','0'],
            ['--kernel-name','A','--launch-skip','-1'],
            ['--kernel-name','regex:A'],
            ['--kernel-regex','regex:A'],
            ['--phase','bad'],
            ['--level','full'],
            ['--kernel-name'],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result, commands = self.run_wrapper(
                    'profile_offline_ncu.sh', ['--model','bf16',*arguments])
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(commands, [])


if __name__ == '__main__':
    unittest.main()
