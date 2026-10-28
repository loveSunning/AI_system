"""CPU-only contracts for the shared non-server NSYS/NCU workload."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('offline_workload', SCRIPTS / 'profile_kernels.py')
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


class OfflineWorkloadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.snapshot = self.base / 'frozen'
        self.snapshot.mkdir()
        self.weights = self.base / 'weights'
        configs = {
            'Qwen3-8B': {},
            'Qwen3-8B-AWQ': {'quantization_config': {'quant_method': 'awq', 'bits': 4}},
            'Qwen3-8B-FP8': {'quantization_config': {
                'quant_method': 'fp8', 'fmt': 'e4m3', 'activation_scheme': 'dynamic',
                'weight_block_size': [128, 128]}},
            'Qwen3-8B-FP8-local': {'quantization_config': {
                'quant_method': 'compressed-tensors', 'config_groups': {'g': {
                    'weights': {'num_bits': 8, 'type': 'float'},
                    'input_activations': {'num_bits': 8, 'type': 'float', 'dynamic': True}}}}},
        }
        for name, config in configs.items():
            directory = self.weights / name
            directory.mkdir(parents=True)
            (directory / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        self.reference = str(self.base / 'frozen_tokenizer_reference')
        (self.snapshot / 'model.json').write_text(json.dumps({'path': self.reference}), encoding='utf-8')
        (self.snapshot / 'workloads.yaml').write_text(json.dumps({'cases': []}), encoding='utf-8')
        rows = [dict(id=f'w128_o128-{i:04}', case='w128_o128',
                     phase='warmup' if i < 5 else 'measure', input_tokens=128,
                     output_tokens=128, prompt_token_ids=[1] * 128) for i in range(8)]
        (self.snapshot / 'requests.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        manifest = {'files': {name: hashlib.sha256((self.snapshot / name).read_bytes()).hexdigest()
                             for name in ('model.json', 'workloads.yaml', 'requests.jsonl')}}
        (self.snapshot / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.events = []
        self.engine = {}
        self.fail_measured = False
        owner = self

        class FakeLLM:
            def __init__(self, **kwargs):
                owner.engine = kwargs
                self.active = False
                owner.events.append('initialize')
            def generate(self, items, sampling_params, use_tqdm=False):
                owner.events.append(('generate', self.active))
                if self.active and owner.fail_measured:
                    raise RuntimeError('measured generation failed')
                assert all(len(item['prompt_token_ids']) == 128 for item in items)
                output = SimpleNamespace(token_ids=[2] * sampling_params.max_tokens)
                return [SimpleNamespace(outputs=[output]) for _ in items]
            def start_profile(self):
                self.active = True
                owner.events.append('start')
            def stop_profile(self):
                self.active = False
                owner.events.append('stop')

        fake = ModuleType('vllm')
        fake.LLM = FakeLLM
        fake.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        self.fake_vllm = fake
        self.environment = dict(MODEL_ROOT=str(self.weights), VLLM_TORCH_CUDA_PROFILE='',
                                VLLM_TORCH_PROFILER_DIR='', VLLM_PROFILER_DELAY_ITERS='',
                                VLLM_PROFILER_MAX_ITERS='')
        self.versions = {'vllm': '0.14.1', 'torch': '2.9.1+cu128', 'triton': '3.5.1'}

    def invoke(self, extra=(), profile='fp8-local'):
        @contextmanager
        def annotate(**kwargs):
            yield
        fake_nvtx = SimpleNamespace(annotate=annotate)
        arguments = ['--profile', profile, '--snapshot', str(self.snapshot),
                     '--output', str(self.base / 'result.json'), *extra]
        with patch.dict(os.environ, self.environment), \
             patch.dict(sys.modules, {'vllm': self.fake_vllm}), \
             patch.object(importlib.metadata, 'version', side_effect=self.versions.__getitem__), \
             patch.object(workload.trace, 'require_nvtx', return_value=fake_nvtx), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            workload.main(arguments)
        path = self.base / 'result.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def test_all_four_models_share_frozen_inputs_and_reference_tokenizer(self):
        for profile in ('bf16', 'awq', 'fp8-official', 'fp8-local'):
            with self.subTest(profile=profile):
                self.events.clear()
                result = self.invoke(profile=profile)
                self.assertEqual(self.engine['tokenizer'], self.reference)
                self.assertEqual(self.engine['max_model_len'], 256)
                self.assertEqual(result['records'][0]['output_tokens'], [128])
                self.assertEqual(result['workload']['snapshot_validation'], 'manifest file hashes passed')
                self.assertEqual(self.events, ['initialize'] + [('generate', False)] * 5
                                 + ['start', ('generate', True), 'stop'])
                (self.base / 'result.json').unlink()

    def test_graph_worker_annotations_and_decode_gate_are_shared(self):
        result = self.invoke(['--graph', '--nvtx', '--delay-iters', '8', '--max-iters', '2',
                              '--event-dir', str(self.base / 'events')])
        self.assertFalse(self.engine['enforce_eager'])
        self.assertEqual(self.engine['worker_cls'], 'profiling_worker.ProfilingWorker')
        self.assertEqual(self.engine['profiler_config'], {
            'profiler': 'cuda', 'delay_iterations': 8, 'max_iterations': 2})
        self.assertTrue((self.base / 'events' / 'controller.jsonl').exists())
        self.assertEqual(result['workload']['measured_ids'], ['w128_o128-0005'])

    def test_full_capture_or_baseline_does_not_arm_cuda_profiler(self):
        self.invoke(['--profiler', 'none', '--nvtx'])
        self.assertNotIn('profiler_config', self.engine)
        self.assertNotIn('start', self.events)
        self.assertNotIn('stop', self.events)
        self.assertEqual(self.events.count(('generate', False)), 6)

    def test_profiler_stops_when_measured_generation_fails(self):
        self.fail_measured = True
        with self.assertRaisesRegex(RuntimeError, 'measured generation failed'):
            self.invoke()
        self.assertEqual(self.events[-1], 'stop')

    def test_snapshot_tampering_is_rejected_before_model_initialization(self):
        with (self.snapshot / 'requests.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'Snapshot hash mismatch'):
            self.invoke()
        self.assertEqual(self.events, [])

    def test_plan_only_does_not_import_or_initialize_vllm(self):
        self.assertIsNone(self.invoke(['--plan-only']))
        self.assertEqual(self.events, [])

    def test_reduced_generation_keeps_original_engine_length(self):
        self.invoke(['--output-tokens', '1'])
        self.assertEqual(self.engine['max_model_len'], 256)


    def test_nsys_and_ncu_engine_configuration_differs_only_in_profiler_gate(self):
        base = ['--profile', 'fp8-local', '--snapshot', str(self.snapshot),
                '--nvtx', '--graph', '--output', str(self.base / 'unused.json')]
        with patch.dict(os.environ, self.environment):
            ns = workload.prepare(workload.parse_args(base + ['--delay-iters', '0', '--max-iters', '0']))
            nc = workload.prepare(workload.parse_args(base + ['--delay-iters', '8', '--max-iters', '2']))
        ns_engine, nc_engine = dict(ns[0]), dict(nc[0])
        ns_engine.pop('profiler_config')
        nc_engine.pop('profiler_config')
        self.assertEqual(ns_engine, nc_engine)
        self.assertEqual(ns[1], nc[1])


if __name__ == '__main__':
    unittest.main()
