"""Shared non-server vLLM workload for NSYS, NCU and unprofiled runs.

Both tools call the same LLM.generate path. CUDA profiler control is delegated
to the real GPU worker; controller NVTX scopes are not GPU duration timers.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from experiment_utils import ROOT, digest, load_snapshot, read
from model_profiles import (resolve_model_alias, validate_model_profile,
                            resolve_runtime_options, fp8_backend_environment)
import profiling_trace as trace


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=('bf16', 'awq', 'fp8-official', 'fp8-local'), required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--snapshot', type=Path, help='validated frozen snapshot directory')
    source.add_argument('--requests-file', type=Path, help='requests.jsonl inside its validated snapshot')
    p.add_argument('--case', default='w128_o128')
    p.add_argument('--count', type=int, default=1, help='measured generate calls')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--output-tokens', type=int)
    p.add_argument('--batch-tokens', type=int, help='override config max_num_batched_tokens')
    p.add_argument('--tokenizer', help='explicit override of the frozen reference tokenizer path')
    p.add_argument('--graph', action='store_true', help='allow compiled / CUDA Graph execution')
    p.add_argument('--nvtx', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--event-dir', type=Path, help='controller and worker stage JSONL directory')
    p.add_argument('--profiler', choices=('none', 'cuda', 'torch'), default='cuda')
    p.add_argument('--delay-iters', type=int, default=0)
    p.add_argument('--max-iters', type=int, default=0)
    p.add_argument('--torch-dir', type=Path)
    p.add_argument('--unique-output', action='store_true')
    p.add_argument('--plan-only', action='store_true', help='validate inputs and print plan without importing vLLM')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if min(a.count, a.batch_size) < 1 or a.warmup < 5:
        p.error('count and batch-size must be positive; warmup must be at least five')
    if (a.batch_tokens is not None and a.batch_tokens < 1
            or a.output_tokens is not None and a.output_tokens < 1):
        p.error('batch-tokens and output-tokens must be positive')
    if min(a.delay_iters, a.max_iters) < 0:
        p.error('iteration gates must be nonnegative')
    if a.profiler == 'torch' and a.torch_dir is None:
        p.error('torch profiler requires --torch-dir')
    if a.requests_file and a.requests_file.name != 'requests.jsonl':
        p.error('--requests-file must be the requests.jsonl belonging to a frozen snapshot')
    return a


def prepare(a):
    snapshot = a.snapshot or (a.requests_file.parent if a.requests_file else ROOT / 'data/frozen')
    snapshot = snapshot.resolve()
    _, rows, protocol = load_snapshot(snapshot)
    source_model = read(snapshot / 'model.json')
    select = lambda stage: sorted(
        (r for r in rows if r['case'] == a.case and r['phase'] == stage),
        key=lambda r: r['id'])
    warm, measured = select('warmup'), select('measure')
    if len(warm) < 5 or len(measured) < a.count * a.batch_size:
        raise ValueError('insufficient frozen warmup/measured requests')
    sample = measured[0]
    output_tokens = a.output_tokens or sample['output_tokens']
    max_len = sample['input_tokens'] + sample['output_tokens']
    if sample['input_tokens'] + output_tokens > max_len:
        raise ValueError('diagnostic output exceeds the original engine length limit')
    selected = warm + measured[:a.count * a.batch_size]
    if any(len(r['prompt_token_ids']) != sample['input_tokens']
           or r['input_tokens'] != sample['input_tokens']
           or r['output_tokens'] != sample['output_tokens'] for r in selected):
        raise ValueError('inconsistent frozen input/output lengths within the selected case')
    filename = 'fp8.yaml' if a.profile == 'fp8-local' else a.profile + '.yaml'
    config_path = ROOT / 'configs' / filename
    cfg = read(config_path)
    model = resolve_model_alias(a.profile)
    actual = read(Path(model) / 'config.json')
    validate_model_profile(actual, a.profile)
    options = resolve_runtime_options(actual, cfg['dtype'], cfg.get('quantization') or 'auto')
    kwargs = dict(
        model=model, tokenizer=a.tokenizer or cfg.get('tokenizer') or source_model['path'],
        dtype=options['dtype'], tensor_parallel_size=1, max_model_len=max_len,
        gpu_memory_utilization=cfg['gpu_memory_utilization'], max_num_seqs=a.batch_size,
        max_num_batched_tokens=a.batch_tokens or cfg['max_num_batched_tokens'],
        enable_chunked_prefill=True, enable_prefix_caching=False, kv_cache_dtype='auto',
        enforce_eager=not a.graph, seed=42, generation_config='vllm')
    if options['quantization'] != 'auto':
        kwargs['quantization'] = options['quantization']
    if a.nvtx:
        kwargs['worker_cls'] = 'profiling_worker.ProfilingWorker'
    if a.profiler != 'none':
        kwargs['profiler_config'] = dict(
            profiler=a.profiler, delay_iterations=a.delay_iters, max_iterations=a.max_iters)
        if a.profiler == 'torch':
            kwargs['profiler_config'].update(
                torch_profiler_dir=str(a.torch_dir.resolve()),
                torch_profiler_record_shapes=True, torch_profiler_with_stack=True,
                torch_profiler_with_memory=False, ignore_frontend=True)
    identity = dict(
        snapshot=str(snapshot), manifest_sha256=digest(snapshot / 'manifest.json'),
        requests_file_sha256=digest(snapshot / 'requests.jsonl'),
        source_model_sha256=digest(snapshot / 'model.json'),
        config_sha256=digest(config_path), actual_model_config_sha256=digest(Path(model) / 'config.json'),
        snapshot_validation='manifest file hashes passed',
        weight_verification='model format checked; full weight hashes not rechecked',
        case=a.case, input_tokens=sample['input_tokens'], output_tokens=output_tokens,
        frozen_output_tokens=sample['output_tokens'],
        warmup_ids=[[warm[(i+j) % len(warm)]['id'] for j in range(a.batch_size)]
                    for i in range(a.warmup)],
        measured_ids=[r['id'] for r in measured[:a.count * a.batch_size]])
    return kwargs, identity, warm, measured, output_tokens


def run(a):
    if a.unique_output:
        a.output = a.output.with_name(a.output.stem + '_' + str(os.getpid()) + a.output.suffix)
    if a.output.exists():
        raise FileExistsError(a.output)
    kwargs, identity, warm, measured, output_tokens = prepare(a)
    plan = dict(engine=kwargs, workload=identity, args=vars(a))
    print(json.dumps(plan, ensure_ascii=False, default=str), flush=True)
    if a.plan_only:
        return
    versions = {name: importlib.metadata.version(name) for name in ('vllm', 'torch', 'triton')}
    if versions['vllm'].split('+')[0] != '0.14.1':
        raise ValueError('worker instrumentation and iteration gates require vLLM 0.14.1')
    overrides = ('VLLM_TORCH_CUDA_PROFILE', 'VLLM_TORCH_PROFILER_DIR',
                 'VLLM_PROFILER_DELAY_ITERS', 'VLLM_PROFILER_MAX_ITERS')
    if any(os.environ.get(name) for name in overrides):
        raise ValueError('unset legacy profiler overrides: ' + ', '.join(overrides))
    os.environ.update(fp8_backend_environment('auto'))
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    scripts = str(Path(__file__).resolve().parent)
    os.environ['PYTHONPATH'] = scripts + os.pathsep + os.environ.get('PYTHONPATH', '')
    if a.event_dir:
        a.event_dir.mkdir(parents=True, exist_ok=True)
        os.environ['QUANT_PROFILE_EVENT_DIR'] = str(a.event_dir.resolve())
        trace.configure_logging(a.event_dir / 'controller.jsonl')
    else:
        os.environ.pop('QUANT_PROFILE_EVENT_DIR', None)
        trace.configure_logging()
    if a.nvtx:
        trace.require_nvtx()
    with trace.stage('offline/import_vllm', nvtx=a.nvtx):
        from vllm import LLM, SamplingParams
    with trace.stage('offline/initialize_engine', nvtx=a.nvtx):
        llm = LLM(**kwargs)
    params = SamplingParams(
        temperature=0, top_p=1, top_k=-1, repetition_penalty=1, seed=42,
        ignore_eos=True, min_tokens=output_tokens, max_tokens=output_tokens)

    def generate(items):
        result = llm.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in items],
                              sampling_params=params, use_tqdm=False)
        if len(result) != len(items):
            raise ValueError('output request count mismatch')
        if any(len(r.outputs[0].token_ids) != output_tokens for r in result):
            raise ValueError('fixed output length mismatch')
        return result

    with trace.stage('offline/warmup', nvtx=a.nvtx, calls=a.warmup):
        for i in range(a.warmup):
            generate([warm[(i+j) % len(warm)] for j in range(a.batch_size)])
    records = []
    if a.profiler != 'none':
        llm.start_profile()
    try:
        # Start after ProfilerStart and close before ProfilerStop so that NSYS
        # steady captures contain a completed measured scope in the controller.
        with trace.stage('offline/measure', nvtx=a.nvtx, calls=a.count):
            for i in range(a.count):
                items = measured[i*a.batch_size:(i+1)*a.batch_size]
                with trace.stage('offline/generate', nvtx=a.nvtx,
                                 request_ids=[r['id'] for r in items]):
                    start = time.perf_counter()
                    outputs = generate(items)
                    records.append(dict(ids=[r['id'] for r in items],
                                        wall_seconds=time.perf_counter()-start,
                                        output_tokens=[len(r.outputs[0].token_ids) for r in outputs]))
    finally:
        if a.profiler != 'none':
            llm.stop_profile()
    total_seconds = sum(record['wall_seconds'] for record in records)
    total_tokens = sum(sum(record['output_tokens']) for record in records)
    summary = dict(generate_calls=len(records),
                   requests=sum(len(record['ids']) for record in records),
                   generated_tokens=total_tokens, generate_wall_seconds=total_seconds,
                   output_tokens_per_second=total_tokens / total_seconds if total_seconds else None)
    controls = ('CUDA_VISIBLE_DEVICES', 'VLLM_TEST_FORCE_FP8_MARLIN',
                'VLLM_BATCH_INVARIANT', 'VLLM_USE_DEEP_GEMM', 'VLLM_WORKER_MULTIPROC_METHOD')
    report = dict(profile=a.profile, case=a.case, engine=kwargs, args=vars(a),
                  workload=identity, records=records, summary=summary, versions=versions,
                  execution_environment={name: os.environ.get(name) for name in controls},
                  requests_file_sha256=identity['requests_file_sha256'],
                  timing_note='offline LLM.generate wall time; profiler/replay overhead possible; not HTTP TTFT/TPOT')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+'\n',
                        encoding='utf-8')


def main(argv=None):
    try:
        run(parse_args(argv))
    finally:
        trace.close_logging()


if __name__ == '__main__':
    main()
