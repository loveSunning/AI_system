"""Opt-in NVTX worker for the audited vLLM 0.14.1 V1 GPU worker interface.

Ranges measure host submission scopes, not CUDA-synchronized kernel duration.
No site-packages patch, forward hook, or per-step CUDA synchronization is used.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import profiling_trace as trace
from profiling_stages import StepClassifier


def _configure_worker_logging():
    directory = os.environ.get("QUANT_PROFILE_EVENT_DIR")
    path = Path(directory)/f"worker_{os.getpid()}.jsonl" if directory else None
    trace.configure_logging(path)


# This opt-in module is loaded by vLLM. Any earlier vLLM imports remain outside
# this range; report cache state rather than claiming to time the entire import.
_configure_worker_logging()
trace.log_event("worker_module_entry", module=__name__)
with trace.stage("worker/nvtx_init"):
    trace.require_nvtx()
with trace.stage("worker/import_gpu_worker", nvtx=True,
                 module="vllm.v1.worker.gpu_worker",
                 already_imported="vllm.v1.worker.gpu_worker" in sys.modules):
    from vllm.v1.worker.gpu_worker import Worker


class ProfilingWorker(Worker):
    def __init__(self, *args, **kwargs):
        _configure_worker_logging()
        self._stage_classifier = StepClassifier()
        self._stage_step = 0
        self._stage_last_phase = None
        self._stage_log_interval = max(1, int(os.environ.get("QUANT_PROFILE_LOG_EVERY_N_STEPS", "32")))
        with trace.stage("worker/base_init", nvtx=True):
            super().__init__(*args, **kwargs)
        trace.log_event("worker_instrumentation_ready", rank=self.rank,
                        nvtx_domain=trace.DOMAIN, worker_class=type(self).__name__)

    def init_device(self, *args, **kwargs):
        with trace.stage("worker/init_device", nvtx=True, rank=self.rank):
            return super().init_device(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        with trace.stage("worker/load_model", nvtx=True, rank=self.rank):
            return super().load_model(*args, **kwargs)

    def determine_available_memory(self, *args, **kwargs):
        with trace.stage("worker/profile_memory", nvtx=True, rank=self.rank):
            return super().determine_available_memory(*args, **kwargs)

    def initialize_from_config(self, *args, **kwargs):
        with trace.stage("worker/initialize_kv_cache", nvtx=True, rank=self.rank):
            return super().initialize_from_config(*args, **kwargs)

    def compile_or_warm_up_model(self, *args, **kwargs):
        with trace.stage("worker/compile_and_graph_capture", nvtx=True, rank=self.rank):
            return super().compile_or_warm_up_model(*args, **kwargs)

    @contextmanager
    def annotate_profile(self, scheduler_output):
        summary = self._stage_classifier.classify(scheduler_output)
        self._stage_step += 1
        emit = (summary.phase != "decode" or self._stage_last_phase != "decode"
                or self._stage_step % self._stage_log_interval == 0)
        self._stage_last_phase = summary.phase
        with super().annotate_profile(scheduler_output):
            with trace.stage("worker/"+summary.phase, nvtx=True, emit_log=emit,
                             payload=self._stage_step, step=self._stage_step,
                             rank=self.rank, **summary.fields()):
                yield

    def sample_tokens(self, *args, **kwargs):
        with trace.nvtx_range("worker/sampling", enabled=True, payload=self._stage_step):
            return super().sample_tokens(*args, **kwargs)
