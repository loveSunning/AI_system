"""CPU-only checks for real scheduler classification and balanced annotations."""
from contextlib import contextmanager
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
import benchmark as bench
import profiling_trace as trace
from profiling_stages import StepClassifier
import run_model
import test_server


def output(new=(), cached=(), positions=(), scheduled=None, finished=()):
    return SimpleNamespace(scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(req_ids=list(cached), num_computed_tokens=list(positions)),
        num_scheduled_tokens=scheduled or {}, finished_req_ids=set(finished))


def new_request(key="a", prompt=128, position=0):
    return SimpleNamespace(req_id=key,prompt_token_ids=list(range(prompt)),
                           num_computed_tokens=position,prompt_embeds=None)


class FakeNVTX:
    def __init__(self):self.active=set();self.started=[];self.ranges=[];self.open_ranges=[];self.next_id=0
    def start_range(self,message,**kwargs):
        self.next_id+=1;self.active.add(self.next_id);self.started.append(message)
        return self.next_id
    def end_range(self,handle):self.active.remove(handle)
    @contextmanager
    def annotate(self,message,**kwargs):
        self.ranges.append(message)
        self.open_ranges.append(message)
        try:
            yield
        finally:
            self.open_ranges.pop()


class StageClassifierTests(unittest.TestCase):
    def test_chunked_prefill_one_token_tail_then_decode(self):
        tracker=StepClassifier()
        self.assertEqual(tracker.classify(output([new_request(prompt=129)],scheduled={"a":128})).phase,"prefill")
        tail=tracker.classify(output(cached=["a"],positions=[128],scheduled={"a":1}))
        self.assertEqual((tail.phase,tail.prefill_tokens,tail.decode_tokens),("prefill",1,0))
        step=tracker.classify(output(cached=["a"],positions=[129],scheduled={"a":1}))
        self.assertEqual((step.phase,step.prefill_tokens,step.decode_tokens),("decode",0,1))

    def test_mixed_batches_resume_and_metadata_gap(self):
        tracker=StepClassifier()
        tracker.classify(output([new_request()],scheduled={"a":128}))
        mixed=tracker.classify(output([new_request("b",64)],cached=["a"],positions=[128],scheduled={"a":1,"b":64}))
        self.assertEqual((mixed.phase,mixed.prefill_tokens,mixed.decode_tokens),("mixed",64,1))
        resumed=tracker.classify(output(cached=["b"],positions=[32],scheduled={"b":16}))
        self.assertEqual(resumed.phase,"prefill")
        self.assertEqual(tracker.classify(output(scheduled={"missing":1})).phase,"unknown")
        tracker.classify(output(finished=["a","b"]))
        self.assertEqual(tracker.prompt_lengths,{})

    def test_empty_or_crossing_step_is_not_guessed_as_decode(self):
        tracker=StepClassifier()
        self.assertEqual(tracker.classify(output()).phase,"idle")
        crossing=tracker.classify(output([new_request(position=127)],scheduled={"a":3}))
        self.assertEqual((crossing.phase,crossing.prefill_tokens,crossing.decode_tokens),("mixed",1,2))


class InstrumentationTests(unittest.TestCase):
    def setUp(self):
        kill_signal=patch.object(bench.signal,"SIGKILL",9,create=True)
        kill_signal.start()
        self.addCleanup(kill_signal.stop)

    def tearDown(self):trace.close_logging()

    def test_disabled_annotations_do_not_import_optional_backend(self):
        with patch.object(trace,"require_nvtx") as backend:
            with trace.nvtx_range("disabled"):pass
            span=trace.ProcessRange("disabled").start();span.finish()
        backend.assert_not_called()

    def test_stage_range_balances_when_body_raises(self):
        fake=FakeNVTX()
        with patch.object(trace,"require_nvtx",return_value=fake):
            with self.assertRaisesRegex(ValueError,"boom"):
                with trace.stage("worker/prefill",nvtx=True):raise ValueError("boom")
        self.assertEqual(fake.ranges,["worker/prefill"])
        self.assertFalse(fake.open_ranges)

    def test_request_first_text_callback_and_failure_end_all_ranges(self):
        fake=FakeNVTX()
        class Client:
            fail=False
            def __init__(self,args):self.args=args
            def stream_chat(self,payload,path):
                self.args.on_first_text(.1)
                if self.fail:raise RuntimeError("stream failed")
                return {"usage":{"prompt_tokens":128,"completion_tokens":128},
                        "server_metrics":{"input_tokens":128,"output_tokens":128,
                           "inference_seconds":.3,"output_tokens_per_second":128/.3,"request_id":"server-a"},
                        "client_metrics":{"end_to_end_seconds":.4,"time_to_first_text_seconds":.1,
                                          "time_to_last_text_seconds":.35}}
        config={"profile":"bf16","served_model_name":"local-model","nvtx":True}
        item={"id":"w128-0","case":"w128_o128","input_tokens":128,"output_tokens":128,"prompt_token_ids":[0]*128}
        for fail in (False,True):
            Client.fail=fail
            with patch.object(trace,"require_nvtx",return_value=fake),patch.object(bench,"Client",Client):
                result=bench.perform(SimpleNamespace(),config,item,1,"measure")
            self.assertEqual(result["ok"],not fail)
            self.assertFalse(fake.active)
        self.assertEqual(fake.started.count("client/wait_first_text"),2)
        self.assertEqual(fake.started.count("client/first_text_to_stream_end"),2)

    def test_first_text_callback_fires_once_after_nonempty_completion_text(self):
        callback=Mock()
        client=test_server.Client(SimpleNamespace(base_url="http://localhost",timeout=1,
              api_key_env="UNSET_TEST_API_KEY",no_proxy=True,on_first_text=callback))
        response=Mock();response.__enter__=Mock(return_value=SimpleNamespace(headers={"Content-Type":"text/event-stream"}))
        response.__exit__=Mock(return_value=False)
        events=[json.dumps({"choices":[{"text":""}]}),
                json.dumps({"choices":[{"text":"a"}]}),
                json.dumps({"choices":[{"text":"b","finish_reason":"length"}],
                            "usage":{"completion_tokens":2}}),"[DONE]"]
        with patch.object(client,"open",return_value=response),patch.object(test_server,"iter_sse_data",return_value=iter(events)):
            result=client.stream_chat({},path="/v1/completions")
        callback.assert_called_once()
        self.assertEqual(result["text"],"ab")

    def test_nvtx_plan_is_read_only_and_does_not_load_backend(self):
        from experiment_utils import write,write_rows,digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol=json.loads((SCRIPTS.parent/'configs/workloads.yaml').read_text())
            protocol.update(requests=1,cases=protocol['cases'][:1])
            write(root/'workloads.yaml',protocol)
            write(root/'model.json',{"path":"/unused/base","config":{},"files":{}})
            write_rows(root/'requests.jsonl',[{"id":f"w128_o128-{i:04d}","case":"w128_o128",
                "phase":"warmup" if i<5 else "measure","input_tokens":128,"output_tokens":128,
                "prompt_token_ids":[0]*128} for i in range(6)])
            write(root/'manifest.json',{"files":{f.name:digest(f) for f in root.iterdir()}})
            buf=io.StringIO()
            with patch.object(bench,"require_nvtx") as backend,contextlib.redirect_stdout(buf),contextlib.redirect_stderr(io.StringIO()):
                code=bench.main(["--snapshot",str(root),"--model","bf16","--requests","1","--nvtx","--plan-only"])
            backend.assert_not_called()
            self.assertEqual(code,0)
            plan=json.loads(buf.getvalue())
            self.assertTrue(plan['commands'][0]['nvtx_enabled'])
            self.assertIn('--nvtx',plan['commands'][0]['command'])

    def test_worker_class_is_only_selected_explicitly(self):
        plain=run_model.server_command(run_model.parse_args(["--model","bf16","--serve"]))
        marked=run_model.server_command(run_model.parse_args(["--model","bf16","--serve","--nvtx"]))
        self.assertNotIn('--worker-cls',plain)
        self.assertEqual(marked[marked.index('--worker-cls')+1],"profiling_worker.ProfilingWorker")

    def test_shutdown_does_not_kill_workers_as_soon_as_wrapper_exits(self):
        process=SimpleNamespace(pid=12345,poll=Mock(return_value=0),wait=Mock())
        with patch.object(bench.os,'killpg',create=True) as killpg, \
             patch.object(bench.time,'monotonic',side_effect=[0,.1,.2,15.1]), \
             patch.object(bench.time,'sleep'):
            bench.stop_process(process)
        self.assertEqual([c.args for c in killpg.call_args_list],
                         [(12345,signal.SIGTERM),(12345,0),(12345,0),(12345,signal.SIGKILL)])
        self.assertEqual(process.poll.call_count,2)

    def test_shutdown_avoids_sigkill_when_owned_group_exits(self):
        process=SimpleNamespace(pid=12345,poll=Mock(return_value=0),wait=Mock())
        with patch.object(bench.os,'killpg',side_effect=[None,None,ProcessLookupError()],create=True) as killpg, \
             patch.object(bench.time,'monotonic',side_effect=[0,.1,.2]),patch.object(bench.time,'sleep'):
            bench.stop_process(process)
        self.assertNotIn((12345,signal.SIGKILL),[c.args for c in killpg.call_args_list])

    def test_worker_hooks_preserve_base_calls_and_use_scheduler_phases(self):
        class BaseWorker:
            def __init__(self):self.rank=0
            def load_model(self):return 'loaded'
            @contextmanager
            def annotate_profile(self,out):yield
            def execute_model(self,out):
                with self.annotate_profile(out):return 'forward'
            def sample_tokens(self,arg):return arg
        modules={name:ModuleType(name) for name in ('vllm','vllm.v1','vllm.v1.worker','vllm.v1.worker.gpu_worker')}
        modules['vllm.v1.worker.gpu_worker'].Worker=BaseWorker
        spec=importlib.util.spec_from_file_location('test_quant_worker',SCRIPTS/'profiling_worker.py')
        module=importlib.util.module_from_spec(spec)
        fake=FakeNVTX()
        with patch.dict(sys.modules,modules),patch.object(trace,'require_nvtx',return_value=fake), \
             patch.object(trace,'configure_logging'):
            spec.loader.exec_module(module)
            worker=module.ProfilingWorker()
            self.assertEqual(worker.load_model(),'loaded')
            self.assertEqual(worker.execute_model(output([new_request()],scheduled={'a':128})),'forward')
            self.assertEqual(worker.execute_model(output(cached=['a'],positions=[128],scheduled={'a':1})),'forward')
            self.assertEqual(worker.sample_tokens('sample'),'sample')
        self.assertEqual(fake.ranges,['worker/load_model','worker/prefill','worker/decode','worker/sampling'])


if __name__=='__main__':unittest.main()
