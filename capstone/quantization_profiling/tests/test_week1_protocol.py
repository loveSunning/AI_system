"""Protocol tests use synthetic HTTP responses, never report them as GPU results."""
import asyncio
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark import phase, perform, client_args, validate, launch_command
from evaluate_quality import score
from experiment_utils import ROOT, read, rows, digest, write, load_snapshot, failure_kind
from prepare_workloads import exact_prompt
from profiling_middleware import ProfilingMiddleware


class Handler(BaseHTTPRequestHandler):
    missing = False
    wrong = False
    def log_message(self, *args):
        pass
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/completions"
        assert isinstance(payload["prompt"], list)
        assert payload["ignore_eos"] and payload["min_tokens"] == payload["max_tokens"]
        n = payload["max_tokens"]
        data = [dict(id="test", choices=[dict(index=0, text="hello", finish_reason=None)]),
                dict(id="test", choices=[dict(index=0, text=" world", finish_reason="length")]),
                dict(id="test", choices=[], usage=dict(prompt_tokens=len(payload["prompt"])+(1 if self.wrong else 0), completion_tokens=n))]
        body = b"".join(b"data: " + json.dumps(x).encode() + b"\n\n" for x in data) + b"data: [DONE]\n\n"
        async def app(scope, receive, send):
            await send(dict(type="http.response.start", status=200, headers=[(b"content-type", b"text/event-stream")]))
            for part in (body[:17], body[17:]):
                await send(dict(type="http.response.body", body=part, more_body=True))
            await send(dict(type="http.response.body", body=b"", more_body=False))
        async def send(message):
            if message["type"] == "http.response.start":
                self.send_response(message["status"])
                for k,v in message["headers"]: self.send_header(k.decode(),v.decode())
                self.end_headers()
            else: self.wfile.write(message.get("body",b""))
        runner = app if self.missing else ProfilingMiddleware(app)
        asyncio.run(runner(dict(type="http",method="POST",path=self.path),None,send))


class WeekOneTests(unittest.TestCase):
    def test_holdout_counts_reference_answers_and_strict_grading(self):
        tasks=rows(ROOT / "data/quality_holdout.jsonl")
        self.assertEqual(len({t["id"] for t in tasks}),100)
        self.assertEqual(set(Counter(t["category"] for t in tasks).values()),{25})
        for t in tasks:
            answer=json.dumps(t["expected"]) if t["scorer"]=="json_exact" else t["expected"]
            self.assertTrue(score(t,answer,"stop"))
            self.assertFalse(score(t,answer,"length"))
            self.assertFalse(score(t,answer+" explanation","stop"))
        task=next(t for t in tasks if t["scorer"]=="json_exact")
        bad=dict(task["expected"],active=1)
        self.assertFalse(score(task,json.dumps(bad),"stop"))
        self.assertFalse(score(task,'{"name":"user_00","age":20,"active":true,"active":true}',"stop"))

    def test_exact_prompt_checks_full_template(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                self.kwargs=kwargs
                return "BEGIN " + messages[0]["content"] + " END ASSISTANT THINK_OFF"
            def encode(self,text,**kwargs): return list(range(len(text.split())))
        tok=Tokenizer()
        for target in (128,1024,4096,8192):
            item=exact_prompt(tok,target,0)
            self.assertEqual(len(item["prompt_token_ids"]),target)
            self.assertTrue(item["rendered_prompt"].endswith("END ASSISTANT THINK_OFF"))
            self.assertFalse(tok.kwargs["enable_thinking"])

    def test_tampered_snapshot_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp); write(path/"workloads.yaml",{})
            write(path/"manifest.json",{"files":{"workloads.yaml":digest(path/"workloads.yaml")}})
            write(path/"workloads.yaml",{"changed":True})
            with self.assertRaisesRegex(ValueError,"hash mismatch"): load_snapshot(path)

    def test_bf16_rejects_quantized_weights_and_small_protocol(self):
        config=read(ROOT/"configs/bf16.yaml"); protocol=read(ROOT/"configs/workloads.yaml")
        validate(config,protocol,{"config":{}})
        with self.assertRaises(ValueError): validate(config,protocol,{"config":{"quantization_config":{"quant_method":"awq"}}})
        with self.assertRaises(ValueError): validate(config,dict(protocol,rounds=0),{"config":{}})
        self.assertIn("8320",launch_command(config,protocol["cases"][3]))

    def test_sse_middleware_to_benchmark_rounds_and_failure_records(self):
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        config=read(ROOT/"configs/bf16.yaml"); config["port"]=server.server_port
        args=client_args(config)
        item={"id":"test", "case":"test", "prompt_token_ids":[1,2,3], "input_tokens":3,"output_tokens":4}
        try:
            with tempfile.TemporaryDirectory() as temp:
                output=Path(temp)/"requests.jsonl"
                for round_id in range(1,4):
                    summary=phase(args,config,[dict(item,id=str(i)) for i in range(4)],round_id,"measure",2,output)
                    self.assertEqual(summary["successful_requests"],4)
                    self.assertEqual(summary["output_tokens"],16)
                    self.assertGreater(summary["output_tokens_per_second"],0)
                self.assertEqual(len(rows(output)),12)
                with patch.object(Handler,"missing",True):
                    bad=perform(args,config,item,1,"warmup")
                    self.assertFalse(bad["ok"])
                    self.assertEqual(bad["failure_kind"],"missing_metrics")
                with patch.object(Handler,"wrong",True):
                    bad=perform(args,config,item,1,"measure")
                    self.assertFalse(bad["ok"])
                    self.assertIn("token mismatch",bad["error"])
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_error_classification_preserves_capacity_and_timeout(self):
        self.assertEqual(failure_kind("CUDA out of memory"),"oom")
        self.assertEqual(failure_kind("No available memory for KV cache"),"kv_capacity")
        self.assertEqual(failure_kind("read timed out"),"timeout")
        self.assertEqual(failure_kind("HTTP 400: too long"),"rejected")

if __name__ == "__main__": unittest.main()
