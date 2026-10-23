#!/usr/bin/env python3
"""Run the frozen quality holdout on BF16 and quantized models, then compare pairs."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import benchmark as bench
import evaluate_quality as evaluator
from experiment_utils import ROOT, digest, load_snapshot, read, rows, write, write_rows
from model_profiles import FP8_BACKENDS, canonical_profile, fp8_backend_environment
from test_server import Client, redact

INPUT_FILES = ("manifest.json", "model.json", "quality_holdout.jsonl",
               "quality_protocol.json", "chat_template.jinja")
LABELS = {"bf16": "BF16", "awq": "AWQ/W4A16", "fp8-official": "官方 FP8", "fp8-local": "本地 FP8"}


def check_holdout(snapshot):
    """Keep the existing scorer byte-for-byte compatible with historical snapshots."""
    manifest = read(snapshot / "manifest.json")
    if manifest["scorer_sha256"] != digest(Path(evaluator.__file__)):
        raise ValueError("Scorer changed after freeze; use the original scorer or prepare a new snapshot")
    for name in INPUT_FILES[1:]:
        if manifest["files"].get(name) != digest(snapshot / name):
            raise ValueError("Snapshot hash mismatch: " + name)
    tasks = rows(snapshot / "quality_holdout.jsonl")
    if len(tasks) != 100 or len({t["id"] for t in tasks}) != 100:
        raise ValueError("Require exactly 100 unique quality task IDs")
    if dict(Counter(t["category"] for t in tasks)) != dict.fromkeys(
            ("arithmetic", "extraction", "choice", "json"), 25):
        raise ValueError("Require four frozen categories, 25 tasks each")
    for task in tasks:
        answer = json.dumps(task["expected"]) if task["scorer"] == "json_exact" else task["expected"]
        if not evaluator.score(task, answer, "stop"):
            raise ValueError("Invalid reference answer: " + task["id"])
    protocol = read(snapshot / "quality_protocol.json")
    expected = {"temperature": 0, "top_p": 1, "top_k": -1, "seed": 42,
                "max_tokens": 128, "ignore_eos": False, "thinking": False}
    if protocol["generation"] != expected or protocol["denominator"] != 100:
        raise ValueError("Unsupported quality generation protocol")
    if protocol["quality_target_drop_percentage_points"] != 2:
        raise ValueError("Quality gate must be frozen at 2 percentage points before evaluation")
    return tasks, protocol


def prepare_inputs(snapshot, source, tasks, max_model_len):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source["path"], local_files_only=True)
    if tokenizer.get_chat_template() != (snapshot / "chat_template.jinja").read_text(encoding="utf-8"):
        raise ValueError("Reference chat template differs from the frozen template")
    requests = []
    for task in tasks:
        messages = [{"role": "user", "content": task["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + 128 > max_model_len:
            raise ValueError("Quality prompt plus output budget exceeds max-model-len: " + task["id"])
        requests.append(dict(id=task["id"], messages=messages, rendered_prompt=text,
                             prompt_token_ids=ids, input_tokens=len(ids)))
    return requests


def launch_command(config, max_model_len):
    # Reuse the server settings. This sum is a context budget, not a forced
    # quality input/output length; the evaluation payload uses normal EOS.
    return bench.launch_command(config, {"input_tokens": max_model_len - 128, "output_tokens": 128})


def wait_port(port, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Port {port} is busy; stop the existing server before running quality comparison")
            time.sleep(1)


def run_model(config, model, snapshot, directory, max_model_len):
    directory.mkdir(parents=True)
    record = {"profile": config["profile"], "status": "failed", "precision_verified": False,
              "command": launch_command(config, max_model_len), "config": config, "model": model}
    process = None
    try:
        wait_port(config["port"])
        with (directory / "server.log").open("w", encoding="utf-8") as log:
            env = fp8_backend_environment(config.get("fp8_backend", "auto"))
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            env["QUANT_PROFILE_NVTX"] = "0"
            env.pop("QUANT_PROFILE_EVENT_DIR", None)
            process = subprocess.Popen(record["command"], stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=env)
            args = bench.client_args(config)
            probe = Client(SimpleNamespace(**dict(vars(args), timeout=2)))
            deadline = time.monotonic() + config["startup_timeout_seconds"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Server exited during startup; inspect server.log")
                try:
                    probe.health()
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server startup timeout; inspect server.log")
                    time.sleep(1)
            record["engine_verification"] = bench.verify_engine(
                (directory / "server.log").read_text(encoding="utf-8"), config, model)
            if config["served_model_name"] not in probe.models()["model_ids"]:
                raise ValueError("Served model name does not match the selected deployment")
            record["precision_verified"] = True
            print(f"{config['profile']}: evaluating 100 quality tasks (normal EOS, concurrency 1)", flush=True)
            status = evaluator.main(["--snapshot", str(snapshot), "--base-url", args.base_url,
                "--model", config["served_model_name"], "--label", config["profile"],
                "--output", str(directory / "evaluation")])
            record["status"] = "complete" if status == 0 else "incomplete"
    except Exception as error:
        record["error"] = redact(str(error), os.environ.get("VLLM_API_KEY", ""))
    finally:
        # Clean up only this run's process group, including API/worker children.
        try:
            bench.stop_process(process)
        except Exception as error:
            record.update(status="failed", cleanup_error=str(error))
        record["artifacts_sha256"] = {
            path.relative_to(directory).as_posix(): digest(path)
            for path in (directory / "server.log", directory / "evaluation/results.jsonl",
                         directory / "evaluation/summary.json") if path.is_file()}
        write(directory / "deployment.json", record)
    return record


def read_result(directory, task_list, requests, manifest_sha):
    """Re-score raw outputs; infrastructure failures never become quality scores."""
    record = read(directory / "deployment.json")
    errors = []
    for name, sha in record.get("artifacts_sha256", {}).items():
        if digest(directory / name) != sha:
            raise ValueError("Result artifact hash mismatch: " + name)
    if record.get("status") != "complete" or not record.get("precision_verified"):
        errors.append(record.get("error", record.get("cleanup_error", "Deployment did not complete")))
    results_path = directory / "evaluation/results.jsonl"
    results = rows(results_path) if results_path.exists() else []
    tasks = {task["id"]: task for task in task_list}
    if len(results) != 100 or len({r["id"] for r in results}) != 100 or {r["id"] for r in results} != tasks.keys():
        errors.append("Require the same 100 unique task IDs as the holdout")
    summary_path = directory / "evaluation/summary.json"
    if summary_path.exists():
        saved = read(summary_path)
        if saved.get("manifest_sha256") != manifest_sha or saved.get("label") != record["profile"]:
            errors.append("Evaluation snapshot or model label mismatch")
    else:
        errors.append("Missing evaluation summary")
    indexed = {}
    for item in results:
        task = tasks.get(item["id"])
        if task is None:
            continue
        if item.get("category") != task["category"] or item.get("expected") != task["expected"]:
            errors.append("Task metadata mismatch: " + task["id"])
        correct = False
        if item.get("ok") is True:
            response = item["result"]
            if (response.get("usage") or {}).get("prompt_tokens") != requests[task["id"]]["input_tokens"]:
                errors.append("Rendered input token count mismatch: " + task["id"])
            correct = evaluator.score(task, response["text"], response["finish_reason"])
            if item.get("correct") is not correct:
                errors.append("Saved score differs from frozen scorer: " + task["id"])
        indexed[item["id"]] = dict(item, correct=correct)
    infrastructure = sum(item.get("ok") is not True for item in results)
    if infrastructure:
        errors.append(f"{infrastructure} infrastructure failures; quality score withheld")
    complete = not errors
    categories = {}
    for category, total in sorted(Counter(t["category"] for t in task_list).items()):
        count = sum(r["correct"] for r in indexed.values() if r["category"] == category)
        categories[category] = {"correct": count if complete else None, "total": total,
                                "accuracy": count / total if complete else None}
    correct = sum(r["correct"] for r in indexed.values()) if complete else None
    return {"profile": record["profile"], "complete": complete, "status": "complete" if complete else "incomplete",
            "precision_verified": record.get("precision_verified", False), "total": 100,
            "received": len(results), "correct": correct, "accuracy": correct / 100 if complete else None,
            "infrastructure_failures": infrastructure,
            "truncated": sum((r.get("result") or {}).get("finish_reason") == "length" for r in results),
            "errors": errors, "categories": categories,
            "base_source_verification": record.get("model", {}).get("base_source_verification", "unknown")}, indexed


def csv_file(path, records, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in fields} for record in records)


def summarize(directory):
    run = read(directory / "run.json")
    inputs = directory / "inputs"
    if digest(inputs / "manifest.json") != run["manifest_sha256"]:
        raise ValueError("Archived snapshot manifest changed")
    tasks, protocol = check_holdout(inputs)
    if digest(directory / "quality_requests.jsonl") != run["quality_requests_sha256"]:
        raise ValueError("Quality request snapshot changed")
    requests = {r["id"]: r for r in rows(directory / "quality_requests.jsonl")}
    if set(requests) != {t["id"] for t in tasks}:
        raise ValueError("Quality request IDs differ from holdout")
    models, indexed = {}, {}
    for profile in run["profiles"]:
        models[profile], indexed[profile] = read_result(directory / profile, tasks, requests, run["manifest_sha256"])
    baseline = models["bf16"]
    pairs, regressions, comparisons = [], [], {}
    target = protocol["quality_target_drop_percentage_points"]
    for profile, model in models.items():
        if profile == "bf16":
            continue
        result = {"delta_percentage_points": None, "gate_passed": None,
                  "regressions": None, "improvements": None, "categories": {}}
        if baseline["complete"] and model["complete"]:
            counts = Counter()
            for task in tasks:
                base, quant = indexed["bf16"][task["id"]], indexed[profile][task["id"]]
                change = ("both_correct" if base["correct"] and quant["correct"] else
                          "regression" if base["correct"] else "improvement" if quant["correct"] else "both_wrong")
                counts[change] += 1
                pair = {"profile": profile, "id": task["id"], "category": task["category"],
                        "prompt": task["prompt"], "expected": task["expected"], "change": change,
                        "bf16": base["result"], "quantized": quant["result"]}
                pairs.append(pair)
                if change == "regression":
                    regressions.append(pair)
            # Integer counts keep the inclusive 2 pp boundary exact.
            delta = (model["correct"] - baseline["correct"]) * 100 / 100
            result.update(delta_percentage_points=delta, gate_passed=delta >= -target,
                          regressions=counts["regression"], improvements=counts["improvement"],
                          both_correct=counts["both_correct"], both_wrong=counts["both_wrong"])
            for category, values in model["categories"].items():
                base = baseline["categories"][category]
                result["categories"][category] = {
                    "delta_percentage_points": (values["correct"] - base["correct"]) * 100 / base["total"]}
        comparisons[profile] = result
    complete = all(model["complete"] for model in models.values())
    gate = all(c["gate_passed"] is True for c in comparisons.values()) if complete else None
    summary = {"run_id": run["run_id"], "manifest_sha256": run["manifest_sha256"],
               "quality_requests_sha256": run["quality_requests_sha256"], "complete": complete,
               "status": "passed" if gate is True else "failed" if gate is False else "incomplete",
               "quality_target_drop_percentage_points": target, "all_quality_gates_passed": gate,
               "models": models, "comparisons": comparisons}
    write(directory / "summary.json", summary)
    write_rows(directory / "paired_results.jsonl", pairs)
    write_rows(directory / "regressions.jsonl", regressions)
    table, categories = [], []
    for profile, model in models.items():
        comparison = comparisons.get(profile, {})
        table.append(dict(profile=profile, status=model["status"], correct=model["correct"], total=100,
            accuracy_percent=model["correct"], delta_percentage_points=comparison.get("delta_percentage_points"),
            gate_passed=comparison.get("gate_passed"), regressions=comparison.get("regressions"),
            improvements=comparison.get("improvements"), infrastructure_failures=model["infrastructure_failures"],
            truncated=model["truncated"]))
        for category, values in model["categories"].items():
            categories.append(dict(profile=profile, category=category, **values,
                delta_percentage_points=comparison.get("categories", {}).get(category, {}).get("delta_percentage_points")))
    csv_file(directory / "quality.csv", table, list(table[0]))
    csv_file(directory / "quality_categories.csv", categories, list(categories[0]))
    report = ["# 四模型质量对比报告", "", f"实验 ID：{run['run_id']}", "",
              f"快照 SHA256：{run['manifest_sha256']}", "",
              "固定 100 条任务，每类 25 条；greedy、seed=42、thinking=false、正常 EOS，最多 128 个输出 token。",
              f"预先冻结的门槛：相对 BF16 的准确率下降不超过 {target} 个百分点（含边界）。", "",
              "| 模型 | 状态 | 正确/总数 | 准确率 | 相对 BF16 (pp) | 新增错误 | 修复错误 | 门槛 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in table:
        available = row["correct"] is not None
        comparison = comparisons.get(row["profile"], {})
        delta, passed = comparison.get("delta_percentage_points"), comparison.get("gate_passed")
        values = [LABELS[row["profile"]], row["status"],
                  f"{row['correct']}/100" if available else f"—/100（收到 {models[row['profile']]['received']} 条）",
                  f"{row['correct']:.1f}%" if available else "—", f"{delta:+.1f}" if delta is not None else "—",
                  comparison.get("regressions"), comparison.get("improvements"),
                  "基线" if row["profile"] == "bf16" else "通过" if passed is True else "未通过" if passed is False else "未完成"]
        report.append("| " + " | ".join("—" if value is None else str(value) for value in values) + " |")
    report += ["", "## 分项得分", "", "| 模型 | 类型 | 正确/总数 | 相对 BF16 (pp) |", "| --- | --- | --- | --- |"]
    for row in categories:
        delta = row["delta_percentage_points"]
        report.append(f"| {LABELS[row['profile']]} | {row['category']} | "
                      f"{row['correct'] if row['correct'] is not None else '—'}/{row['total']} | "
                      f"{f'{delta:+.1f}' if delta is not None else '—'} |")
    report += ["", "## 失败与证据", ""]
    for profile, model in models.items():
        report.append(f"- {LABELS[profile]}：基础设施失败 {model['infrastructure_failures']}；截断 {model['truncated']}；"
                      f"模型来源核验：{model['base_source_verification']}。")
        report.extend(f"  - {error}" for error in model["errors"])
    report += ["", f"最终状态：**{summary['status']}**。", "",
               "逐题原始回答见各模型 evaluation/results.jsonl；逐题配对见 paired_results.jsonl；",
               "BF16 答对而量化答错的完整问题、参考答案和双方回答见 regressions.jsonl。",
               "服务命令、模型文件指纹及引擎 dtype/量化方法见 deployment.json 和 server.log。", "",
               "100 条自编基础合成任务仅用于本项目的回归检查，可能出现满分和不敏感的情况；",
               "1 题等于 1 个百分点，分项 1 题等于 4 个百分点。通过不证明广泛任务无退化，也不保证统计显著。",
               "净正确数可能抵消新增错误与修复，需结合配对记录解读。官方量化权重只核验结构兼容，",
               "不据此保证与 BF16 来自同一原始 revision。此报告不替代性能或 GPU kernel 分析。", ""]
    (directory / "Quality_Report.md").write_text("\n".join(report), encoding="utf-8")
    return 0 if gate is True else 2 if gate is False else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/frozen")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", action="append", help="repeat; include bf16 and at least one quantized model")
    selection.add_argument("--all-models", action="store_true", help="default: BF16, AWQ, official FP8, local FP8")
    parser.add_argument("--max-model-len", type=bench.positive_int, default=2048)
    parser.add_argument("--enforce-eager", type=bench.boolean_value, default=False)
    parser.add_argument("--fp8-backend", choices=FP8_BACKENDS, default="auto")
    parser.add_argument("--output", type=Path, help="new result directory; default: log/quality_<ID>")
    parser.add_argument("--plan-only", action="store_true", help="verify snapshot and print commands without loading models")
    parser.add_argument("--summarize", type=Path, metavar="RUN", help="rebuild a managed quality report without CUDA or Transformers")
    args = parser.parse_args(argv)
    if args.summarize:
        return summarize(args.summarize)
    try:
        manifest, _, performance_protocol = load_snapshot(args.snapshot)
        tasks, protocol = check_holdout(args.snapshot)
        source = read(args.snapshot / "model.json")
        args.model = [canonical_profile(name) for name in args.model] if args.model else None
        args.config, args.all_models, args.nvtx = None, args.model is None, False
        configs = bench.load_configs(args, source)
        profiles = [config["profile"] for config in configs]
        if "bf16" not in profiles or len(profiles) < 2:
            raise ValueError("Comparison requires bf16 and at least one quantized model")
        configs.sort(key=lambda config: config["profile"] != "bf16")
        profiles = [config["profile"] for config in configs]
        if args.max_model_len <= protocol["generation"]["max_tokens"]:
            raise ValueError("max-model-len must exceed the 128-token output budget")
        for config in configs:
            bench.validate(config, dict(performance_protocol, concurrency=1))
        plan = {"profiles": profiles, "tasks_per_model": 100, "total_requests": len(configs) * 100,
                "generation": protocol["generation"], "quality_target_drop_percentage_points": 2,
                "max_model_len": args.max_model_len,
                "commands": [launch_command(config, args.max_model_len) for config in configs]}
        if args.plan_only:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if sys.platform != "linux":
            raise ValueError("Managed model serving requires the Linux GPU host; use --plan-only or --summarize locally")
        runtime = bench.check_runtime_versions()
        if args.fp8_backend == "marlin" and runtime["versions"]["vllm"].split("+")[0] != "0.14.1":
            raise ValueError("Explicit FP8 Marlin requires the unified vLLM 0.14.1 environment")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = args.output or ROOT / "log" / ("quality_" + run_id)
    if directory.exists():
        parser.error("Output exists; use a new directory to preserve previous results")
    directory.mkdir(parents=True)
    print("Artifacts:", directory, flush=True)
    inputs = directory / "inputs"
    inputs.mkdir()
    for name in INPUT_FILES:
        shutil.copyfile(args.snapshot / name, inputs / name)
    run = dict(plan, run_id=run_id, snapshot=str(args.snapshot.resolve()), runtime=runtime,
               manifest_sha256=digest(args.snapshot / "manifest.json"),
               scorer_sha256=manifest["scorer_sha256"],
               scripts_sha256={path.name: digest(path) for path in (ROOT / "scripts").glob("*.py")})
    write(directory / "run.json", run)
    try:
        # Verify frozen base tokenizer files before rendering the common requests.
        baseline_model = bench.verify_model(configs[0], source)
        requests = prepare_inputs(args.snapshot, source, tasks, args.max_model_len)
        write_rows(directory / "quality_requests.jsonl", requests)
        run["quality_requests_sha256"] = digest(directory / "quality_requests.jsonl")
        write(directory / "run.json", run)
        with (directory / "environment.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/collect_environment.py"), "--output",
                str(directory / "environment.json")], stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=60, check=True)
        (directory / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        for config in configs:
            model_dir = directory / config["profile"]
            try:
                model = baseline_model if config["profile"] == "bf16" else bench.verify_model(config, source)
                bench.validate(config, dict(performance_protocol, concurrency=1), model)
                run_model(config, model, args.snapshot, model_dir, args.max_model_len)
            except Exception as error:
                write(model_dir / "deployment.json", {"profile": config["profile"], "status": "failed",
                    "precision_verified": False, "error": redact(str(error), os.environ.get("VLLM_API_KEY", ""))})
        result = summarize(directory)
    except Exception as error:
        write(directory / "failure.json", {"error": redact(str(error), os.environ.get("VLLM_API_KEY", ""))})
        print("Quality run incomplete; inspect failure.json:", error, file=sys.stderr)
        return 1
    print("Report:", directory / "Quality_Report.md", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
