#!/usr/bin/env python3
"""Create an auditable week-1 report from one managed benchmark run."""
import argparse
from pathlib import Path
from experiment_utils import ROOT, read, rows, write_rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, help="default: RUN/Week1_Report.md")
    a = p.parse_args(argv)
    run, summary = read(a.run / "run.json"), read(a.run / "summary.json")
    lines = ["# 第一周 BF16 实验报告", "", f"实验 ID：{run['run_id']}",
        f"快照 SHA256：{run['manifest_sha256']}", "",
        "| 负载 | 状态 | 轮次 | 成功/总数 | 吞吐 token/s | E2E p95(s) | TTFT文本事件 p95(s) |", "|---|---|---|---|---|---|---|"]
    records = []
    for case in summary["cases"]:
        path = a.run / case["case"] / "requests.jsonl"
        if path.exists():
            records.extend(dict(e, run_id=run["run_id"]) for e in rows(path))
        measured = [x for x in case["rounds"] if x["phase"] == "measure"]
        for s in measured:
            lines.append(f"|{case['case']}|{case['status']}|{s['round']}|{s['successful_requests']}/{s['total_requests']}|{s['output_tokens_per_second']}|{s['latency_p95_seconds']}|{s['time_to_first_text_seconds_p95']}|")
        if not measured:
            lines.append(f"|{case['case']}|{case['status']}|未测|—|—|—|—|")
    complete = (len(summary["cases"]) == 5 and all(c["status"] == "passed" and
        len([x for x in c["rounds"] if x["phase"] == "measure"]) >= 3 for c in summary["cases"]))
    lines += ["", f"五组全通过且每组至少三轮：{complete}", "",
        "显存见每组 summary.json（分阶段/UUID 的设备采样峰值）和 gpu.jsonl；包括其它进程，可能漏掉瞬时峰值。",
        "TTFT/TPOT 使用客户端文本事件近似；包含网络与排队，不能解释为纯 GPU prefill/decode。",
        "服务端 profiling 为请求墙钟时间，不是 CUDA kernel 时间。正式性能输出不用于质量评分。",
        "", "## 失败与限制"]
    for case in summary["cases"]:
        if case["status"] != "passed":
            lines += [f"- {case['case']}: {case.get('failure_kind', 'request_failure')}; 查看该组 server.log / requests.jsonl / summary.json。"]
    lines += ["", "## 尚需人工验收", "- 结合 environment.json 核对目标 GPU、版本、其它进程与显存预算。",
        "- 若 source_revision 为 unknown，补查来源；本次以权重文件 SHA256 固定本地内容。",
        "- 本报告不包含量化对照、Nsight 或质量回归结果，不代表整个项目完成。"]
    output = a.output or a.run / "Week1_Report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_rows(a.run / "performance.jsonl", records)
    print(output)

if __name__ == "__main__":
    main()
