#!/usr/bin/env python3
"""Create an auditable report from one managed benchmark run, including legacy BF16 runs."""
import argparse
from pathlib import Path, PurePosixPath, PureWindowsPath
from experiment_utils import read, rows, write_rows


LABELS = {"bf16": "BF16", "awq": "AWQ/W4A16",
          "fp8-official": "官方 FP8", "fp8-local": "本地 FP8"}


def cell(value):
    """Keep metadata in one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def profile_of(case):
    return case.get("model_profile", "bf16")


def request_path(directory, case):
    """Read only artifacts contained within this run, including after resolving symlinks."""
    artifact = str(case.get("artifact_directory", case["case"]))
    relative = PurePosixPath(artifact.replace("\\", "/"))
    if (not artifact or relative.is_absolute() or PureWindowsPath(artifact).drive
            or any(part == ".." for part in relative.parts)):
        raise ValueError("Artifact directory must be a relative path inside the run: " + artifact)
    root = directory.resolve()
    path = (root.joinpath(*relative.parts) / "requests.jsonl").resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Artifact directory escapes the run: " + artifact) from error
    return path


def completion(run, summary):
    """Require the selected model/case matrix and every requested formal round."""
    profiles = [entry["profile"] for entry in run.get("models", [{"profile": "bf16"}])]
    cases = run.get("cases", [])
    expected = {(profile, case) for profile in profiles for case in cases}
    protocol = run["protocol"]
    count, rounds = protocol["requests"], protocol["rounds"]
    issues = []
    if (not expected or len(profiles) != len(set(profiles)) or len(cases) != len(set(cases))
            or isinstance(count, bool) or not isinstance(count, int) or count < 1
            or isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1):
        return False, ["模型、负载或正式轮次/请求数配置为空、重复或无效。"]
    seen = set()
    for case in summary.get("cases", []):
        key = (profile_of(case), case["case"])
        label = "/".join(map(str, key))
        if key in seen or key not in expected:
            issues.append(label + "：重复或不属于本次选择的实验组。")
        seen.add(key)
        measured = [entry for entry in case.get("rounds", []) if entry.get("phase") == "measure"]
        valid_rounds = (len(measured) == rounds
                        and {entry.get("round") for entry in measured} == set(range(1, rounds + 1)))
        valid_counts = all(entry.get("total_requests") == count
                           and entry.get("successful_requests") == count
                           and entry.get("failed_requests", 0) == 0 for entry in measured)
        if case.get("status") != "passed" or not valid_rounds or not valid_counts:
            issues.append(label + "：状态、正式轮次或成功请求数未达到本次协议。")
    for profile, case in sorted(expected - seen):
        issues.append(f"{profile}/{case}：缺少实验结果。")
    if summary.get("error"):
        issues.append("实验预检/执行错误：" + str(summary["error"]))
    return not issues, issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, help="default: RUN/Week1_Report.md")
    args = parser.parse_args(argv)
    run = read(args.run / "run.json")
    summary = read(args.run / "summary.json")
    profiles = [entry["profile"] for entry in run.get("models", [{"profile": "bf16"}])]
    title = ("第一周 BF16 实验报告" if profiles == ["bf16"] else
             "多模型量化性能实验报告" if len(profiles) > 1 else
             LABELS.get(profiles[0], profiles[0]) + " 实验报告" if profiles else "性能实验报告")
    protocol = run["protocol"]
    lines = ["# " + title, "", f"实验 ID：{run['run_id']}",
             f"快照 SHA256：{run['manifest_sha256']}"]
    if run.get("selected_requests_sha256"):
        lines.append(f"本次请求子集 SHA256：{run['selected_requests_sha256']}")
    lines += [f"本次协议：每个模型/负载 {protocol['rounds']} 轮，每轮 {protocol['requests']} 个正式请求；"
              f"每轮预热 {protocol.get('warmup', '未记录')} 个请求，并发 {protocol.get('concurrency', '未记录')}。"]
    if run.get("snapshot_protocol"):
        frozen = run["snapshot_protocol"]
        lines.append(f"原冻结协议：每组 {frozen['rounds']} 轮、每轮 {frozen['requests']} 个正式请求；"
                     "本次验收以上述有效协议为准，快照保持原始内容。")
    if protocol["rounds"] == 1:
        lines.append("本次仅一轮，无法据此评估多轮之间的波动。")
    configured = [d for d in run.get("models", []) if "enforce_eager" in d.get("config", {})]
    if configured:
        lines += ["", "| 模型 | enforce_eager | FP8 后端请求 | 执行精度约定 |",
                  "|---|---|---|---|"]
        for descriptor in configured:
            config = descriptor["config"]
            backend = config.get("fp8_backend", "auto")
            precision = "W8A16（Marlin）" if backend == "marlin" else "按 checkpoint / 平台分派"
            lines.append("|" + "|".join(cell(x) for x in (
                LABELS.get(descriptor["profile"], descriptor["profile"]),
                config["enforce_eager"], backend, precision)) + "|")
        lines += ["", "后端列记录请求配置；Marlin 初始化核验见各组 summary.json，实际 CUDA kernel 仍需 Nsight 确认。"]
    lines += ["", "| 模型 | 负载 | 状态 | 轮次 | 成功/总数 | 吞吐 token/s | E2E p95(s) | TTFT文本事件 p95(s) |",
              "|---|---|---|---|---|---|---|---|"]
    records, paths = [], set()
    for case in summary.get("cases", []):
        profile = profile_of(case)
        path = request_path(args.run, case)
        if path.exists() and path not in paths:
            records.extend(dict(entry, run_id=run["run_id"], model_profile=profile) for entry in rows(path))
            paths.add(path)
        measured = [entry for entry in case.get("rounds", []) if entry.get("phase") == "measure"]
        for entry in measured:
            values = [LABELS.get(profile, profile), case["case"], case.get("status", "incomplete"),
                      entry["round"], f"{entry.get('successful_requests', 0)}/{entry.get('total_requests', 0)}",
                      entry.get("output_tokens_per_second", "—"), entry.get("latency_p95_seconds", "—"),
                      entry.get("time_to_first_text_seconds_p95", "—")]
            lines.append("|" + "|".join(cell(value) for value in values) + "|")
        if not measured:
            lines.append("|" + "|".join(map(cell, [LABELS.get(profile, profile), case["case"],
                                                  case.get("status", "incomplete"), "未测", "—", "—", "—", "—"])) + "|")
    complete, issues = completion(run, summary)
    lines += ["", f"本次所选模型/负载全部满足有效协议：{complete}", "",
              "各模型单独列示，吞吐与延迟按各自轮次计算，未跨精度合并。",
              "显存见各模型/负载的 summary.json（分阶段/UUID 的设备采样峰值）和 gpu.jsonl；包括其它进程，可能漏掉瞬时峰值。",
              "TTFT/TPOT 使用客户端文本事件近似；包含网络与排队，不能解释为纯 GPU prefill/decode。",
              "服务端 profiling 为请求墙钟时间，不是 CUDA kernel 时间。正式性能输出不用于质量评分。",
              "", "## 失败与限制"]
    lines.extend("- " + cell(issue) for issue in issues)
    for case in summary.get("cases", []):
        if case.get("status") != "passed":
            error = f"；{case['error']}" if case.get("error") else ""
            lines.append(f"- {cell(profile_of(case))}/{cell(case['case'])}: "
                         f"{cell(case.get('failure_kind', 'request_failure'))}{cell(error)}；"
                         "查看该组 server.log / requests.jsonl / summary.json。")
    if not issues and all(case.get("status") == "passed" for case in summary.get("cases", [])):
        lines.append("- 本次记录未发现请求失败或协议缺项。")
    lines += ["", "## 尚需人工验收", "- 结合 environment.json 核对目标 GPU、版本、其它进程与显存预算。",
              "- 结合 run.json 与各组服务启动日志核对权重身份、实际 dtype 与量化后端；source_revision 为 unknown 时补查来源。",
              "- 本报告不包含质量回归、Nsight 或原生 GPU kernel 确认结果，不代表整个项目完成。"]
    output = args.output or args.run / "Week1_Report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_rows(args.run / "performance.jsonl", records)
    print(output)


if __name__ == "__main__":
    main()
