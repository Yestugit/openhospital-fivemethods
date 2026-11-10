"""Audit a completed pilot without making model calls."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import json
from mematlas.core.files import atomic_write_json
from mematlas.core.telemetry import UsageLedger, UsageRecord


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_report(directory):
    root = Path(directory).resolve()
    manifest = read(root / "suite.json", {})
    if not manifest.get("pilot"):
        raise ValueError("The report command requires a pilot suite")
    records, cases, encounters, tests, curves = {}, {}, {}, {}, []
    for condition in manifest.get("jobs", {}):
        for attempt in sorted((root / condition).glob("attempt-*")):
            output = attempt / "evaluation"
            for path in (output / "runs").glob("*/usage.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    prior = records.get(row["record_id"])
                    if prior is not None and prior != row:
                        raise ValueError("Conflicting usage record identity")
                    records[row["record_id"]] = row
            for path in (output / "case_usage").glob("*.json"):
                cases[(condition, path.stem)] = read(path)
            for path in (output / "encounters").glob("*.json"):
                encounters[(condition, path.stem)] = read(path)
            for path in (output / "test_cases").glob("*.json"):
                tests[(condition, path.stem)] = read(path)
        final = Path(manifest["jobs"][condition]["output"]) / "runs/001-mem0/curve.json"
        curves.extend(read(final, []))
    grouped = defaultdict(list)
    for row in records.values():
        grouped[(row.get("stage"), row.get("phase"), row.get("resource_kind"))].append(row)
    usage = {}
    for group, rows in sorted(grouped.items(), key=str):
        ledger = UsageLedger()
        for row in rows:
            ledger.add(UsageRecord(**row), imported=True)
        summary = ledger.summary()
        usage["/".join(str(value) for value in group)] = {key: summary.get(key) for key in (
            "tracked_calls", "prompt_tokens", "completion_tokens", "failed_calls", "token_coverage",
            "known_cached_prompt_tokens", "known_cache_creation_prompt_tokens", "prompt_cache_hit_rate")}
    times = sorted((row for row in cases.values() if row.get("response_time_s") is not None), key=lambda row: row["position"])
    recovered = [step for bundle in encounters.values() for step in bundle.get("case_state", {}).get("timeline", ())]
    frozen = [row for row in curves if row.get("split") == "primary_test"]
    expected = manifest.get("pilot_validation", {})
    train_s = sum(row["response_time_s"] for row in times)
    test_s = sum(row.get("window_wall_time_ms", 0) for row in frozen) / 1000
    projection = None
    if len(times) == 20 and len(tests) == 2:
        projection = {"basis": "Arithmetic scaling from 20 training and 2 held-out cases; not a runtime guarantee. The held-out sample includes one retried request.",
            "mean_training_s": train_s / 20, "mean_heldout_including_scoring_s": test_s / 2,
            "formal_750_training_serial_hours": 750 * train_s / 20 / 3600,
            "formal_350_heldout_serial_hours": 350 * test_s / 2 / 3600,
            "formal_total_serial_hours": (750 * train_s / 20 + 350 * test_s / 2) / 3600,
            "formal_transfer_job_hours_at_pilot_rates": (250 * train_s / 20 + 350 * test_s / 2) / 3600}
    failures = [row for row in records.values() if row.get("status") != "success"]
    bootstrap_path = root.parent / "provider-bootstrap/repeated/memory-responses.jsonl"
    bootstrap_rows = [json.loads(line) for line in bootstrap_path.read_text(encoding="utf-8").splitlines()] if bootstrap_path.is_file() else []
    report = {"status": manifest.get("status"), "pilot_validation": expected,
        "failed_request_count": len(failures),
        "unknown_usage_request_count": sum(not row.get("usage_known") for row in records.values()),
        "bootstrap_probe_llm_responses": len(bootstrap_rows),
        "bootstrap_probe_known_llm_tokens": sum((row.get("usage") or {}).get("total_tokens",0) for row in bootstrap_rows),
        "unique_usage_records": len(records), "training_encounters": len(encounters), "heldout_encounters": len(tests),
        "frozen_verified": bool(frozen) and all(row.get("frozen_verified") for row in frozen),
        "clinical_metrics": {key: frozen[-1].get(key) for key in (
            "diagnosis_accuracy", "examination_precision", "treatment_overall_score")} if frozen else {},
        "mean_first_5_training_s": mean(row["response_time_s"] for row in times[:5]) if len(times)>=5 else None,
        "mean_last_5_training_s": mean(row["response_time_s"] for row in times[-5:]) if len(times)>=5 else None,
        "training_timing_samples": len(times),
        "training_json_fallbacks": sum(bool(step.get("json_fallback")) for step in recovered),
        "training_json_repairs": sum(step.get("json_repair_count",0) for step in recovered),
        "test_json_repairs": sum(row.get("json_recovery",{}).get("repairs",0) for row in tests.values()),
        "preparation_usage": manifest.get("provider_check", {}).get("usage"),
        "resume_event_audit": read(root / "resume-event-audit.json", {}),
        "formal_schedule_audit": read(root / "formal-plan-audit.json", {}).get("ready"),
        "cache_equivalence_audit": read(root / "cache-equivalence.json", {}),
        "training_total_s": train_s, "heldout_total_including_scoring_s": test_s,
        "runtime_projection": projection,
        "test_json_fallbacks": sum(row.get("json_recovery",{}).get("fallbacks",0) for row in tests.values()),
        "usage_by_stage_phase_kind": usage,
        "embedding_cache": {key: manifest.get("local_embedding_final",{}).get(key) for key in ("exact_cache_hits","inferred_texts","inferred_tokens","saved_tokens")},
        "price_estimate": None, "price_note": "No verified model price configured; tokens are not converted to a bill.",
        "formal_run_started": False}
    atomic_write_json(root / "pilot-report.json", report)
    lines = ["# Mem0 × OpenHospital 预跑报告", "", f"状态：{report['status']}；训练 {len(encounters)}/20，冻结测试 {len(tests)}/2。",
        f"冻结验证：{report['frozen_verified']}；第10例后恢复：{expected.get('resume_verified',False)}。", "",
        f"前5例平均耗时：{report['mean_first_5_training_s']} 秒；后5例：{report['mean_last_5_training_s']} 秒。",
        f"训练 JSON 修复：{report['training_json_repairs']}；训练兜底：{report['training_json_fallbacks']}；测试修复：{report['test_json_repairs']}；测试兜底：{report['test_json_fallbacks']}。", "",
        "| 阶段 / 来源 / 类型 | 调用 | 输入 token | 输出 token | 失败 | 已报告缓存命中 token |", "|---|---:|---:|---:|---:|---:|"]
    for key, row in usage.items():
        lines.append(f"| {key} | {row['tracked_calls']} | {row['prompt_tokens']} | {row['completion_tokens']} | {row['failed_calls']} | {row['known_cached_prompt_tokens']} |")
    lines += ["", f"冻结测试：诊断准确率 {report['clinical_metrics'].get('diagnosis_accuracy')}，检查精确率 {report['clinical_metrics'].get('examination_precision')}，治疗总体得分 {report['clinical_metrics'].get('treatment_overall_score')}。仅2例，不作方法效果结论。",
        f"Embedding服务累计：{report['embedding_cache']}（包含连通性检查和2次合成缓存校验；病例用量以表格为准）。",
        f"准备阶段用量单列于JSON报告preparation_usage；此前bootstrap检查另有{len(bootstrap_rows)}条LLM响应、已知{report['bootstrap_probe_known_llm_tokens']} token，原文保存在provider-bootstrap目录。",
        f"共记录{len(failures)}次请求失败，{report['unknown_usage_request_count']}次请求用量未知；失败请求仍保留在统计中，表中已知输入输出不能代表完整账单。",
        (f"按预跑均值纯算术外推：750次训练串行约{projection['formal_750_training_serial_hours']:.1f}小时，350次测试约{projection['formal_350_heldout_serial_hours']:.1f}小时，合计约{projection['formal_total_serial_hours']:.1f}小时；三组并发时跨科组自身约{projection['formal_transfer_job_hours_at_pilot_rates']:.1f}小时。" if projection else "样本尚不足以计算工期。")
            + "模型限流、病例复杂度和共享GPU会改变耗时；两个测试病例且其中一次重试，不足以作可靠工期预测。",
        "医生含诊疗与训练反思；patient/examination 为医院模拟，judge 为评分。各阶段分别统计，恢复后的同一 record_id 只计一次。",
        "费用未换算：未配置已核实单价。本次短预跑只用于运行与成本观察，不能证明正式实验效果或确定总费用。正式1100次实验未启动。"]
    (root / "pilot-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
