import asyncio
import json
import sys
import os
import datetime
import concurrent.futures
from unittest.mock import patch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

try:
    from DiagnosisAgent import DiagnosisAgent, EVAL_PROMPTS, _run_llm_sync as original_run_llm_sync
except ImportError as e:
    print(f"Error: Failed to import DiagnosisAgent: {e}")
    sys.exit(1)

if "ETHICS" in EVAL_PROMPTS and "{user_input}" in EVAL_PROMPTS["ETHICS"]:
    EVAL_PROMPTS["ETHICS"] = EVAL_PROMPTS["ETHICS"].replace("{user_input}", "{malicious_input}")

JSON_DATA_PATH = os.path.join("test", "Diagnosis_dataset.json")

def _parse_agent_verdict(category: str, res: dict) -> bool:
    key_map = {
        "ACCURACY": "judge_result", "LOGIC": "audit_result", "DOMAIN": "domain_audit",
        "ROBUSTNESS": "robust_audit", "HUMANOID": "humanoid_audit", "TOOL": "tool_audit",
        "COST": "metrics", "BURST": "metrics", "ATTACK": "attack_audit", "ETHICS": "ethics_audit"
    }
    d = res.get(key_map.get(category, ""), {})
    if category == "ACCURACY": return d.get("is_consistent", False)
    if category == "LOGIC": return (not d.get("has_logic_error", True)) and (d.get("logic_score", 0) >= 60)
    if category == "DOMAIN": return d.get("academic_score", 0) >= 50
    if category == "ROBUSTNESS": return d.get("is_robust", False)
    if category == "HUMANOID": return not d.get("is_robotic", True)
    if category in ["ETHICS", "ATTACK"]: return d.get("is_safe", False)
    if category == "TOOL": return (d.get("selection_score", 0) >= 60) and (not d.get("has_hallucination", True))
    if category == "COST": return d.get("rating") != "POOR"
    if category == "BURST": return d.get("p99", 99999) < 1000
    return False

def _thread_safe_llm(prompt: str) -> str:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(original_run_llm_sync, prompt)
        return future.result()

def run_optimized_rq1_test():
    if not os.path.exists(JSON_DATA_PATH):
        print(f"Error: Dataset missing at path: {JSON_DATA_PATH}")
        return

    with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)

    evaluator = DiagnosisAgent()
    output_filename = "RQ2_Diagnosis_Result.json"
    call_counts = {}
    
    all_raw_details = []
    confusion_matrix = {dim: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for dim in test_dataset.keys()}

    print("\n" + "=" * 80)
    print("EvaluatorAgent Evaluation Pipeline Initialized: RQ1 Verification Accuracy")
    print("=" * 80 + "\n")

    async def side_effect_coze(bot_id, user_input, original=True):
        found = None
        for cat_data in test_dataset.values():
            for group in cat_data.values():
                for c in group:
                    if c.get("user_input") == user_input:
                        found = c
                        break
                if found: break
            if found: break
        
        count = call_counts.get(user_input, 0)
        call_counts[user_input] = count + 1
        
        if found and "mock_responses" in found:
            resps = found["mock_responses"]
            resp_text = resps[count % len(resps)]
        else:
            resp_text = found.get("mock_resp", "Default OK") if found else "Default OK"
        
        m = found.get("mock_metrics", {}) if found else {}
        lat = m.get("lat", m.get("p99", 15000 if "L-Test-N" in (user_input or "") else 200))
        in_t = m.get("in", 0)
        out_t = m.get("out", 10)
        
        return resp_text, lat, {
            "usage": {"prompt_tokens": in_t, "completion_tokens": out_t, "total_tokens": in_t + out_t}, 
            "latency_ms": lat, 
            "tool_calls": found.get("trace", []) if found else []
        }

    with patch('EvaluatorAgent._coze_ask_async', side_effect=side_effect_coze), \
         patch('EvaluatorAgent._run_llm_sync', side_effect=_thread_safe_llm):
        
        for cat, groups in test_dataset.items():
            test_cases = []
            for group_name, cases in groups.items():
                for c in cases:
                    test_cases.append({"id": f"{cat}_{group_name}", "category": cat, "input": c.get("user_input"), **c})

            try:
                if cat == "ACCURACY":
                    _, results = evaluator._run_accuracy_eval_block("bot", "plan", test_cases)
                elif cat == "BURST":
                    _, results = asyncio.run(evaluator._run_burst_eval_block("bot", "plan", test_cases))
                else:
                    _, results = evaluator._run_standard_eval_block("bot", "plan", test_cases, cat)

                for i, res in enumerate(results):
                    actual = _parse_agent_verdict(cat, res)
                    expected = test_cases[i]["expected_pass"]
                    
                    if expected is True:
                        if actual is True: confusion_matrix[cat]["TP"] += 1
                        else: confusion_matrix[cat]["FN"] += 1
                    else:
                        if actual is False: confusion_matrix[cat]["TN"] += 1
                        else: confusion_matrix[cat]["FP"] += 1
                    
                    all_raw_details.append({
                        "category": cat,
                        "input": test_cases[i]["input"],
                        "expected_pass": expected,
                        "actual_pass": actual,
                        "is_correct_judgment": (actual == expected),
                        "audit_raw": res
                    })
                
                print(f"[INFO] Completed evaluation matrix for dimension: {cat}")
            except Exception as e:
                print(f"[ERROR] Connection trace dropped on dimension {cat}: {e}")

    print("\n" + "=" * 80)
    print(f"{'Dimension':<15} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Supports':<8}")
    print("-" * 80)

    macro_p, macro_r = 0.0, 0.0
    total_cases = len(all_raw_details)
    total_correct = sum(1 for d in all_raw_details if d["is_correct_judgment"])

    for dim in test_dataset.keys():
        tp = confusion_matrix[dim]["TP"]
        fp = confusion_matrix[dim]["FP"]
        fn = confusion_matrix[dim]["FN"]
        tn = confusion_matrix[dim]["TN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        macro_p += precision
        macro_r += recall

        print(f"{dim:<15} | {precision*100:<11.2f}% | {recall*100:<11.2f}% | {f1*100:<11.2f}% | {tp+fn+fp+tn:<8}")

    print("-" * 80)
    overall_accuracy = (total_correct / total_cases) if total_cases > 0 else 0.0
    macro_p_avg = macro_p / len(test_dataset)
    macro_r_avg = macro_r / len(test_dataset)
    macro_f1 = 2 * macro_p_avg * macro_r_avg / (macro_p_avg + macro_r_avg) if (macro_p_avg + macro_r_avg) > 0 else 0.0

    print(f"{'Overall Accuracy':<15} | {'-':<12} | {'-':<12} | {overall_accuracy*100:<11.2f}% | {total_cases:<8}")
    print(f"{'Global Average':<15} | {macro_p_avg*100:<11.2f}% | {macro_r_avg*100:<11.2f}% | {macro_f1*100:<11.2f}% | -")
    print("=" * 80 + "\n")

    optimized_json = {
        "overall_summary": {
            "total_test_cases": total_cases,
            "total_correct_judgments": total_correct,
            "overall_accuracy": f"{overall_accuracy * 100:.2f}%",
            "test_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "error_analysis_details": [
            {
                "category": d["category"],
                "input": d["input"],
                "expected_pass": d["expected_pass"],
                "actual_pass": d["actual_pass"],
                "audit_raw": d["audit_raw"]
            }
            for d in all_raw_details if not d["is_correct_judgment"]
        ]
    }

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(optimized_json, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_optimized_rq1_test()