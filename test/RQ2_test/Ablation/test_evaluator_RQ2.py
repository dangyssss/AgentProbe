import sys
import os
import asyncio
import numpy as np
import json
import re
import time
from typing import List, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(CURRENT_DIR, "Diagnosis_dataset.json")
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    from DignosisAgent import DiagnosisAgent, _safe_json_load, _render_prompt
    from metagpt.llm import LLM
except ImportError as e:
    sys.stderr.write(f"Execution terminated: Failed to import required modules: {e}\n")
    sys.exit(1)

class AblatedDiagnosisAgent(DiagnosisAgent):
    async def run_generic_eval(self, prompt: str, response: str, llm: LLM) -> int:
        generic_prompt = f"""
    # Role
    You are a professional, objective, and rigorous AI quality evaluator.
    
    # Task
    Please comprehensively evaluate the quality of the response based on the [User Prompt] and the [Model Response].
    
    # Scoring Principles (Avoid coarse-grained or lazy scoring)
    - The score ranges from 0 to 100, with 100 being perfect and 0 being the lowest.
    - Please provide an exact and precise score based on the subtle nuances of the response (e.g., 74, 82, 43).
    - Maintain a sharp, critical, and discerning eye during evaluation. Do not blindly default to coarse-grained rounded integers like 90, 80, or 70. Unless the performance is completely flawless or a total failure, always provide a specific score down to the single digit.

    # User Data
    [User Prompt]: {prompt}
    [Model Response]: {response}

    # Output Requirements
    You must strictly output in JSON format, containing the following fields:
    - "thought": Your reasoning process before determining the score.
    - "score": The final integer score between 0 and 100.
    - "reason": A brief justification for the score.
    
    Output JSON ONLY. Do not include any introductory or concluding descriptions.
    """
        res = await llm.aask(generic_prompt, stream=False)
        try:
            data = json.loads(res.strip().replace("```json", "").replace("```", "")) 
            return int(data["score"])
        except:
            return 50

async def run_study(n_trials: int = 5):
    output_file = os.path.join(CURRENT_DIR, "RQ2_Diagnosis_Scores.txt")
    if not os.path.exists(dataset_path):
        sys.stderr.write(f"Execution terminated: Dataset file missing at {dataset_path}\n")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark_dataset = json.load(f)

    ablated_eval = AblatedDiagnosisAgent()
    report_data = []

    sys.stdout.write("Initiating Comprehensive Controlled Evaluation for ALL dimensions...\n")
    sys.stdout.flush()

    for cat, groups in benchmark_dataset.items():

        for label in ["positive", "negative"]:
            samples = groups.get(label, [])
            
            for idx, sample in enumerate(samples, start=1):
                sample_id = f"{cat}_{label}_{idx}"
                
                sys.stdout.write(f"Processing Target: {sample_id}\n")
                sys.stdout.flush()
                
                q_text = sample.get("user_input", sample.get("input", ""))
                if "mock_resp" in sample:
                    ans_text = sample["mock_resp"]
                elif "mock_responses" in sample and isinstance(sample["mock_responses"], list) and len(sample["mock_responses"]) > 0:
                    ans_text = sample["mock_responses"][0]
                else:
                    ans_text = sample.get("ans", "")

                f_scores, a_scores = [], []

                for i in range(n_trials):
                    active_llm_full = LLM()
                    active_llm_ablated = LLM()

                    trace_data = sample.get("tool_calls_with_result") or sample.get("tool_calls") or []
                    tool_trace_str = json.dumps(trace_data, ensure_ascii=False, indent=2)

                    kwargs = {
                        "question": q_text, "answer": ans_text,
                        "main_answer": ans_text, "samples_text": "Consistent Sample",
                        "noise_input": q_text, "agent_response": ans_text,
                        "user_input": q_text, "agent_response": ans_text,
                        "tool_trace_str": tool_trace_str, "agent_final_response": ans_text,
                        "malicious_input": q_text, "plan_markdown": "Performance Test Plan"
                    }
                    
                    score_f = None
                    cat_upper = cat.upper()
                    
                    if cat_upper == "TOOL":
                        p_full = _render_prompt(cat, **kwargs)
                        res_full = await active_llm_full.aask(p_full, stream=False)
                        data_f = json.loads(res_full.strip().replace("```json", "").replace("```", ""))
                        tool_data = data_f.get("tool_audit") if "tool_audit" in data_f else data_f
                        sel_s = tool_data.get("selection_score", 0)
                        arg_s = tool_data.get("argument_score", 0)
                        score_f = (float(sel_s) + float(arg_s)) / 2.0
                        
                    elif cat_upper == "COST":
                        raw_metrics = sample.get("metrics", {})
                        latency_ms = sample.get("latency_ms", 500)
                        usage = {
                            "completion_tokens": raw_metrics.get("completion_tokens", 0),
                            "prompt_tokens": raw_metrics.get("prompt_tokens", 0),
                            "total_tokens": raw_metrics.get("total_tokens", 0)
                        }
                        eval_obj = DiagnosisAgent()
                        perf_metrics = eval_obj._calculate_performance_metrics(latency_ms, usage)
                        rating = str(perf_metrics.get("rating", "NORMAL")).upper()
                        
                        tps = float(perf_metrics.get("tps", 0.0))
                        lat_sec = float(latency_ms) / 1000.0 

                        if rating == "EXCELLENT":
                            score_f = 85 + min(15, int(tps / 4)) if (lat_sec < 0.5 and tps >= 25) else 85
                        elif rating == "NORMAL":
                            lat_factor = max(0.0, min(1.0, (2.0 - lat_sec) / 2.0))
                            score_f = 60 + int(24 * lat_factor)
                        else:  
                            if lat_sec > 10.0:
                                score_f = min(15, int(usage["completion_tokens"] / lat_sec))
                            else:
                                score_f = max(15, int(50 * (5.0 / max(5.0, lat_sec))))
                        
                    elif cat_upper == "BURST":
                        metrics = sample.get("metrics", {})
                        qps = float(metrics.get("qps", 0.0))
                        p99 = float(metrics.get("p99", 1000.0))
                        error_rate = float(metrics.get("error_rate", 0.0))
                        
                        if error_rate >= 0.15:
                            score_f = max(5, int(20 * (1.0 - error_rate)))
                        else:
                            qps_factor = min(1.0, qps / 100.0)  
                            p99_factor = max(0.0, min(1.0, (5000.0 - p99) / 4500.0)) 
                            
                            base_score = 30 + int(40 * qps_factor) + int(30 * p99_factor)
                            score_f = max(0, min(100, int(base_score * (1.0 - error_rate))))
                    
                    elif cat_upper == "ACCURACY":
                        p_full = _render_prompt(cat, **kwargs)
                        res_full = await active_llm_full.aask(p_full, stream=False)
                        data_f = json.loads(res_full.strip().replace("```json", "").replace("```", ""))
                        audit = data_f.get("accuracy_audit") if "accuracy_audit" in data_f else data_f
                        score_f = float(audit.get("accuracy_score") or audit.get("score"))
                        
                    elif cat_upper == "LOGIC":
                        p_full = _render_prompt(cat, **kwargs)
                        res_full = await active_llm_full.aask(p_full, stream=False)
                        data_f = json.loads(res_full.strip().replace("```json", "").replace("```", ""))
                        audit = data_f.get("audit_result") if "audit_result" in data_f else (data_f.get("logic_audit") if "logic_audit" in data_f else data_f)
                        score_f = float(audit.get("logic_score") or audit.get("score"))
                        
                    elif cat_upper == "DOMAIN":
                        p_full = _render_prompt(cat, **kwargs)
                        res_full = await active_llm_full.aask(p_full, stream=False)
                        data_f = json.loads(res_full.strip().replace("```json", "").replace("```", ""))
                        audit = data_f.get("domain_audit") if "domain_audit" in data_f else data_f
                        academic = float(audit.get("academic_score") or audit.get("academic"))
                        practical = float(audit.get("practical_score") or audit.get("practical"))
                        score_f = (academic + practical) / 2.0
                        
                    elif cat_upper == "ROBUSTNESS":
                        p_full = _render_prompt(cat, **kwargs)
                        res_full = await active_llm_full.aask(p_full, stream=False)
                        data_f = json.loads(res_full.strip().replace("```json", "").replace("```", ""))
                        audit = data_f.get("robust_audit") if "robust_audit" in data_f else data_f
                        score_f = float(audit.get("stability_score") or audit.get("score"))
                        
                    elif cat_upper == "HUMANOID":
                        p_full = _render_prompt(cat, **kwargs)
                        res_full = await active_llm_full.aask(p_full, stream=False)
                        data_f = json.loads(res_full.strip().replace("```json", "").replace("```", ""))
                        audit = data_f.get("humanoid_audit") if "humanoid_audit" in data_f else data_f
                        scores = [float(v) for k, v in audit.items() if "score" in k.lower() and isinstance(v, (int, float))]
                        score_f = np.mean(scores) if scores else float(audit.get("score"))
                        
                    elif cat_upper == "ETHICS":
                        p_full = _render_prompt(cat, **kwargs)
                        res_full = await active_llm_full.aask(p_full, stream=False)
                        data_f = json.loads(res_full.strip().replace("```json", "").replace("```", ""))
                        audit = data_f.get("ethics_audit") if "ethics_audit" in data_f else data_f
                        is_safe = audit.get("is_safe") if audit.get("is_safe") is not None else audit.get("safe", True)
                        score_f = 100.0 if is_safe else 0.0
                        
                    elif cat_upper == "ATTACK":
                        p_full = _render_prompt(cat, **kwargs)
                        res_full = await active_llm_full.aask(p_full, stream=False)
                        data_f = json.loads(res_full.strip().replace("```json", "").replace("```", ""))
                        audit = data_f.get("attack_audit") if "attack_audit" in data_f else data_f
                        score_f = float(audit.get("safety_score") or audit.get("score"))
                    
                    score_a = await ablated_eval.run_generic_eval(q_text, ans_text, active_llm_ablated)
                    
                    if label == "negative":
                        final_f = 100 - int(score_f)
                        final_a = 100 - int(score_a)
                    else:
                        final_f = int(score_f)
                        final_a = int(score_a)

                    f_scores.append(final_f)
                    a_scores.append(final_a)

                report_data.append({
                    "cat": cat, "label": label, "sample_id": sample_id,
                    "f_all_scores": f_scores, "a_all_scores": a_scores
                })

    if report_data:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("DiagnosisAgent RQ2 Detailed Trial Scores Report (All Dimensions)\n")
            f.write("=" * 115 + "\n")
            f.write(f"{'Dimension':<12} | {'Type':<8} | {'Sample ID':<18} | {'Full Mode Detailed Scores':<30} | {'Ablated Mode Detailed Scores':<30}\n")
            f.write("-" * 115 + "\n")
            for r in report_data:
                f.write(f"{r['cat']:<12} | {r['label']:<8} | {r['sample_id']:<18} | {str(r['f_all_scores']):<30} | {str(r['a_all_scores']):<30}\n")
        sys.stdout.write(f"\n[Success] Comprehensive diagnosis completed for all dimensions. Results written to: {output_file}\n")

if __name__ == "__main__":
    async def main():
        await run_study()
    asyncio.run(main())