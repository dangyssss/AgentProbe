import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Any

from PlannerAgent import _classify_tests_async

STANDARD_DIMENSIONS = {
    "ACCURACY", "LOGIC", "DOMAIN", "COST", "ROBUSTNESS", 
    "HUMANOID", "TOOL", "ETHICS", "ATTACK", "BURST"
}

def _display_execution_telemetry(current: int, total: int, start_time: float) -> None:
    if total == 0: 
        return
    progress_ratio = current / total
    percent = progress_ratio * 100
    bar_length = 25
    filled_length = int(round(bar_length * progress_ratio))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    elapsed_time = time.perf_counter() - start_time
    average_latency = elapsed_time / current if current > 0 else 0.0
    estimated_time_arrival = (total - current) * average_latency
    
    elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
    eta_str = time.strftime("%M:%S", time.gmtime(estimated_time_arrival)) if estimated_time_arrival > 0 else "00:00"
    
    sys.stdout.write(
        f"\rPipeline Execution: |{bar}| {current}/{total} ({percent:.1f}%) | "
        f"Elapsed: {elapsed_str} | ETA: {eta_str}"
    )
    sys.stdout.flush()

async def run_comprehensive_experiment() -> None:
    input_file = "ground_truth_dataset.json"
    report_file = "evaluation_metrics_report.json"
    conflict_file = "conflict.json"
    
    if not os.path.exists(input_file):
        sys.stderr.write(f"Critical Error: Source evaluation dataset '{input_file}' not found.\n")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        dataset: Dict[str, Any] = json.load(f)

    total_cases_count = 0
    for agent_data in dataset.values():
        for case in agent_data.get("test_cases_to_label", []):
            ground_truth_dim = str(case.get("ground_truth", "CUSTOM")).upper().strip()
            primary_ground_truth = ground_truth_dim.split(',')[0].strip()
            if primary_ground_truth != "CUSTOM":
                total_cases_count += 1

    dimension_metrics = {
        dim: {
            "true_positive_match": 0, 
            "false_negative_error": 0, 
            "false_positive_error": 0
        } for dim in STANDARD_DIMENSIONS
    }

    misclassification_logs: List[Dict[str, Any]] = []
    current_count = 0
    start_time = time.perf_counter()
    
    print("Execution Initialization: Domain-Specific Absolute Alignment Pipeline Started.")
    print(f"Filtering Parameters: Excluded out-of-domain instances | Evaluation Target Size: {total_cases_count} items.")

    for agent_key, agent_data in dataset.items():
        domain = agent_data.get("domain", "")
        system_prompt = agent_data.get("system_prompt", "")
        tools_list = agent_data.get("tools", [])
        
        agent_context_profile = (
            f"System Persona Configuration:\n{system_prompt}\n\n"
            f"Accessible Operational Tools:\n{json.dumps(tools_list, ensure_ascii=False)}"
        )
        agent_name = "LCNC_Optimization_Assistant" if "Coze" in str(system_prompt) else agent_data.get("agent_name", "")
            
        test_cases = agent_data.get("test_cases_to_label", [])
        
        for idx, case in enumerate(test_cases, start=1):
            ground_truth_dim = str(case.get("ground_truth", "CUSTOM")).upper().strip()
            primary_ground_truth = ground_truth_dim.split(',')[0].strip()

            if primary_ground_truth == "CUSTOM":
                continue

            user_input = case.get("req_text", " ")
            
            try:
                predicted_labels = await _classify_tests_async(agent_context_profile, user_input)
                raw_prediction = str(predicted_labels[0]).strip() if (predicted_labels and isinstance(predicted_labels, list)) else "CUSTOM"
            except Exception as e:
                raw_prediction = f"ERROR_CRASH: {str(e)}"
            
            normalized_prediction = raw_prediction.upper().strip()
            if normalized_prediction.startswith("CUSTOM:") or (normalized_prediction not in STANDARD_DIMENSIONS):
                predicted_dimension = "CUSTOM"
            else:
                predicted_dimension = normalized_prediction

            is_aligned = (predicted_dimension == primary_ground_truth)

            if is_aligned:
                if predicted_dimension in dimension_metrics:
                    dimension_metrics[predicted_dimension]["true_positive_match"] += 1
            else:
                if primary_ground_truth in dimension_metrics:
                    dimension_metrics[primary_ground_truth]["false_negative_error"] += 1
                if predicted_dimension in dimension_metrics:
                    dimension_metrics[predicted_dimension]["false_positive_error"] += 1
                
                misclassification_logs.append({
                    "req_id": case.get("req_id"),
                    "belonging_agent": f"{domain}_{agent_name}",
                    "user_test_request": user_input,
                    "ground_truth": primary_ground_truth,
                    "planner_predicted_dimension": raw_prediction
                })

            current_count += 1
            _display_execution_telemetry(current_count, total_cases_count, start_time)

    final_report_payload = {
        "experiment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_tested": total_cases_count,
            "overall_matched": sum(metrics["true_positive_match"] for metrics in dimension_metrics.values()),
            "overall_conflicts": len(misclassification_logs)
        },
        "dimension_breakdown_metrics": dimension_metrics
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_report_payload, f, ensure_ascii=False, indent=4)
    with open(conflict_file, "w", encoding="utf-8") as f:
        json.dump(misclassification_logs, f, ensure_ascii=False, indent=4)

    print("\n\nEvaluation Summary Statistics:")
    print(f"Misclassification Log exported to: {conflict_file}")
    print(f"Denoised Quantitative Evaluation Report generated at: {report_file}")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_experiment())