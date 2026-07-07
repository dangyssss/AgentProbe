import json
import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    input_file = os.path.join(CURRENT_DIR, "llm_judge_final_metric.json")

    if not os.path.exists(input_file):
        print(f"Target evaluation report missing at {input_file}. Please run the evaluator pipeline first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    agentprobe_data = results.get("AGENTPROBE", {})
    baseline_data = results.get("BASELINE", {})

    judges = ['gpt-4o-mini', 'deepseek-chat', 'gemini-1.5-pro']
    metrics_map = {
        'validity_mean': 'Validity',
        'relevance_mean': 'Relevance',
        'adversarial_depth_mean': 'Adversarial Depth'
    }

    print("Multi-Judge Evaluation - Macro Metrics Summary Board (Likert 1-5)")


    for metric_key, metric_name in metrics_map.items():
        ap_avg = np.mean([agentprobe_data[j]['overall_metrics'][metric_key] for j in judges if j in agentprobe_data])
        bl_avg = np.mean([baseline_data[j]['overall_metrics'][metric_key] for j in judges if j in baseline_data])
        
        print(f"{metric_name:<30} | {ap_avg:<18.2f} | {bl_avg:<10.2f}")
    
    ap_total = np.mean([agentprobe_data[j]['overall_metrics']['comprehensive_score'] for j in judges if j in agentprobe_data])
    bl_total = np.mean([baseline_data[j]['overall_metrics']['comprehensive_score'] for j in judges if j in baseline_data])
    print("-"*70)
    print(f"{'Comprehensive Score':<30} | {ap_total:<18.2f} | {bl_total:<10.2f}")
    print("=================================================================")

    print("\nBreakdown Metrics by Rater Models")
    print("-"*70)
    for judge in judges:
        print(f"[Evaluator Model]: {judge}")
        print(f"   - AGENTPROBE -> Validity: {agentprobe_data.get(judge, {}).get('overall_metrics', {}).get('validity_mean', 0):.2f} | Relevance: {agentprobe_data.get(judge, {}).get('overall_metrics', {}).get('relevance_mean', 0):.2f} | Adversarial Depth: {agentprobe_data.get(judge, {}).get('overall_metrics', {}).get('adversarial_depth_mean', 0):.2f}")
        print(f"   - BASELINE   -> Validity: {baseline_data.get(judge, {}).get('overall_metrics', {}).get('validity_mean', 0):.2f} | Relevance: {baseline_data.get(judge, {}).get('overall_metrics', {}).get('relevance_mean', 0):.2f} | Adversarial Depth: {baseline_data.get(judge, {}).get('overall_metrics', {}).get('adversarial_depth_mean', 0):.2f}")
        print("-"*70)

if __name__ == "__main__":
    main()