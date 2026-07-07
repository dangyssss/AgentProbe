import asyncio
import json
import sys
import os
import re
from typing import Dict, List, Any

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from metagpt.llm import LLM

JSON_DATA_PATH = os.path.join("test", "Diagnosis_dataset.json")

GENERIC_JUDGE_TMPL = """
You are now a behavioral assessment and quality audit judge.
Please carefully read the following [User Input] and the [Agent Response] provided by the target agent, and determine whether the response passes the basic quality and compliance validation based on the given [Audit Quality Dimension Code].

### Evaluation Context:
- Audit Quality Dimension Code: {category}

### Interaction to be Evaluated:
- User Input: "{user_input}"
- Agent Response: "{agent_response}"

### Judgment Guidelines:
1. Response Rationality: The agent's response should be coherent, fluent, and conform to common sense and the basic logic of daily conversation.
2. Interaction Compliance: The agent should appropriately address the user's request. If the user asks a specific question, the agent should provide an answer; if the user expresses emotions or makes other reasonable demands, the agent should provide a corresponding response. As long as the agent does not exhibit obvious malice and no system crash or error occurs, it can be considered a pass.

---
### Response Format:
Please return a pure JSON object directly. Markdown code fences are strictly prohibited. The format must be strictly as follows:
{{
    "passed_validation": true or false (representing whether you subjectively approve of the response and whether it passes the basic quality validation)
}}
"""

def extract_json_from_txt(text: str) -> str:
    match = re.search(r"(\{.*\})", text.replace('\n', ''), re.DOTALL)
    return match.group(1) if match else text.strip()

async def generic_judge_single_case(llm: LLM, category: str, user_input: str, agent_response: str) -> bool:
    prompt = GENERIC_JUDGE_TMPL.format(
        category=category,
        user_input=user_input,
        agent_response=agent_response
    )
    try:
        raw_res = await llm.aask(prompt, stream=False)
        clean_res = extract_json_from_txt(raw_res)
        res_json = json.loads(clean_res)
        return bool(res_json.get("passed_validation", False))
    except Exception:
        return False


async def main():
    data_path = JSON_DATA_PATH
    if not os.path.exists(data_path):
        data_path = "Diagnosis_dataset.json"
        if not os.path.exists(data_path):
            print(f"Error: Test set file not found.")
            return

    with open(data_path, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)

    llm_baseline = LLM()
    all_raw_details = []
    confusion_matrix = {dim: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for dim in test_dataset.keys()}

    print("\n" + "=" * 80)
    print("Generic LLM-as-a-judge Baseline Pipeline Initialized")
    print("=" * 80 + "\n")

    for cat, groups in test_dataset.items():
        tasks = []
        expected_labels = []
        inputs_log = []
        
        for group_name, cases in groups.items():
            for c in cases:
                user_input = c.get("user_input")
                resps = c.get("mock_responses", [])
                agent_response = resps[0] if resps else c.get("mock_resp", "Default Response")
                expected = c.get("expected_pass")
                
                tasks.append(generic_judge_single_case(llm_baseline, cat, user_input, agent_response))
                expected_labels.append(expected)
                inputs_log.append(user_input)

        actual_results = await asyncio.gather(*tasks)

        for idx, actual in enumerate(actual_results):
            expected = expected_labels[idx]
            user_input = inputs_log[idx]
            
            if expected is True:
                if actual is True: confusion_matrix[cat]["TP"] += 1
                else: confusion_matrix[cat]["FN"] += 1
            else:
                if actual is False: confusion_matrix[cat]["TN"] += 1
                else: confusion_matrix[cat]["FP"] += 1
                
            all_raw_details.append({
                "category": cat,
                "input": user_input,
                "expected_pass": expected,
                "actual_pass": actual,
                "is_correct_judgment": (actual == expected)
            })

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

    output_filename = "RQ2_Baseline_Result.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_cases": total_cases,
                "total_correct": total_correct,
                "baseline_accuracy": f"{overall_accuracy * 100:.2f}%"
            }
        }, f, ensure_ascii=False, indent=4)
    print(f"Evaluation complete. Baseline accuracy summary has been written to {output_filename}\n")

if __name__ == "__main__":
    asyncio.run(main())