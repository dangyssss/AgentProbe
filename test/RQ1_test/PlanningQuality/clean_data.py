import json
import os
import sys
from typing import List, Dict, Any

def extract_discrepant_cases(input_file: str, output_file: str) -> None:
    """
    Parses the combined dataset to extract instances where annotations from DeepSeek 
    and GPT exhibit discrepancies, generating a flattened JSON format for specialized audit.
    """
    if not os.path.exists(input_file):
        sys.stderr.write(f"Critical Error: Target input file '{input_file}' not found.\n")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset: Dict[str, Any] = json.load(f)
    except Exception as e:
        sys.stderr.write(f"Critical Error: Failed to deserialize JSON dataset -> {e}\n")
        return

    audit_payload: List[Dict[str, Any]] = []
    total_scanned: int = 0

    for agent_key, agent_data in dataset.items():
        domain: str = agent_data.get("domain", "")
        agent_name: str = agent_data.get("agent_name", "")
        test_cases: List[Dict[str, Any]] = agent_data.get("test_cases_to_label", [])

        for case in test_cases:
            total_scanned += 1
            gt_ds: str = case.get("ground_truth_ds", "").strip()
            gt_gpt: str = case.get("ground_truth_gpt", "").strip()

            if gt_ds != gt_gpt:
                audit_payload.append({
                    "id": case.get("req_id", ""),
                    "belonging_agent": f"{domain}_{agent_name}",
                    "test_requirement": case.get("req_text", ""),
                    "ground_truth_ds": gt_ds,
                    "ground_truth_gpt": gt_gpt,
                    "final_audited_dimension": ""
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(audit_payload, f, ensure_ascii=False, indent=4)

    print(f"Scan Completed: Successfully processed {total_scanned} test records.")
    print(f"Discrepancy Detection: Found {len(audit_payload)} conflicting dual-track annotation entries.")
    print(f"Persistence Layer Matrix: Conflict dataset exported to target audit path: {output_file}")

if __name__ == "__main__":
    INPUT_DATASET: str = "final_cleaned_balanced_dataset.json"
    OUTPUT_AUDIT: str = "conflict_cases_for_audit.json"
    
    extract_discrepant_cases(INPUT_DATASET, OUTPUT_AUDIT)