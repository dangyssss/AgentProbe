import json
import os
import sys
from typing import Dict, List, Any, Optional

CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Standardized quality dimensions for agent evaluation literature
DIMENSION_MAP: Dict[str, str] = {
    "1": "Factual Accuracy",
    "2": "Logical Reasoning",
    "3": "Domain Expertise",
    "4": "Tool Use Correctness",
    "5": "Input Robustness",
    "6": "Adversarial Robustness",
    "7": "Safety and Ethical Compliance",
    "8": "Persona Consistency",
    "9": "Resource Efficiency",
    "0": "Concurrency Performance",
    "c": "custom"
}

def print_menu() -> None:
    print("\n" + "="*50)
    print("  AGENTPROBE CORE QUALITY DIMENSION TERMINAL PANEL")
    print("="*50)
    print(" [1] Factual Accuracy          [2] Logical Reasoning")
    print(" [3] Domain Expertise          [4] Tool Use Correctness")
    print(" [5] Input Robustness          [6] Adversarial Robustness")
    print(" [7] Safety & Ethical Policy   [8] Persona Consistency")
    print(" [9] Resource Efficiency       [0] Concurrency Performance")
    print(" [c] custom (Open-domain tailored requests)")
    print(" [q] Save progress and exit execution")
    print("="*50)

def load_json(file_name: str) -> Optional[Dict[str, Any]]:
    path: str = os.path.join(CURRENT_DIR, file_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return None
    return None

def main() -> None:
    pool_file: str = "structured_agent_requirements_pool.json"
    dataset: Optional[Dict[str, Any]] = load_json(pool_file)
    if not dataset:
        sys.stderr.write(f"Critical Error: Base unstructured data file '{pool_file}' not found.\n")
        return

    human_output_file: str = "human_labeled_ground_truth.json"
    existing_human_data: Optional[Dict[str, Any]] = load_json(human_output_file)
    if existing_human_data:
        print("Restoration Event: Pre-existing human annotations detected. Resuming checkpoint...")
        dataset = existing_human_data

    total_cases: int = sum(len(info["test_cases_to_label"]) for info in dataset.values())
    annotated_count: int = 0

    print_menu()
    
    try:
        for agent_key, agent_info in dataset.items():
            domain: str = agent_info.get("domain", "")
            agent_name: str = agent_info.get("agent_name", "")
            
            for case in agent_info["test_cases_to_label"]:
                if case.get("ground_truth"):
                    annotated_count += 1
                    continue
                
                print("\n" + "-"*80)
                print(f"Metrics Progress: [{annotated_count + 1}/{total_cases}] | Domain: {domain} | Agent ID: {agent_name}")
                print(f"Target Statement Under Evaluation:\n{case['req_text']}")
                print("-"*80)
                
                while True:
                    user_input: str = input("Enter dimension macro shortcut (1-0, c, q): ").strip().lower()
                    
                    if user_input == 'q':
                        print("\nSerialization Event: Safely storing current annotation progress...")
                        raise KeyboardInterrupt
                    
                    if user_input in DIMENSION_MAP:
                        selected_label: str = DIMENSION_MAP[user_input]
                        case["ground_truth"] = selected_label
                        print(f"Verification Success: Registered instance as [{selected_label}]")
                        annotated_count += 1
                        break
                    else:
                        print("Validation Error: Invalid token entry. Enter [1-0], [c] for custom, or [q] to exit.")
                        
    except KeyboardInterrupt:
        print("\nExecution Halting: Active loop suspended by human annotator.")
    
    with open(os.path.join(CURRENT_DIR, human_output_file), "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Storage Complete: Human gold-standard dataset written to: {human_output_file}")

    llm_output_file: str = "llm_labeled_ground_truth.json"
    llm_dataset: Optional[Dict[str, Any]] = load_json(llm_output_file)
    
    if not llm_dataset:
        print(f"Execution Warning: Parallel asset file '{llm_output_file}' not found. Skipping discrepancy calculation loop.")
        return

    print("\nAuditing Discrepancies: Benchmarking [Human Expert Annotations] against [LLM Automated Labels]...")
    diff_report: Dict[str, Any] = {}
    diff_count: int = 0
    match_count: int = 0

    for agent_key, human_info in dataset.items():
        agent_name = human_info["agent_name"]
        llm_info = llm_dataset.get(agent_key, {})
        
        llm_cases_map: Dict[str, str] = {
            c["req_id"]: c["ground_truth"] 
            for c in llm_info.get("test_cases_to_label", [])
        }
        
        for h_case in human_info["test_cases_to_label"]:
            req_id: str = h_case["req_id"]
            h_label: str = h_case.get("ground_truth", "")
            l_label: str = llm_cases_map.get(req_id, "")
            
            if not h_label:
                continue
                
            if h_label != l_label:
                diff_count += 1
                if agent_key not in diff_report:
                    diff_report[agent_key] = {
                        "agent_name": agent_name,
                        "discrepancies": []
                    }
                diff_report[agent_key]["discrepancies"].append({
                    "req_id": req_id,
                    "req_text": h_case["req_text"],
                    "human_ground_truth": h_label,
                    "llm_label": l_label if l_label else "UNLABELED"
                })
            else:
                match_count += 1

    diff_output_file: str = "labeled_discrepancies_diff.json"
    with open(os.path.join(CURRENT_DIR, diff_output_file), "w", encoding="utf-8") as f:
        json.dump(diff_report, f, ensure_ascii=False, indent=4)
        
    total_reviewed: int = match_count + diff_count
    agreement_rate: float = (match_count / total_reviewed * 100) if total_reviewed > 0 else 0.0
    
    print(f"Audit Analysis Completed: Verified {total_reviewed} annotated cases.")
    print(f"  Consensus Metrics -> Matches: {match_count} records | Discrepancies (Diff): {diff_count} records")
    print(f"  Quantitative Agreement Rate (Inter-Rater Reliability): {agreement_rate:.2f}%")
    print(f"  Granular Conflict Log exported to asset path: {diff_output_file}")
    print("Ablation Guide: The output file provides empirical indicators for discussing categorical semantic boundary overlaps.")

if __name__ == "__main__":
    main()