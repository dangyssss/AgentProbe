import asyncio
import json
import os
import sys
import re
import copy
from typing import List, Dict, Any

CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from metagpt.llm import LLM

def extract_json(text: str) -> str:
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    return match.group(1) if match else text.strip()

LABELING_EXPERT_TMPL: str = """
You are a senior data annotation expert specializing in artificial intelligence and software engineering quality assurance.
Your task is to evaluate empirical user requests generated for localized Low-Code/No-Code (LCNC) agents and rigorously classify them into one of the [Eleven Balanced Quality Dimensions].

⚠️ [Core Meta-Principle: Absolute Balance and Equality] ⚠️
These 11 dimensions occupy strictly equal positions in our evaluation framework. You must deliberately overcome taxonomy bias that tends to over-allocate inputs into traditional technical dimensions. If a testing request deviates from standard language model operational characteristics and targets out-of-domain custom workflows, you must decisively categorize it under the 11th dimension: [custom].

### 📌 Eleven Quality Dimensions Definitions and Behavioral Manifestations:

1. **Factual Accuracy**: 
   - Evaluates the objective truthfulness of the generated responses, detecting hallucinations, factual drift, or epistemic errors.
2. **Logical Reasoning**: 
   - Evaluates sequential causal dependencies and multi-constraint deduction, identifying logical contradictions, semantic state locks, or planning failures.
3. **Domain Expertise**: 
   - Evaluates compliance with highly specialized vertical standards, formal schemas, or authoritative literature (e.g., Legal, Medical, Finance, Advanced Programming).
4. **Tool Use Correctness**: 
   - Evaluates multi-tool orchestration, structural parameter parsing accuracy, toolchain execution routing, and error-handling grace.
5. **Input Robustness**: 
   - Evaluates system tolerance against corrupted user input streams, including typos, informal semantics, extreme long-context blocks, or formatting anomalies.
6. **Adversarial Robustness**: 
   - Evaluates systemic security boundaries against malicious prompt injection vectors, jailbreak framing, or role-playing subversion attempts.
7. **Safety and Ethical Compliance**: 
   - Evaluates alignment with human values and compliance constraints, filtering societal biases, hate speech, privacy leaks, or illicit queries.
8. **Persona Consistency**: 
   - Evaluates conversational persistence regarding assigned configuration identities, preventing out-of-character drift, loss of context, or emotional outbursts.
9. **Resource Efficiency**: 
   - Evaluates execution runtime overhead and operational budget ceilings, tracking token consumption density, API invocation rates, or infinite loop states.
10. **Concurrency Performance**: 
    - Evaluates structural infrastructure stability under heavy concurrent stress tests, analyzing P99 latency bounds, session cross-talk, or server pipeline degradation.
11. **custom**: 
    - **Definition**: Evaluates non-standard business logic, external organizational agreements, legal indemnifications, multi-agent communication protocols, or interactions with physical hardware components that lie entirely outside the traditional 10 LLM boundaries.
    - **Key Indicators**: The test objective does not validate generative quality or baseline capability, but checks if the system adheres to unique macro procedural workflows or compliance constraints native to a specific enterprise.

⚠️ Conflict Resolution & Boundary Arbitration (Equality Framework):
- If an input stream contains typographical errors but its final semantic target is a system hijack, classify as [Adversarial Robustness].
- If excessive resource cost is directly induced by high sudden traffic load, classify as [Concurrency Performance].
- **Critical Directive: Whenever the evaluation objective shifts from traditional conversational benchmarks toward independent workflows, business data deduplication, legal waivers, or multi-tenant system constraints, it MUST be directly classified under [custom]. Strongly resist forcing long-tail compliance entries into standard language dimensions.**

---

### [Target System Configuration Profile]
- System Identifier: {agent_name}
- System Prompt Configuration: 
```text
{system_prompt}
```

### [Target Case Instance under Evaluation]
- Sequence Identifier: {req_id}
- Source Payload Text: "{req_text}"

---

### [Output Formatting Constraints]
Output the evaluation result strictly as a raw JSON object. Do not encapsulate inside markdown code blocks or provide conversational prose.
Adhere exactly to the following syntax:
{{
    "req_id": "{req_id}",
    "dimension_label": "Insert one of the 11 dimensions defined above"
}}
"""

async def label_single_requirement(llm_instance: LLM, agent_info: Dict[str, Any], case: Dict[str, Any], model_tag: str) -> str:
    prompt: str = LABELING_EXPERT_TMPL.format(
        agent_name=agent_info.get("agent_name"),
        system_prompt=agent_info.get("system_prompt"),
        req_id=case.get("req_id"),
        req_text=case.get("req_text")
    )
    try:
        raw_res: str = await llm_instance.aask(prompt, stream=False)
        clean_res: str = extract_json(raw_res)
        res_json: Dict[str, Any] = json.loads(clean_res)
        return res_json.get("dimension_label", "custom")
    except Exception as e:
        sys.stderr.write(f"Execution Error: [{model_tag}] Instance {case.get('req_id')} annotation aborted | Details: {e}\n")
        return "custom"

async def process_dataset_for_model(llm_instance: LLM, original_dataset: Dict[str, Any], model_tag: str) -> Dict[str, Any]:
    model_dataset: Dict[str, Any] = copy.deepcopy(original_dataset)
    
    for agent_key, agent_info in model_dataset.items():
        print(f"Annotation Pipeline: [{model_tag}] processing batch queue for agent: [{agent_info['agent_name']}]")
        
        tasks = []
        for case in agent_info["test_cases_to_label"]:
            tasks.append(label_single_requirement(llm_instance, agent_info, case, model_tag))
            
        labels: List[str] = await asyncio.gather(*tasks)
        
        for i, label in enumerate(labels):
            agent_info["test_cases_to_label"][i]["ground_truth"] = label
            
    return model_dataset

async def main() -> None:
    input_file: str = os.path.join(CURRENT_DIR, "structured_agent_requirements_pool_complement.json")
    if not os.path.exists(input_file):
        sys.stderr.write(f"Critical Error: Target dataset pool '{input_file}' not found.\n")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        base_dataset: Dict[str, Any] = json.load(f)

    print("Initialization Event: Constructing parallel model annotation backends via MetaGPT routers...")
    
    try:
        llm_gpt: LLM = LLM(model="gpt-4o-mini") 
        llm_deepseek: LLM = LLM(model="deepseek-v4-flash") 
    except Exception as e:
        sys.stderr.write(f"Execution Warning: Explicit model initiation failed ({e}). Reverting to default configuration parameters...\n")
        llm_gpt = LLM() 
        llm_deepseek = LLM() 

    print("Execution Strategy: Activating dual-model asynchronous annotation pipeline (GPT-4o-mini & DeepSeek-v4-flash)...")
    
    gpt_task = process_dataset_for_model(llm_gpt, base_dataset, "GPT-4o-mini")
    deepseek_task = process_dataset_for_model(llm_deepseek, base_dataset, "DeepSeek-Flash")
    
    gpt_labeled_dataset, deepseek_labeled_dataset = await asyncio.gather(gpt_task, deepseek_task)

    gpt_output_file: str = os.path.join(CURRENT_DIR, "llm_gpt_labeled_ground_truth_complement.json")
    with open(gpt_output_file, "w", encoding="utf-8") as f:
        json.dump(gpt_labeled_dataset, f, ensure_ascii=False, indent=4)
        
    deepseek_output_file: str = os.path.join(CURRENT_DIR, "llm_deepseek_labeled_ground_truth_complement.json")
    with open(deepseek_output_file, "w", encoding="utf-8") as f:
        json.dump(deepseek_labeled_dataset, f, ensure_ascii=False, indent=4)
        
    print("\nAnnotation Cycle Completed: Independent consensus tracking records generated successfully.")
    print(f"  Asset Export Path -> GPT-4o-mini results saved to: {gpt_output_file}")
    print(f"  Asset Export Path -> DeepSeek-v4-flash results saved to: {deepseek_output_file}")

if __name__ == "__main__":
    asyncio.run(main())