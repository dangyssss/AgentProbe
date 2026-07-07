import asyncio
import json
import os
import sys
import re
from typing import List, Dict, Any

CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
from metagpt.llm import LLM

def load_external_agent_profiles(file_name: str = "agent_profiles.json") -> Dict[str, Any]:
    file_path: str = os.path.join(CURRENT_DIR, file_name)
    if not os.path.exists(file_path):
        sys.stderr.write(f"Critical Error: External asset file '{file_path}' not found.\n")
        sys.exit(1)
        
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Critical Error: Failed to deserialize JSON dataset {e}\n")
            sys.exit(1)

def extract_json(text: str) -> str:
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    return match.group(1) if match else text.strip()

REQUIREMENT_GEN_TMPL: str = """
You are to simulate an end-user scenario acting as a "Citizen Developer" on a Low-Code/No-Code (LCNC) platform (e.g., Coze). 
You possess extensive domain expertise in your respective vertical but lack a formal software engineering background and technical testing expertise. Your method for validating Large Language Model (LLM) agents involves constructing qualitative, localized black-box scenarios derived from actual operational vulnerabilities.

Examine the system specifications of the target agent model under evaluation:

### Target Agent System Profile:
- Identifier: {agent_name}
- System Prompt Configuration: 
```text
{system_prompt}
```
- Mounted Plugins / Toolkits:
{tools_description}

---

### Generation Criteria (Rigorous Scientific Standards):
1. End-User Persona Alignment: Do not incorporate technical terminology (e.g., "boundary values", "robustness", "API schema tuning"). Formulate empirical requirements utilizing plain, natural language scenarios native to your specified industry vertical.
2. Semantic Atomicity and Singularity: Each generated test request must be atomic, containing a solitary assessment parameter. Complex or multi-label semantic expressions are strictly forbidden.
   - Non-compliant (Composite): "I want to test if the system remains polite when users use offensive language and whether it crashes if the database returns an error."
   - Compliant (Atomic): "I want to verify if the agent retains an appropriate tone and policy alignment when exposed to highly informal, sarcastic user inputs."
3. Domain Boundary Derivation: Incorporate domain-specific failures that yield catastrophic operational liabilities (e.g., medical dosage hallucinations, financial miscalculations, or structural output corruption).
4. Coverage Density and Distinctness: Generate an exhaustive collection of diverse scenarios. Ensure test cases are distinct, non-redundant, and densely aligned with the agent profile.

Output the compiled atomic requirements strictly as a raw JSON array of strings. Do not encapsulate the response within markdown code fences or append introductory/explanatory text.

Strictly adhere to the following output syntax:
[
    "Atomic Test Requirement Statement 1",
    "Atomic Test Requirement Statement 2",
    "Atomic Test Requirement Statement 3",
    "..."
]
"""

async def generate_single_llm_requirements(llm_instance: LLM, profile: Dict[str, Any]) -> List[str]:
    tools_str: str = ""
    for t in profile.get("tools", []):
        tools_str += f"  * Tool Name: {t.get('tool_name', 'Unnamed')} | Description: {t.get('description', 'None')}\n"
        
    prompt: str = REQUIREMENT_GEN_TMPL.format(
        agent_name=profile.get("agent_name", "Unnamed_Agent"),
        system_prompt=profile.get("system_prompt", "No configuration prompt provided."),
        tools_description=tools_str if tools_str else "  * No external tools bound."
    )
    
    try:
        raw_res: str = await llm_instance.aask(prompt, stream=False)
        clean_res: str = extract_json(raw_res)
        reqs: Any = json.loads(clean_res)
        return reqs if isinstance(reqs, list) else []
    except Exception as e:
        sys.stderr.write(f"Execution Warning: Synthesis extraction failed for [{profile.get('agent_name', 'Unknown')}] | Details: {e}\n")
        return []

async def cross_review_and_clean(llm_instance: LLM, all_reqs: List[str], profile: Dict[str, Any]) -> List[str]:
    REVIEW_TMPL: str = """
    You are the final arbitrator for an automated LCNC requirement refinement review.
    For the targeted system configuration [{agent_name}], an raw evaluation candidate pool has been synthesized via multi-model execution:
    {reqs_json}
    
    Perform strict syntactic filtration, pruning, and deduplication:
    1. Filter out all instances with composite semantic definitions containing conjunctions (e.g., "and", "furthermore", "simultaneously checking"). Ensure structural atomicity.
    2. Eliminate semantically redundant suggestions, keeping only the most representative localized single-point requests.
    3. Output the sanitized collection strictly as a raw JSON array of strings, devoid of markdown code fences or natural prose explanations:
    """
    prompt: str = REVIEW_TMPL.format(agent_name=profile.get("agent_name", "Unnamed"), reqs_json=json.dumps(all_reqs, ensure_ascii=False))
    try:
        raw_res: str = await llm_instance.aask(prompt, stream=False)
        clean_res: str = extract_json(raw_res)
        final_list: Any = json.loads(clean_res)
        return final_list if isinstance(final_list, list) else all_reqs[:10]
    except Exception as e:
        sys.stderr.write(f"Execution Warning: Synthesis arbitration failed for [{profile.get('agent_name', 'Unknown')}] | Details: {e}\n")
        return list(set(all_reqs))[:6]

async def main() -> None:
    agent_profiles: Dict[str, Any] = load_external_agent_profiles("agent_profiles.json")

    llm_gpt: LLM = LLM()       
    llm_claude: LLM = LLM()    
    llm_gemini: LLM = LLM()    

    new_labeled_dataset: Dict[str, Any] = {}

    total_agents: int = sum([len(agents) for agents in agent_profiles.values()])
    print(f"Dataset Pipeline Engaged: Successfully loaded external asset profiles with {len(agent_profiles)} structural domains and {total_agents} target agents.")
    print("Execution Strategy: Initializing MetaGPT framework multi-model cross-review consensus mechanism...")
    
    for domain_key, agent_list in agent_profiles.items():
        print(f"\nProcessing Operational Domain: [{domain_key}] | Elements to evaluate: {len(agent_list)}")
        
        for idx, profile in enumerate(agent_list):
            agent_name: str = profile.get("agent_name", f"Unnamed_Agent_{idx}")
            agent_id_seed: str = f"{domain_key}_{idx+1:02d}"
            
            print(f"  Deconstructing Configuration Architecture for target: [{agent_name}]")
            
            tasks: List[Any] = [
                generate_single_llm_requirements(llm_gpt, profile),
                generate_single_llm_requirements(llm_claude, profile),
                generate_single_llm_requirements(llm_gemini, profile)
            ]
            results: List[List[str]] = await asyncio.gather(*tasks)
            
            raw_pool: List[str] = []
            for r in results:
                raw_pool.extend(r)
                
            print(f"    Initial Consensus Yield: Generated {len(raw_pool)} requirements. Initiating filtration pass...")
            
            refined_reqs: List[str] = await cross_review_and_clean(llm_gpt, raw_pool, profile)
            
            print(f"    Refinement Phase Finalized: Retained {len(refined_reqs)} validated atomic test records.")
            
            new_labeled_dataset[f"{domain_key}_{agent_name}"] = {
                "domain": domain_key,
                "agent_name": agent_name,
                "system_prompt": profile.get("system_prompt", ""),
                "tools": profile.get("tools", []),
                "test_cases_to_label": [
                    {"req_id": f"{agent_id_seed}_REQ_{i+1:03d}", "req_text": text, "ground_truth": ""} 
                    for i, text in enumerate(refined_reqs)
                ]
            }

    output_path: str = os.path.join(CURRENT_DIR, "structured_agent_requirements_pool.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_labeled_dataset, f, ensure_ascii=False, indent=4)
        
    print(f"\nExecution Complete: Refactored target evaluation suite exported to asset layer path: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())