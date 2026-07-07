import os
import sys
import json
import datetime
import argparse
import contextlib
import io
import logging
import asyncio
import hashlib
import textwrap

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PlannerAgent import PlannerAgent, PlannerAgentRuntimeError
from DiagnosisAgent import DiagnosisAgent
from selfevolve_manager import SelfevolveRubricManager

BOT_ID = os.getenv("BOT_ID")
SELFEVOLVERUBRIC_FILE = os.path.join(CURRENT_DIR, "selfevolve_rubric.py")
HASH_CACHE_FILE = os.path.join(CURRENT_DIR, ".last_rubric_hash")
FIXED_OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")


CASE_NUM_MAP = {
   "ACCURACY": 0,
   "LOGIC": 0,
   "DOMAIN": 0,
   "COST": 0,
   "ROBUSTNESS": 0,
   "HUMANOID": 0,
   "TOOL": 0,
   "ETHICS": 0,
   "ATTACK": 0,
   "BURST": 0,
}

def get_file_hash(filepath):
    if not os.path.exists(filepath): return ""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def check_rubric_modified():
    """Detect whether the SelfEvolveRubric file has undergone physical changes since the last execution."""
    current_hash = get_file_hash(SELFEVOLVERUBRIC_FILE)
    last_hash = ""
    if os.path.exists(HASH_CACHE_FILE):
        with open(HASH_CACHE_FILE, "r") as f:
            last_hash = f.read().strip()
    return current_hash != last_hash, current_hash

def update_hash_cache(current_hash):
    with open(HASH_CACHE_FILE, "w") as f:
        f.write(current_hash)

async def run_evaluation_flow(is_modified: bool):
    """Core workflow: Planning -> Execution -> Reporting"""
    print("\n[Step 1/3] Please enter the functional profile/positioning of the target agent:")

    agent_desc = input("> ").strip() or "General Assistant"
    user_test_request = ""
    per_num = 3
    target_map = None

    active_dims = [k for k, v in CASE_NUM_MAP.items() if v > 0]
    
    if active_dims:
        print(f"Detected active dimensions: {active_dims}. Proceeding with automated full-dimensional evaluation.")
        user_test_request = "Please perform full-dimensional evaluation based on the preset configuration."
        target_map = CASE_NUM_MAP
        per_num = None
    else:
        print("\nNo preset dimensions are currently active. Please describe your primary concerns or testing requirements for this run:")
        user_test_request = input("> ").strip()
        if not user_test_request:
            print("Requirements cannot be empty."); return
            
        print("Please enter the number of test cases to generate per strategy:")
        num_txt = input("> ").strip()
        per_num = int(num_txt) if num_txt.isdigit() else 3
        target_map = None

    print("\n[Step 2/3] Generating test plans and test cases...")
    planner = PlannerAgent()
    try:
        plan_md, raw_cases, debug_payload = await planner.plan_tests(
            agent_desc=agent_desc,
            user_test_request=user_test_request,
            num_cases_map=target_map,
            num_cases_per_strategy=per_num
        )
        print("\n" + "="*50)
        print("Test Plan Preview (Trunacted):")
        preview = (plan_md[:800] + '...') if len(plan_md) > 800 else plan_md
        print(textwrap.indent(preview, "   ")) 
        print("="*50)
        print(f"System has successfully provisioned {len(raw_cases)} test cases.")
        
        confirm = input("\nDo you want to proceed with the evaluation using the above plan? (y/N): ").strip().lower()
        if confirm not in ('y', 'yes'):
            print("Execution canceled. Returning to main menu.")
            return

    except PlannerAgentRuntimeError as e:
        print("\n" + "!"*40)
        print("SelfEvolve Rubric Audit Intercepted")
        print("-" * 40)
        feedback = str(e).replace("Test requirement failed to meet mandatory checklist criteria after 3 RLCF feedback iterations.", "").strip()
        print(f"Rejection Reason:\n{feedback}")
        print("-" * 40)
        print("!"*40 + "\n")
        return
    
    except Exception as e:
        print(f"Planning failed: {e}"); return

    print(f"\n[Step 3/3] Commencing evaluation flow (Shadow Mode: {'ON' if is_modified else 'OFF'})...")
    evaluator = DiagnosisAgent()
    report, _ = await evaluator.evaluate(
        bot_id=BOT_ID,
        plan_markdown=plan_md,
        cases=raw_cases,
        planner_debug=debug_payload,
        is_rubric_modified=is_modified
    )

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(FIXED_OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(FIXED_OUTPUT_DIR, f"Report_{now}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nEvaluation completed. The report has been saved to:\n{report_path}")

async def main():
    mgr = SelfevolveRubricManager()

    while True:
        is_modified, current_hash = check_rubric_modified()
        
        print("\n--- Current System Status ---")
        if is_modified:
            print("SelfEvolveRubric status: MODIFIED (Changes will trigger Dual-Track Audit)")
        else:
            print("SelfEvolveRubric status: ALIGNED (Standard clean report will be generated)")
        
        print("\nAvailable Operations:")
        print("1. Start automated evaluation task")
        print("2. Submit feedback to upgrade the SelfEvolveRubric")
        print("3. Roll back the last rubric modification")
        print("4. Reset the SelfEvolveRubric to base initial settings")
        print("5. Exit system")
        
        choice = input("\nPlease enter command: ").strip()

        if choice == "1":
            await run_evaluation_flow(is_modified)
            update_hash_cache(current_hash)
            
        elif choice == "2":
            feedback = input("\nPlease enter your optimization suggestions for the audit criteria:\n> ").strip()
            if feedback:
                await mgr.upgrade_rubric(feedback)
                print("SelfEvolveRubric has evolved successfully.")
            
        elif choice == "3":
            if await mgr.rollback():
                print("Successfully rolled back to the previous version.")
            
        elif choice == "4":
            confirm = input("Are you sure you want to clear all evolution history and reset to v1.0.0? (y/N): ")
            if confirm.lower() == 'y':
                await mgr.reset_to_base()
                print("Factory reset complete.")
                
        elif choice == "5":
            break
        else:
            print("Invalid selection, please try again.")

if __name__ == "__main__":
    asyncio.run(main())