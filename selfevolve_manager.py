from __future__ import annotations
import sys
import os
import re
import shutil
import datetime
import textwrap
import selfevolve_rubric
import asyncio
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from metagpt.llm import LLM
import importlib


class SelfevolveRubricManager:
    def __init__(self, file_path="selfevolve_rubric.py"):
        self.file_path = os.path.join(CURRENT_DIR, file_path)
        self.backup_dir = os.path.join(CURRENT_DIR, "backups")
        self.base_backup_path = os.path.join(self.backup_dir, "SelfevolveRubric_BASE.py.bak")
        self.llm = LLM()
        
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

        if not os.path.exists(self.base_backup_path) and os.path.exists(self.file_path):
            shutil.copy2(self.file_path, self.base_backup_path)
            print(f"[SelfevolveRubricManager] Base snapshot (Initial Seed) created for safety.")

    def _commit(self, version: str):
        """
        Store the current SelfevolveRubric into the backup repository before executing physical modifications.
        """
        if os.path.exists(self.file_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"SelfevolveRubric_v{version}_{timestamp}.py.bak"
            backup_path = os.path.join(self.backup_dir, backup_name)
            shutil.copy2(self.file_path, backup_path)
            print(f"[SelfevolveRubricManager] Snapshot created: {backup_name}")

    async def rollback(self):
        """
        One-Click Rollback: Restore to the most recent .bak backup version.
        """
        backups = sorted([
            f for f in os.listdir(self.backup_dir) 
            if f.endswith(".bak") and "BASE" not in f
        ])
        
        if not backups:
            print("[SelfevolveRubricManager] No intermediate backups found. Rollback failed.")
            return False
        
        latest_backup = backups[-1]
        backup_path = os.path.join(self.backup_dir, latest_backup)
        
        shutil.copy2(backup_path, self.file_path)
        importlib.reload(selfevolve_rubric)
        print(f"[Rollback] SelfevolveRubric successfully restored from: {latest_backup}")
        return True

    async def reset_to_base(self):
        """
        Factory Reset: Bypass all intermediate versions and restore directly to the initial seed version.
        """
        if not os.path.exists(self.base_backup_path):
            print("[SelfevolveRubricManager] Critical Error: Base backup not found.")
            return False
        
        shutil.copy2(self.base_backup_path, self.file_path)
        importlib.reload(selfevolve_rubric)
        print(f"[FACTORY RESET] SelfevolveRubric has been restored to the INITIAL version (v1.0.0).")
        return True

    async def upgrade_rubric(self, user_feedback: str):
        """
        Incorporate user feedback to automatically rewrite and upgrade SelfevolveRubrical rules via LLM.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Cannot find {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            current_code = f.read()

        version_match = re.search(r'SelfevolveRubric_VERSION\s*=\s*"(.*?)"', current_code)
        current_version = version_match.group(1) if version_match else "unknown"

        self._commit(current_version)

        prompt = textwrap.dedent(f"""
            You are a "Low-Level System Rules Architect". Your task is to modify the Audit SelfevolveRubric based on user feedback.
            
            [Current Code Content]:
            {current_code}
            
            [User Modification Intent / Feedback]:
            "{user_feedback}"
            
            ### Strict Rewriting Requirements:
            1. Perform logical additions, deletions, or modifications only on the articles within the SelfevolveRubric_RULES dictionary.
            2. You must increment the version string in SelfevolveRubric_VERSION.
            3. You must preserve the full integrity of the get_SelfevolveRubric_text function.
            4. Output only the raw Python code content. Do not include Markdown code block fences (such as ```python).
            
            Please begin rewriting:
        """)

        print(f"[SelfevolveRubricManager] Automatically upgrading the SelfevolveRubric based on user feedback...")
        new_code = await self.llm.aask(prompt, stream=False)
        
        new_code = new_code.strip().replace("```python", "").replace("```", "")
        
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        importlib.reload(selfevolve_rubric)
        print(f"SelfevolveRubric successfully upgraded and saved to {self.file_path}")


if __name__ == "__main__":  
    async def init_manager():
        mgr = SelfevolveRubricManager()
        print("[SelfevolveRubricManager] Initialization complete")
        print(f"Backup directory status: {'Ready' if os.path.exists(mgr.backup_dir) else 'Failed'}")    
    asyncio.run(init_manager())