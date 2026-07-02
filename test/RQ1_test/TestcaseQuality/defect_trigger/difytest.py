import asyncio
import json
import os
import sys
import time
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional

DIFY_BASE_URL: str = "https://api.dify.ai/v1"

DIFY_AGENT_API_KEYS: Dict[str, str] = {
    "Agent_1_Medical":         "app-6PLoF29m0X9KKHxku0YNi9qY",
    "Agent_2_Financial":       "app-vLXePUcGHbpMzzFDPF9mSCoN",
    "Agent_3_Ecommerce":       "app-gtr4xdRTVY3Diou3gR61g7w5",
    "Agent_4_Psychological":   "app-Cf6deH9TpAakObim2BHvqC4o", 
    "Agent_5_Event_Scheduler": "app-KlVeLdGxG48iPCoX74Sw6E16", 
}

DIFY_AGENT_APP_TYPES: Dict[str, str] = {
    "Agent_1_Medical":         "workflow",
    "Agent_2_Financial":       "workflow",
    "Agent_3_Ecommerce":       "workflow",
    "Agent_4_Psychological":   "chatflow",
    "Agent_5_Event_Scheduler": "workflow",
}

REQUEST_TIMEOUT: int = 120
MAX_CONCURRENT: int = 3
RETRY_COUNT: int = 2
RETRY_DELAY: int = 5

TEST_USER_ID: str = "benchmark-2026"

AGENT_DIMENSION_MATRIX: Dict[str, Dict[str, Any]] = {
    "Agent_1_Medical": {
        "name_en": "Medical_Triage_Agent",
        "dimensions": ["ACCURACY", "DOMAIN", "ROBUSTNESS"],
    },
    "Agent_2_Financial": {
        "name_en": "Financial_Audit_Assistant",
        "dimensions": ["LOGIC", "COST", "TOOL"],
    },
    "Agent_3_Ecommerce": {
        "name_en": "Ecommerce_Customer_Support",
        "dimensions": ["TOOL", "ATTACK", "LOGIC"],
    },
    "Agent_4_Psychological": {
        "name_en": "Psychological_Counseling_Support",
        "dimensions": ["ETHICS", "HUMANOID", "ATTACK"],
    },
    "Agent_5_Event_Scheduler": {
        "name_en": "Executive_Schedule_Assistant",
        "dimensions": ["BURST", "ROBUSTNESS", "ETHICS"],
    },
}

PLANNERAGENT_UNIFIED_FILE: str = "./test_case_for_PA/planneragent_full_matrix.json"
PLANNERAGENT_FILES: Dict[str, Optional[str]] = {
    "Agent_1_Medical":         None,
    "Agent_2_Financial":       None,
    "Agent_3_Ecommerce":       None,
    "Agent_4_Psychological":   None,
    "Agent_5_Event_Scheduler": None,
}

BASELINE_UNIFIED_FILE: str = "./test_case_for_PA/baseline_full_matrix.json"
BASELINE_FILES: Dict[str, Optional[str]] = {
    "Agent_1_Medical":         None,
    "Agent_2_Financial":       None,
    "Agent_3_Ecommerce":       None,
    "Agent_4_Psychological":   None,
    "Agent_5_Event_Scheduler": None,
}

def load_test_cases_from_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Anomalous evaluation suite structural format: {filepath}")

def load_unified_file(filepath: str) -> Dict[str, Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[str, Dict[str, Any]] = {}
    for agent_key, agent_data in data.items():
        if isinstance(agent_data, dict) and "cases" in agent_data:
            cases = agent_data["cases"]
            metadata = {
                "name": agent_data.get("name", ""),
                "total_cases_generated": agent_data.get("total_cases_generated", 0),
                "score": agent_data.get("score", 0),
                "dimension_distribution": agent_data.get("dimension_distribution", {}),
            }
            result[agent_key] = {
                "metadata": metadata,
                "cases": cases,
            }
            print(f"  Injesting Node -> {agent_key}: Consolidated {len(cases)} instances (Source quality score: {metadata['score']})")
        elif isinstance(agent_data, list):
            result[agent_key] = {"metadata": {}, "cases": agent_data}
            print(f"  Injesting Node -> {agent_key}: Consolidated {len(agent_data)} instances")
        else:
            sys.stderr.write(f"Execution Warning: Skipping unparseable agent entity sequence block: {agent_key}\n")

    return result

def load_all_test_cases(
    unified_file: Optional[str],
    per_agent_files: Dict[str, Optional[str]],
    source_label: str
) -> Dict[str, Dict[str, Any]]:
    if unified_file and os.path.exists(unified_file):
        print(f"Deserialization Layer: Loading consolidated validation suite for [{source_label}] via: {unified_file}")
        return load_unified_file(unified_file)

    all_cases: Dict[str, Dict[str, Any]] = {}
    for agent_key, filepath in per_agent_files.items():
        if filepath and os.path.exists(filepath):
            cases = load_test_cases_from_file(filepath)
            all_cases[agent_key] = {"metadata": {}, "cases": cases}
            print(f"  Local Node -> {agent_key}: Resolved {len(cases)} instances extracted from target path: {filepath}")
        else:
            print(f"  Local Node -> {agent_key}: Target evaluation asset not assigned or missing: {filepath}")

    return all_cases

def extract_cases(all_cases: Dict[str, Dict[str, Any]], agent_key: str) -> List[Dict[str, Any]]:
    agent_data = all_cases.get(agent_key)
    if not agent_data:
        return []
    if isinstance(agent_data, dict):
        return agent_data.get("cases", [])
    elif isinstance(agent_data, list):
        return agent_data
    return []

def extract_metadata(all_cases: Dict[str, Dict[str, Any]], agent_key: str) -> Dict[str, Any]:
    agent_data = all_cases.get(agent_key, {})
    if isinstance(agent_data, dict):
        return agent_data.get("metadata", {})
    return {}

class DifyAgentCaller:
    def __init__(self, base_url: str):
        self.base_url: str = base_url.rstrip("/")
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    def _get_endpoint(self, app_type: str) -> str:
        if app_type == "chatflow":
            return f"{self.base_url}/chat-messages"
        elif app_type == "workflow":
            return f"{self.base_url}/workflows/run"
        else:
            raise ValueError(f"Unsupported Dify systemic architecture application schema type: {app_type}")

    def _build_payload(self, app_type: str, query: str, conversation_id: str = "") -> Dict[str, Any]:
        if app_type == "chatflow":
            payload: Dict[str, Any] = {
                "inputs": {},
                "query": query,
                "response_mode": "blocking",
                "user": TEST_USER_ID,
            }
            if conversation_id:
                payload["conversation_id"] = conversation_id
            return payload
        elif app_type == "workflow":
            return {
                "inputs": {"userinput": query},
                "response_mode": "blocking",
                "user": TEST_USER_ID,
            }
        else:
            raise ValueError(f"Unsupported Dify systemic architecture application schema type: {app_type}")

    def _normalize_response(self, app_type: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if app_type == "chatflow":
            metadata: Dict[str, Any] = body.get("metadata", {})
            return {
                "answer": body.get("answer", ""),
                "conversation_id": body.get("conversation_id", ""),
                "message_id": body.get("message_id", ""),
                "task_id": body.get("task_id", ""),
                "metadata": metadata,
                "usage": metadata.get("usage", {}),
            }
        elif app_type == "workflow":
            data: Dict[str, Any] = body.get("data", {})
            outputs: Dict[str, Any] = data.get("outputs", {})
            answer_text: str = str(
                outputs.get("answer", "")
                or outputs.get("text", "")
                or outputs.get("result", "")
                or (str(outputs) if outputs else "")
            )
            return {
                "answer": answer_text,
                "conversation_id": "",
                "message_id": "",
                "task_id": data.get("task_id", ""),
                "metadata": {
                    "total_tokens": data.get("total_tokens", 0),
                    "total_steps": data.get("total_steps", 0),
                    "created_at": data.get("created_at", 0),
                },
                "usage": {
                    "total_tokens": data.get("total_tokens", 0),
                },
                "workflow_outputs_raw": outputs,
            }
        return body

    async def send_single(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        app_type: str,
        query: str,
        conversation_id: str = ""
    ) -> Dict[str, Any]:
        endpoint: str = self._get_endpoint(app_type)
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = self._build_payload(app_type, query, conversation_id)
        last_error: Optional[Dict[str, Any]] = None

        for attempt in range(1, RETRY_COUNT + 1):
            async with self.semaphore:
                start_time: float = time.monotonic()
                try:
                    async with session.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                    ) as resp:
                        elapsed: float = round(time.monotonic() - start_time, 3)

                        if resp.status == 200:
                            body: Dict[str, Any] = await resp.json()
                            normalized: Dict[str, Any] = self._normalize_response(app_type, body)
                            return {
                                "status": "success",
                                "app_type": app_type,
                                **normalized,
                                "latency_sec": elapsed,
                                "http_status": 200,
                            }
                        else:
                            body_text: str = await resp.text()
                            last_error = {
                                "status": "error",
                                "app_type": app_type,
                                "http_status": resp.status,
                                "error_detail": body_text[:500],
                                "latency_sec": elapsed,
                                "attempt": attempt,
                            }
                            sys.stderr.write(f"  Network warning: HTTP {resp.status} received during retry execution loop (attempt {attempt}/{RETRY_COUNT})\n")

                except asyncio.TimeoutError:
                    elapsed = round(time.monotonic() - start_time, 3)
                    last_error = {
                        "status": "timeout",
                        "app_type": app_type,
                        "error_detail": f"Network execution limit reached ({REQUEST_TIMEOUT}s)",
                        "latency_sec": elapsed,
                        "attempt": attempt,
                    }
                    sys.stderr.write(f"  Network warning: Time limit reached (attempt {attempt}/{RETRY_COUNT})\n")

                except Exception as e:
                    elapsed = round(time.monotonic() - start_time, 3)
                    last_error = {
                        "status": "exception",
                        "app_type": app_type,
                        "error_detail": str(e)[:500],
                        "latency_sec": elapsed,
                        "attempt": attempt,
                    }
                    sys.stderr.write(f"  Network warning: Uncaught structural hazard: {e} (attempt {attempt}/{RETRY_COUNT})\n")

                if attempt < RETRY_COUNT:
                    await asyncio.sleep(RETRY_DELAY)

        return last_error or {"status": "unknown_error", "app_type": app_type}

async def execute_agent_tests(
    agent_key: str,
    agent_info: Dict[str, Any],
    test_cases: List[Dict[str, Any]],
    source_label: str,
    caller: DifyAgentCaller,
    session: aiohttp.ClientSession,
) -> Dict[str, Any]:
    api_key: Optional[str] = DIFY_AGENT_API_KEYS.get(agent_key)
    app_type: str = DIFY_AGENT_APP_TYPES.get(agent_key, "workflow")

    if not api_key:
        sys.stderr.write(f"Execution Warning: Target credential credentials missing for [{agent_key}] pipeline, skipping configuration node.\n")
        return {"agent_key": agent_key, "error": "missing_api_key", "results": []}

    agent_name: str = agent_info["name_en"]
    results: List[Dict[str, Any]] = []

    print(f"Target Agent Initialization Pass: Domain Profile -> [{source_label}] | Active Subsystem: {agent_name} ({agent_key})")
    print(f"  Architecture Type: {app_type} | Evaluation Manifest Size: {len(test_cases)} units")

    for idx, case in enumerate(test_cases, 1):
        case_id: str = case.get("id", f"unknown-{idx:03d}")
        case_category: str = case.get("category", "unclassified")
        case_input: str = case.get("input", "")

        if not case_input.strip():
            results.append({
                "id": case_id,
                "category": case_category,
                "input": case_input,
                "response": {"status": "skipped", "error_detail": "empty input"},
                "source": source_label,
            })
            continue

        print(f"  Queue Progression Track -> [{idx}/{len(test_cases)}] Tracking Node ID [{case_id}] ({case_category}) transmitting string payload...")

        response: Dict[str, Any] = await caller.send_single(
            session=session,
            api_key=api_key,
            app_type=app_type,
            query=case_input,
            conversation_id="",
        )

        latency: float = response.get("latency_sec", 0.0)
        print(f"    Transmission Feedback Status: Result [{response.get('status')}] received in {latency}s")

        results.append({
            "id": case_id,
            "category": case_category,
            "input": case_input,
            "source": source_label,
            "response": response,
        })

    success_count: int = sum(1 for r in results if r["response"].get("status") == "success")
    print(f"Subsystem Finalization Profile -> {agent_name} [{source_label}] complete: {success_count}/{len(results)} metrics passed.")

    return {
        "agent_key": agent_key,
        "agent_name": agent_name,
        "app_type": app_type,
        "source": source_label,
        "total_cases": len(test_cases),
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "results": results,
    }

async def run_full_benchmark() -> None:
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("System Mapping Configurations:")
    for ak in AGENT_DIMENSION_MATRIX:
        at: str = DIFY_AGENT_APP_TYPES.get(ak, "?")
        ep: str = "/workflows/run" if at == "workflow" else "/chat-messages"
        print(f"  Mapping Profile -> {ak}: Schema Variant [{at}] bound to target API endpoint: {ep}")

    planner_cases: Dict[str, Dict[str, Any]] = load_all_test_cases(
        unified_file=PLANNERAGENT_UNIFIED_FILE,
        per_agent_files=PLANNERAGENT_FILES,
        source_label="PlannerAgent"
    )

    baseline_cases: Dict[str, Dict[str, Any]] = load_all_test_cases(
        unified_file=BASELINE_UNIFIED_FILE,
        per_agent_files=BASELINE_FILES,
        source_label="Baseline"
    )

    total_planner: int = sum(len(extract_cases(planner_cases, ak)) for ak in AGENT_DIMENSION_MATRIX)
    total_baseline: int = sum(len(extract_cases(baseline_cases, ak)) for ak in AGENT_DIMENSION_MATRIX)
    print(f"\nEvaluation Suite Metric Consolidation: PlannerAgent={total_planner} units | Baseline={total_baseline} units | Cumulative Portfolio Total={total_planner + total_baseline} units")

    if total_planner == 0 and total_baseline == 0:
        sys.stderr.write("Critical Error: Evaluation assets empty. Verify directory mapping values.\n")
        return

    caller: DifyAgentCaller = DifyAgentCaller(base_url=DIFY_BASE_URL)
    all_results: Dict[str, Any] = {}

    async with aiohttp.ClientSession() as session:
        for agent_key, agent_info in AGENT_DIMENSION_MATRIX.items():
            agent_results: Dict[str, Any] = {"planneragent": None, "baseline": None}

            cases: List[Dict[str, Any]] = extract_cases(planner_cases, agent_key)
            metadata: Dict[str, Any] = extract_metadata(planner_cases, agent_key)
            if cases:
                agent_results["planneragent"] = await execute_agent_tests(
                    agent_key=agent_key,
                    agent_info=agent_info,
                    test_cases=cases,
                    source_label="PlannerAgent",
                    caller=caller,
                    session=session,
                )
                agent_results["planneragent"]["source_metadata"] = metadata

            cases = extract_cases(baseline_cases, agent_key)
            metadata = extract_metadata(baseline_cases, agent_key)
            if cases:
                agent_results["baseline"] = await execute_agent_tests(
                    agent_key=agent_key,
                    agent_info=agent_info,
                    test_cases=cases,
                    source_label="Baseline",
                    caller=caller,
                    session=session,
                )
                agent_results["baseline"]["source_metadata"] = metadata

            all_results[agent_key] = {
                "agent_name": agent_info["name_en"],
                "dimensions": agent_info["dimensions"],
                "app_type": DIFY_AGENT_APP_TYPES.get(agent_key, "workflow"),
                "planneragent_run": agent_results["planneragent"],
                "baseline_run": agent_results["baseline"],
            }

    output_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    detail_file: str = os.path.join(output_dir, f"dify_benchmark_detail_{timestamp}.json")
    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Data Export Notification: Meticulous record logs generated at: {detail_file}")

    summary: Dict[str, Any] = {
        "benchmark_metadata": {
            "timestamp": timestamp,
            "dify_base_url": DIFY_BASE_URL,
            "agent_app_types": DIFY_AGENT_APP_TYPES,
            "test_user_id": TEST_USER_ID,
        },
        "agents": {}
    }

    total_success: int = 0
    total_cases: int = 0

    for agent_key, agent_data in all_results.items():
        agent_summary: Dict[str, Any] = {
            "agent_name": agent_data["agent_name"],
            "dimensions": agent_data["dimensions"],
            "app_type": agent_data.get("app_type", "unknown"),
        }
        for source_key in ["planneragent_run", "baseline_run"]:
            run_data: Optional[Dict[str, Any]] = agent_data.get(source_key)
            if run_data:
                agent_summary[source_key] = {
                    "total_cases": run_data.get("total_cases", 0),
                    "success_count": run_data.get("success_count", 0),
                    "failure_count": run_data.get("failure_count", 0),
                    "source_score": run_data.get("source_metadata", {}).get("score", "N/A"),
                }
                total_success += run_data.get("success_count", 0)
                total_cases += run_data.get("total_cases", 0)
            else:
                agent_summary[source_key] = None

        summary["agents"][agent_key] = agent_summary

    summary["overall"] = {
        "total_cases_executed": total_cases,
        "total_success": total_success,
        "total_failure": total_cases - total_success,
        "success_rate": f"{(total_success / total_cases * 100):.1f}%" if total_cases > 0 else "N/A",
    }

    summary_file: str = os.path.join(output_dir, f"dify_benchmark_summary_{timestamp}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"Data Export Complete: Aggregated metric overview written to path: {summary_file}")

if __name__ == "__main__":
    asyncio.run(run_full_benchmark())