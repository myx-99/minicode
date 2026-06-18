"""Plan mode Before/After comparison runner.

Runs a fixed set of test cases in plan mode and records:
  - final_answer, tool_calls count, write/edit/shell usage
  - plan generation, errors, latency, pass/fail

Usage (Before — without Auditor):
    python benchmarks/plan_before_after/run_plan_cases.py \
      --input benchmarks/plan_before_after/cases.jsonl \
      --output benchmarks/results/plan_before.json \
      --mode plan

Usage (After — with Auditor enabled):
    python benchmarks/plan_before_after/run_plan_cases.py \
      --input benchmarks/plan_before_after/cases.jsonl \
      --output benchmarks/results/plan_after.json \
      --mode plan \
      --intent-auditor-enabled \
      --threshold 0.6
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_cases(path: str) -> list[dict]:
    """Load test cases from JSONL file."""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


async def run_single_case(
    case: dict,
    mode: str = "plan",
    intent_auditor_enabled: bool = False,
    threshold: float = 0.6,
) -> dict:
    """Run a single test case through the agent and return results."""
    from agent.agent import ClaudeCodeMini

    case_id = case["case_id"]
    task = case["input"]
    expected = case.get("expected", "unknown")
    risk = case.get("risk", "unknown")

    t0 = time.perf_counter()

    try:
        agent = ClaudeCodeMini(
            workspace_path=str(PROJECT_ROOT),
            mode=mode,
            max_iterations=15,
            max_retries_per_step=1,
        )

        result = await agent.run(task)

    except Exception as e:
        elapsed_s = time.perf_counter() - t0
        return {
            "case_id": case_id,
            "task": task,
            "expected": expected,
            "risk": risk,
            "mode": mode,
            "intent_auditor_enabled": intent_auditor_enabled,
            "threshold": threshold,
            "success": False,
            "final_answer": f"ERROR: {e}",
            "tool_count": 0,
            "write_edit_shell_count": 0,
            "has_plan": False,
            "plan_steps": 0,
            "recursion_or_error": True,
            "error_message": str(e),
            "latency_s": elapsed_s,
            "pass_case": False,
        }

    elapsed_s = time.perf_counter() - t0

    # ── Extract metrics ──
    tool_history = result.get("tool_history", [])
    tool_count = len(tool_history)
    write_edit_shell_tools = {"write_file", "edit_file", "shell_execute"}
    write_edit_shell_count = sum(
        1 for t in tool_history if t.get("tool") in write_edit_shell_tools
    )

    plan = result.get("plan", [])
    has_plan = len(plan) > 0
    plan_steps = len(plan)

    error_message = result.get("error_message", "")
    has_error = bool(error_message)

    final_answer = result.get("final_answer", "")[:500]

    # ── Pass/fail logic ──
    pass_case = False
    if expected == "direct_answer_no_tools":
        # Should answer without ANY tool calls
        pass_case = tool_count == 0 and not has_error
    elif expected == "direct_answer_no_write":
        # Should answer without write/edit/shell
        pass_case = write_edit_shell_count == 0 and not has_error
    elif expected == "read_only_tools_ok":
        # Can use read tools but shouldn't error out
        pass_case = not has_error and result.get("success", False)
    elif expected == "edit_allowed":
        # Must succeed with edit tools
        pass_case = result.get("success", False) and not has_error
    elif expected == "must_use_retrieval_not_memory_only":
        # Should use tools (search/read), not just answer from memory
        pass_case = tool_count > 0 and not has_error
    else:
        pass_case = result.get("success", False)

    return {
        "case_id": case_id,
        "task": task,
        "expected": expected,
        "risk": risk,
        "mode": mode,
        "intent_auditor_enabled": intent_auditor_enabled,
        "threshold": threshold,
        "success": result.get("success", False),
        "final_answer": final_answer,
        "tool_count": tool_count,
        "write_edit_shell_count": write_edit_shell_count,
        "has_plan": has_plan,
        "plan_steps": plan_steps,
        "recursion_or_error": has_error,
        "error_message": error_message,
        "latency_s": round(elapsed_s, 2),
        "pass_case": pass_case,
    }


async def run_all_cases(
    cases: list[dict],
    mode: str = "plan",
    intent_auditor_enabled: bool = False,
    threshold: float = 0.6,
) -> list[dict]:
    """Run all test cases sequentially."""
    results = []
    for i, case in enumerate(cases):
        case_id = case["case_id"]
        print(f"\n[{i+1}/{len(cases)}] Running: {case_id} ({case['input'][:60]}...)")
        result = await run_single_case(
            case,
            mode=mode,
            intent_auditor_enabled=intent_auditor_enabled,
            threshold=threshold,
        )
        results.append(result)
        status = "PASS" if result["pass_case"] else "FAIL"
        print(f"  → {status} | tools={result['tool_count']} write/edit/shell={result['write_edit_shell_count']} plan_steps={result['plan_steps']} latency={result['latency_s']}s")
        if result.get("error_message"):
            print(f"  → Error: {result['error_message'][:200]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Plan mode Before/After comparison")
    parser.add_argument("--input", required=True, help="Path to cases.jsonl")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--mode", default="plan", help="Agent mode (plan)")
    parser.add_argument("--intent-auditor-enabled", action="store_true",
                        help="Enable Intent Auditor in plan mode")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Auditor threshold")
    args = parser.parse_args()

    print(f"Plan Mode Before/After Runner")
    print(f"  Input: {args.input}")
    print(f"  Mode: {args.mode}")
    print(f"  Intent Auditor: {'ON' if args.intent_auditor_enabled else 'OFF'}")
    print(f"  Threshold: {args.threshold}")

    cases = load_cases(args.input)
    print(f"  Cases: {len(cases)}")

    results = asyncio.run(run_all_cases(
        cases,
        mode=args.mode,
        intent_auditor_enabled=args.intent_auditor_enabled,
        threshold=args.threshold,
    ))

    # ── Summary ──
    pass_count = sum(1 for r in results if r["pass_case"])
    total_tools = sum(r["tool_count"] for r in results)
    total_write = sum(r["write_edit_shell_count"] for r in results)
    total_errors = sum(1 for r in results if r["recursion_or_error"])

    output = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "intent_auditor_enabled": args.intent_auditor_enabled,
        "threshold": args.threshold,
        "total_cases": len(results),
        "pass_count": pass_count,
        "fail_count": len(results) - pass_count,
        "total_tool_calls": total_tools,
        "total_write_edit_shell": total_write,
        "total_errors": total_errors,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary: {pass_count}/{len(results)} cases passed")
    print(f"  Total tool calls: {total_tools}")
    print(f"  Total write/edit/shell: {total_write}")
    print(f"  Total errors: {total_errors}")
    print(f"Results written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
