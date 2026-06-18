"""Graph builder — constructs the full LangGraph StateGraph for the Coding Agent.

V3 enhancements:
  - Three-mode graph construction: "ask" (read-only React), "agent" (full React),
    and "plan" (V1 Plan-and-Execute + enhanced routing).
  - "react" is accepted as a deprecated alias for "agent".
  - execute → finish/replan shortcuts (LLM can declare completion or request replan).
  - Routing extracted to graph/routing.py for testability.
  - Node factories accept optional ContextManager and MemoryManager.
  - No intent_class routing — model decides tool usage via mode + tool registry.
"""

from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from agent.state import AgentState
from graph.nodes import (
    init_node,
    plan_node,
    audit_plan_node,
    create_execute_node,
    create_tool_node,
    create_reflect_node,
    replan_node,
    finish_node,
)
from graph.routing import route_after_execute, route_after_reflect, route_after_audit, normalize_mode


# ═══════════════════════════════════════════════════════════════════
# ── Graph Construction ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

def build_graph(
    llm: BaseChatModel,
    tool_registry,
    mode: str = "agent",
    context_manager=None,
    memory_manager=None,
) -> StateGraph:
    """Build and compile the full Coding Agent StateGraph.

    Args:
        llm: A LangChain ChatModel instance (without tools bound).
        tool_registry: A ToolRegistry instance with mode-appropriate tools.
        mode: "ask" (read-only React), "agent" (full React, default),
              or "plan" (Plan-and-Execute). "react" is accepted as "agent" alias.
        context_manager: Optional ContextManager for message compression (V2).
        memory_manager: Optional MemoryManager for cross-turn memory (V2.1).

    Returns:
        A compiled LangGraph graph.

    Ask / Agent mode graph (same structure, different tool registry):
        START → [init] → [execute] ←─────────────────────┐
                            │                             │
                  ┌─────────┴─────────┐                  │
                  ▼                   ▼                  │
               [tools]          [route_execute]          │
                  │                   │                  │
                  │         ┌─────────┼─────────┐        │
                  │         ▼         ▼         ▼       │
                  │      [replan]  [finish]  [execute]   │
                  │         │         │      (continue)  │
                  └──→ execute ←──────┘                 │
                            ▲                             │
                            └─────────────────────────────┘
        (No plan, no reflect — model decides when to use tools)

    Plan mode graph:
        START → [init] → [plan] → [execute] ←──────────────────┐
                                    │                         │
                          ┌─────────┴─────────┐               │
                          ▼                   ▼               │
                       [tools]          [route_execute]        │
                          │                   │               │
                          │         ┌─────────┼─────────┐     │
                          │         ▼         ▼         ▼     │
                          │    [reflect]  [replan]  [finish]  │
                          │         │         │               │
                          └──→ execute ←──────┘               │
                                    ▲                         │
                                    └─────────────────────────┘
    """
    # ── Normalize mode: react → agent (deprecated alias) ─────
    mode = normalize_mode(mode)

    # ── Tool-aware LLM ──────────────────────────────────────
    tool_schemas = tool_registry.get_openai_schemas()
    llm_with_tools = llm.bind_tools(tool_schemas)

    # ── Create node functions (injecting dependencies) ──────
    _execute_node = create_execute_node(
        llm_with_tools,
        context_manager=context_manager,
    )

    _tool_node = create_tool_node(tool_registry)
    _reflect_node = create_reflect_node(llm)

    async def _plan_node(state: AgentState):
        return await plan_node(state, llm)

    async def _replan_node(state: AgentState):
        return await replan_node(state, llm)

    async def _finish_node(state: AgentState):
        return await finish_node(
            state, llm, memory_manager=memory_manager
        )

    async def _init_node(state: AgentState):
        return await init_node(state, memory_manager=memory_manager)

    async def _audit_plan_node(state: AgentState):
        return await audit_plan_node(state, llm, settings=None)

    # ── Build ───────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    # Common nodes (all modes)
    workflow.add_node("init", _init_node)
    workflow.add_node("execute", _execute_node)
    workflow.add_node("tools", _tool_node)
    workflow.add_node("replan", _replan_node)
    workflow.add_node("finish", _finish_node)

    if mode == "plan":
        # ── Plan mode: Plan-and-Execute + reflect + audit ──
        workflow.add_node("plan", _plan_node)
        workflow.add_node("audit_plan", _audit_plan_node)
        workflow.add_node("reflect", _reflect_node)

        workflow.set_entry_point("init")
        # V3: plan mode with optional Intent Auditor
        workflow.add_edge("init", "plan")
        workflow.add_edge("plan", "audit_plan")
        # audit_plan → execute OR finish (if all steps rejected)
        workflow.add_conditional_edges(
            "audit_plan",
            route_after_audit,
            {
                "execute": "execute",
                "finish": "finish",
            },
        )

        # execute → tools / reflect / replan / finish
        workflow.add_conditional_edges(
            "execute",
            route_after_execute,
            {
                "tools": "tools",
                "reflect": "reflect",
                "replan": "replan",
                "finish": "finish",
                "execute": "reflect",  # fallback
            },
        )

        # tools → back to execute (ReAct observe → act)
        workflow.add_edge("tools", "execute")

        # reflect → execute / replan / finish
        workflow.add_conditional_edges(
            "reflect",
            route_after_reflect,
            {
                "execute": "execute",
                "replan": "replan",
                "finish": "finish",
            },
        )

        # replan → execute (with revised plan)
        workflow.add_edge("replan", "execute")

        # finish → END
        workflow.add_edge("finish", END)

    else:
        # ── Ask / Agent mode: React loop (no plan/reflect) ──
        # Both modes use the same graph structure — the difference is
        # in the tool registry (ask=3 read-only, agent=6 full tools).

        workflow.set_entry_point("init")
        workflow.add_edge("init", "execute")

        # execute → tools / finish / replan / execute (free loop)
        workflow.add_conditional_edges(
            "execute",
            route_after_execute,
            {
                "tools": "tools",
                "finish": "finish",
                "replan": "replan",
                "execute": "execute",    # ← self-loop: continue thinking
                "reflect": "finish",     # fallback (shouldn't happen in ask/agent)
            },
        )

        # tools → back to execute (ReAct observe → act)
        workflow.add_edge("tools", "execute")

        # replan → execute (with revised steps or empty plan)
        workflow.add_edge("replan", "execute")

        # finish → END
        workflow.add_edge("finish", END)

    return workflow.compile()
