import os
import json
import time
import random
from dotenv import load_dotenv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import TypedDict, Annotated, Literal, Optional

from tools import (
    get_merchant_status,
    find_alternatives,
    assign_alternative_driver,
    notify_customer,
    issue_refund,
    collect_evidence,
    create_support_ticket,
    get_traffic_status,
)
from router import classify_incident
from logger import save_decision_trace

# MODIFIED: Import Gemini instead of OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# 1) Load config
load_dotenv()
# MODIFIED: Check for GOOGLE_API_KEY
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Missing GOOGLE_API_KEY. Please set it in a .env file.")

with open("policies.json", "r") as f:
    POLICIES = json.load(f)

# 2) State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    incident: dict
    steps: list
    result: str
    refund_amount: Optional[float]

# 3) LLM
# MODIFIED: Instantiate Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    timeout=30.0, # Added timeout to prevent hanging
)

# 4) Utilities
IST = ZoneInfo("Asia/Kolkata")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _now_ist_iso() -> str:
    return datetime.now(IST).isoformat()

def _append_step(state: State, thought: str, action: str, observation):
    return {
        "step": len(state["steps"]) + 1,
        "thought": thought,
        "action": action,
        "observation": observation,
        "at_utc": _now_utc_iso(),
        "at_ist": _now_ist_iso(),
    }

def _local_summary(incident, steps):
    itype = incident.get("type", "incident")
    oid = incident.get("order_id", "")
    acts = [s.get("action", "") for s in steps]
    return f"Resolved {itype} for order {oid}; actions: " + ", ".join(acts)

# MODIFIED: Updated type hint to ChatGoogleGenerativeAI
def _invoke_with_retry(model: ChatGoogleGenerativeAI, messages, max_retries=5, max_tokens=96):
    for i in range(max_retries):
        try:
            config = {"max_output_tokens": max_tokens}
            out = model.invoke(messages, generation_config=config)
            return out
        except Exception as e:
            emsg = str(e)
            low = emsg.lower()
            if "quota" in low or "rate" in low or "429" in low or "resource_exhausted" in low or "timeout" in low:
                sleep = (2 ** i) + random.random()
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError("LLM unavailable after retries")

def compute_refund_amount(incident: dict, evidence: dict | None) -> float:
    claimed = None
    if evidence and isinstance(evidence, dict):
        claimed = evidence.get("claimed_amount")
    if isinstance(claimed, (int, float)) and claimed >= 0:
        amount = float(claimed)
    else:
        amount = 0.0
        if evidence and isinstance(evidence, dict):
            items = evidence.get("line_items") or []
            if isinstance(items, list):
                for it in items:
                    qty = int(it.get("missing_qty", 0) or 0)
                    unit = float(it.get("unit_price", 0.0) or 0.0)
                    amount += qty * unit
        if amount <= 0.0:
            if POLICIES.get("refund_flat_amount") is not None:
                amount = float(POLICIES["refund_flat_amount"])
            elif incident.get("type") in {"missing_item", "item_out_of_stock"}:
                unit_fb = float(POLICIES.get("missing_item_unit_price_fallback", 100.0))
                missing_qty = 1
                if evidence and isinstance(evidence, dict):
                    missing_qty = int(evidence.get("total_missing_qty", 1) or 1)
                amount = unit_fb * max(1, missing_qty)
            else:
                amount = 0.0
    order_total = incident.get("order_total")
    if isinstance(order_total, (int, float)):
        pct = POLICIES.get("refund_percent_cap")
        if isinstance(pct, (int, float)) and pct >= 0:
            cap = float(order_total) * (float(pct) / 100.0)
            amount = min(amount, cap)
    max_cap = POLICIES.get("refund_max_amount")
    if isinstance(max_cap, (int, float)) and max_cap >= 0:
        amount = min(amount, float(max_cap))
    return round(max(0.0, amount), 2)

# 5) Nodes
def check_merchant(state: State) -> State:
    inc = state["incident"]
    status = get_merchant_status(inc["merchant_id"])
    if status.get("prep_delay_min", 0) > POLICIES.get("max_prep_delay_before_notify", 30):
        notify_customer(inc["merchant_id"], "Delay exceeds policy threshold")
    step = _append_step(
        state,
        "Checking merchant status and applying notification policy if needed",
        "call_tool:get_merchant_status",
        status,
    )
    return {
        "messages": state["messages"] + [AIMessage(content=f"Merchant status: {status}")],
        "incident": state["incident"],
        "steps": state["steps"] + [step],
        "result": json.dumps(status),
        "refund_amount": state.get("refund_amount"),
    }

def assign_driver_node(state: State) -> State:
    inc = state["incident"]
    assignment = assign_alternative_driver(inc["order_id"])
    step = _append_step(
        state,
        "Assigning alternative driver to reduce idle time",
        "call_tool:assign_alternative_driver",
        assignment,
    )
    return {
        "messages": state["messages"] + [AIMessage(content=f"Driver reassigned: {assignment}")],
        "incident": state["incident"],
        "steps": state["steps"] + [step],
        "result": json.dumps(assignment),
        "refund_amount": state.get("refund_amount"),
    }

def traffic_check_node(state: State) -> State:
    inc = state["incident"]
    traffic = get_traffic_status(inc["order_id"])
    step = _append_step(
        state,
        "Checking traffic status to inform ETA and routing decisions",
        "call_tool:get_traffic_status",
        traffic,
    )
    return {
        "messages": state["messages"] + [AIMessage(content=f"Traffic status: {traffic}")],
        "incident": state["incident"],
        "steps": state["steps"] + [step],
        "result": json.dumps(traffic),
        "refund_amount": state.get("refund_amount"),
    }

def alternatives_node(state: State) -> State:
    inc = state["incident"]
    alternatives = find_alternatives(inc["merchant_id"], inc)
    step = _append_step(
        state,
        "Finding alternative merchants/items based on incident context",
        "call_tool:find_alternatives",
        alternatives,
    )
    return {
        "messages": state["messages"] + [AIMessage(content=f"Alternatives: {alternatives}")],
        "incident": state["incident"],
        "steps": state["steps"] + [step],
        "result": json.dumps(alternatives),
        "refund_amount": state.get("refund_amount"),
    }

def collect_evidence_node(state: State) -> State:
    inc = state["incident"]
    evidence = collect_evidence(inc["order_id"])
    amount = compute_refund_amount(inc, evidence)
    step = _append_step(
        state,
        "Collecting evidence and computing tentative refund amount",
        "call_tool:collect_evidence + compute_amount",
        {"evidence": "omitted_for_brevity", "tentative_amount": amount},
    )
    return {
        "messages": state["messages"] + [AIMessage(content=f"Evidence collected; tentative refund: {amount}")],
        "incident": state["incident"],
        "steps": state["steps"] + [step],
        "result": json.dumps(evidence),
        "refund_amount": amount,
    }

def refund_node(state: State) -> State:
    inc = state["incident"]
    evidence = {}
    try:
        evidence = json.loads(state.get("result") or "{}")
    except Exception:
        pass
    amount = state.get("refund_amount")
    if amount is None:
        amount = compute_refund_amount(inc, evidence)
    refund = issue_refund(inc["order_id"], amount)
    step = _append_step(
        state,
        "Issuing refund or partial credit per policy thresholds",
        "call_tool:issue_refund",
        {"amount": amount, "response": refund},
    )
    return {
        "messages": state["messages"] + [AIMessage(content=f"Refund issued: {refund} (amount={amount})")],
        "incident": state["incident"],
        "steps": state["steps"] + [step],
        "result": json.dumps(refund),
        "refund_amount": amount,
    }

def support_ticket_node(state: State) -> State:
    inc = state["incident"]
    ticket = create_support_ticket(inc)
    step = _append_step(
        state,
        "Creating support ticket for follow-up or escalation",
        "call_tool:create_support_ticket",
        ticket,
    )
    return {
        "messages": state["messages"] + [AIMessage(content=f"Support ticket: {ticket}")],
        "incident": state["incident"],
        "steps": state["steps"] + [step],
        "result": json.dumps(ticket),
        "refund_amount": state.get("refund_amount"),
    }

def final_decision_node(state: State) -> State:
    inc = state["incident"]
    summary_prompt = f"""
    You are Synapse-OPS.
    Summarize the resolution of this incident in one sentence for the ops dashboard.
    Incident: {state["incident"]}
    Steps Taken: {json.dumps(state["steps"], indent=2)}
    """
    degraded = False
    try:
        output = _invoke_with_retry(
            llm, [HumanMessage(content=summary_prompt)], max_retries=5, max_tokens=96
        )
        summary = output.content
        action = "call_llm:gemini-1.5-flash"
    except Exception:
        degraded = True
        summary = _local_summary(state["incident"], state["steps"])
        action = "llm_summary_fallback"

    step = _append_step(
        state,
        "Produce final summary; degraded mode used" if degraded else "Produce final summary",
        action,
        {"degraded": degraded},
    )
    return {
        "messages": state["messages"] + [AIMessage(content=summary)],
        "incident": state["incident"],
        "steps": state["steps"] + [step],
        "result": summary,
        "refund_amount": state.get("refund_amount"),
    }

# 6) Routing logic
def route_after_check(state: State) -> Literal[
    "assign_driver_node",
    "alternatives_node",
    "collect_evidence_node",
    "final_decision_node",
]:
    inc = state["incident"]
    itype = inc.get("type", "")
    try:
        status = json.loads(state.get("result") or "{}")
    except Exception:
        status = {}
    prep_delay = int(status.get("prep_delay_min", 0)) if isinstance(status, dict) else 0

    if itype in {"merchant_backlog", "courier_delay", "driver_unassigned"} or (
        prep_delay >= POLICIES.get("max_prep_delay_before_reassign", 30)
    ):
        return "assign_driver_node"
    elif itype in {"item_out_of_stock", "missing_item", "damaged_item"}:
        return "collect_evidence_node"
    elif itype in {"merchant_closed", "merchant_unreachable"}:
        return "alternatives_node"
    else:
        return "final_decision_node"

def route_after_assign_driver(state: State) -> Literal["traffic_check_node", "final_decision_node"]:
    check = POLICIES.get("check_traffic_after_reassign", True)
    return "traffic_check_node" if check else "final_decision_node"

def route_after_evidence(state: State) -> Literal["refund_node", "support_ticket_node"]:
    amount = state.get("refund_amount") or 0.0
    auto = POLICIES.get("auto_refund_for_missing_item", True)
    min_amt = float(POLICIES.get("auto_refund_min_amount", 0.0))
    max_amt = float(POLICIES.get("auto_refund_max_amount", 1e9))
    if auto and (min_amt <= float(amount) <= max_amt):
        return "refund_node"
    else:
        return "support_ticket_node"

def route_after_alternatives(state: State) -> Literal["support_ticket_node", "final_decision_node"]:
    try:
        alts = json.loads(state.get("result") or "[]")
    except Exception:
        alts = []
    auto = POLICIES.get("auto_rebook_to_alternative", False)
    return "final_decision_node" if (auto and alts) else "support_ticket_node"

# 7) Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("check_merchant", check_merchant)
graph_builder.add_node("assign_driver_node", assign_driver_node)
graph_builder.add_node("traffic_check_node", traffic_check_node)
graph_builder.add_node("alternatives_node", alternatives_node)
graph_builder.add_node("collect_evidence_node", collect_evidence_node)
graph_builder.add_node("refund_node", refund_node)
graph_builder.add_node("support_ticket_node", support_ticket_node)
graph_builder.add_node("final_decision_node", final_decision_node)

graph_builder.add_edge(START, "check_merchant")
graph_builder.add_conditional_edges(
    "check_merchant",
    route_after_check,
    {
        "assign_driver_node": "assign_driver_node",
        "alternatives_node": "alternatives_node",
        "collect_evidence_node": "collect_evidence_node",
        "final_decision_node": "final_decision_node",
    },
)
graph_builder.add_conditional_edges(
    "assign_driver_node",
    route_after_assign_driver,
    {
        "traffic_check_node": "traffic_check_node",
        "final_decision_node": "final_decision_node",
    },
)
graph_builder.add_edge("traffic_check_node", "final_decision_node")
graph_builder.add_conditional_edges(
    "collect_evidence_node",
    route_after_evidence,
    {
        "refund_node": "refund_node",
        "support_ticket_node": "support_ticket_node",
    },
)
graph_builder.add_edge("refund_node", "final_decision_node")
graph_builder.add_conditional_edges(
    "alternatives_node",
    route_after_alternatives,
    {
        "support_ticket_node": "support_ticket_node",
        "final_decision_node": "final_decision_node",
    },
)
graph_builder.add_edge("support_ticket_node", "final_decision_node")
graph_builder.add_edge("final_decision_node", END)

graph = graph_builder.compile()

# 8) Orchestrator
def resolve_incident(incident: dict) -> str:
    raw_desc = incident.get("description", incident.get("type", ""))
    category, confidence = classify_incident(raw_desc)
    if category == "unknown":
        save_decision_trace(incident, [], {"to_human": True}, POLICIES["policy_version"])
        return "Escalated to human"
    
    incident["type"] = category

    initial_state: State = {
        "messages": [HumanMessage(content=f"Resolve incident: {incident}")],
        "incident": incident,
        "steps": [],
        "result": "",
        "refund_amount": None,
    }
    
    final_state = graph.invoke(initial_state)

    save_decision_trace(
        incident,
        final_state["steps"],
        {"to_human": False},
        POLICIES["policy_version"],
    )
    return final_state["result"]

# 9) Example run
if __name__ == "__main__":
    example_incidents = [
        {
            "order_id": "ORD-001",
            "merchant_id": "M129",
            "type": "merchant_backlog",
            "description": "Merchant M129 reported 45 min prep delay.",
            "order_total": 850.0,
        },
        # You can uncomment these to test other scenarios
        # {
        #     "order_id": "ORD-003",
        #     "merchant_id": "M130",
        #     "type": "item_out_of_stock",
        #     "description": "Customer reports missing drink from order.",
        #     "order_total": 420.0,
        # },
        # {
        #     "order_id": "ORD-004",
        #     "merchant_id": "M131",
        #     "type": "merchant_closed",
        #     "description": "Restaurant unexpectedly closed before pickup.",
        #     "order_total": 1200.0,
        # },
    ]
    print("\n=== Running Synapse-OPS ===")
    for ei in example_incidents:
        # raw_desc = ei.get("description", ei.get("type", ""))
        # category, confidence = classify_incident(raw_desc)
        # print(category,confidence)
        result = resolve_incident(ei)

        print(f"\n=== Final Resolution ===\n{result}")