"""
reasoner.py - An agentic orchestrator for the Synapse-OPS system using LangGraph's iterative planning.
This version is updated to use the full suite of policy-driven tools and includes a pre-execution self-check.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import time

from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

# --- 1. SETUP AND IMPORTS ---
from tools import (
    get_merchant_status,
    find_alternatives,
    assign_alternative_driver,
    notify_customer,
    calculate_and_issue_refund,
    collect_incident_evidence,
    create_support_ticket,
    get_traffic_status,
    find_temporary_task_for_idle_driver,
    compute_refund_amount
)
from router import classify_incident
from logger import save_decision_trace

import nest_asyncio
nest_asyncio.apply()
load_dotenv()

with open("policies.json", "r") as f:
    POLICIES = json.load(f)


# --- 2. DEFINE TOOLS AND LLM ---
tools = [
    get_merchant_status,
    find_alternatives,
    assign_alternative_driver,
    notify_customer,
    calculate_and_issue_refund,
    collect_incident_evidence,
    create_support_ticket,
    get_traffic_status,
    find_temporary_task_for_idle_driver,
    compute_refund_amount
]

llm = ChatMistralAI(
    model="mistral-medium",
    api_key=os.getenv("MISTRAL_API_KEY3"),
    temperature=0
)

# bind_tools (if available) lets LLM call tools via LangGraph prebuilt ToolNode style
try:
    llm_with_tools = llm.bind_tools(tools)
except Exception:
    # fallback to llm only; ToolNode will still use the functions directly
    llm_with_tools = llm


# --- 3. DEFINE AGENT STATE ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    incident: dict


# --- 4. DEFINE THE AGENT NODES AND EDGES ---
async def call_model(state: AgentState):
    messages = state["messages"]
    # Use llm_with_tools if bound, else llm
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    else:
        return END


async def llm_classify_incident(description: str) -> Optional[str]:
    """
    Asks the LLM to classify an incident description into a predefined category.

    This function is used as a fallback when the primary ML model is not confident.
    It validates the LLM's output to ensure it's one of the allowed categories.
    """
    valid_categories = {
        "merchant_backlog",
        "merchant_closed",
        "driver_issue",
        "item_out_of_stock",
        "missing_item",
        "packaging_issue",
        "payment_issue",
    }

    prompt = f"""
    You are an expert incident classifier for a logistics company. Your task is to classify the following incident description into one of the predefined categories.

    Here are the only valid categories:
    {', '.join(sorted(list(valid_categories)))}

    CRITICAL: Respond with ONLY the single, most appropriate category name and nothing else. Do not add explanations or punctuation.

    Incident Description: "{description}"
    Category:
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        predicted_category = response.content.strip().lower().replace("_", " ").replace(" ", "_")
        if predicted_category in valid_categories:
            return predicted_category
        else:
            print(f"[LLM-CLASSIFY-WARN] LLM returned an invalid category: '{predicted_category}'")
            return None
    except Exception as e:
        print(f"[LLM-CLASSIFY-ERROR] The API call failed: {e}")
        return None


# --- 5. BUILD THE AGENTIC GRAPH ---
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("action", tool_node)
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "action": "action",
        END: END,
    },
)
graph_builder.add_edge("action", "agent")
agent_graph = graph_builder.compile()


# --- UPDATED --- Helper function to create a proper audit trail from the conversation.
def _parse_steps_from_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Parses the message history to create a step-by-step trace for logging, 
    enforcing escalation lock and consolidating notifications."""
    steps = []
    step_counter = 1
    escalated = False
    notification_buffer = []

    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                observation = "No result found."
                for next_msg in messages[i+1:]:
                    if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id == tool_call["id"]:
                        # --- SAFE JSON PARSING ---
                        raw_content = next_msg.content
                        if isinstance(raw_content, (dict, list)):
                            observation = raw_content  # Already structured
                        else:
                            try:
                                observation = json.loads(raw_content)
                            except (json.JSONDecodeError, TypeError):
                                observation = {"raw_content": raw_content}
                        break
                
                # Wrap list outputs
                if isinstance(observation, list):
                    observation = {"items": observation}
                elif not isinstance(observation, (dict, str)):
                    # If it's any other non-conforming type (like int, float, bool, None),
                    # wrap it in a dictionary to make it valid for the log.
                    observation = {"value": observation}

                # --- ESCALATION LOCK ---
                if tool_call['name'] == "create_support_ticket":
                    escalated = True
                    

                # --- SKIP FURTHER STEPS IF ESCALATED ---
                if escalated and tool_call['name'] != "create_support_ticket":
                    continue  # Ignore anything after escalation

                # --- BUFFER NOTIFICATIONS ---
                if tool_call['name'] == "notify_customer":
                    notification_buffer.append({
                        "step": step_counter,
                        "thought": msg.content,
                        "action": f"call_tool:{tool_call['name']}",
                        "observation": observation
                    })
                else:
                    steps.append({
                        "step": step_counter,
                        "thought": msg.content or f"Decided to call tool: {tool_call['name']}",
                        "action": f"call_tool:{tool_call['name']}",
                        "observation": observation,
                    })
                step_counter += 1

    # --- CONSOLIDATE NOTIFICATIONS AT END ---
    if notification_buffer:
        final_notification = notification_buffer[-1]  # Use the latest notification
        final_notification['step'] = step_counter
        final_notification['thought'] = "Consolidated final customer notification after all resolution actions."
        steps.append(final_notification)

    return steps



# --- 6. UTILITIES ---
from bs4 import BeautifulSoup
from markdown import markdown


def remove_markdown(md_text: str) -> str:
    """
    Converts markdown text to plain text by first converting it to HTML,
    then stripping all HTML tags.
    """
    html = markdown(md_text)
    soup = BeautifulSoup(html, "html.parser")
    plain_text = soup.get_text()
    return plain_text


# --- SELF-CHECK VALIDATOR FUNCTION ---
def validate_llm_plan_for_trace(planned_steps: List[Dict[str, Any]], incident: dict):
    """
    Validates the LLM's planned steps against critical policies and tool usage rules.
    Auto-corrects invalid plans when possible.

    Returns:
        corrected_steps (List[Dict]): A potentially corrected version of the plan.
        passed_check (bool): Whether the original plan passed all checks.
    """
    corrected: List[Dict[str, Any]] = []
    passed_check = True
    available_time = int(incident.get("available_time_min", 0) or 0)
    temp_threshold = int(POLICIES.get("min_prep_delay_for_temp_task_min", 20))
    reassign_threshold = int(POLICIES.get("max_prep_delay_before_reassign", 30))

    # Extract flags from the planned steps
    names = [s.get("action", "") for s in planned_steps]
    has_get_status = any("get_merchant_status" in n for n in names)
    has_collect_evidence = any("collect_incident_evidence" in n for n in names)
    has_refund = any("calculate_and_issue_refund" in n for n in names)
    has_assign_driver = any("assign_alternative_driver" in n for n in names)
    has_temp_task = any("find_temporary_task_for_idle_driver" in n for n in names)
    has_create_ticket = any("create_support_ticket" in n for n in names)

    order_total = float(incident.get("order_total", 0.0) or 0.0)
    max_auto_refund = float(POLICIES.get("auto_refund_max_amount", 500))
    refund_pct_cap = float(POLICIES.get("refund_percent_cap", 25)) / 100.0
    flat_fallback = float(POLICIES.get("refund_flat_amount", 0))

    # Helper: conservative refund estimate
    estimated_refund = min(order_total * refund_pct_cap,
                           flat_fallback if flat_fallback > 0 else order_total * refund_pct_cap)

    # ---------------------------
    # Critical Checks & Fixes
    # ---------------------------

    for step in planned_steps:
        action = step.get("action", "")

        # 1) create_support_ticket must be exclusive
        if "call_tool:create_support_ticket" in action and len(planned_steps) > 1:
            corrected = [{
                "step": 1,
                "thought": "Escalation forced because create_support_ticket must be exclusive.",
                "action": "call_tool:create_support_ticket",
                "observation": {}
            }]
            passed_check = False
            # print("[SELF-CHECK] create_support_ticket found with other steps — forcing escalation.")
            return corrected, False

    # 2) If refund planned but no evidence, insert evidence collection
    if has_refund and not has_collect_evidence:
        passed_check = False
        corrected.append({
            "step": len(corrected) + 1,
            "thought": "Auto-inserted evidence collection as per refund policy.",
            "action": "call_tool:collect_incident_evidence",
            "observation": {"auto_added": True}
        })
        # print("[SELF-CHECK] Inserted collect_incident_evidence before refund.")

    # 3) Enforce driver strategy hierarchy (temp task before reassignment)
    if has_assign_driver and not has_temp_task:
        passed_check = False
        corrected.insert(0, {
            "step": len(corrected) + 1,
            "thought": "Inserted temporary task assignment before driver reassignment per strategy hierarchy.",
            "action": "call_tool:find_temporary_task_for_idle_driver",
            "observation": {"auto_added": True}
        })
        # print("[SELF-CHECK] Inserted find_temporary_task_for_idle_driver before assign_alternative_driver.")

    # 4) Ensure get_merchant_status for backlog/driver issues
    if incident.get("type") in {"merchant_backlog", "driver_issue"} and not has_get_status:
        passed_check = False
        corrected.insert(0, {
            "step": len(corrected) + 1,
            "thought": "Auto-inserted merchant status check as per triage policy.",
            "action": "call_tool:get_merchant_status",
            "observation": {"auto_added": True}
        })
        # print("[SELF-CHECK] Inserted get_merchant_status for triage on backlog/driver_issue incident.")

    # 5) Refund cap enforcement
    if has_refund and estimated_refund > max_auto_refund:
        passed_check = False
        # print(f"[SELF-CHECK] Estimated refund ({estimated_refund}) exceeds max_auto_refund ({max_auto_refund}). Escalating.")
        return [{
            "step": 1,
            "thought": f"Refund estimate {estimated_refund} exceeds policy cap {max_auto_refund}. Escalation required.",
            "action": "call_tool:create_support_ticket",
            "observation": {}
        }], False

    # 6) Temporary task check against threshold
    if has_temp_task and available_time < temp_threshold:
        passed_check = False
        # print(f"[SELF-CHECK] Plan includes temp task but available_time ({available_time}) < temp threshold ({temp_threshold}). Escalating.")
        return [{
            "step": 1,
            "thought": f"Not enough available_time ({available_time}) for temporary task (threshold {temp_threshold}). Escalation required.",
            "action": "call_tool:create_support_ticket",
            "observation": {"available_time": available_time}
        }], False

    # 7) Driver reassignment check against threshold
    if has_assign_driver and available_time < reassign_threshold:
        passed_check = False
        # print(f"[SELF-CHECK] assign_alternative_driver planned but available_time ({available_time}) < reassign threshold ({reassign_threshold}). Escalating.")
        return [{
            "step": 1,
            "thought": f"Reassignment requested but available_time ({available_time}) insufficient (threshold {reassign_threshold}). Escalation required.",
            "action": "call_tool:create_support_ticket",
            "observation": {}
        }], False

    # ---------------------------
    # Append original steps (avoiding duplicates)
    # ---------------------------
    existing_actions = {s["action"] for s in corrected}
    for s in planned_steps:
        if s["action"] in existing_actions:
            continue
        corrected.append(s)

    return corrected, passed_check



# --- 7. REFACTORED ORCHESTRATOR ---
async def resolve_incident(incident: dict) -> str:
    raw_desc = incident.get("description", "")
    category, confidence = classify_incident(raw_desc)
    print(f"[CLASSIFIER] ML suggests '{category}' (conf={confidence:.2f})")

    llm_category_override = None
    if category.lower() == "unknown" or confidence < 0.4:
        print("[CLASSIFIER] Confidence is low. Querying LLM for a second opinion...")
        llm_category_override = await llm_classify_incident(raw_desc)
        # time.sleep(10)
        if llm_category_override:
            print(f"[LLM-FALLBACK] LLM classified the incident as '{llm_category_override}'. Overriding initial classification.")
            category = llm_category_override

    if category.lower() == "unknown":
        print("[STATUS] Escalating to human: Both classifiers failed to identify the incident.")
        save_decision_trace(incident, [], {"to_human": True}, POLICIES["policy_version"])
        return "Escalated to human"

    incident["type"] = category
    merchant_id = incident.get("merchant_id")
    merchant_status: Dict[str, Any] = {}
    try:
        merchant_status = get_merchant_status(merchant_id) if merchant_id else {}
    except Exception as e:
        print(f"[WARN] get_merchant_status failed: {e}")
        merchant_status = {}

    prep_delay = int(merchant_status.get("prep_delay_min", merchant_status.get("minimum_prep_delay", 0) or 0))
    driver_idle = int(incident.get("driver_idle_min", 0) or 0)
    travel_buffer = int(POLICIES.get("driver_travel_buffer_min", 0) or 0)

    available_time = max(0, prep_delay - driver_idle - travel_buffer)

    # attach values to incident (tools can read these)
    incident["merchant_status"] = merchant_status
    incident["prep_delay_min"] = prep_delay
    incident["driver_idle_min"] = driver_idle
    incident["available_time_min"] = available_time

    print(f"[INFO] merchant_status={merchant_status}, prep_delay={prep_delay}, driver_idle={driver_idle}, available_time={available_time}")

    # Build prompt that includes policies summary and self-check instruction
    initial_prompt = f"""
<role>
You are Synapse-OPS, an expert incident resolution agent. Your goal is to resolve the provided incident by thinking step-by-step and calling tools from the available tool list. You must strictly follow all rules and examples provided.
</role>

<policies>
These are the current operational policies that MUST be respected:
{json.dumps(POLICIES, indent=2)}
</policies>

<rules>
<critical_safety_rules>
    1.  Before you take any action, you MUST double-check if your planned action contradicts a specific negative constraint in the customer's request (e.g., "do not refund my credit card", "do not call me"). If it does, your only option is to escalate by creating a support ticket.
    2.  If you use `create_support_ticket`, that MUST be your final action. Do not call any other tools after escalating.
    3.  If you detect a direct contradiction between a tool's output and the incident description, escalate immediately by creating a support ticket.
    4. Refunds are also the final action. If you issue a refund, that must be your final action.
</critical_safety_rules>

<standard_operating_procedures>
    1.  **Triage:** For any issue involving a delay or an unavailable merchant, your first action MUST be `get_merchant_status`.
    2.  **Refunds:** Before issuing a refund for `missing_item` or `packaging_issue`, you MUST `collect_incident_evidence` first.
    3.  **Alternatives:** After using `find_alternatives`, you are authorized to rebook.
</standard_operating_procedures>

<strategy_hierarchy>
    1.  **Driver Delays:** To handle a long merchant delay, you have two strategies. You must follow this logic:
        - **Step A:** First, try `find_temporary_task_for_idle_driver`.
        - **Step B:** If and ONLY IF that tool returns `status: "no_suitable_task_found"`, then you are authorized to consider using `assign_alternative_driver`. DO NOT use both.
    2. Refund is final. If you issue a refund, that must be your final action.
</strategy_hierarchy>
</rules>

<examples>
<example_1 name="Standard Refund">
    Incident: {{
        "order_id": "ORD-115", "description": "Food arrived soggy, refund requested.", "type": "packaging_issue"
    }}
    Thought: The user has a packaging issue and wants a refund. As per my rules, I must collect evidence first.
    Action: `collect_incident_evidence`
    Observation: {{ "evidence_type": "photo", "verified": True }}
    Thought: Evidence is verified. Now I can issue a refund.
    Action: `calculate_and_issue_refund`
    Observation: {{ "ok": True, "amount_refunded": 225.0 }}
    Thought: The refund was successful. The incident is resolved.
    Final Answer: The packaging issue for ORD-115 has been resolved by issuing a refund of 225.0.
</example_1>

<example_2 name="Escalation on Invalid Request">
    Incident: {{
        "order_id": "ORD-604", "description": "My drink was missing, but please credit my account wallet, do not refund my credit card.", "type": "missing_item"
    }}
    Thought: The user has a missing item but has a specific negative constraint: "do not refund my credit card". My `calculate_and_issue_refund` tool does not allow specifying a refund destination. This contradicts my capabilities. According to my critical safety rules, I must escalate.
    Action: `create_support_ticket`
</example_2>
</examples>

<current_incident>
Incident details: {json.dumps(incident)}
</current_incident>

<self_check_instruction>
Before finalizing your plan, double-check that:
- You do NOT violate any of the above policies.
- You follow the tool order hierarchy: get_merchant_status first (for delays), evidence before refunds, find_temporary_task before assign_alternative_driver.
- If your plan violates any rule, FIX it before outputting your final sequence.
</self_check_instruction>
"""

    initial_state: AgentState = {
        "messages": [HumanMessage(content=initial_prompt)],
        "incident": incident,
    }

    print("Invoking agent graph to obtain plan...")
    # First invocation yields the LLM's plan and any tool_calls in conversation
    raw_state = await agent_graph.ainvoke(initial_state)

    # Parse planned steps from the agent messages BEFORE executing them (we only parse the plan)
    planned_steps = _parse_steps_from_messages(raw_state["messages"])

    # Validate and possibly correct the plan
    corrected_steps, passed_check = validate_llm_plan_for_trace(planned_steps, incident)

    if not passed_check:
        # print("[SELF-CHECK] Detected violations in LLM plan. Correcting plan before execution...")
        # Build a corrected plan message and re-run the agent with the corrected plan
        corrected_plan_message = HumanMessage(content=f"""
<self_correction>
Your previous plan violated policies. Here is a corrected plan that follows rules:
{json.dumps(corrected_steps, indent=2)}
Now execute only this corrected plan step-by-step.
</self_correction>
""")
        # Re-invoke the agentgraph with the corrected plan appended
        corrected_state = await agent_graph.ainvoke({
            "messages": raw_state["messages"] + [corrected_plan_message],
            "incident": incident,
        })
        final_state = corrected_state
        executed_steps = _parse_steps_from_messages(corrected_state["messages"])
        # print("[SELF-CHECK] Corrected plan executed.")
    else:
        final_state = raw_state
        executed_steps = planned_steps
        # print("[SELF-CHECK] Plan passed validation, executing as-is.")

    final_resolution = final_state["messages"][-1].content
    plain_text = remove_markdown(final_resolution)
    print("Agent graph invocation complete.")

    # Append a self-check step to executed_steps for logging
    executed_steps.append({
        "step": len(executed_steps) + 1,
        "thought": "Performed self-check on LLM plan pre-execution.",
        "action": "policy_self_check",
        "observation": {"passed": passed_check}
    })

    escalated_to_human = any(step.get("action") == "call_tool:create_support_ticket" for step in executed_steps)

    # Save decision trace (logger expected signature in your repo: (incident, steps, escalation, policy_version, summary) )
    save_decision_trace(
        incident,
        executed_steps,
        {"to_human": escalated_to_human},
        POLICIES["policy_version"],
        plain_text
    )

    return plain_text


# --- 8. EXAMPLE RUN ---
if __name__ == "__main__":
    async def main():
        import sys
        try:
            incidents_to_run = json.load(sys.stdin)
        except Exception as e:
            print(f"Failed to read incident data from stdin: {e}")
            return
        
        for ei in incidents_to_run:
            print(f"\n--- Resolving Incident: {ei['order_id']} ---")
            result = await resolve_incident(ei)
            # Print the final summary to stdout so the GUI can capture it.
            print(result) 

    # Run the main async function
    # import asyncio
    asyncio.run(main())
