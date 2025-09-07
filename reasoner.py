"""
reasoner.py - An agentic orchestrator for the Synapse-OPS system using LangGraph's iterative planning.
This version is updated to use the full suite of policy-driven tools.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

# --- 1. SETUP AND IMPORTS ---
# --- UPDATED --- Importing the new and renamed tools.
from tools import (
    get_merchant_status,
    find_alternatives,
    assign_alternative_driver,
    notify_customer,
    calculate_and_issue_refund,
    collect_incident_evidence,
    create_support_ticket,
    get_traffic_status,
)
from router import classify_incident
from logger import save_decision_trace

import nest_asyncio
nest_asyncio.apply()
load_dotenv()

with open("policies.json", "r") as f:
    POLICIES = json.load(f)


# --- 2. DEFINE TOOLS AND LLM ---
# --- UPDATED --- The tool list now reflects the refined tools from tools.py.
tools = [
    get_merchant_status,
    find_alternatives,
    assign_alternative_driver,
    notify_customer,
    calculate_and_issue_refund,
    collect_incident_evidence,
    create_support_ticket,
    get_traffic_status,
]

llm = ChatMistralAI(
    model="mistral-medium-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


# --- 3. DEFINE AGENT STATE ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    incident: dict


# --- 4. DEFINE THE AGENT NODES AND EDGES ---
async def call_model(state: AgentState):
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "action"
    else:
        return END


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
        END: END, # The key END now matches the value returned by the function
    },
)
graph_builder.add_edge("action", "agent")
agent_graph = graph_builder.compile()


# --- NEW --- Helper function to create a proper audit trail from the conversation.
def _parse_steps_from_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Parses the message history to create a step-by-step trace for logging."""
    steps = []
    step_counter = 1
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                observation = "No result found."
                for next_msg in messages[i+1:]:
                    if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id == tool_call["id"]:
                        observation = json.loads(next_msg.content)
                        break
                
                # --- ADD THIS LOGIC ---
                # If the tool output is a list, wrap it in a dictionary to conform to the schema.
                if isinstance(observation, list):
                    observation = {"items": observation}
                # --- END OF CHANGE ---
                
                steps.append({
                    "step": step_counter,
                    "thought": msg.content or f"Decided to call tool: {tool_call['name']}",
                    "action": f"call_tool:{tool_call['name']}",
                    "observation": observation,
                })
                step_counter += 1
    return steps

# --- 6. REFACTORED ORCHESTRATOR ---
# In reasoner.py
from bs4 import BeautifulSoup
from markdown import markdown

def remove_markdown(md_text: str) -> str:
    """
    Converts markdown text to plain text by first converting it to HTML,
    then stripping all HTML tags.
    """
    # 1. Convert Markdown to HTML
    html = markdown(md_text)

    # 2. Use BeautifulSoup to parse the HTML and extract the text
    soup = BeautifulSoup(html, "html.parser")
    plain_text = soup.get_text()

    return plain_text

async def resolve_incident(incident: dict) -> str:
    # ... (classification logic is unchanged) ...
    raw_desc = incident.get("description", "")
    category, confidence = classify_incident(raw_desc)
    print(f"[CLASSIFIER] ML suggests '{category}' (conf={confidence:.2f})")

    if category == "unknown":
        print("[STATUS] Escalating to human: Classifier could not identify the incident.")
        # --- MODIFIED --- Pass an appropriate summary for escalation logs.
        save_decision_trace(
            incident,
            [],
            {"to_human": True},
            POLICIES["policy_version"],
            "N/A - Escalated before summary generation"
        )
        return "Escalated to human"
    
    incident["type"] = category

    # In reasoner.py -> resolve_incident function

    initial_prompt = f"""
    You are Synapse-OPS, an expert incident resolution agent. Your goal is to resolve the following incident.
    You will reason about the problem, decide on an action (a tool to call), observe the result, and repeat until the incident is resolved.

    --- RULES ---
    1.  Always check the merchant status (`get_merchant_status`) first for any issue involving a delay or an unavailable merchant.
    2.  Before issuing a refund, you MUST `collect_incident_evidence` first, if applicable.
    3.  If a tool returns an error or fails (e.g., a refund is denied by policy), you MUST create a support ticket (`create_support_ticket`) for human review. Do not retry the failed action.
    4.  Only provide your final summary answer after all necessary actions are complete.

    --- EXAMPLE OF A GOOD RESOLUTION ---
    Incident: {{
        "order_id": "ORD-115",
        "description": "Food arrived soggy due to poor packaging, refund requested.",
        "initial_classification": "packaging_issue"
    }}
    Thought: The user is reporting a packaging issue and requesting a refund. According to my rules, I must collect evidence before issuing a refund.
    Action: `collect_incident_evidence` with incident details.
    Observation: {{ "evidence_type": "photo", "verified": True }}
    Thought: Evidence has been collected and verified. The incident is eligible for a refund. I should now calculate and issue the correct refund based on policy.
    Action: `calculate_and_issue_refund` with incident details.
    Observation: {{ "ok": True, "amount_refunded": 225.0 }}
    Thought: The refund was successful. The incident is now fully resolved. I can provide my final summary.
    Final Answer: The packaging issue for order ORD-115 has been resolved by issuing a refund of 225.0.

    --- CURRENT INCIDENT ---
    Incident details: {json.dumps(incident)}
    """
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content=initial_prompt)],
        "incident": incident,
    }
    
    print("Invoking agent graph...")
    final_state = await agent_graph.ainvoke(initial_state)
    final_resolution = final_state["messages"][-1].content
    plain_text=remove_markdown(final_resolution)
    print("Agent graph invocation complete.")
    
    steps_for_log = _parse_steps_from_messages(final_state["messages"])
    
    escalated_to_human = any(step.get("action") == "call_tool:create_support_ticket" for step in steps_for_log)

    # --- MODIFIED --- Pass the 'final_resolution' string to the logger.
    save_decision_trace(
        incident,
        steps_for_log,
        {"to_human": escalated_to_human},
        POLICIES["policy_version"],
        plain_text
    )
    
    return plain_text

    
# --- 7. EXAMPLE RUN ---
if __name__ == "__main__":
    async def main():
        example_incidents = [
            # {
            #     "order_id": "ORD-115", "merchant_id": "M514", "type": "packaging_issue",
            #     "description": "Food arrived soggy due to poor packaging, refund requested.",
            #     "order_total": 900.0,
            # },
            # {
            #     "order_id": "ORD-102", "merchant_id": "M129", "type": "merchant_backlog",
            #     "description": "Merchant reported 60 min prep delay, drivers waiting outside.",
            #     "order_total": 1200.0,
            # },
            # --- NEW --- Test case to check the refund policy cap.
            {
                "order_id": "ORD-301", "merchant_id": "M999", "type": "packaging_issue",
                "description": "Customer's entire main course was missing from a large order.",
                "order_total": 3000.0, # High value order
            },
        ]

        for ei in example_incidents:
            print(f"\n--- Resolving Incident: {ei['order_id']} ---")
            result = await resolve_incident(ei)
            print(f"--- Final Resolution for {ei['order_id']} ---\n{result}")
            
    asyncio.run(main())