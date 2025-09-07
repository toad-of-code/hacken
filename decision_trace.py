"""
decision_trace_schema.py - Defines the schema for Synapse-OPS decision trace.
Ensures all agent outputs are structured and auditable.
"""

from typing import List, Dict, Union
from pydantic import BaseModel, Field
from datetime import datetime


class Observation(BaseModel):
    """Observation returned by a tool call."""
    result: Dict


class ThoughtStep(BaseModel):
    """One step in the chain of thought."""
    step: int = Field(..., description="Sequential step number")
    thought: str = Field(..., description="LLM's reasoning for this step")
    action: str = Field(..., description="Tool or decision taken")
    observation: Union[Dict, str] = Field(..., description="Result from tool or observation")


class DecisionTrace(BaseModel):
    """Full trace of the agent's reasoning for one incident."""
    order_id: str
    incident_type: str
    chain_of_thought: List[ThoughtStep]
    escalation: Dict
    policy_version: str = Field(..., description="Version of policy file used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


if __name__ == "__main__":
    # Example usage
    trace = DecisionTrace(
        order_id="ORD-001",
        incident_type="merchant_backlog",
        chain_of_thought=[
            ThoughtStep(step=1, thought="Check merchant status", action="call_tool:get_merchant_status",
                        observation={"prep_delay_min": 45}),
            ThoughtStep(step=2, thought="Notify customer", action="call_tool:notify_customer",
                        observation={"ok": True}),
        ],
        escalation={"to_human": False},
        policy_version="1.0.0"
    )

    print(trace.json(indent=2))
