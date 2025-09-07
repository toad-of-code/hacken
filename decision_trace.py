"""
decision_trace_schema.py - Defines the schema for Synapse-OPS decision trace.
Ensures all agent outputs are structured and auditable.
"""

from typing import List, Dict, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime


# - REMOVED: The 'Observation' class was defined but not used.
# class Observation(BaseModel):
#     """Observation returned by a tool call."""
#     result: Dict


class ThoughtStep(BaseModel):
    """One step in the chain of thought."""
    step: int = Field(..., description="Sequential step number")
    thought: str = Field(..., description="The reasoning for this step")
    action: str = Field(..., description="The tool or decision taken")
    observation: Union[Dict[str, Any], str] = Field(..., description="The result from the tool or decision")


# + ADDED: A specific model for the escalation status for better type safety.
class EscalationInfo(BaseModel):
    """Details about the escalation decision."""
    to_human: bool = Field(..., description="True if the incident was escalated to a human agent")


class DecisionTrace(BaseModel):
    """Full trace of the agent's reasoning for one incident."""
    order_id: str
    incident_type: str
    chain_of_thought: List[ThoughtStep]
    escalation: EscalationInfo
    # --- ADDED --- New field to store the final summary.
    final_summary: str = Field(..., description="The final, human-readable summary from the agent.")
    policy_version: str = Field(..., description="Version of policy file used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

