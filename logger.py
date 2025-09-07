"""
logger.py - Centralized logging for Synapse-OPS
Validates decision traces using Pydantic schema and saves them to the /logs directory.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import the Pydantic models from your schema file
from decision_trace import DecisionTrace, ThoughtStep, EscalationInfo

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def save_decision_trace(
    incident: Dict[str, Any],
    steps: List[Dict[str, Any]],
    escalation: Dict[str, bool],
    policy_version: str,
    final_summary: str
) -> Optional[str]:
    """
    Validates the incident data against the Pydantic schema and saves it as a JSON trace file.

    Returns the filename on success, None on failure.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    order_id = incident.get("order_id", "UNKNOWN_ORDER")

    try:
        # Create Pydantic models from the raw dictionaries for validation
        chain_of_thought = [ThoughtStep(**s) for s in steps]
        escalation_model = EscalationInfo(**escalation)

        # Create the final, validated trace object
        trace = DecisionTrace(
            order_id=order_id,
            incident_type=incident.get("type", "unknown"),
            chain_of_thought=chain_of_thought,
            escalation=escalation_model,
            final_summary=final_summary, # Include the final summary
            policy_version=policy_version,
        )

        # Save the validated data to a JSON file
        filename = os.path.join(LOG_DIR, f"decision_{order_id}_{ts}.json")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(trace.model_dump_json(indent=2))

        print(f"✅ Decision trace saved successfully: {filename}")
        return filename
    
    except Exception as e:
        # If validation or saving fails, save a raw log for debugging
        print(f"❌ [ERROR] Failed to save decision trace for order {order_id}: {e}")
        
        fallback_file = os.path.join(LOG_DIR, f"FAILED_{order_id}_{ts}.json")
        try:
            with open(fallback_file, "w", encoding="utf-8") as f:
                fallback_data = {
                    "error": str(e),
                    "failed_incident_data": incident,
                    "failed_steps_data": steps,
                    "failed_escalation_data": escalation,
                    "failed_summary": final_summary,
                }
                json.dump(fallback_data, f, indent=2)
            print(f"⚠️  Fallback log for failed trace saved: {fallback_file}")
        except Exception as fallback_e:
            print(f"🚨 [CRITICAL] Could not even save a fallback log for order {order_id}: {fallback_e}")
            
        return None