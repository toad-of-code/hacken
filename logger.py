"""
logger.py - Centralized logging for Synapse-OPS
Validates decision traces using Pydantic schema and saves them to /logs.
"""

import os
import json
from datetime import datetime
from decision_trace import DecisionTrace, ThoughtStep

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def save_decision_trace(incident, steps, escalation, policy_version):
    """
    Save decision trace as validated JSON.
    
    incident: dict with keys order_id, type
    steps: list of dicts with keys step, thought, action, observation
    escalation: dict like {"to_human": False}
    policy_version: str (e.g. "1.0.0")
    """
    try:
        trace = DecisionTrace(
            order_id=incident["order_id"],
            incident_type=incident["type"],
            chain_of_thought=[ThoughtStep(**s) for s in steps],
            escalation=escalation,
            policy_version=policy_version
        )

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(LOG_DIR, f"decision_{incident['order_id']}_{ts}.json")
        with open(filename, "w") as f:
            f.write(trace.model_dump_json(indent=2))


        print(f"[INFO] Decision trace saved successfully: {filename}")
        return filename

    except Exception as e:
        print(f"[ERROR] Failed to save decision trace: {e}")
        # Optionally escalate to human
        fallback_file = os.path.join(LOG_DIR, f"FAILED_{incident['order_id']}_{ts}.json")
        with open(fallback_file, "w") as f:
            json.dump({"incident": incident, "error": str(e)}, f, indent=2)
        print(f"[WARN] Fallback log saved: {fallback_file}")
        return None
