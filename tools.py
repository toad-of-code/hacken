"""
tools.py - Deterministic API Layer for Synapse-OPS 🛠️

This version's tools are fully deterministic, returning fixed values to ensure
predictable and repeatable behavior for testing the agent's reasoning logic.
"""

from typing import Dict, Any, List
import json

# --- DETERMINISTIC CHANGE --- Removed 'random' and 'datetime' imports as they are no longer needed.

with open("policies.json", "r") as f:
    POLICIES = json.load(f)

# --- REFINED HELPER ---
def _compute_refund_amount(incident: dict) -> float:
    """Calculates the appropriate refund amount based on the incident and policies."""
    incident_type = incident.get("type", "")
    order_total = incident.get("order_total", 0.0)
    
    if incident_type in {"item_out_of_stock", "missing_item"} and not POLICIES.get("auto_refund_for_missing_item", False):
        return 0.0

    if incident_type in {"item_out_of_stock", "missing_item", "packaging_issue"}:
        refund_pct = POLICIES.get("refund_percent_cap", 25)
        amount = order_total * (refund_pct / 100.0)
        return amount
    
    if POLICIES.get("refund_flat_amount"):
        return float(POLICIES["refund_flat_amount"])

    return 0.0


# --- DETERMINISTIC & POLICY-DRIVEN TOOLS ---

def get_merchant_status(merchant_id: str) -> Dict[str, Any]:
    """Checks a merchant's current operational status, including prep delays."""
    # --- DETERMINISTIC CHANGE --- Removed time-based logic for peak hours. Delays are now constant.
    cases = {
        "M129": {"prep_delay_min": 45, "accepts_orders": True},
        "M130": {"prep_delay_min": 15, "accepts_orders": True},
        "M999": {"prep_delay_min": 0, "accepts_orders": False},
    }
    return cases.get(merchant_id, {"prep_delay_min": 10, "accepts_orders": True})


def find_alternatives(incident: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Suggests alternative merchants, filtered by an allowed list in the policies."""
    allowed_merchants = POLICIES.get("reroute_allowed_merchants", [])
    
    if not allowed_merchants:
        return []

    # --- DETERMINISTIC CHANGE --- Replaced random ETA difference with fixed values.
    # We use a simple loop to make it look dynamic but the output is always the same.
    alternatives = []
    for i, merchant in enumerate(allowed_merchants):
        eta_diff = -10 + i * 2 # e.g., -10, -8, -6...
        alternatives.append({"merchant_id": merchant, "eta_diff_min": eta_diff})
    return alternatives


def assign_alternative_driver(order_id: str) -> Dict[str, str]:
    """Reassigns the order to a new driver."""
    # --- DETERMINISTIC CHANGE --- Replaced random values with fixed ones.
    return {"new_driver_id": "DRV-888", "new_eta": "20m"}


def notify_customer(order_id: str, message: str) -> Dict[str, Any]:
    """Sends a proactive notification message to the customer regarding their order."""
    print(f"--- NOTIFICATION to {order_id}: {message} ---")
    return {"ok": True, "notified": True, "message": message}


def calculate_and_issue_refund(incident: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates and issues a refund, respecting the automatic refund policy caps."""
    order_id = incident.get("order_id", "N/A")
    amount = _compute_refund_amount(incident)

    if amount == 0:
        return {"ok": False, "reason": "Incident not eligible for a refund per policy."}

    max_auto_refund = POLICIES.get("auto_refund_max_amount", 500)
    if amount > max_auto_refund:
        return {"ok": False, "reason": f"Calculated refund of {amount:.2f} requires manual approval (cap is {max_auto_refund})."}
    
    return {"ok": True, "refund_id": f"R-{order_id}", "amount_refunded": round(amount, 2)}


def collect_incident_evidence(incident: Dict[str, Any]) -> Dict[str, Any]:
    """Collects evidence for an incident. The type of evidence depends on the incident type."""
    order_id = incident.get("order_id", "N/A")
    incident_type = incident.get("type", "")
    if incident_type == "packaging_issue":
        return {"evidence_type": "photo", "photo_url": f"http://example.com/evidence_{order_id}.jpg", "verified": True}
    elif incident_type in {"missing_item", "item_out_of_stock"}:
        return {"evidence_type": "chat_log", "log": "CUSTOMER: My drink was missing.", "verified": True}
    return {"evidence_type": "none", "error": "No evidence available for this incident type."}


def create_support_ticket(incident: Dict[str, Any]) -> Dict[str, Any]:
    """Escalates an issue to a human agent by creating a support ticket."""
    order_id = incident.get("order_id", "N/A")
    issue_type = incident.get("type", "generic")
    summary = incident.get("description", "No description provided.")
    return {"ticket_id": f"T-{order_id}", "status": "OPEN", "issue_type": issue_type, "summary": summary}


def get_traffic_status(order_id: str) -> Dict[str, Any]:
    """Checks for traffic congestion along the delivery route."""
    try:
        order_num = int(order_id.split('-')[-1])
        if order_num % 2 == 0:
            # --- DETERMINISTIC CHANGE --- Replaced random delay with a fixed value.
            return {"jam": True, "extra_delay_min": 15}
    except (ValueError, IndexError):
        pass
    return {"jam": False, "extra_delay_min": 0}