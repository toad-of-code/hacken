"""
tools.py - Fully Policy-Integrated and Deterministic API Layer for Synapse-OPS 🛠️
"""

from typing import Dict, Any, List
import json

with open("policies.json", "r") as f:
    POLICIES = json.load(f)

# --- REFINED HELPER ---
def compute_refund_amount(incident: dict) -> float:
    """Calculates the appropriate refund amount based on all relevant policies."""
    incident_type = incident.get("type", "")
    order_total = incident.get("order_total", 0.0)
    amount = 0.0

    # --- POLICY IMPLEMENTED ---
    if incident_type in {"item_out_of_stock", "missing_item"} and not POLICIES.get("auto_refund_for_missing_item", False):
        return 0.0

    # --- POLICY IMPLEMENTED ---
    if incident_type in {"item_out_of_stock", "missing_item", "packaging_issue"}:
        # Missing item can be refunded fully if policy allows
        if incident_type == "missing_item" and POLICIES.get("refund_full_amount_for_missing_item", False):
            item_price = incident.get("item_price")
            amount = float(item_price) if item_price is not None else order_total
        else:
            refund_pct = POLICIES.get("refund_percent_cap", 25)
            amount = order_total * (refund_pct / 100.0)

    # Fallback logic
    if amount == 0:
        if incident_type == "missing_item" and POLICIES.get("refund_full_amount_for_missing_item", False):
            amount = float(POLICIES.get("missing_item_unit_price_fallback", 0))
        elif incident_type in {"missing_item", "item_out_of_stock"} and POLICIES.get("missing_item_unit_price_fallback"):
            amount = float(POLICIES["missing_item_unit_price_fallback"])
        elif POLICIES.get("refund_flat_amount"):
            amount = float(POLICIES["refund_flat_amount"])

    # Minimum threshold check
    if amount < POLICIES.get("auto_refund_min_amount", 0):
        return 0.0

    return amount


# --- POLICY-DRIVEN TOOLS ---

def get_merchant_status(merchant_id: str) -> Dict[str, Any]:
    """Checks a merchant's current operational status, including prep delays."""
    cases = {
        "M129": {"prep_delay_min": 60, "accepts_orders": True},
        "M130": {"prep_delay_min": 20, "accepts_orders": True},
        "M999": {"prep_delay_min": 0, "accepts_orders": False},
    }
    return cases.get(merchant_id, {"prep_delay_min": 10, "accepts_orders": True})


def find_alternatives(incident: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Suggests alternative merchants, filtered by allowed list in policies."""
    allowed_merchants = POLICIES.get("reroute_allowed_merchants", [])
    if not allowed_merchants:
        return []
    return [{"merchant_id": m, "eta_diff_min": -10 + i * 2} for i, m in enumerate(allowed_merchants)]


def assign_alternative_driver(order_id: str, available_time_minutes: int) -> Dict[str, str]:
    """
    Reassigns the order to a new driver IF policy threshold is met.
    Returns a dict with driver info OR indicates that reassignment was skipped.
    """
    threshold = POLICIES.get("max_prep_delay_before_reassign", 30)
    if available_time_minutes < threshold:
        return {
            "status": "skipped",
            "reason": f"Available time ({available_time_minutes} min) is below reassignment threshold ({threshold} min)."
        }

    return {
        "status": "driver_reassigned",
        "new_driver_id": "DRV-888",
        "new_eta": "20m"
    }


def notify_customer(order_id: str, message: str) -> Dict[str, Any]:
    """Sends a proactive notification message to the customer regarding their order."""
    # print(f"--- NOTIFICATION to {order_id}: {message} ---")
    return {"ok": True, "notified": True, "message": message}


def calculate_and_issue_refund(incident: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates and issues a refund, respecting automatic refund policy caps."""
    order_id = incident.get("order_id", "N/A")
    amount = compute_refund_amount(incident)

    if amount == 0:
        return {"ok": False, "reason": "Incident not eligible for a refund per policy."}

    max_auto_refund = POLICIES.get("auto_refund_max_amount", 500)
    if amount > max_auto_refund:
        return {"ok": False, "reason": f"Calculated refund of {amount:.2f} exceeds auto-refund cap ({max_auto_refund}). Manual approval required."}

    return {"ok": True, "refund_id": f"R-{order_id}", "amount_refunded": round(amount, 2)}


def collect_incident_evidence(incident: Dict[str, Any]) -> Dict[str, Any]:
    """Collects evidence for an incident. Type depends on incident type."""
    order_id = incident.get("order_id", "N/A")
    incident_type = incident.get("type", "")
    description = incident.get("description", "No details provided.")

    if incident_type == "packaging_issue":
        return {"evidence_type": "photo", "photo_url": f"http://example.com/evidence_{order_id}.jpg", "verified": True}
    if incident_type in {"missing_item", "item_out_of_stock"}:
        return {"evidence_type": "chat_log", "log": f"CUSTOMER: Issue reported - {description}", "verified": True}

    return {"evidence_type": "none", "error": "No evidence available for this incident type."}


def create_support_ticket(incident: Dict[str, Any]) -> Dict[str, Any]:
    """Escalates an issue to a human agent by creating a support ticket."""
    return {
        "ticket_id": f"T-{incident.get('order_id', 'N/A')}",
        "status": "OPEN",
        "issue_type": incident.get("type", "generic"),
        "summary": incident.get("description", "No description provided.")
    }


def find_temporary_task_for_idle_driver(incident: Dict[str, Any], available_time_minutes: int) -> Dict[str, Any]:
    """Finds a temporary task for a driver waiting for a delayed order."""
    threshold = POLICIES.get("min_prep_delay_for_temp_task_min", 20)

    if available_time_minutes > threshold:
        temp_task_id = f"TEMP-{incident.get('order_id', 'N/A')[-3:]}"
        return {
            "status": "temporary_task_assigned",
            "temporary_task_id": temp_task_id,
            "instructions": f"Driver should complete short delivery {temp_task_id} and then return to the merchant."
        }

    return {
        "status": "no_suitable_task_found",
        "reason": f"Available time ({available_time_minutes} min) is below threshold ({threshold} min)."
    }


def get_traffic_status(order_id: str) -> Dict[str, Any]:
    """Checks for traffic congestion along the delivery route."""
    try:
        order_num = int(order_id.split('-')[-1])
        if order_num % 2 == 0:
            return {"jam": True, "extra_delay_min": 15}
    except (ValueError, IndexError):
        pass
    return {"jam": False, "extra_delay_min": 0}
