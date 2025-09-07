"""
tools.py - Deterministic Simulated API Layer for Synapse-OPS
Each tool mimics a real-world logistics API but returns predictable outputs for demo stability.
"""

def get_merchant_status(merchant_id: str):
    """Check merchant's order acceptance and preparation delay."""
    cases = {
        "M129": {"prep_delay_min": 45, "accepts_orders": False},  # Backlog case
        "M130": {"prep_delay_min": 10, "accepts_orders": True},   # Normal
        "M999": {"prep_delay_min": 0, "accepts_orders": False},   # Closed
    }
    return cases.get(merchant_id, {"prep_delay_min": 15, "accepts_orders": True})


def find_alternatives(merchant_id: str):
    """Suggest alternative merchants with faster prep time."""
    return [
        {"merchant_id": f"{merchant_id}_ALT1", "eta_diff_min": -5},
        {"merchant_id": f"{merchant_id}_ALT2", "eta_diff_min": -3}
    ]


def assign_alternative_driver(order_id: str):
    """Reassign a driver to prevent idle time."""
    return {"new_driver_id": "D42", "new_eta": "12m"}


def notify_customer(order_id: str, message: str = "Your order is delayed. We've reassigned a driver."):
    """Notify customer proactively."""
    return {"ok": True, "notified": True, "message": message}


def issue_refund(order_id: str, amount: float):
    """Issue a partial or full refund."""
    if amount > 50:  # Example policy cap
        return {"ok": False, "reason": "Refund exceeds policy cap"}
    return {"ok": True, "refund_id": f"R-{order_id}", "amount": amount}


def collect_evidence(order_id: str):
    """Collect photo or text evidence of the issue."""
    return {"photo_url": "http://example.com/evidence.jpg", "verified": True}


def create_support_ticket(order_id: str, issue_type: str, summary: str):
    """Escalate to ops team for manual handling."""
    return {"ticket_id": f"T-{order_id}", "status": "OPEN", "issue_type": issue_type, "summary": summary}


def get_traffic_status(route_id: str):
    """Check for traffic congestion."""
    if route_id.endswith("X"):
        return {"jam": True, "extra_delay_min": 15}
    return {"jam": False, "extra_delay_min": 0}
