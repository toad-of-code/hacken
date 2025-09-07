#!/usr/bin/env python3
# run_demo.py - Batch runner that uses reasoner.resolve_incident
# Usage:
#   python run_demo.py            # uses scenarios.json
#   python run_demo.py -s path    # use a custom scenarios file

import json
import time
import os
import glob
import argparse
import traceback

from reasoner import resolve_incident  # uses router, graph, and logger internally

def load_scenarios(path):
    with open(path, "r") as fh:
        return json.load(fh)

def find_latest_log_for_order(order_id):
    # looks for logs/decision_<order_id>_*.json and returns the newest
    matches = sorted(glob.glob(os.path.join("logs", f"decision_{order_id}_*.json")))
    return matches[-1] if matches else None

def run_batch(scenarios_path):
    scenarios = load_scenarios(scenarios_path)
    print(f"\nLoaded {len(scenarios)} scenarios from {scenarios_path}\n")

    for s in scenarios:
        order_id = s.get("order_id", "UNKNOWN")
        desc = s.get("description", s.get("type", ""))
        print("─" * 72)
        print(f"Running scenario: {order_id} — {desc}")
        start = time.time()
        try:
            result = resolve_incident(s)
            elapsed = time.time() - start
            # result may be a string or structured output depending on your reasoner
            print(f"\n[RESULT] Order: {order_id}  |  Time: {elapsed:.2f}s")
            print(f"{result}\n")
            log_file = find_latest_log_for_order(order_id)
            if log_file:
                print(f"[LOG] Decision trace saved: {log_file}")
            else:
                print("[LOG] No decision trace file found for this order (check logger).")
        except Exception as exc:
            print(f"[ERROR] Exception while processing {order_id}: {exc}")
            traceback.print_exc()

    print("\nBatch run completed. Inspect the logs/ folder for saved decision traces.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch Synapse-OPS demo scenarios via reasoner.")
    parser.add_argument("-s", "--scenarios", default="scenarios.json", help="Path to scenarios JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.scenarios):
        print(f"[FATAL] Scenarios file not found: {args.scenarios}")
        raise SystemExit(1)

    # ensure logs directory exists (reasoner/logger will also create it, but this is safe)
    os.makedirs("logs", exist_ok=True)

    run_batch(args.scenarios)
