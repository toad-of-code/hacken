import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import subprocess
import glob
from ttkthemes import ThemedTk
import sys
import threading
import queue


class IncidentResolverApp(ThemedTk):
    def __init__(self):
        super().__init__(theme="equilux")
        self.title("Synapse-OPS Agent Interface")
        self.geometry("800x650")

        style = ttk.Style(self)
        style.configure("TLabel", foreground="white")
        style.configure("TButton", foreground="white")
        style.configure("TLabelframe.Label", foreground="white")

        # Queue to communicate between subprocess thread and GUI
        self.log_queue = queue.Queue()
        self.create_widgets()
        self.after(100, self.process_queue)  # Start checking the queue

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(main_frame, text="Create Scenario", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Order ID:").grid(row=0, column=0, sticky="w", pady=2, padx=5)
        self.order_id_entry = ttk.Entry(input_frame)
        self.order_id_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(input_frame, text="Merchant ID:").grid(row=1, column=0, sticky="w", pady=2, padx=5)
        self.merchant_id_entry = ttk.Entry(input_frame)
        self.merchant_id_entry.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(input_frame, text="Type:").grid(row=2, column=0, sticky="w", pady=2, padx=5)
        self.type_entry = ttk.Entry(input_frame)
        self.type_entry.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(input_frame, text="Description:").grid(row=3, column=0, sticky="w", pady=2, padx=5)
        self.description_entry = ttk.Entry(input_frame)
        self.description_entry.grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(input_frame, text="Order Total:").grid(row=4, column=0, sticky="w", pady=2, padx=5)
        self.amount_entry = ttk.Entry(input_frame)
        self.amount_entry.grid(row=4, column=1, sticky="ew", pady=2)

        self.submit_button = ttk.Button(
            input_frame, text="Resolve Incident", command=self.submit_scenario, style="Accent.TButton"
        )
        self.submit_button.grid(row=5, column=0, columnspan=2, pady=10)

        # --- Live Agent Output Frame ---
        summary_frame = ttk.LabelFrame(main_frame, text="Live Agent Output", padding="10")
        summary_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        self.summary_text = tk.Text(
            summary_frame,
            wrap=tk.WORD,
            height=10,
            font=("Segoe UI", 9),
            bg="#464646",
            fg="white",
            insertbackground="white",
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # --- Decision Trace Frame ---
        trace_frame = ttk.LabelFrame(main_frame, text="Decision Trace (from logs)", padding="10")
        trace_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.grid_rowconfigure(2, weight=1)

        self.trace_text = tk.Text(
            trace_frame,
            wrap=tk.WORD,
            height=10,
            font=("Consolas", 9),
            bg="#464646",
            fg="white",
            insertbackground="white",
        )
        self.trace_text.pack(fill=tk.BOTH, expand=True)

    def submit_scenario(self):
        order_id = self.order_id_entry.get()
        if not order_id:
            messagebox.showerror("Input Error", "Order ID is required.")
            return

        # ✅ Safely cast order_total to float, default 0.0
        try:
            order_total = float(self.amount_entry.get()) if self.amount_entry.get() else 0.0
        except ValueError:
            messagebox.showerror("Input Error", "Order Total must be a valid number.")
            return

        # Build scenario dictionary
        scenario = {
            "order_id": order_id,
            "merchant_id": self.merchant_id_entry.get(),
            "description": self.description_entry.get(),
            "order_total": order_total,
        }

        # ✅ Only include 'type' if provided
        type_value = self.type_entry.get()
        if type_value:
            scenario["type"] = type_value

        # Convert to JSON string
        self.scenario_json = json.dumps([scenario])

        # Clear outputs & disable button
        self.summary_text.delete("1.0", tk.END)
        self.trace_text.delete("1.0", tk.END)
        self.submit_button.config(state="disabled")

        threading.Thread(target=self.run_reasoner_thread, daemon=True).start()

    def run_reasoner_thread(self):
        try:
            incidents = json.loads(self.scenario_json)
        except json.JSONDecodeError as e:
            self.summary_text.insert(tk.END, f"Invalid JSON: {e}")
            self.submit_button.config(state="normal")
            return

    results = []

    for ei in incidents:
        self.trace_text.insert(tk.END, f"\n--- Resolving Incident: {ei['order_id']} ---\n")
        self.trace_text.see(tk.END)

        try:
            # ✅ Safely run the async coroutine in a fresh event loop
            result = asyncio.run(resolve_incident(ei))
        except RuntimeError:
            # Fallback: if asyncio.run() complains about a running loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(resolve_incident(ei))
            loop.close()

        results.append(result)
        self.trace_text.insert(tk.END, result + "\n")
        self.trace_text.see(tk.END)

    # Show combined summary in summary_text box
    self.summary_text.insert(tk.END, "\n".join(results))
    self.summary_text.see(tk.END)

    # Re-enable button after execution
    self.submit_button.config(state="normal")


    def process_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message is None:
                    self.submit_button.config(state="normal")
                    self.display_trace()
                    return
                else:
                    self.summary_text.insert(tk.END, message)
                    self.summary_text.see(tk.END)
        except queue.Empty:
            pass

        self.after(100, self.process_queue)

    def display_trace(self):
        order_id = self.order_id_entry.get()
        if not order_id:
            return

        log_files = glob.glob(f"logs/decision_{order_id}_*.json")
        if not log_files:
            self.trace_text.insert(tk.END, "No trace file found for this order ID yet.")
            return

        latest_log = max(log_files, key=os.path.getctime)
        with open(latest_log, "r") as f:
            trace_data = json.load(f)
            self.trace_text.delete("1.0", tk.END)
            self.trace_text.insert(tk.END, json.dumps(trace_data, indent=2))


if __name__ == "__main__":
    app = IncidentResolverApp()
    app.mainloop()
