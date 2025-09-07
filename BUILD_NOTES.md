
# 📝 Synapse-OPS Build Notes / Tracker

Use this as a living document while developing the prototype.

---

## 1. Project Scope
- ✅ Focus: **GrabFood & GrabMart only**
- ✅ Goal: **Policy-aware, tool-using agent** that handles as many real disruption cases as possible.
- ✅ Must show:
  - Natural language input  
  - Step-by-step reasoning (chain_of_thought)  
  - Tool calls + observations  
  - Policy compliance  
  - Structured JSON output  

---

## 2. Tech Stack
- **Language:** Python 3.x  
- **Framework:** LangChain (LLM agent orchestration)  
- **LLM:** OpenAI GPT-4o-mini (temperature = 0 for deterministic demo)  
- **Tools:** Deterministic simulated APIs (8–10 variety)  
- **Policy Engine:** JSON-based rules  
- **Validation:** Pydantic schema for decision_trace  
- **Optional:** Scikit-learn router for case classification  

---

## 3. Core Components to Build
| **Component** | **Status** | **Notes** |
|---------------|-----------|-----------|
| `tools.py` | ⬜ | Implement 8–10 deterministic tool stubs |
| `policies.json` | ⬜ | Editable, with max_voucher, escalate thresholds, etc. |
| `reasoner.py` | ⬜ | Main orchestrator (LLM + tools + loop) |
| `decision_trace schema` | ⬜ | Pydantic validation |
| `router.py` | ⬜ | (Optional) ML classifier for case routing |
| `logger.py` | ⬜ | Save trace as JSON after each run |
| `scenarios/` | ⬜ | 10–20 seed scenarios for testing |
| `run_demo.py` | ⬜ | CLI / notebook to run 3 cases for judges |

---

## 4. Must-Have Tools
- [ ] `get_merchant_status()`
- [ ] `find_alternatives()`
- [ ] `assign_alternative_driver()`
- [ ] `notify_customer()`
- [ ] `issue_refund()`
- [ ] `collect_evidence()`
- [ ] `create_support_ticket()`
- [ ] `get_traffic_status()`  

---

## 5. Test Coverage (Scenarios)
- [ ] Merchant backlog (45m delay)  
- [ ] Merchant closed unexpectedly  
- [ ] Item out of stock (OOS)  
- [ ] Packaging damage (refund flow)  
- [ ] Driver idle → reroute  
- [ ] Driver breakdown → reassign  
- [ ] Payment failure → refund  
- [ ] Customer unreachable → escalate  

---

## 6. Output Format
- ✅ **Slim JSON for PPT** (3–4 steps only)
- ✅ Full decision_trace saved to `/logs/`
- ✅ Human-readable summary for console output

---

## 7. Demo Plan
- [ ] Run **3 scenarios live** (merchant backlog, item OOS, packaging damage)
- [ ] Show reasoning steps printing one by one  
- [ ] Open final JSON log in Colab or VSCode  
- [ ] Point out where policy prevented or escalated a case  
- [ ] Show edit of policy.json → re-run → see different behavior  

---

## 8. USPs to Mention
- Auditable reasoning → structured chain_of_thought  
- Tool-validated actions → no hallucinations  
- Policy-safe automation → editable guardrails  

---

## 9. Future Work
- Real API integration  
- Multi-modal (image evidence auto-classification)  
- Continuous learning from escalated cases  
- Dashboard for Ops managers  
- Multi-agent coordination  

---

## 10. Build Timeline (Optional)
| **Day** | **Goal** |
|--------|-----------|
| Day 1 | Set up repo + write tools.py + policies.json |
| Day 2 | Build deterministic reasoner + JSON schema validation |
| Day 3 | Add 8–10 scenario seeds + run_demo script |
| Day 4 | Integrate LangChain agent, test loop |
| Day 5 | Polish outputs, rehearse demo, prepare PPT |

---

## 11. Gotchas to Watch
- ✅ Keep **tools deterministic** (avoid random values)
- ✅ Set **temperature=0** for LLM → reproducible output
- ✅ Validate JSON after every run (fallback to escalation if invalid)
- ✅ Keep **scenario IDs consistent** so you know which case triggered which result
