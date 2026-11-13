"""Option 2: run the agent yourself and add patients by hand.

    poetry run python src/agent_cli.py                 # start from the PDF department and patients
    poetry run python src/agent_cli.py --empty         # same department, no patients yet
    poetry run python src/agent_cli.py --given         # use the scenario's written predictions

The agent deliberates once, then waits for you. Every change you make (a new
patient, new vital signs, a resource lost or back, a discharge) is observed by
the agent, which re-runs the models, re-reasons, searches a new allocation and
plans again, and shows you what it changed. Execute actions one by one to see
how far the plan has gone when the next change arrives: patients not yet
transferred can lose a reserved bed to someone more urgent; patients already
transferred keep theirs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import Agent
from agent.display import show_cycle, show_environment, show_step, show_world, title
from environment.scenario import Scenario
from patient import Patient
from patient_cli import HISTORY_COLUMNS, ask, bars, enter_patient, load_patient
from reasoning.printer import to_pdf

ROOT = Path(__file__).resolve().parents[1]

HELP = """
  n  add a new patient (you type the data)      f  add a new patient from a JSON file
  s  execute the next action                    r  execute several actions
  a  execute the whole plan                     p  show the current plan
  w  show who holds what                        v  new vital signs for a patient
  o  put a resource out of service              b  put a resource back in service
  d  discharge a patient                        e  ask the knowledge base (proof or query)
  h  this help                                  q  quit
"""


def deliberate_and_show(agent: Agent, verbose: bool = False) -> None:
    result = agent.step()                      # the agent sees the pending change and deliberates
    if result.cycle:
        show_cycle(agent, result.cycle, verbose=verbose)
    show_world(agent)


def preview_models(agent: Agent, p: Patient) -> None:
    """Per-level probabilities for the new patient, before the agent reasons."""
    if agent.ml:
        print("\n  ML model, current snapshot:")
        bars(agent.ml.probabilities(p))
    if agent.dl and len(p.history) >= 2:
        print("  DL model, evolution over the hours entered:")
        bars(agent.dl.probabilities(p))
    elif agent.dl:
        print("  DL model: no earlier measurements, so no DL_Risk for this patient.")


def add_patient(agent: Agent, p: Patient) -> None:
    title(f"New patient {p.pid} arrives", "#")
    preview_models(agent, p)
    agent.observe({"type": "new_patient", "patient": p.to_dict(), "description": f"new patient {p.pid} arrives"})
    deliberate_and_show(agent)


def pick(prompt: str, options: list[str]) -> str | None:
    if not options:
        print("  Nothing to choose from.")
        return None
    choice = ask(f"{prompt} ({', '.join(options)})").upper()
    if choice not in options:
        print("  Not one of the options.")
        return None
    return choice


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=str(ROOT / "scenarios" / "pdf_example.json"))
    ap.add_argument("--empty", action="store_true", help="start with the department only, no patients")
    ap.add_argument("--given", action="store_true", help="use predictions written in the scenario file")
    args = ap.parse_args(argv)

    scenario = Scenario.from_json(args.scenario)
    scenario.events = []
    if args.empty:
        scenario.patients, scenario.predictions, scenario.assignments = [], {}, []
    title(f"Interactive agent: {scenario.name}")
    print("  A simulation for teaching: no real diagnosis or treatment advice.")
    agent = Agent(scenario, use_models=not args.given)
    show_environment(agent)
    deliberate_and_show(agent, verbose=False)
    print(HELP)

    while True:
        cmd = ask("\ncommand (h for help)").lower()[:1]
        try:
            if cmd == "q":
                break
            elif cmd == "h":
                print(HELP)
            elif cmd == "n":
                add_patient(agent, enter_patient(scenario.next_patient_id()))
            elif cmd == "f":
                path = ask("JSON file", str(ROOT / "scenarios" / "patient_example.json"))
                add_patient(agent, load_patient(path, scenario.next_patient_id()))
            elif cmd == "s":
                show_step(agent.step()) if agent.plan else print("  Nothing left to do.")
            elif cmd == "r":
                n = int(ask("how many actions", "3"))
                agent.run(max_steps=n, on_step=show_step)
                show_world(agent)
            elif cmd == "a":
                agent.run(on_step=show_step)
                show_world(agent)
            elif cmd == "p":
                print("\n".join(f"  {i:>2}. {a}" for i, a in enumerate(agent.plan, 1)) or "  Nothing left to do.")
            elif cmd == "w":
                show_world(agent)
            elif cmd == "v":
                pid = pick("patient", [p.pid for p in scenario.patients])
                if pid:
                    print(f"  One line: {' '.join(c.upper() for c in HISTORY_COLUMNS)}  (- for missing)")
                    parts = ask("  values").replace(",", " ").split()
                    values = {k: float(v) for k, v in zip(HISTORY_COLUMNS, parts) if v != "-"}
                    agent.observe({"type": "new_measurement", "pid": pid, "measurement": values,
                                   "description": f"new vital signs for {pid}"})
                    deliberate_and_show(agent)
            elif cmd in ("o", "b"):
                out = cmd == "o"
                ids = [r.rid for r in scenario.resources if (r.rid not in agent.out_of_service) == out]
                rid = pick("resource", ids)
                if rid:
                    agent.observe({"type": "resource_unavailable" if out else "resource_available", "id": rid})
                    deliberate_and_show(agent)
            elif cmd == "d":
                pid = pick("patient", [p.pid for p in scenario.patients])
                if pid:
                    agent.observe({"type": "discharge", "pid": pid})
                    deliberate_and_show(agent)
            elif cmd == "e":
                q = ask("query, e.g. RequiresICU(P1) or RequiresEquipment(P1, t)")
                if q:
                    if "(" in q and any(w.strip()[:1].islower() and len(w.strip()) <= 2
                                        for w in q[q.find("(") + 1:q.rfind(")")].split(",")):
                        rows = agent.kb.query(q)
                        print("  " + ("\n  ".join(str(r) for r in rows) if rows else "no answers"))
                    else:
                        print(agent.kb.format_proof(q))
            elif cmd:
                print("  Unknown command; h for help.")
        except (ValueError, KeyError) as e:
            print(f"  Could not do that: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
