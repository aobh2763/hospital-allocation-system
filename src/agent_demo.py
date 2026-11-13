"""Option 1: the complete scenario from end to end.

    poetry run python src/agent_demo.py                 # PDF scenario, ML/DL computed by the models
    poetry run python src/agent_demo.py --given         # use the PDF's predictions (0.87, 0.91)
    poetry run python src/agent_demo.py --pause         # wait for Enter between stages (live demo)
    poetry run python src/agent_demo.py --scenario scenarios/icu_reallocation.json

What happens (the PDF's "complete end-to-end example"):
  1. the environment is loaded (Block 1)
  2. the agent deliberates: ML and DL predictions (Block 3) become facts, the
     rules derive risk and needs (Block 2), A* chooses an allocation (Block 4),
     the planner orders the actions (Block 5)
  3. the plan is executed step by step and stopped just before P1's transfer
  4. the scenario's events happen one after another (ICU1 lost, a second
     Critical patient, P2 deteriorating); after each one the agent detects the
     change, deliberates again and continues with the new plan (Block 6)
  5. a summary of every decision
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import Agent
from agent.display import show_cycle, show_environment, show_step, show_summary, show_world, title
from environment.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=str(ROOT / "scenarios" / "pdf_example.json"))
    ap.add_argument("--given", action="store_true", help="use predictions written in the scenario file")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between stages")
    ap.add_argument("--steps-between-events", type=int, default=3)
    args = ap.parse_args(argv)

    def pause() -> None:
        if args.pause:
            input("\n[Enter to continue] ")

    scenario = Scenario.from_json(args.scenario)
    events, scenario.events = scenario.events, []
    title(f"Integrated agent: {scenario.name}")
    print("  A simulation for teaching: no real diagnosis or treatment advice.")
    agent = Agent(scenario, use_models=not args.given)
    show_environment(agent)
    pause()

    show_cycle(agent, agent.step().cycle, verbose=True)
    pause()

    first = next((a for a in agent.plan if a.name == "TransferPatient"), None)
    title("Execution" + (f" until just before {first}" if first and events else ""))
    agent.run(until=(lambda a: a == first) if first and events else None, on_step=show_step)
    show_world(agent)
    pause()

    for i, ev in enumerate(events, 1):
        title(f"EVENT {i}: {ev.get('description', ev['type'])}", "#")
        agent.observe(ev)
        result = agent.step()                      # the agent notices and deliberates
        show_cycle(agent, result.cycle, verbose=False)
        pause()
        last = i == len(events)
        title("Execution" + ("" if last else f" ({args.steps_between_events} actions)"))
        agent.run(max_steps=10_000 if last else args.steps_between_events, on_step=show_step)
        show_world(agent)
        pause()

    if not events:
        agent.run(on_step=show_step)
        show_world(agent)
    show_summary(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
