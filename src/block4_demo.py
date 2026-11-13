"""Block 4 demo: Block 2's requirements -> Uniform Cost Search and A* -> allocation.

    poetry run python src/block4_demo.py                  # PDF example, its 3 events, ICU reallocation
    poetry run python src/block4_demo.py --models         # predictions from the trained ML/DL models
    poetry run python src/block4_demo.py --scenario scenarios/icu_reallocation.json --no-events

For each situation: what Block 2 says each patient needs, both searches side by
side (cost, states expanded and generated, largest frontier, time), the chosen
allocation with its cost breakdown and announcements, and an independent check
of the two validity rules. The scaling experiment is src/search_compare.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from environment.scenario import Scenario
from reasoning.adapters import DLAdapter, MLAdapter, run_models
from reasoning.events import apply_event
from reasoning.kb import KnowledgeBase
from search import allocate, check_allocation
from search.problem import COST_TABLE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = [ROOT / "scenarios" / "pdf_example.json", ROOT / "scenarios" / "icu_reallocation.json"]


def title(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def needs_table(req: dict) -> None:
    print(f"  {'Patient':<8}{'Risk':<10}{'Status':<21}{'Bed':<11}{'Doctor':<16}{'Nurse':<7}Equipment")
    for pid, p in req["patients"].items():
        status = "waiting" if p["waiting"] else f"holds {', '.join(p['assigned'])}" if p["assigned"] else "treated"
        print(f"  {pid:<8}{p['risk'] or '-':<10}{status:<21}{p['bed'] or '-':<11}"
              f"{p['specialist'] or p['doctor'] or '-':<16}{'yes' if p['nurse'] else '-':<7}"
              f"{', '.join(p['equipment']) or '-'}")


def solve(label: str, req: dict, rows: list) -> None:
    title(label)
    needs_table(req)
    decisions = {alg: allocate(req, alg) for alg in ("ucs", "astar")}
    d = decisions["astar"]

    print(f"\n  {'':<5}{'valid':<7}{'cost':>6}{'expanded':>10}{'generated':>11}{'frontier':>10}{'h(start)':>10}{'ms':>9}")
    for alg, dec in decisions.items():
        r = dec.result
        print(f"  {r.algorithm:<5}{'yes' if dec.valid else 'NO':<7}{r.cost:>6}{r.expanded:>10}{r.generated:>11}"
              f"{r.max_frontier:>10}{r.h_start:>10}{r.seconds * 1000:>9.2f}")
    same = decisions["ucs"].result.cost == d.result.cost
    print(f"  Same cost for both: {'yes' if same else 'NO'}")

    if not d.valid:
        print("\n  NO VALID ALLOCATION: not every High/Critical patient can be served at once.")
        for msg in d.diagnosis:
            print(f"    - {msg}")
        print("  Best allocation that leaves some High/Critical patients waiting (escalate):")

    alloc = d.allocation
    print("\n  Allocation (A*):")
    for pid, a in alloc.get("patients", {}).items():
        eq = ", ".join(f"{t}={rid}" for t, rid in a["equipment"].items())
        print(f"    {pid} ({a['risk']}): bed {a['bed']}, doctor {a['doctor']}"
              + (f", nurse {a['nurse']}" if a["nurse"] else "") + (f", {eq}" if eq else ""))
    for m in alloc.get("relocations", []):
        print(f"    move {m['patient']}: {m['from']} -> {m['to']}")
    if alloc.get("waiting"):
        print(f"    left waiting: {', '.join(alloc['waiting'])}")
    print(f"  Actions: {' '.join(str(a) for a in d.result.actions)}")
    print(f"  Cost {d.result.cost} = " + " + ".join(f"{v} {k}" for k, v in sorted(d.result.breakdown.items())))
    for note in d.result.announcements:
        print(f"  ANNOUNCE: {note}")
    errors = check_allocation(d.problem, d.result)
    print(f"  Check: {'valid, no resource used twice, every High/Critical patient fully served' if not errors else '; '.join(errors)}")
    rows.append((label, decisions))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", help="scenario JSON (repeatable); default: PDF + reallocation")
    ap.add_argument("--models", action="store_true", help="compute ML/DL predictions with the trained models")
    ap.add_argument("--no-events", action="store_true")
    args = ap.parse_args(argv)

    title("Cost of each action (g)")
    for what, cost in COST_TABLE:
        print(f"  {str(cost):>4}  {what}")
    print("  A* adds h: a lower bound of the cost still to pay (see docs/block4_search.md).")

    ml = dl = None
    if args.models:
        ml, dl = MLAdapter(), DLAdapter()
    rows: list = []
    for path in [Path(p) for p in args.scenario] if args.scenario else DEFAULT_SCENARIOS:
        scenario = Scenario.from_json(path)
        if args.models:
            scenario.predictions = {}
        kb = KnowledgeBase()
        kb.load_scenario(scenario)
        if args.models:
            run_models(kb, scenario.patients, ml, dl)
        kb.update()
        solve(f"{scenario.name}: start", kb.requirements(), rows)
        if args.no_events:
            continue
        for ev in scenario.events:
            apply_event(kb, ev, ml, dl)
            solve(f"{scenario.name}: {ev.get('description', ev['type'])}", kb.requirements(), rows)

    title("UCS vs A* on these situations")
    print(f"  {'Situation':<52}{'valid':<7}{'cost':>5}{'UCS exp':>9}{'A* exp':>8}{'UCS ms':>8}{'A* ms':>7}")
    for label, dec in rows:
        u, a = dec["ucs"].result, dec["astar"].result
        print(f"  {label[:51]:<52}{'yes' if dec['astar'].valid else 'NO':<7}{a.cost:>5}{u.expanded:>9}"
              f"{a.expanded:>8}{u.seconds * 1000:>8.2f}{a.seconds * 1000:>7.2f}")
    print("\n  Larger departments: poetry run python src/search_compare.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
