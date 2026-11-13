"""Block 2 demo: from patient data to risk levels, requirements and proofs.

    python src/block2_demo.py                         # PDF example, predictions as in the PDF
    python src/block2_demo.py --models                # PDF example, predictions from the trained ML/DL models
    python src/block2_demo.py --hospital FAC0001 --date 2023-01-02
                                                      # real hospital resources + Synthea/time-series patients
    python src/block2_demo.py --json out.json         # also write the Search contract to a file

Sections: scenario, predictions, deduction, risk table, explanation,
resolution, unification, requirements for Search, then the dynamic events.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from environment.scenario import Scenario
from reasoning.adapters import DLAdapter, MLAdapter, dl_unavailable_reason, run_models
from reasoning.events import apply_event
from reasoning.kb import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]

UNIFICATION_EXAMPLES = [
    ("RequiresSpecialist(x, Cardiology)", "RequiresSpecialist(P1, Cardiology)"),
    ("ML_Risk(x, l, p)", "ML_Risk(P1, High, 0.87)"),
    ("Type(b, ICU)", "Type(EB1, Emergency)"),
    ("Assigned(x, x)", "Assigned(P1, ICU1)"),
    ("x", "f(x)"),
]


def title(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def load(args) -> Scenario:
    if args.hospital:
        from environment.hospital import from_hospital_data

        return from_hospital_data(
            args.hospital,
            args.date,
            args.shift,
            seed=args.seed,
            extra_specialists=args.on_call,
        )
    return Scenario.from_json(args.scenario)


def risk_table(kb: KnowledgeBase) -> None:
    reqs = kb.requirements()["patients"]
    preds = {}
    for src in ("ML_Risk", "DL_Risk"):
        for r in kb.query(f"{src}(x, l, p)"):
            preds.setdefault(r["x"], {})[src] = f"{r['l']} {float(r['p']):.2f}"
    head = f"{'Patient':<8}{'ML':<16}{'DL':<16}{'Risk':<10}{'Prio':<6}{'Bed':<11}{'Doctor':<16}Equipment"
    print(head + "\n" + "-" * len(head))
    for pid, p in reqs.items():
        doc = p["specialist"] or p["doctor"] or "-"
        flags = (" [reassess]" if p["reassess"] else "") + (
            " [no data]" if p["insufficient_data"] else ""
        )
        print(
            f"{pid:<8}{preds.get(pid, {}).get('ML_Risk', '-'):<16}{preds.get(pid, {}).get('DL_Risk', '-'):<16}"
            f"{p['risk'] or '-':<10}{p['priority'] or '-':<6}{p['bed'] or '-':<11}{doc:<16}"
            f"{', '.join(p['equipment']) or '-'}{flags}"
        )


def print_changes(changes) -> None:
    if not changes:
        print("  no requirement changed")
    for pid, c in changes.items():
        if c["risk"] and c["risk"][0] is None:
            print(f"  {pid}: new patient, risk {c['risk'][1]}")
        elif c["risk"]:
            print(f"  {pid}: risk {c['risk'][0]} -> {c['risk'][1]}")
        for f in c["added"]:
            print(f"  {pid}: + {f}")
        for f in c["removed"]:
            print(f"  {pid}: - {f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenario", default=str(ROOT / "scenarios" / "pdf_example.json"))
    ap.add_argument("--hospital", help="facility id from data/hospital, e.g. FAC0001")
    ap.add_argument("--date", default="2023-01-02")
    ap.add_argument("--shift", default="Day", choices=["Day", "Evening", "Night"])
    ap.add_argument(
        "--on-call", nargs="*", default=[], help="extra specialists, e.g. cardiology"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--models",
        action="store_true",
        help="run the ML/DL models instead of given predictions",
    )
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--json", help="write the requirements for Search to this file")
    args = ap.parse_args(argv)

    scenario = load(args)
    use_models = args.models or bool(args.hospital)
    if use_models:
        scenario.predictions = {}

    title(f"1. Scenario: {scenario.name}")
    for n in scenario.notes:
        print(f"  note: {n}")
    for r in scenario.resources:
        print(
            f"  {r.rid:<6}{r.kind:<10}{r.subtype or '':<18}{'available' if r.available else 'UNAVAILABLE'}"
        )
    for p in scenario.patients:
        src = p.source.get("dataset", "form")
        print(
            f"  {p.pid}: age {p.age}, vitals {p.vitals}, symptoms {p.symptoms}, "
            f"conditions {p.conditions}, {len(p.history)} hourly steps [{src}]"
        )

    kb = KnowledgeBase()
    kb.load_scenario(scenario)

    ml = dl = None
    title("2. Learned evidence (Block 3)")
    if use_models:
        ml = MLAdapter()
        missing = dl_unavailable_reason()
        if missing:
            print(f"  DL model skipped: {missing}. Using ML and rules only.")
        else:
            dl = DLAdapter()
        out = run_models(kb, scenario.patients, ml, dl)
        for pid, o in out.items():
            parts = [
                f"{s.upper()}_Risk({pid}, {lv}, {pr:.2f})" for s, (lv, pr) in o.items()
            ]
            cov = ml.coverage.get(pid)
            print(
                f"  {', '.join(parts) or 'no prediction'}"
                + (
                    f"   (ML features known: {cov:.0%})"
                    if cov is not None and cov < 1
                    else ""
                )
            )
    else:
        for pid, by in scenario.predictions.items():
            for s, pr in by.items():
                print(
                    f"  {s.upper()}_Risk({pid}, {pr.level}, {pr.prob:.2f})   (given in the scenario file)"
                )

    first = kb.update()
    print(f"\n  {len(kb.base_facts())} base facts -> {first.derived} derived facts")

    focus = next(
        (
            pid
            for pid, p in kb.requirements()["patients"].items()
            if p["risk"] == "Critical"
        ),
        kb.patients()[0],
    )
    title(f"3. Deduction (forward chaining), facts derived for {focus}")
    print(kb.format_deductions(patient=focus))

    title("4. Risk levels and requirements")
    risk_table(kb)

    title(f"5. Why does {focus} need what it needs? (deduction tree)")
    goal = (
        f"RequiresICU({focus})"
        if kb.holds(f"RequiresICU({focus})")
        else f"RiskLevel({focus}, {kb.risk(focus)})"
    )
    print(kb.format_explanation(goal))

    title("6. Resolution (SLD refutation with negation as failure)")
    print(kb.format_proof(f"RequiresICU({focus})"))
    other = next((p for p in kb.patients() if not kb.holds(f"RequiresICU({p})")), None)
    if other:
        print()
        print(kb.format_proof(f"RequiresICU({other})"))

    title("7. Unification")
    for a, b in UNIFICATION_EXAMPLES:
        u = kb.unify(a, b)
        res = f"θ = {u['theta']}" if u["unified"] else f"fails: {u['reason']}"
        print(f"  {a:<36} with {b:<36} {res}")

    title("8. Requirements for Search (Block 4)")
    reqs = kb.requirements()
    for pid, p in reqs["patients"].items():
        print(
            f"  {pid}: {json.dumps({k: v for k, v in p.items() if k != 'candidates'})}"
        )
        print(f"       candidates: {json.dumps(p['candidates'])}")
    for s in reqs["shortages"]:
        print(
            f"  SHORTAGE: {s['resource']} needed by {', '.join(s['patients'])}; {s['available']} available"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(reqs, indent=2))
        print(f"  written to {args.json}")

    if not args.no_events and scenario.events:
        title("9. Dynamic environment")
        for i, ev in enumerate(scenario.events, 1):
            print(f"\n  Event {i}: {ev.get('description', ev['type'])}")
            res = apply_event(kb, ev, ml, dl)
            print_changes(res.changes)
            for s in kb.requirements()["shortages"]:
                print(
                    f"  SHORTAGE: {s['resource']} needed by {', '.join(s['patients'])}; {s['available']} available"
                )
        print()
        risk_table(kb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
