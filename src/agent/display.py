"""Text output shared by the end-to-end demo and the interactive mode."""

from __future__ import annotations

from planning import describe
from reasoning.facts import patient_facts
from reasoning.printer import to_pdf

WIDTH = 78


def title(text: str, char: str = "=") -> None:
    print(f"\n{char * WIDTH}\n{text}\n{char * WIDTH}")


def section(text: str) -> None:
    print(f"\n--- {text} " + "-" * max(0, WIDTH - len(text) - 5))


def show_environment(agent) -> None:
    sc = agent.scenario
    section("Block 1 · environment")
    for note in sc.notes:
        print(f"  note: {note}")
    by_kind: dict[str, list[str]] = {}
    for r in sc.resources:
        state = "" if r.available else " (busy)" if r.rid not in agent.out_of_service else " (out of service)"
        by_kind.setdefault(r.kind, []).append(f"{r.rid}{'=' + r.subtype if r.subtype else ''}{state}")
    for kind, items in by_kind.items():
        label = kind if kind == "equipment" else kind + "s"
        print(f"  {label:<11}{', '.join(items)}")
    for p in sc.patients:
        print(f"  {p.pid}: age {p.age:g}, {', '.join(p.symptoms) or 'no symptoms'}; "
              f"vitals {_vitals(p.vitals)}; {len(p.history)} hourly steps; {p.status.replace('_', ' ')}"
              if p.age is not None else f"  {p.pid}: {p.status}")
    n = sum(len(patient_facts(p)) for p in sc.patients)
    print(f"  -> {len(agent.kb.base_facts())} base facts in the knowledge base ({n} about patients)")


def show_cycle(agent, cycle, verbose: bool = True) -> None:
    title(f"Deliberation {cycle.number}: " + "; ".join(cycle.reasons))
    if cycle.invalid_step:
        i, action, missing = cycle.invalid_step
        print(f"  The old plan is broken: step {action} needs {', '.join(missing)}, which no longer holds.")
    elif cycle.number > 1:
        print("  The old plan could still run, but the situation changed: re-deciding.")
    for pid, c in cycle.changes.items():
        if c["risk"] and c["risk"][0] is None:
            print(f"  {pid}: new patient, rated {c['risk'][1]}")
        elif c["risk"]:
            print(f"  {pid}: risk {c['risk'][0]} -> {c['risk'][1]}")
        needs = [f for f in c["added"] if f.startswith("Requires")]
        if needs:
            print(f"  {pid}: now {', '.join(needs)}")

    req, d = cycle.requirements, cycle.decision
    section("Block 3 · learning (ML on the current state, DL on the hourly evolution)")
    preds = {r["x"]: {} for r in agent.kb.query("Patient(x)")}
    for src in ("ML", "DL"):
        for r in agent.kb.query(f"{src}_Risk(x, l, p)"):
            preds[r["x"]][src] = f"{r['l']} {float(r['p']):.2f}"
    for pid in req["patients"]:
        mark = "  (new)" if pid in cycle.predictions else ""
        print(f"  {pid}: ML_Risk {preds[pid].get('ML', '-'):<15} DL_Risk {preds[pid].get('DL', '-'):<15}{mark}")

    section("Block 2 · reasoning (facts + rules -> risk and needs)")
    print(f"  {'Patient':<8}{'Risk':<10}{'Status':<17}{'Bed':<11}{'Doctor':<16}Flags")
    for pid, p in req["patients"].items():
        status = "waiting" if p["waiting"] else "under treatment"
        flags = ", ".join(f for f, on in (("urgent", p["urgent_evaluation"]), ("reassess", p["reassess"])) if on)
        print(f"  {pid:<8}{p['risk'] or '-':<10}{status:<17}{p['bed'] or '-':<11}"
              f"{p['specialist'] or p['doctor'] or '-':<16}{flags or '-'}")
    if verbose:
        critical = next((pid for pid, p in req["patients"].items() if p["risk"] == "Critical" and p["waiting"]), None)
        if critical:
            print()
            print("  " + agent.kb.format_explanation(f"RequiresICU({critical})").replace("\n", "\n  "))

    section("Block 4 · search (A*, UCS for comparison)")
    for name, dec in (("A*", d), ("UCS", cycle.ucs)):
        if dec:
            r = dec.result
            print(f"  {name:<4} cost {r.cost:<5} states expanded {r.expanded:<6} {r.seconds * 1000:7.2f} ms")
    if not d.valid:
        print("  NO VALID ALLOCATION: " + "; ".join(d.diagnosis))
    for pid, a in d.allocation.get("patients", {}).items():
        eq = ", ".join(a["equipment"].values())
        print(f"  {pid} ({a['risk']}): bed {a['bed']}, doctor {a['doctor']}"
              + (f", nurse {a['nurse']}" if a["nurse"] else "") + (f", {eq}" if eq else ""))
    for m in d.allocation.get("relocations", []):
        print(f"  move {m['patient']}: {m['from']} -> {m['to']}")
    if d.allocation.get("waiting"):
        print(f"  waiting: {', '.join(d.allocation['waiting'])}")
    for note in d.result.announcements:
        print(f"  ANNOUNCE: {note}")
    if cycle.allocation_changes:
        print("  Changed since the last decision: " + "; ".join(cycle.allocation_changes))

    section("Block 5 · planning (ordered actions, preconditions checked)")
    if not cycle.plan.steps:
        print("  Nothing to do.")
    for i, a in enumerate(cycle.plan.steps, 1):
        print(f"  {i:>2}. " + (describe(a) if verbose else str(a)))
    status = "valid: every precondition holds in order and every goal is reached" \
        if not cycle.plan_errors else "INVALID: " + "; ".join(cycle.plan_errors)
    print(f"  Plan {status} ({cycle.plan.expanded} states expanded, {cycle.plan.seconds * 1000:.1f} ms)")


def show_step(result) -> None:
    if result.kind == "executed":
        a = result.action
        effects = ", ".join([f"+{x}" for x in sorted(a.add)] + [f"-{x}" for x in sorted(a.delete)])
        print(f"  ▶ {a}   [{effects}]")
    elif result.kind == "done":
        print("  Plan complete.")


def show_world(agent) -> None:
    req = agent.kb.requirements()
    section("Current state")
    for pid, p in req["patients"].items():
        where = ", ".join(p["assigned"]) or "-"
        print(f"  {pid:<5}{p['risk'] or '-':<10}{'waiting' if p['waiting'] else 'under treatment':<17}holds {where}")
    free = [r["id"] for k in ("doctors", "nurses", "beds", "equipment") for r in req["resources"][k] if r["available"]]
    print(f"  free: {', '.join(free) or 'nothing'}")
    if agent.out_of_service:
        print(f"  out of service: {', '.join(sorted(agent.out_of_service))}")
    if agent.plan:
        print(f"  next: {agent.plan[0]}  ({len(agent.plan)} actions left)")


def show_summary(agent) -> None:
    title("Summary of the agent's decisions")
    print(f"  {'#':<3}{'trigger':<46}{'valid':<7}{'A* exp':>7}{'UCS exp':>8}{'plan':>6}")
    for c in agent.cycles:
        u = c.ucs.result.expanded if c.ucs else "-"
        print(f"  {c.number:<3}{'; '.join(c.reasons)[:45]:<46}{'yes' if c.decision.valid else 'NO':<7}"
              f"{c.decision.result.expanded:>7}{u:>8}{len(c.plan.steps):>6}")
    print(f"  Actions executed: {len(agent.executed)}")


def _vitals(v: dict) -> str:
    names = {"hr": "HR", "map": "MAP", "spo2": "SpO2", "temp": "T", "rr": "RR", "lactate": "lact"}
    return " ".join(f"{names[k]} {v[k]:g}" for k in names if k in v)


def pdf(text: str) -> str:
    return to_pdf(text)
