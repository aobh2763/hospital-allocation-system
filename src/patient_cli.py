"""Enter one patient yourself, see the ML and DL results, then the Prolog reasoning.

    poetry run python src/patient_cli.py                       # asks the questions
    poetry run python src/patient_cli.py --file scenarios/patient_example.json
    poetry run python src/patient_cli.py --save my_patient.json   # keep what you typed

Steps shown:
  1. the patient as entered, and the base facts it becomes
  2. ML model: probability of each risk level (snapshot of the current state)
  3. DL model: probability of each risk level (evolution over the past hours)
  4. Prolog: every fact derived for the patient, layer by layer
  5. final risk level, needs and candidate resources (what Search receives)
  6. why: the deduction tree, then a resolution proof of the risk level
  7. your own queries, e.g. RequiresICU(P1) or RequiresEquipment(P1, t)

The patient is placed in the PDF's emergency department (2 doctors, 2 nurses,
1 ICU bed, 2 emergency beds, 1 exam room, monitors, pump, defibrillator,
ventilator) so the matching predicates have resources to work with.
This is a simulation for teaching, not medical advice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from environment.scenario import Scenario
from patient import SYMPTOMS, SYNTHEA_CONDITIONS, Patient, mean_arterial_pressure
from reasoning.adapters import LEVELS, DLAdapter, MLAdapter, dl_unavailable_reason
from reasoning.facts import patient_facts
from reasoning.kb import KnowledgeBase
from reasoning.printer import to_pdf

ROOT = Path(__file__).resolve().parents[1]
DEPARTMENT = ROOT / "scenarios" / "pdf_example.json"
PID = "P1"
CONDITIONS = tuple(SYNTHEA_CONDITIONS.values())

# Current vitals asked for: (key, question, unit, plausible range)
CURRENT_VITALS = [
    ("hr", "Heart rate", "bpm", (20, 250)),
    ("sbp", "Systolic blood pressure", "mmHg", (40, 260)),
    ("dbp", "Diastolic blood pressure", "mmHg", (20, 160)),
    ("spo2", "Oxygen saturation (SpO2)", "%", (50, 100)),
    ("temp", "Temperature", "°C", (30, 43)),
    ("rr", "Respiratory rate", "breaths/min", (4, 60)),
    ("lactate", "Lactate", "mmol/L", (0, 20)),
]
HISTORY_COLUMNS = ["hr", "map", "spo2", "temp", "rr", "lactate"]   # DL model inputs


# --------------------------------------------------------------- input helpers

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        print()
        sys.exit("Input ended.")
    return answer or default


def ask_number(prompt: str, rng: tuple[float, float] | None = None, required: bool = False) -> float | None:
    while True:
        raw = ask(prompt + ("" if required else " (Enter to skip)"))
        if not raw:
            if required:
                print("  A value is required.")
                continue
            return None
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            print("  Please type a number.")
            continue
        if rng and not rng[0] <= value <= rng[1]:
            print(f"  Expected between {rng[0]} and {rng[1]}.")
            continue
        return value


def ask_choices(prompt: str, options: tuple[str, ...]) -> list[str]:
    print(f"{prompt}  (numbers or names, comma separated, Enter for none)")
    for i, o in enumerate(options, 1):
        print(f"  {i:>2}. {o}")
    while True:
        raw = ask("Your choice")
        if not raw:
            return []
        chosen, bad = [], []
        for part in (x.strip() for x in raw.split(",") if x.strip()):
            name = part.lower().replace(" ", "_")
            if part.isdigit() and 1 <= int(part) <= len(options):
                chosen.append(options[int(part) - 1])
            elif name in options:
                chosen.append(name)
            else:
                bad.append(part)
        if bad:
            print(f"  Not in the list: {', '.join(bad)}")
            continue
        return list(dict.fromkeys(chosen))


def ask_history() -> list[dict[str, float]]:
    print("\nEarlier hourly measurements for the DL model, oldest first (up to 9).")
    print("One line per hour:  HR MAP SpO2 Temp RR Lactate   (use - for a missing value)")
    print("Example:            105 82 95 38.2 20 1.8")
    print("Press Enter on an empty line when done.")
    rows: list[dict[str, float]] = []
    while len(rows) < 9:
        raw = ask(f"  hour {len(rows) + 1}")
        if not raw:
            break
        parts = raw.replace(",", " ").split()
        if len(parts) != len(HISTORY_COLUMNS):
            print(f"  Expected {len(HISTORY_COLUMNS)} values, got {len(parts)}.")
            continue
        try:
            row = {k: float(v) for k, v in zip(HISTORY_COLUMNS, parts) if v != "-"}
        except ValueError:
            print("  Values must be numbers or -.")
            continue
        rows.append(row)
    return rows


def enter_patient() -> Patient:
    print("New patient (P1). Press Enter to skip any optional value.\n")
    age = ask_number("Age (years)", (0, 120), required=True)
    sex = ask("Sex (m/f)", "").lower()[:1] or None
    print()
    symptoms = ask_choices("Symptoms", SYMPTOMS)
    print()
    conditions = ask_choices("Known conditions (medical history)", CONDITIONS)
    print("\nCurrent vital signs:")
    vitals = {}
    for key, label, unit, rng in CURRENT_VITALS:
        v = ask_number(f"  {label} ({unit})", rng)
        if v is not None:
            vitals[key] = v
    low_bp = False
    if "sbp" not in vitals:
        low_bp = ask("  Blood pressure low? (y/n)", "n").lower().startswith("y")
    history = ask_history()
    return build_patient(age, sex, symptoms, conditions, vitals, low_bp, history)


def build_patient(age, sex, symptoms, conditions, vitals, low_bp, history) -> Patient:
    """The current vitals become the last hour of the timeline the DL model reads."""
    current = {k: v for k, v in vitals.items() if k in HISTORY_COLUMNS}
    m = mean_arterial_pressure(vitals.get("sbp"), vitals.get("dbp"))
    if m is not None:
        current["map"] = m
    timeline = list(history) + ([current] if history and current else [])
    if not vitals and history:                 # only a timeline given: its last hour is "now"
        vitals = {k: v for k, v in history[-1].items()}
    return Patient(pid=PID, age=age, sex=sex if sex in ("m", "f") else None, vitals=dict(vitals),
                   symptoms=symptoms, conditions=conditions, low_bp=low_bp,
                   history=timeline, source={"dataset": "typed in"})


def load_patient(path: str) -> Patient:
    d = json.loads(Path(path).read_text())
    return build_patient(d.get("age"), d.get("sex"), d.get("symptoms", []), d.get("conditions", []),
                         d.get("vitals", {}), d.get("low_bp", False), d.get("history", []))


def to_file(p: Patient, history_before_now: list[dict[str, float]], path: str) -> None:
    d = {"age": p.age, "sex": p.sex, "symptoms": p.symptoms, "conditions": p.conditions,
         "vitals": {k: v for k, v in p.vitals.items() if k != "map" or "sbp" not in p.vitals},
         "low_bp": p.low_bp, "history": history_before_now}
    Path(path).write_text(json.dumps(d, indent=2))
    print(f"\nSaved to {path}")


# --------------------------------------------------------------- output

def title(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def bars(probs: dict[str, float]) -> None:
    best = max(probs, key=probs.get)
    for level in LEVELS:
        pr = probs.get(level, 0.0)
        mark = "  <- prediction" if level == best else ""
        print(f"  {level:<9}{pr:6.1%}  {'█' * round(pr * 40)}{mark}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="read the patient from a JSON file instead of asking")
    ap.add_argument("--save", help="write the patient you typed to this JSON file")
    ap.add_argument("--no-queries", action="store_true", help="skip the interactive query step")
    args = ap.parse_args(argv)

    if args.file:
        patient = load_patient(args.file)
        earlier = patient.history[:-1]
    else:
        patient = enter_patient()
        earlier = patient.history[:-1]
        if args.save:
            to_file(patient, earlier, args.save)

    # ---- 1. patient and base facts
    title("1. Patient and the base facts it becomes")
    print(f"  Age {patient.age}, sex {patient.sex or '-'}")
    print(f"  Symptoms:   {', '.join(patient.symptoms) or '-'}")
    print(f"  Conditions: {', '.join(patient.conditions) or '-'}")
    print(f"  Current vitals: {patient.vitals}")
    print(f"  Timeline for the DL model: {len(patient.history)} hourly steps")
    print()
    for f in patient_facts(patient):
        print(f"  {to_pdf(f)}")

    kb = KnowledgeBase()
    dept = Scenario.from_json(DEPARTMENT)
    kb.load_scenario(Scenario("typed-in patient", patients=[patient], resources=dept.resources))

    # ---- 2. ML
    title("2. ML model (gradient boosting): risk from the current snapshot")
    ml = MLAdapter()
    ml_probs = ml.probabilities(patient)
    bars(ml_probs)
    ml_level = max(ml_probs, key=ml_probs.get)
    print(f"\n  Features known: {ml.coverage[PID]:.0%} of the model's {len(ml.features)} "
          f"(the rest are Synthea medians, so this evidence is weaker)")
    kb.set_prediction(PID, "ml", ml_level, round(ml_probs[ml_level], 2))
    print(f"  Inserted: ML_Risk({PID}, {ml_level}, {ml_probs[ml_level]:.2f})")

    # ---- 3. DL
    title("3. DL model (GRU): risk from the evolution over the past hours")
    reason = dl_unavailable_reason()
    if reason:
        print(f"  DL model unavailable: {reason}")
    elif len(patient.history) < 2:
        print("  No earlier measurements entered, so there is no evolution to analyse: no DL_Risk fact.")
    else:
        dl = DLAdapter()
        dl_probs = dl.probabilities(patient)
        bars(dl_probs)
        dl_level = max(dl_probs, key=dl_probs.get)
        if len(patient.history) < dl.T:
            print(f"\n  {len(patient.history)} steps given; the model reads {dl.T}, "
                  f"so the last one is repeated to fill the gap.")
        kb.set_prediction(PID, "dl", dl_level, round(dl_probs[dl_level], 2))
        print(f"  Inserted: DL_Risk({PID}, {dl_level}, {dl_probs[dl_level]:.2f})")

    # ---- 4. Prolog deduction
    result = kb.update()
    title(f"4. Prolog: facts derived for {PID} (forward chaining, {result.derived} in total)")
    print(kb.format_deductions(patient=PID))

    # ---- 5. outcome
    req = kb.requirements()["patients"][PID]
    title("5. Result: risk level and what the patient needs")
    print(f"  Risk level:   {req['risk']}   (priority {req['priority']}, max wait {req['max_wait']} min)")
    sources = kb.query(f"SourceOpinion({PID}, s, l)")
    print("  Opinions:     " + ", ".join(f"{r['s']} says {r['l']}" for r in sources))
    flags = [name for name, on in (("urgent evaluation", req["urgent_evaluation"]),
                                   ("reassessment (models unsure or disagree)", req["reassess"]),
                                   ("insufficient data", req["insufficient_data"])) if on]
    print(f"  Flags:        {', '.join(flags) or '-'}")
    print(f"  Bed:          {req['bed']}")
    print(f"  Doctor:       {req['specialist'] or req['doctor']}{' (specialist)' if req['specialist'] else ''}")
    print(f"  Nurse:        {'yes' if req['nurse'] else 'no'}")
    print(f"  Equipment:    {', '.join(req['equipment']) or '-'}")
    print(f"  Monitoring:   {req['monitoring']}")
    print(f"  Candidates:   {json.dumps(req['candidates'])}")

    # ---- 6. why
    risk_goal = f"{req['risk']}Risk({PID})"
    title(f"6. Why {risk_goal}?")
    print("Deduction tree:")
    print(kb.format_explanation(risk_goal))
    print("\nResolution proof:")
    print(kb.format_proof(risk_goal))

    # ---- 7. queries
    if not args.no_queries and not args.file:
        title("7. Ask your own queries")
        print("Examples: RequiresICU(P1)   CriticalRisk(P1)   RequiresEquipment(P1, t)   Evidence(P1, l, s)")
        print("A query with lowercase variables lists all answers; a ground query shows its proof.")
        while True:
            q = ask("\nquery (Enter to quit)")
            if not q:
                break
            try:
                if any(c.islower() for c in _variables(q)):
                    rows = kb.query(q)
                    print("  " + ("\n  ".join(str(r) for r in rows) if rows else "no answers"))
                else:
                    print(kb.format_proof(q))
            except Exception as e:  # bad syntax, unknown predicate arity, ...
                print(f"  Could not run that query: {e}")
    return 0


def _variables(query: str) -> str:
    """The argument words of a query that are PDF-style variables (single lowercase letters)."""
    inside = query[query.find("(") + 1: query.rfind(")")] if "(" in query else ""
    return "".join(w for w in (a.strip() for a in inside.split(",")) if len(w) <= 2 and w[:1].islower())


if __name__ == "__main__":
    sys.exit(main())
