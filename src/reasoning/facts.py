"""Turn Patient records and Scenario resources into base facts (Prolog strings)."""

from __future__ import annotations

from environment.scenario import Resource, Scenario
from patient import Patient
from reasoning.printer import fact, to_atom

VITAL_PREDICATES = {
    "hr": "heart_rate",
    "sbp": "systolic_bp",
    "dbp": "diastolic_bp",
    "map": "mean_arterial_pressure",
    "spo2": "oxygen_saturation",
    "temp": "temperature",
    "rr": "respiratory_rate",
    "lactate": "lactate",
}


def patient_facts(p: Patient) -> list[str]:
    facts = [fact("patient", p.pid)]
    if p.age is not None:
        facts.append(fact("age", p.pid, round(float(p.age), 1)))
    if p.sex:
        facts.append(fact("sex", p.pid, p.sex))
    for key, value in p.vitals.items():
        facts.append(fact(VITAL_PREDICATES[key], p.pid, round(float(value), 2)))
    facts += [fact("has_symptom", p.pid, s) for s in p.symptoms]
    facts += [fact("has_condition", p.pid, c) for c in p.conditions]
    if p.low_bp:
        facts.append(fact("low_blood_pressure", p.pid))
    if p.status == "waiting":
        facts.append(fact("waiting", p.pid))
    elif p.status == "under_treatment":
        facts.append(fact("under_treatment", p.pid))
    return facts


def resource_facts(r: Resource) -> list[str]:
    if r.kind == "doctor":
        facts = [fact("doctor", r.rid)]
        if r.subtype:
            facts.append(fact("specialty", r.rid, r.subtype))
    elif r.kind == "nurse":
        facts = [fact("nurse", r.rid)]
    elif r.kind == "bed":
        facts = [fact("bed", r.rid), fact("type", r.rid, r.subtype)]
    else:
        facts = [fact("equipment", r.rid), fact("equipment_type", r.rid, r.subtype)]
    if r.available:
        facts.append(fact("available", r.rid))
    return facts


def prediction_fact(pid: str, source: str, level: str, prob: float) -> str:
    if source not in ("ml", "dl"):
        raise ValueError("source must be 'ml' or 'dl'")
    return fact(f"{source}_risk", pid, to_atom(level), round(float(prob), 4))


def scenario_facts(s: Scenario) -> list[str]:
    facts: list[str] = []
    for p in s.patients:
        facts += patient_facts(p)
    for r in s.resources:
        facts += resource_facts(r)
    for pid, rid in s.assignments:
        facts.append(fact("assigned", pid, rid))
    for pid, by_source in s.predictions.items():
        for source, pred in by_source.items():
            facts.append(prediction_fact(pid, source, pred.level, pred.prob))
    if s.surge_day:
        facts.append("surge_day")
    if s.diversion:
        facts.append("diversion")
    return facts
