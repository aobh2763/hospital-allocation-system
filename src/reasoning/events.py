"""Apply an environment change to the Knowledge Base (dynamic environment).

Supported events (as written in scenarios/*.json):

    {"type": "resource_unavailable", "id": "ICU1"}
    {"type": "resource_available",   "id": "ICU1"}
    {"type": "new_patient", "patient": {...Patient...}, "predictions": {"dl": {"level": "...", "prob": ...}}}
    {"type": "new_measurement", "pid": "P2", "measurement": {"hr": 116, ...}, "predictions": {...}}
    {"type": "assign", "pid": "P1", "id": "ICU1"}
    {"type": "discharge", "pid": "P3"}

Predictions in the event are used as given; otherwise, when adapters are
passed, the models run again on the patient's new data.
"""

from __future__ import annotations

from typing import Any

from patient import Patient
from reasoning.adapters import run_models
from reasoning.kb import KnowledgeBase, UpdateResult


def apply_event(kb: KnowledgeBase, event: dict[str, Any], ml=None, dl=None) -> UpdateResult:
    kind = event["type"]
    given = event.get("predictions", {})

    if kind == "resource_unavailable":
        kb.set_available(event["id"], False)
    elif kind == "resource_available":
        kb.set_available(event["id"], True)
    elif kind == "new_patient":
        p = Patient.from_dict(event["patient"])
        kb.add_patient(p)
        _predict(kb, p, given, ml, dl)
    elif kind == "new_measurement":
        p = kb.scenario.patient(event["pid"])
        p.add_measurement(**event["measurement"])
        kb.refresh_patient(p)
        _predict(kb, p, given, ml, dl)
    elif kind == "assign":
        kb.assign(event["pid"], event["id"])
    elif kind == "discharge":
        kb.remove_patient(event["pid"])
        kb.scenario.patients = [p for p in kb.scenario.patients if p.pid != event["pid"]]
    else:
        raise ValueError(f"unknown event type {kind!r}")
    return kb.update()


def _predict(kb: KnowledgeBase, p: Patient, given: dict[str, Any], ml, dl) -> None:
    for source, pred in given.items():
        kb.set_prediction(p.pid, source, pred["level"], pred["prob"])
    run_models(kb, [p], ml, dl, skip={p.pid: given})
