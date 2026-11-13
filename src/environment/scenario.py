"""The emergency-department state the agent reasons about.

A Scenario holds patients, resources, current assignments, optional
pre-computed predictions and a list of scripted events (for the dynamic
demo). It can be loaded from JSON (``scenarios/*.json``) or built from the
HLT-010 hospital dataset (``environment.hospital.from_hospital_data``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patient import Patient

RESOURCE_KINDS = ("doctor", "nurse", "bed", "equipment")
JSON_KEYS = {"doctor": "doctors", "nurse": "nurses", "bed": "beds", "equipment": "equipment"}
BED_TYPES = ("icu", "emergency", "exam_room")
RISK_LEVELS = ("Low", "Medium", "High", "Critical")


@dataclass
class Resource:
    rid: str                    # "D1", "ICU1", "V1" ...
    kind: str                   # doctor / nurse / bed / equipment
    subtype: str | None = None  # specialty, bed type or equipment type
    available: bool = True
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in RESOURCE_KINDS:
            raise ValueError(f"{self.rid}: unknown resource kind {self.kind!r}")
        if self.kind == "bed" and self.subtype not in BED_TYPES:
            raise ValueError(f"{self.rid}: bed type must be one of {BED_TYPES}")


@dataclass
class Prediction:
    level: str      # Low / Medium / High / Critical
    prob: float

    def __post_init__(self) -> None:
        if self.level not in RISK_LEVELS:
            raise ValueError(f"unknown risk level {self.level!r}")
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"probability out of range: {self.prob}")


@dataclass
class Scenario:
    name: str
    patients: list[Patient] = field(default_factory=list)
    resources: list[Resource] = field(default_factory=list)
    assignments: list[tuple[str, str]] = field(default_factory=list)   # (patient, resource)
    predictions: dict[str, dict[str, Prediction]] = field(default_factory=dict)  # pid -> {"ml"/"dl": ...}
    events: list[dict[str, Any]] = field(default_factory=list)
    surge_day: bool = False
    diversion: bool = False
    notes: list[str] = field(default_factory=list)

    def patient(self, pid: str) -> Patient:
        return next(p for p in self.patients if p.pid == pid)

    def resource(self, rid: str) -> Resource:
        return next(r for r in self.resources if r.rid == rid)

    def next_patient_id(self) -> str:
        nums = [int(p.pid[1:]) for p in self.patients if p.pid[1:].isdigit()]
        return f"P{max(nums, default=0) + 1}"

    # ------------------------------------------------------------------ JSON

    @classmethod
    def from_json(cls, path: str | Path) -> "Scenario":
        d = json.loads(Path(path).read_text())
        resources = []
        for kind in RESOURCE_KINDS:
            for r in d.get("resources", {}).get(JSON_KEYS[kind], []):
                resources.append(Resource(rid=r["id"], kind=kind, subtype=r.get("type"),
                                          available=r.get("available", True)))
        preds = {
            pid: {src: Prediction(**v) for src, v in by_src.items()}
            for pid, by_src in d.get("predictions", {}).items()
        }
        return cls(
            name=d.get("name", Path(path).stem),
            patients=[Patient.from_dict(p) for p in d.get("patients", [])],
            resources=resources,
            assignments=[tuple(a) for a in d.get("assignments", [])],
            predictions=preds,
            events=d.get("events", []),
            surge_day=d.get("surge_day", False),
            diversion=d.get("diversion", False),
            notes=d.get("notes", []),
        )

    def to_json(self) -> dict[str, Any]:
        res: dict[str, list] = {JSON_KEYS[k]: [] for k in RESOURCE_KINDS}
        for r in self.resources:
            res[JSON_KEYS[r.kind]].append({"id": r.rid, "type": r.subtype, "available": r.available})
        return {
            "name": self.name,
            "patients": [p.to_dict() for p in self.patients],
            "resources": res,
            "assignments": [list(a) for a in self.assignments],
            "predictions": {pid: {s: vars(p) for s, p in d.items()} for pid, d in self.predictions.items()},
            "events": self.events,
            "surge_day": self.surge_day,
            "diversion": self.diversion,
            "notes": self.notes,
        }
