"""One patient, whatever the data source.

The same record feeds the three consumers of Block 2:

* the ML model reads a snapshot (``ml_features`` or the latest vitals),
* the DL model reads the vitals timeline (``history``),
* the Knowledge Base reads the latest vitals, symptoms and conditions as facts.

Vital keys follow the DL dataset: ``hr``, ``map``, ``spo2``, ``temp``, ``rr``,
``lactate``, plus ``sbp`` and ``dbp`` from Synthea.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

VITALS = ("hr", "sbp", "dbp", "map", "spo2", "temp", "rr", "lactate")

# Column names of the DL time-series dataset -> vital keys.
TIMESERIES_COLUMNS = {
    "HR": "hr",
    "MAP": "map",
    "SpO2": "spo2",
    "Temp": "temp",
    "RespRate": "rr",
    "Lactate": "lactate",
}

# Synthea risk-factor flags (rf_* columns of the ML dataset) -> condition names.
SYNTHEA_CONDITIONS = {
    "rf_hypertension": "hypertension",
    "rf_diabetes": "diabetes",
    "rf_heart_disease": "heart_disease",
    "rf_copd": "copd",
    "rf_obesity": "obesity",
    "rf_smoker": "smoker",
    "rf_stroke": "stroke",
    "rf_ckd": "ckd",
    "rf_cancer": "cancer",
    "rf_asthma": "asthma",
}

# Synthea observation columns -> vital keys.
SYNTHEA_VITALS = {
    "obs_Heart_rate": "hr",
    "obs_Systolic_Blood_Pressure": "sbp",
    "obs_Diastolic_Blood_Pressure": "dbp",
    "obs_Respiratory_rate": "rr",
    "obs_Body_temperature": "temp",
}

SYMPTOMS = (
    "chest_pain",
    "shortness_of_breath",
    "abdominal_pain",
    "fever",
    "confusion",
    "headache",
    "bleeding",
    "trauma",
)


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def mean_arterial_pressure(sbp: float | None, dbp: float | None) -> float | None:
    if sbp is None or dbp is None:
        return None
    return round((sbp + 2 * dbp) / 3, 1)


@dataclass
class Patient:
    pid: str                                    # "P1", "P2", ... (PDF notation)
    age: float | None = None
    sex: str | None = None                      # "m" / "f"
    vitals: dict[str, float] = field(default_factory=dict)        # latest values
    history: list[dict[str, float]] = field(default_factory=list)  # hourly, oldest first
    symptoms: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    low_bp: bool = False                        # "Blood Pressure = Low" from a form
    status: str = "waiting"                     # waiting / under_treatment
    source: dict[str, Any] = field(default_factory=dict)
    ml_features: dict[str, float] | None = None  # full Synthea feature row, if known

    def __post_init__(self) -> None:
        self.vitals = {k: v for k, v in ((k, _num(v)) for k, v in self.vitals.items()) if v is not None}
        if "map" not in self.vitals:
            m = mean_arterial_pressure(self.vitals.get("sbp"), self.vitals.get("dbp"))
            if m is not None:
                self.vitals["map"] = m
        unknown = set(self.vitals) - set(VITALS)
        if unknown:
            raise ValueError(f"{self.pid}: unknown vitals {sorted(unknown)}")

    # ------------------------------------------------------------------ updates

    def add_measurement(self, **values: float) -> None:
        """Append one time step (e.g. hr=130, map=58) and make it the latest vitals."""
        step = {k: _num(v) for k, v in values.items() if _num(v) is not None}
        self.history.append(step)
        self.vitals.update(step)

    # ----------------------------------------------------------- constructors

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Patient":
        d = dict(d)
        history = d.pop("history", [])
        p = cls(**d)
        p.history = [{k: _num(v) for k, v in step.items()} for step in history]
        if p.history and not p.vitals:
            p.vitals = {k: v for k, v in p.history[-1].items() if v is not None}
        return p

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_synthea(cls, pid: str, raw: Any, features: Any | None = None, row: int | None = None) -> "Patient":
        """One encounter of ``data/patients.csv`` (raw) and, optionally, the
        matching row of ``data/patient_preprocessed.csv`` (ML features)."""
        vitals = {key: _num(raw.get(col)) for col, key in SYNTHEA_VITALS.items()}
        conditions = [name for col, name in SYNTHEA_CONDITIONS.items() if _num(raw.get(col)) == 1]
        sex = str(raw.get("GENDER", "")).lower() or None
        ml = None
        if features is not None:
            ml = {k: _num(v) for k, v in dict(features).items()
                  if k not in ("PATIENT", "risk_score", "risk_class")}
        age = _num(raw.get("age"))
        return cls(
            pid=pid,
            age=round(age, 1) if age is not None else None,
            sex=sex if sex in ("m", "f") else None,
            vitals={k: v for k, v in vitals.items() if v is not None},
            conditions=conditions,
            source={"dataset": "synthea", "patient": raw.get("PATIENT"), "row": row,
                    "label": raw.get("risk_class")},
            ml_features=ml,
        )

    @classmethod
    def from_timeseries(cls, pid: str, rows: Any) -> "Patient":
        """All time steps of one ``Patient_ID`` from the DL time-series dataset."""
        rows = rows.sort_values("Time_Step")
        history = [
            {key: _num(r[col]) for col, key in TIMESERIES_COLUMNS.items() if _num(r[col]) is not None}
            for _, r in rows.iterrows()
        ]
        p = cls(pid=pid, source={"dataset": "timeseries", "patient": rows["Patient_ID"].iloc[0],
                                 "label": rows["Risk"].iloc[-1]})
        p.history = history
        for step in history:
            p.vitals.update(step)
        return p

    def merge_timeseries(self, other: "Patient") -> "Patient":
        """Give this (Synthea) patient the vitals timeline of a time-series patient.

        The datasets share no patients, so this pairing is simulated: it lets
        the ML model (static history) and the DL model (temporal evolution)
        speak about the same person. The latest time step replaces the
        snapshot vitals it covers.
        """
        self.history = [dict(s) for s in other.history]
        if any("map" in s for s in self.history):
            # the snapshot blood pressure belongs to another moment: keep the timeline's MAP only
            self.vitals.pop("sbp", None)
            self.vitals.pop("dbp", None)
        for step in self.history:
            self.vitals.update(step)
        self.source = {**self.source, "timeseries": other.source}
        return self
