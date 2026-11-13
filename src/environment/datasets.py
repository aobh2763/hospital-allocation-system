"""Which file holds what, and how to turn rows into Patient records.

| File                                              | Built by                         | One row is            | Used for                    |
|---------------------------------------------------|----------------------------------|-----------------------|-----------------------------|
| data/patients.csv                                 | prediction/build_dataset.py      | one Synthea encounter | facts (age, vitals, history)|
| data/patient_preprocessed.csv                     | prediction/ml/preprocessing.ipynb| same encounter, encoded | ML model input (55 features)|
| data/synthetic_emergency_timeseries_2000_patients.csv | downloaded                  | one hour of one patient | DL model input (10 steps)   |
| data/hospital/*.csv                               | HLT-010 sample (downloaded)      | facility-day / shift / asset | beds, staff, equipment |

``patients.csv`` and ``patient_preprocessed.csv`` have the same rows in the
same order, so row *i* of one is row *i* of the other.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from patient import Patient

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SYNTHEA_RAW = DATA / "patients.csv"
SYNTHEA_FEATURES = DATA / "patient_preprocessed.csv"
TIMESERIES = DATA / "synthetic_emergency_timeseries_2000_patients.csv"
HOSPITAL = DATA / "hospital"


@lru_cache(maxsize=1)
def synthea_raw() -> pd.DataFrame:
    return pd.read_csv(SYNTHEA_RAW, low_memory=False)


@lru_cache(maxsize=1)
def synthea_features() -> pd.DataFrame:
    return pd.read_csv(SYNTHEA_FEATURES, low_memory=False)


@lru_cache(maxsize=1)
def timeseries() -> pd.DataFrame:
    return pd.read_csv(TIMESERIES).sort_values(["Patient_ID", "Time_Step"])


def synthea_patient(pid: str, row: int) -> Patient:
    """Encounter ``row`` of the Synthea dataset, with its ML feature row."""
    raw = synthea_raw().iloc[row]
    feats = synthea_features().iloc[row]
    if raw["PATIENT"] != feats["PATIENT"]:
        raise RuntimeError("patients.csv and patient_preprocessed.csv are out of sync; "
                           "re-run preprocessing.ipynb")
    return Patient.from_synthea(pid, raw, feats, row=row)


def timeseries_patient(pid: str, patient_id: str) -> Patient:
    """All time steps of ``patient_id`` (e.g. ``"P0001"``) from the DL dataset."""
    ts = timeseries()
    rows = ts[ts["Patient_ID"] == patient_id]
    if rows.empty:
        raise KeyError(patient_id)
    return Patient.from_timeseries(pid, rows)


def sample_rows(risk_class: str, n: int, seed: int) -> list[int]:
    """Random Synthea encounter rows whose label is ``risk_class``."""
    raw = synthea_raw()
    idx = raw.index[raw["risk_class"] == risk_class]
    return [int(i) for i in pd.Series(idx).sample(n=n, random_state=seed)]


def sample_timeseries_ids(risk: str, n: int, seed: int, split: str = "test") -> list[str]:
    """Random time-series patients with label ``risk``, from the held-out split
    by default so the DL model is not shown its own training data."""
    ts = timeseries()
    first = ts.groupby("Patient_ID").first()
    ids = first.index[(first["Risk"] == risk) & (first["Split"] == split)]
    return list(pd.Series(ids).sample(n=n, random_state=seed))


def combined_patient(pid: str, risk_class: str, seed: int) -> Patient:
    """A Synthea encounter paired with a time-series patient of the same label.

    The two datasets have no patient in common, so the pairing is simulated:
    it gives the agent one patient with both a medical history (ML input) and
    an hourly vitals timeline (DL input).
    """
    row = sample_rows(risk_class, 1, seed)[0]
    ts_id = sample_timeseries_ids(risk_class, 1, seed)[0]
    return synthea_patient(pid, row).merge_timeseries(timeseries_patient(pid, ts_id))
