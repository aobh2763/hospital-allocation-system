"""Block 3 -> Block 2: run the ML and DL models and return (level, probability).

ML  (prediction/ml/models/hgb_model.joblib, the best model in the report)
    Input: one snapshot, the 55 Synthea features of patient_preprocessed.csv.
    Synthea patients bring their exact feature row. Patients typed into the
    interface only have age, sex, a few vitals and conditions: the other
    features are filled with the training medians, so their ML evidence is
    weaker. That limitation is reported in ``MLAdapter.coverage``.

DL  (GRU from prediction/dl/dl_risk.py, saved by prediction/dl/export_model.py)
    Run with NumPy (reasoning/numpy_rnn.py), so PyTorch is only needed to train.
    Input: the hourly vitals timeline HR, MAP, SpO2, Temp, RespRate, Lactate.
    Shorter timelines are padded with the last step, missing values filled
    with the training medians, as dl_risk.predirePatient does
    (but keeping the most recent T hours when the timeline is longer).

Both return the class with the highest probability, as the PDF's
``ML_Risk(P1, High, 0.87)`` and ``DL_Risk(P1, Critical, 0.91)``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from patient import Patient, SYNTHEA_CONDITIONS
from reasoning.numpy_rnn import NumpyRNN, load_state_dict

ROOT = Path(__file__).resolve().parents[2]
ML_MODEL = ROOT / "prediction" / "ml" / "models" / "hgb_model.joblib"
DL_DIR = ROOT / "prediction" / "dl"
DL_MODEL = DL_DIR / "models" / "gru.pt"
DL_PREP = DL_DIR / "models" / "gru_prep.json"

LEVELS = ["Low", "Medium", "High", "Critical"]   # class index 0..3 in both models
DL_FEATURES = ["hr", "map", "spo2", "temp", "rr", "lactate"]  # = dl_risk.FEATURES order


class MLAdapter:
    def __init__(self, model_path: Path = ML_MODEL) -> None:
        import joblib

        self.model = joblib.load(model_path)
        self.features = list(self.model.feature_names_in_)
        self.coverage: dict[str, float] = {}

    @staticmethod
    @lru_cache(maxsize=1)
    def _medians() -> dict[str, float]:
        from environment.datasets import synthea_features

        return synthea_features().median(numeric_only=True).to_dict()

    def feature_row(self, p: Patient) -> pd.DataFrame:
        if p.ml_features:
            row = {f: p.ml_features.get(f) for f in self.features}
            self.coverage[p.pid] = 1.0
        else:
            med = self._medians()
            row = {f: med.get(f, 0.0) for f in self.features}
            known = {}
            if p.age is not None:
                known["age_years"] = int(p.age)
                known["age_months"] = int(round((p.age - int(p.age)) * 12))
            if p.sex in ("m", "f"):
                known["gender_m"], known["gender_f"] = int(p.sex == "m"), int(p.sex == "f")
            for key, col in (("hr", "obs_Heart_rate"), ("sbp", "obs_Systolic_Blood_Pressure"),
                             ("dbp", "obs_Diastolic_Blood_Pressure"), ("rr", "obs_Respiratory_rate")):
                if key in p.vitals:
                    known[col] = p.vitals[key]
            for col, name in SYNTHEA_CONDITIONS.items():
                known[col] = int(name in p.conditions)
            row.update({k: v for k, v in known.items() if k in row})
            self.coverage[p.pid] = len([k for k in known if k in row]) / len(self.features)
        return pd.DataFrame([row], columns=self.features)

    def probabilities(self, p: Patient) -> dict[str, float]:
        """Probability of every risk level, e.g. {"Low": 0.10, ..., "Critical": 0.49}."""
        proba = self.model.predict_proba(self.feature_row(p))[0]
        return {LEVELS[int(c)]: float(pr) for c, pr in zip(self.model.classes_, proba)}

    def predict(self, p: Patient) -> tuple[str, float]:
        probs = self.probabilities(p)
        level = max(probs, key=probs.get)
        return level, probs[level]


def dl_unavailable_reason(model_path: Path = DL_MODEL, prep_path: Path = DL_PREP) -> str | None:
    """Why the DL model cannot run here, or None if it can.

    PyTorch is not needed: the weights are read and run with NumPy
    (reasoning/numpy_rnn.py). Only the trained files must exist.
    """
    for f in (model_path, prep_path):
        if not f.exists():
            return f"{f.name} not found (someone with PyTorch runs `python prediction/dl/export_model.py` once)"
    return None


class DLAdapter:
    def __init__(self, model_path: Path = DL_MODEL, prep_path: Path = DL_PREP) -> None:
        reason = dl_unavailable_reason(model_path, prep_path)
        if reason:
            raise RuntimeError(f"DL model unavailable: {reason}")
        prep = json.loads(prep_path.read_text())
        self.T = int(prep["T"])
        self.med = np.array(prep["med"], dtype=np.float32)
        self.mu = np.array(prep["mu"], dtype=np.float32)
        self.sd = np.array(prep["sd"], dtype=np.float32)
        self.kind = prep.get("kind", "GRU")
        self.model = NumpyRNN(load_state_dict(model_path), self.kind)

    def sequence(self, p: Patient) -> np.ndarray | None:
        if not p.history:
            return None
        a = np.array([[step.get(k, np.nan) for k in DL_FEATURES] for step in p.history], dtype=np.float32)
        a = pd.DataFrame(a).ffill().to_numpy(dtype=np.float32)     # carry last measurement forward
        a = a[-self.T:]                                              # most recent T hours
        if len(a) < self.T:
            a = np.vstack([a, np.repeat(a[-1:], self.T - len(a), axis=0)])
        a = np.where(np.isnan(a), self.med, a)
        return (a - self.mu) / self.sd

    def probabilities(self, p: Patient) -> dict[str, float] | None:
        """Probability of every risk level, or None when the patient has no timeline."""
        x = self.sequence(p)
        if x is None:
            return None
        proba = self.model.predict_proba(x[None, :, :])[0]
        return {level: float(pr) for level, pr in zip(LEVELS, proba)}

    def predict(self, p: Patient) -> tuple[str, float] | None:
        probs = self.probabilities(p)
        if probs is None:
            return None
        level = max(probs, key=probs.get)
        return level, probs[level]


def run_models(kb, patients: list[Patient], ml: MLAdapter | None, dl: DLAdapter | None,
               skip: dict[str, dict] | None = None) -> dict[str, dict[str, tuple[str, float]]]:
    """Predict for each patient and insert ML_Risk / DL_Risk facts into the KB.

    ``skip`` maps pid -> sources already provided (e.g. predictions fixed in a
    scenario file); those are left untouched.
    """
    skip = skip or {}
    out: dict[str, dict[str, tuple[str, float]]] = {}
    for p in patients:
        out[p.pid] = {}
        if ml and "ml" not in skip.get(p.pid, {}):
            level, prob = ml.predict(p)
            kb.set_prediction(p.pid, "ml", level, prob)
            out[p.pid]["ml"] = (level, prob)
        if dl and "dl" not in skip.get(p.pid, {}):
            r = dl.predict(p)
            if r:
                kb.set_prediction(p.pid, "dl", *r)
                out[p.pid]["dl"] = r
    return out
