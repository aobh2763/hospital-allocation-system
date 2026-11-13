"""Build a Scenario from the HLT-010 hospital dataset (data/hospital/).

What the dataset gives, per facility and day:

* beds:      ``hospital_resources.csv`` -> free ICU beds = icu_beds_x - icu_occupied,
             free general beds = total_beds - occupied_beds (used as emergency beds)
* staff:     ``staffing.csv`` -> who works the chosen shift, by role
* equipment: ``equipment.csv`` -> each asset, unavailable on a failure or
             unplanned downtime that day
* context:   surge_day_flag, diversion_flag

What it does not give, and how we fill it:

* Staff roles are perioperative (Surgeon, Anesthesiologist, Resident, PA/NP,
  RN Circulator, CRNA). There are no cardiologists, pulmonologists or
  neurologists, so a patient needing one gets ``FallbackDoctor`` options
  unless ``extra_specialists`` adds on-call specialists.
* There are no ED examination rooms: ``caps["exam_room"]`` sets how many.
* A real facility has hundreds of free beds; ``caps`` keeps the search problem
  small enough to explore (defaults below).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from environment import datasets
from environment.scenario import Resource, Scenario

ROLE_TO_SPECIALTY = {
    "Surgeon": "general_surgery",
    "Anesthesiologist": "anesthesiology",
    "Resident": "general",
    "PA/NP": "general",
}
NURSE_ROLES = ("RN Circulator", "CRNA")

EQUIPMENT_TYPES = {  # dataset class -> (our type, id prefix)
    "Ventilator": ("ventilator", "V"),
    "Patient Monitor": ("patient_monitor", "M"),
    "Infusion Pump": ("infusion_pump", "IP"),
    "Defibrillator": ("defibrillator", "DF"),
    "CT Scanner": ("ct_scanner", "CT"),
    "Portable X-Ray": ("portable_xray", "XR"),
    "Ultrasound Unit": ("ultrasound", "US"),
}

DEFAULT_CAPS = {
    "icu": 3,
    "emergency": 4,
    "exam_room": 2,
    "doctors_per_specialty": 2,
    "nurses": 4,
    "equipment_per_type": 2,
}

DEFAULT_MIX = ("Critical", "High", "Medium", "Low", "Medium", "Low")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(datasets.HOSPITAL / f"{name}.csv")


def facility_days() -> pd.DataFrame:
    """The (facility, date) pairs available, with their bed situation."""
    h = _read("hospital_resources")
    h["free_icu"] = h["icu_beds_x"] - h["icu_occupied"]
    h["free_beds"] = h["total_beds"] - h["occupied_beds"]
    return h[["facility_id", "census_date", "free_icu", "free_beds", "surge_day_flag", "diversion_flag"]]


def from_hospital_data(
    facility: str = "FAC0001",
    date: str = "2023-01-02",
    shift: str = "Day",
    caps: dict | None = None,
    patient_mix: Iterable[str] = DEFAULT_MIX,
    extra_specialists: Iterable[str] = (),
    seed: int = 0,
) -> Scenario:
    caps = {**DEFAULT_CAPS, **(caps or {})}
    notes: list[str] = []

    day = _read("hospital_resources")
    day = day[(day.facility_id == facility) & (day.census_date == date)]
    if day.empty:
        raise ValueError(f"no capacity record for {facility} on {date}")
    day = day.iloc[0]

    resources: list[Resource] = []

    # ---- beds
    free_icu = int(day.icu_beds_x - day.icu_occupied)
    free_beds = int(day.total_beds - day.occupied_beds)
    n_icu, n_eb = min(free_icu, caps["icu"]), min(free_beds, caps["emergency"])
    resources += [Resource(f"ICU{i}", "bed", "icu", source={"facility": facility}) for i in range(1, n_icu + 1)]
    resources += [Resource(f"EB{i}", "bed", "emergency", source={"facility": facility}) for i in range(1, n_eb + 1)]
    resources += [Resource(f"EX{i}", "bed", "exam_room", source={"assumed": True})
                  for i in range(1, caps["exam_room"] + 1)]
    notes.append(f"{facility} {date}: {free_icu} free ICU beds, {free_beds} free beds "
                 f"(kept {n_icu} ICU, {n_eb} emergency); {caps['exam_room']} exam rooms assumed")

    # ---- staff on shift
    staff = _read("staffing")
    staff = staff[(staff.facility_id == facility) & (staff.shift_date == date) & (staff.shift_type == shift)]
    staff = staff.drop_duplicates("staff_id").sort_values("staff_id")
    d = 0
    for role, specialty in ROLE_TO_SPECIALTY.items():
        taken = sum(1 for r in resources if r.kind == "doctor" and r.subtype == specialty)
        for sid in staff.loc[staff.staff_role == role, "staff_id"]:
            if taken >= caps["doctors_per_specialty"]:
                break
            d += 1
            taken += 1
            resources.append(Resource(f"D{d}", "doctor", specialty, source={"staff_id": sid, "role": role}))
    for specialty in extra_specialists:
        d += 1
        resources.append(Resource(f"D{d}", "doctor", specialty, source={"on_call": True}))
        notes.append(f"D{d}: on-call {specialty} added (not in the dataset)")
    nurses = staff.loc[staff.staff_role.isin(NURSE_ROLES), "staff_id"].head(caps["nurses"])
    resources += [Resource(f"N{i}", "nurse", None, source={"staff_id": sid}) for i, sid in enumerate(nurses, 1)]

    # ---- equipment
    eq = _read("equipment")
    eq = eq[(eq.facility_id == facility) & (eq.record_date == date)].sort_values("asset_id")
    for cls, (etype, prefix) in EQUIPMENT_TYPES.items():
        assets = eq[eq.equipment_class == cls].head(caps["equipment_per_type"])
        for i, (_, a) in enumerate(assets.iterrows(), 1):
            down = bool(a.failure_flag) or bool(a.unplanned_downtime_flag)
            resources.append(Resource(f"{prefix}{i}", "equipment", etype, available=not down,
                                      source={"asset_id": a.asset_id}))

    # ---- patients: Synthea encounter + time series of the same label
    patients = [datasets.combined_patient(f"P{i}", risk, seed + i)
                for i, risk in enumerate(patient_mix, 1)]

    return Scenario(
        name=f"{facility} {date} {shift} shift",
        patients=patients,
        resources=resources,
        surge_day=bool(day.surge_day_flag),
        diversion=bool(day.diversion_flag),
        notes=notes,
    )
