"""Export the predicate contract (docs/predicates.md) for Search and Planning.

``requirements(kb)`` returns plain JSON-ready data, so Block 4 never has to
call Prolog itself. ``diff(old, new)`` lists, per patient, which contract
predicates appeared or disappeared: the signal Block 6 uses to re-run Search
and Planning after the environment changes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, TYPE_CHECKING

from reasoning.printer import to_pdf

if TYPE_CHECKING:
    from reasoning.kb import KnowledgeBase

# Every contract predicate, with X standing for the patient.
CONTRACT_PATTERNS = (
    "risk_level(X,L)",
    "priority(X,N)",
    "max_wait(X,M)",
    "requires_urgent_evaluation(X)",
    "requires_reassessment(X)",
    "insufficient_data(X)",
    "requires_icu(X)",
    "requires_emergency_bed(X)",
    "requires_exam_room(X)",
    "requires_specialist(X,S)",
    "requires_doctor(X,S)",
    "requires_nurse(X)",
    "requires_equipment(X,T)",
    "requires_monitoring(X,M)",
    "can_treat(D,X)",
    "suitable_bed(B,X)",
    "suitable_equipment(E,X)",
    "can_nurse(N,X)",
    "fallback_doctor(D,X)",
    "fallback_bed(B,X)",
)


def _rows(kb: "KnowledgeBase", pattern: str) -> list[dict[str, str]]:
    return [{k: to_pdf(str(v)) for k, v in row.items()} for row in kb.solutions(pattern)]


def contract_facts(kb: "KnowledgeBase") -> dict[str, set[str]]:
    """Patient -> set of contract facts in PDF notation, e.g. 'RequiresICU(P1)'."""
    out: dict[str, set[str]] = defaultdict(set)
    for pattern in CONTRACT_PATTERNS:
        for row in kb.solutions(pattern):
            ground = pattern
            for var, value in row.items():
                ground = ground.replace(var, str(value), 1) if var != "X" else ground
            ground = ground.replace("X", str(row["X"]))
            out[to_pdf(row["X"])].add(to_pdf(ground))
    return out


def requirements(kb: "KnowledgeBase") -> dict[str, Any]:
    patients: dict[str, dict[str, Any]] = {}
    for pid in kb.patients():
        patients[pid] = {
            "risk": None, "priority": None, "max_wait": None, "waiting": False,
            "urgent_evaluation": False, "reassess": False, "insufficient_data": False,
            "bed": None, "specialist": None, "doctor": None, "nurse": False,
            "equipment": [], "monitoring": None, "assigned": [],
            "candidates": {"doctors": [], "beds": [], "nurses": [], "equipment": {},
                           "fallback_doctors": [], "fallback_beds": []},
        }

    def each(pattern: str):
        for r in _rows(kb, pattern):
            if r["X"] in patients:
                yield patients[r["X"]], r

    for p, r in each("risk_level(X,L)"):
        p["risk"] = r["L"]
    for p, r in each("priority(X,N)"):
        p["priority"] = int(r["N"])
    for p, r in each("max_wait(X,M)"):
        p["max_wait"] = int(r["M"])
    for p, _ in each("waiting(X)"):
        p["waiting"] = True
    for p, _ in each("requires_urgent_evaluation(X)"):
        p["urgent_evaluation"] = True
    for p, _ in each("requires_reassessment(X)"):
        p["reassess"] = True
    for p, _ in each("insufficient_data(X)"):
        p["insufficient_data"] = True
    for p, r in each("needs_bed_type(X,T)"):
        p["bed"] = r["T"]
    for p, r in each("requires_specialist(X,S)"):
        p["specialist"] = r["S"]
    for p, r in each("requires_doctor(X,S)"):
        p["doctor"] = r["S"]
    for p, _ in each("requires_nurse(X)"):
        p["nurse"] = True
    for p, r in each("requires_equipment(X,T)"):
        p["equipment"].append(r["T"])
    for p, r in each("requires_monitoring(X,M)"):
        p["monitoring"] = r["M"]
    for p, r in each("assigned(X,R)"):
        p["assigned"].append(r["R"])

    for p, r in each("can_treat(D,X)"):
        p["candidates"]["doctors"].append(r["D"])
    for p, r in each("suitable_bed(B,X)"):
        p["candidates"]["beds"].append(r["B"])
    for p, r in each("can_nurse(N,X)"):
        p["candidates"]["nurses"].append(r["N"])
    for p, r in each("fallback_doctor(D,X)"):
        p["candidates"]["fallback_doctors"].append(r["D"])
    for p, r in each("fallback_bed(B,X)"):
        p["candidates"]["fallback_beds"].append(r["B"])
    types = {r["E"]: r["T"] for r in _rows(kb, "equipment_type(E,T)")}
    for p, r in each("suitable_equipment(E,X)"):
        p["candidates"]["equipment"].setdefault(types[r["E"]], []).append(r["E"])

    for p in patients.values():
        p["equipment"].sort()
        p["assigned"].sort(key=_rid_key)
        for k, v in p["candidates"].items():
            if isinstance(v, list):
                v.sort(key=_rid_key)

    return {
        "patients": patients,
        "resources": _available_resources(kb),
        "shortages": _shortages(patients, kb),
        "contract": {pid: sorted(f) for pid, f in contract_facts(kb).items()},
    }


def _rid_key(rid: str):
    head = rid.rstrip("0123456789")
    tail = rid[len(head):]
    return (head, int(tail) if tail else 0)


def _available_resources(kb: "KnowledgeBase") -> dict[str, list[dict[str, str]]]:
    avail = {r["R"] for r in _rows(kb, "available(R)")}
    holder = {r["R"]: r["X"] for r in _rows(kb, "assigned(X,R)")}
    spec = {r["D"]: r["S"] for r in _rows(kb, "specialty(D,S)")}
    btype = {r["B"]: r["T"] for r in _rows(kb, "type(B,T)")}
    etype = {r["E"]: r["T"] for r in _rows(kb, "equipment_type(E,T)")}

    def ids(pattern: str, var: str) -> list[str]:
        return sorted((r[var] for r in _rows(kb, pattern)), key=_rid_key)

    return {
        "doctors": [{"id": d, "specialty": spec.get(d), "available": d in avail,
                     "assigned_to": holder.get(d)} for d in ids("doctor(D)", "D")],
        "nurses": [{"id": n, "available": n in avail, "assigned_to": holder.get(n)} for n in ids("nurse(N)", "N")],
        "beds": [{"id": b, "type": btype.get(b), "available": b in avail,
                  "assigned_to": holder.get(b)} for b in ids("bed(B)", "B")],
        "equipment": [{"id": e, "type": etype.get(e), "available": e in avail,
                       "assigned_to": holder.get(e)} for e in ids("equipment(E)", "E")],
    }


def _shortages(patients: dict[str, dict[str, Any]], kb: "KnowledgeBase") -> list[dict[str, Any]]:
    """Needs that more waiting patients share than there are free resources."""
    res = _available_resources(kb)
    free_beds: dict[str, int] = defaultdict(int)
    for b in res["beds"]:
        free_beds[b["type"]] += b["available"]
    need: dict[str, list[str]] = defaultdict(list)
    for pid, p in patients.items():
        if p["waiting"] and p["bed"]:
            need[p["bed"]].append(pid)
    out = []
    for bed_type, pids in sorted(need.items()):
        if len(pids) > free_beds[bed_type]:
            out.append({"resource": f"{bed_type} bed", "patients": pids, "available": free_beds[bed_type]})
    return out


def diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per patient: contract facts added and removed, and the risk change."""
    old_c = old.get("contract", {}) if old else {}
    new_c = new.get("contract", {})
    changes: dict[str, dict[str, Any]] = {}
    for pid in sorted(set(old_c) | set(new_c), key=lambda p: (len(p), p)):
        a, b = set(old_c.get(pid, [])), set(new_c.get(pid, []))
        if a == b:
            continue
        before = old.get("patients", {}).get(pid, {}).get("risk") if old else None
        after = new["patients"].get(pid, {}).get("risk")
        changes[pid] = {"added": sorted(b - a), "removed": sorted(a - b),
                        "risk": (before, after) if before != after else None}
    return changes
