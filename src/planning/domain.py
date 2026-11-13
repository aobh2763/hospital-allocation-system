"""Block 5: the hospital planning domain.

Search (Block 4) chose a configuration: which bed, doctor, nurse and equipment
each patient gets, and who must be relocated. Planning finds an ordered
sequence of actions that reaches it from the current world, respecting each
action's preconditions. Actions, as in the PDF:

    AssignDoctor(P, D)            pre  Waiting(P), Available(D)
                                  add  DoctorAssigned(P, D)          del  Available(D)
    ReserveICUBed(P, B)           pre  Waiting(P), CriticalRisk(P), Type(B, ICU), Available(B)
                                  add  BedReserved(P, B)             del  Available(B)
    ReserveBed(P, B)              pre  Waiting(P), Available(B)       (emergency bed, exam room)
                                  add  BedReserved(P, B)             del  Available(B)
    AssignNurse(P, N)             pre  Waiting(P), Available(N)
                                  add  NurseAssigned(P, N)           del  Available(N)
    AllocateEquipment(P, E)       pre  Waiting(P), Available(E)
                                  add  EquipmentAllocated(P, E)      del  Available(E)
    Relocate(Q, B, B2)            pre  UnderTreatment(Q), In(Q, B), Available(B2)
                                  add  In(Q, B2), Available(B)       del  In(Q, B), Available(B2)
    PerformUrgentEvaluation(P, D) pre  DoctorAssigned(P, D), RequiresUrgentEvaluation(P)
                                  add  Evaluated(P)
    TransferPatient(P, B)         pre  Waiting(P), BedReserved(P, B), DoctorAssigned(P, D),
                                       [NurseAssigned(P, N)], [Evaluated(P) if urgent]
                                  add  In(P, B), UnderTreatment(P)   del  Waiting(P), BedReserved(P, B)
    StartMonitoring(P, M)         pre  In(P, B), EquipmentAllocated(P, M)        (M a patient monitor)
                                  add  Monitored(P)
    Reevaluate(P, D)              pre  In(P, B), DoctorAssigned(P, D), [Monitored(P)]
                                  add  Reassessed(P)       (Critical patients and RequiresReassessment)

So the PDF's ordering constraints hold by construction: a patient cannot be
transferred before their bed is reserved, a busy doctor cannot be assigned
(``Available(D)`` was deleted), and an ICU bed occupied by someone else is only
available after that patient has been relocated.

Goals, per patient Search serves: In(P, B), DoctorAssigned(P, D), the nurse and
every piece of equipment, Evaluated(P) when urgent, Monitored(P) when a monitor
is allocated, Reassessed(P) when Critical or flagged for reassessment; and
In(Q, B2) for each relocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from planning.strips import Action

PRIORITY = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
WEIGHT = {"Critical": 1000, "High": 100, "Medium": 10, "Low": 1}
RANK = {"Relocate": 0, "AssignDoctor": 1, "ReserveICUBed": 2, "ReserveBed": 2, "AssignNurse": 3,
        "AllocateEquipment": 4, "PerformUrgentEvaluation": 5, "TransferPatient": 6,
        "StartMonitoring": 7, "Reevaluate": 8}


@dataclass
class Memory:
    """What the agent has already done that the knowledge base does not record."""
    evaluated: set[str] = field(default_factory=set)
    monitored: set[str] = field(default_factory=set)
    reassessed: set[str] = field(default_factory=set)


@dataclass
class PlanningProblem:
    init: frozenset[str]
    goals: frozenset[str]
    actions: list[Action]
    weights: dict[str, float] = field(default_factory=dict)   # goal -> weight in the heuristic

    def solve(self, limit: int = 50_000):
        from planning.strips import plan

        return plan(self.init, self.goals, self.actions, limit, self.weights)


def world_state(req: dict[str, Any], memory: Memory) -> frozenset[str]:
    """Current facts the planner works from, built from Block 2's contract."""
    s: set[str] = set()
    res = req["resources"]
    for kind in ("doctors", "nurses", "beds", "equipment"):
        for r in res[kind]:
            if r["available"]:
                s.add(f"Available({r['id']})")
    for b in res["beds"]:
        if b["type"] == "ICU":
            s.add(f"Type({b['id']}, ICU)")
    held_atom = {**{r["id"]: "DoctorAssigned" for r in res["doctors"]},
                 **{r["id"]: "NurseAssigned" for r in res["nurses"]},
                 **{r["id"]: "EquipmentAllocated" for r in res["equipment"]},
                 **{r["id"]: "In" for r in res["beds"]}}
    for pid, p in req["patients"].items():
        if p["waiting"]:
            s.add(f"Waiting({pid})")
        else:
            s.add(f"UnderTreatment({pid})")
            for rid in p["assigned"]:
                s.add(f"{held_atom[rid]}({pid}, {rid})")
        if p["risk"] == "Critical":
            s.add(f"CriticalRisk({pid})")
        if p["urgent_evaluation"]:
            s.add(f"RequiresUrgentEvaluation({pid})")
        if p["reassess"]:
            s.add(f"RequiresReassessment({pid})")
        if pid in memory.evaluated:
            s.add(f"Evaluated({pid})")
        if pid in memory.monitored:
            s.add(f"Monitored({pid})")
        if pid in memory.reassessed:
            s.add(f"Reassessed({pid})")
    return frozenset(s)


def build(req: dict[str, Any], allocation: dict[str, Any], memory: Memory,
          kept: set[tuple[str, str]] = frozenset()) -> PlanningProblem:
    """``kept``: (patient, resource) reservations already made in an earlier
    plan and kept by the new allocation; they hold in the initial state."""
    init = set(world_state(req, memory))
    kinds = {r["id"]: k for k in ("doctors", "nurses", "beds", "equipment") for r in req["resources"][k]}
    held_atom = {"doctors": "DoctorAssigned", "nurses": "NurseAssigned", "beds": "BedReserved",
                 "equipment": "EquipmentAllocated"}
    for pid, rid in kept:
        init.discard(f"Available({rid})")
        init.add(f"{held_atom[kinds[rid]]}({pid}, {rid})")
    init = frozenset(init)
    actions: list[Action] = []
    goals: set[str] = set()
    owner: dict[str, str] = {}             # goal -> risk of the patient it serves
    icu_beds = {b["id"] for b in req["resources"]["beds"] if b["type"] == "ICU"}
    monitors = {e["id"] for e in req["resources"]["equipment"] if e["type"] == "PatientMonitor"}

    def act(name, args, pre, add, delete=(), patient="", risk="Low"):
        actions.append(Action(name, tuple(args), frozenset(pre), frozenset(add), frozenset(delete),
                              patient, PRIORITY.get(risk, 9), RANK[name]))

    beneficiary = {a["bed"]: pid for pid, a in allocation.get("patients", {}).items()}
    for m in allocation.get("relocations", []):
        q, b, b2 = m["patient"], m["from"], m["to"]
        for_pid = beneficiary.get(b, q)            # who the move is for (q itself when moving up)
        risk = req["patients"][for_pid]["risk"]
        act("Relocate", (q, b, b2), [f"UnderTreatment({q})", f"In({q}, {b})", f"Available({b2})"],
            [f"In({q}, {b2})", f"Available({b})"], [f"In({q}, {b})", f"Available({b2})"], for_pid, risk)
        goals.add(f"In({q}, {b2})")
        owner[f"In({q}, {b2})"] = risk

    for pid, a in allocation.get("patients", {}).items():
        p, risk = req["patients"][pid], a["risk"]
        bed, doc, nurse = a["bed"], a["doctor"], a["nurse"]
        eqs = list(a["equipment"].values())
        monitor = next((e for e in eqs if e in monitors), None)
        urgent = p["urgent_evaluation"]

        act("AssignDoctor", (pid, doc), [f"Waiting({pid})", f"Available({doc})"],
            [f"DoctorAssigned({pid}, {doc})"], [f"Available({doc})"], pid, risk)
        if bed in icu_beds:
            act("ReserveICUBed", (pid, bed),
                [f"Waiting({pid})", f"CriticalRisk({pid})", f"Type({bed}, ICU)", f"Available({bed})"],
                [f"BedReserved({pid}, {bed})"], [f"Available({bed})"], pid, risk)
        else:
            act("ReserveBed", (pid, bed), [f"Waiting({pid})", f"Available({bed})"],
                [f"BedReserved({pid}, {bed})"], [f"Available({bed})"], pid, risk)
        transfer_pre = [f"Waiting({pid})", f"BedReserved({pid}, {bed})", f"DoctorAssigned({pid}, {doc})"]
        if nurse:
            act("AssignNurse", (pid, nurse), [f"Waiting({pid})", f"Available({nurse})"],
                [f"NurseAssigned({pid}, {nurse})"], [f"Available({nurse})"], pid, risk)
            transfer_pre.append(f"NurseAssigned({pid}, {nurse})")
            goals.add(f"NurseAssigned({pid}, {nurse})")
        for e in eqs:
            act("AllocateEquipment", (pid, e), [f"Waiting({pid})", f"Available({e})"],
                [f"EquipmentAllocated({pid}, {e})"], [f"Available({e})"], pid, risk)
            goals.add(f"EquipmentAllocated({pid}, {e})")
        if urgent:
            if pid not in memory.evaluated:
                act("PerformUrgentEvaluation", (pid, doc),
                    [f"DoctorAssigned({pid}, {doc})", f"RequiresUrgentEvaluation({pid})"],
                    [f"Evaluated({pid})"], (), pid, risk)
            transfer_pre.append(f"Evaluated({pid})")
            goals.add(f"Evaluated({pid})")
        act("TransferPatient", (pid, bed), transfer_pre, [f"In({pid}, {bed})", f"UnderTreatment({pid})"],
            [f"Waiting({pid})", f"BedReserved({pid}, {bed})"], pid, risk)
        goals |= {f"In({pid}, {bed})", f"DoctorAssigned({pid}, {doc})"}
        reeval_pre = [f"In({pid}, {bed})", f"DoctorAssigned({pid}, {doc})"]
        if monitor:
            act("StartMonitoring", (pid, monitor), [f"In({pid}, {bed})", f"EquipmentAllocated({pid}, {monitor})"],
                [f"Monitored({pid})"], (), pid, risk)
            goals.add(f"Monitored({pid})")
            reeval_pre.append(f"Monitored({pid})")
        if risk == "Critical" or p["reassess"]:
            act("Reevaluate", (pid, doc), reeval_pre, [f"Reassessed({pid})"], (), pid, risk)
            goals.add(f"Reassessed({pid})")
        for g in goals:
            if f"({pid}," in g or f"({pid})" in g:
                owner.setdefault(g, risk)

    # unfinished follow-up of patients already under treatment
    moved_to = {m["patient"]: m["to"] for m in allocation.get("relocations", [])}
    kinds = {r["id"]: k for k in ("doctors", "beds", "equipment") for r in req["resources"][k]}
    for pid, p in req["patients"].items():
        if p["waiting"] or pid in allocation.get("patients", {}):
            continue
        held = {kinds.get(rid): rid for rid in p["assigned"] if kinds.get(rid) != "equipment"}
        bed = moved_to.get(pid, held.get("beds"))
        doc = held.get("doctors")
        monitor = next((rid for rid in p["assigned"] if rid in monitors), None)
        if not bed:
            continue
        risk = p["risk"]
        reeval_pre = [f"In({pid}, {bed})"]
        if monitor and pid not in memory.monitored:
            act("StartMonitoring", (pid, monitor), [f"In({pid}, {bed})", f"EquipmentAllocated({pid}, {monitor})"],
                [f"Monitored({pid})"], (), pid, risk)
            goals.add(f"Monitored({pid})")
            owner[f"Monitored({pid})"] = risk
        if monitor:
            reeval_pre.append(f"Monitored({pid})")
        if doc and (risk == "Critical" or p["reassess"]) and pid not in memory.reassessed:
            act("Reevaluate", (pid, doc), reeval_pre + [f"DoctorAssigned({pid}, {doc})"],
                [f"Reassessed({pid})"], (), pid, risk)
            goals.add(f"Reassessed({pid})")
            owner[f"Reassessed({pid})"] = risk

    weights = {g: WEIGHT[owner.get(g, "Low")] for g in goals}
    return PlanningProblem(init, frozenset(goals), actions, weights)


def describe(a: Action) -> str:
    """One action with its preconditions and effects, for traces."""
    eff = [f"+{x}" for x in sorted(a.add)] + [f"-{x}" for x in sorted(a.delete)]
    return f"{a}\n        pre: {', '.join(sorted(a.pre))}\n        eff: {', '.join(eff)}"
