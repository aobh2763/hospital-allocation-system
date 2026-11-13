"""Block 6: the integrated intelligent agent.

One cycle of deliberation chains every block:

    Block 1  environment     the scenario: patients, resources, assignments
    Block 3  learning        ML_Risk / DL_Risk computed for new or changed patients
    Block 2  reasoning       facts + rules -> risk levels and RequiresX (forward chaining)
    Block 4  search          A* chooses the allocation (UCS run alongside for comparison)
    Block 5  planning        STRIPS planner orders the actions; the validator replays them

Then the agent carries the plan out one action at a time. Before each action
it looks at the world. If something changed (``observe``), it checks whether
the rest of the plan is still executable and what the change does to the
patients' requirements, and deliberates again instead of executing an obsolete
plan.

Execution model
---------------
* Reserving a resource for a waiting patient is **tentative**: at each new
  deliberation those reservations are released, so Search can give a bed or a
  doctor to a more urgent patient who just arrived.
* ``TransferPatient`` is the commitment point: the patient is now under
  treatment and keeps their resources. Search can only move them out of a bed
  they no longer need (relocation).
* Evaluations already performed are remembered (``Memory``) and not repeated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from environment.scenario import Scenario
from planning import Memory, build, first_invalid, validate
from planning.strips import Action, PlanResult
from reasoning.adapters import DLAdapter, MLAdapter, dl_unavailable_reason, run_models
from reasoning.events import apply_event
from reasoning.kb import KnowledgeBase
from reasoning.printer import fact, to_atom
from search import Decision, allocate


@dataclass
class Cycle:
    """Everything one deliberation produced, for display and for the report."""
    number: int
    reasons: list[str]                       # why the agent deliberated
    invalid_step: tuple | None               # (index, action, missing preconditions) of the old plan
    changes: dict[str, dict[str, Any]]       # requirement changes caused by the observed events
    predictions: dict[str, dict]             # new ML/DL predictions this cycle
    requirements: dict[str, Any]
    decision: Decision
    ucs: Decision | None
    plan: PlanResult
    plan_errors: list[str]
    allocation_changes: list[str]
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass
class StepResult:
    kind: str                                # "executed", "replanned", "done"
    action: Action | None = None
    cycle: Cycle | None = None


class Agent:
    def __init__(self, scenario: Scenario, use_models: bool = True, compare_ucs: bool = True) -> None:
        self.scenario = scenario
        self.use_models = use_models
        self.compare_ucs = compare_ucs
        self.ml = self.dl = None
        if use_models:
            scenario.predictions = {}
            self.ml = MLAdapter()
            self.dl = None if dl_unavailable_reason() else DLAdapter()
        self.kb = KnowledgeBase()
        self.kb.load_scenario(scenario)
        self.memory = Memory()
        held = {rid for _, rid in scenario.assignments}
        self.out_of_service = {r.rid for r in scenario.resources if not r.available and r.rid not in held}
        self.plan: list[Action] = []
        self.world: frozenset[str] = frozenset()
        self.goals: frozenset[str] = frozenset()
        self.cycles: list[Cycle] = []
        self.executed: list[tuple[int, Action]] = []          # (cycle number, action)
        self._pending: list[str] = []
        self._pending_changes: dict[str, dict[str, Any]] = {}
        self._new_predictions: dict[str, dict] = {}
        self._decision: Decision | None = None
        if use_models:
            self._new_predictions = run_models(self.kb, scenario.patients, self.ml, self.dl)
        self.kb.update()

    # ------------------------------------------------------------------ perception

    def observe(self, event: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Apply an environment change and remember it until the next step.
        Returns the requirement changes it caused (Block 2's diff)."""
        kind = event["type"]
        desc = event.get("description") or _describe(event)
        result = None
        if kind == "resource_unavailable":
            self._take_out_of_service(event["id"])
        elif kind == "resource_available":
            self.out_of_service.discard(event["id"])
            if not self._holder(event["id"]):
                self.kb.set_available(event["id"], True)
        elif kind == "discharge":
            self._discharge(event["pid"])
        elif kind in ("new_patient", "new_measurement"):
            result = apply_event(self.kb, event, self.ml, self.dl, use_given=not self.use_models)
            pid = event["patient"]["pid"] if kind == "new_patient" else event["pid"]
            self._new_predictions[pid] = self._predictions_of(pid)
        else:
            raise ValueError(f"unknown event type {kind!r}")
        result = result or self.kb.update()
        for pid, c in result.changes.items():
            merged = self._pending_changes.setdefault(pid, {"added": [], "removed": [], "risk": None})
            merged["added"] += c["added"]
            merged["removed"] += c["removed"]
            if c["risk"]:
                merged["risk"] = c["risk"] if not merged["risk"] else (merged["risk"][0], c["risk"][1])
        self._pending.append(desc)
        return result.changes

    # ------------------------------------------------------------------ deliberation

    def deliberate(self, reasons: list[str] | None = None) -> Cycle:
        """Reason (Block 2 with Block 3's evidence), search (Block 4), plan (Block 5)."""
        t = {}
        invalid = first_invalid(self.world, self.plan) if self.plan else None
        t0 = time.perf_counter()
        previous = self._release_tentative()
        self.kb.update()
        req = self.kb.requirements()
        t["reasoning"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        decision = allocate(req, "astar")
        t["search"] = time.perf_counter() - t0
        ucs = allocate(req, "ucs") if self.compare_ucs else None

        # reservations the new allocation keeps are not redone
        kept = previous & _pairs(decision.allocation)
        for pid, rid in kept:
            self.kb.add(fact("assigned", pid, rid))
            self.kb.set_available(rid, False)
        if kept:
            self.kb.update()

        t0 = time.perf_counter()
        problem = build(req, decision.allocation, self.memory, kept)
        result = problem.solve()
        errors = validate(problem.init, result.steps, problem.goals) if result.found else ["no plan found"]
        t["planning"] = time.perf_counter() - t0

        cycle = Cycle(len(self.cycles) + 1, reasons or self._pending or ["initial situation"], invalid,
                      self._pending_changes, self._new_predictions, req, decision, ucs, result, errors,
                      _allocation_changes(self._decision, decision), t)
        self.cycles.append(cycle)
        self.plan, self.world, self.goals = list(result.steps), problem.init, problem.goals
        self._decision = decision
        self._pending, self._pending_changes, self._new_predictions = [], {}, {}
        return cycle

    # ------------------------------------------------------------------ execution

    def step(self) -> StepResult:
        """Deliberate if the world changed, otherwise execute the next action."""
        if self._pending or not self.cycles:
            return StepResult("replanned", cycle=self.deliberate())
        if not self.plan:
            return StepResult("done")
        action = self.plan[0]
        missing = action.pre - self.world
        if missing:                                   # should not happen: the world only changes through observe
            return StepResult("replanned", cycle=self.deliberate([f"{action} cannot run: {', '.join(sorted(missing))}"]))
        self.plan.pop(0)
        self.world = action.apply(self.world)
        self._execute(action)
        self.executed.append((len(self.cycles), action))
        return StepResult("executed", action)

    def run(self, max_steps: int = 1000, until: Callable[[Action], bool] | None = None,
            on_step: Callable[[StepResult], None] | None = None) -> list[StepResult]:
        """Step until the plan is done, ``max_steps`` actions ran, or the next
        action satisfies ``until`` (that action is not executed)."""
        out = []
        while len([r for r in out if r.kind == "executed"]) < max_steps:
            if until and self.plan and not self._pending and until(self.plan[0]):
                break
            r = self.step()
            out.append(r)
            if on_step:
                on_step(r)
            if r.kind == "done":
                break
        return out

    # ------------------------------------------------------------------ helpers

    def _execute(self, a: Action) -> None:
        kb = self.kb
        if a.name in ("AssignDoctor", "AssignNurse", "ReserveICUBed", "ReserveBed", "AllocateEquipment"):
            pid, rid = a.args
            kb.add(fact("assigned", pid, rid))
            kb.set_available(rid, False)
        elif a.name == "Relocate":
            q, b, b2 = a.args
            kb.remove(fact("assigned", q, b))
            kb.add(fact("assigned", q, b2))
            kb.set_available(b2, False)
            if b not in self.out_of_service:
                kb.set_available(b, True)
        elif a.name == "PerformUrgentEvaluation":
            self.memory.evaluated.add(a.args[0])
        elif a.name == "TransferPatient":
            pid = a.args[0]
            kb.remove(fact("waiting", pid))
            kb.add(fact("under_treatment", pid))
            self.scenario.patient(pid).status = "under_treatment"
        elif a.name == "StartMonitoring":
            self.memory.monitored.add(a.args[0])
        elif a.name == "Reevaluate":
            self.memory.reassessed.add(a.args[0])
        kb.update()          # keep Block 2's view current, so an event's diff shows only the event

    def _holder(self, rid: str) -> str | None:
        rows = self.kb.solutions(f"assigned(X,{to_atom(rid)})")
        return str(rows[0]["X"]).upper() if rows else None

    def _release_tentative(self) -> set[tuple[str, str]]:
        """Reservations of patients not yet transferred are released before
        re-planning, so Search may give those resources to someone else.
        Returns the released (patient, resource) pairs."""
        released = set()
        for row in self.kb.solutions("assigned(X,R)"):
            pid, rid = str(row["X"]).upper(), str(row["R"]).upper()
            if self.kb.holds(f"Waiting({pid})"):
                self.kb.remove(fact("assigned", pid, rid))
                if rid not in self.out_of_service:
                    self.kb.set_available(rid, True)
                    released.add((pid, rid))
        return released

    def _take_out_of_service(self, rid: str) -> None:
        self.out_of_service.add(rid)
        holder = self._holder(rid)
        if holder:
            self.kb.remove(fact("assigned", holder, rid))
            if self.kb.holds(f"UnderTreatment({holder})") and self.scenario.resource(rid).kind == "bed":
                # lost their bed: back to the waiting list
                self.kb.remove(fact("under_treatment", holder))
                self.kb.add(fact("waiting", holder))
                self.scenario.patient(holder).status = "waiting"
                self.memory.monitored.discard(holder)
        self.kb.set_available(rid, False)
        # anything the old plan built on this resource is gone
        self.world = frozenset(x for x in self.world if f", {rid})" not in x and x != f"Available({rid})")

    def _discharge(self, pid: str) -> None:
        for row in self.kb.solutions(f"assigned({to_atom(pid)},R)"):
            rid = str(row["R"]).upper()
            if rid not in self.out_of_service:
                self.kb.set_available(rid, True)
        self.kb.remove_patient(pid)
        self.scenario.patients = [p for p in self.scenario.patients if p.pid != pid]
        for s in (self.memory.evaluated, self.memory.monitored, self.memory.reassessed):
            s.discard(pid)

    def _predictions_of(self, pid: str) -> dict:
        out = {}
        for src in ("ml", "dl"):
            rows = self.kb.query(f"{src.upper()}_Risk({pid}, l, p)")
            if rows:
                out[src] = (rows[0]["l"], float(rows[0]["p"]))
        return out


def _describe(event: dict[str, Any]) -> str:
    kind = event["type"]
    if kind == "resource_unavailable":
        return f"{event['id']} is out of service"
    if kind == "resource_available":
        return f"{event['id']} is back in service"
    if kind == "new_patient":
        return f"new patient {event['patient']['pid']} arrives"
    if kind == "new_measurement":
        return f"new measurements for {event['pid']}"
    if kind == "discharge":
        return f"{event['pid']} is discharged"
    return kind


def _pairs(allocation: dict[str, Any]) -> set[tuple[str, str]]:
    out = set()
    for pid, a in allocation.get("patients", {}).items():
        for rid in (a["bed"], a["doctor"], a["nurse"], *a["equipment"].values()):
            if rid:
                out.add((pid, rid))
    return out


def _allocation_changes(old: Decision | None, new: Decision) -> list[str]:
    """What the new decision changes for each patient compared with the previous one."""
    if old is None:
        return []
    out = []
    a, b = old.allocation.get("patients", {}), new.allocation.get("patients", {})
    for pid in sorted(set(a) | set(b), key=lambda p: (len(p), p)):
        if pid not in b:
            if pid in new.allocation.get("waiting", []):
                out.append(f"{pid}: now left waiting")
            continue
        if pid not in a:
            out.append(f"{pid}: now served (bed {b[pid]['bed']}, doctor {b[pid]['doctor']})")
            continue
        for key in ("bed", "doctor", "nurse"):
            if a[pid][key] != b[pid][key]:
                out.append(f"{pid}: {key} {a[pid][key]} -> {b[pid][key]}")
        if a[pid]["equipment"] != b[pid]["equipment"]:
            out.append(f"{pid}: equipment {a[pid]['equipment']} -> {b[pid]['equipment']}")
    old_moves = {(m["patient"], m["to"]) for m in old.allocation.get("relocations", [])}
    for m in new.allocation.get("relocations", []):
        if (m["patient"], m["to"]) not in old_moves:
            out.append(f"{m['patient']}: relocated {m['from']} -> {m['to']}")
    return out
