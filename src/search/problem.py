"""Block 4: resource allocation as a state-space search problem.

Input: the Block 2 contract (``KnowledgeBase.requirements()``): each patient's
risk, priority and needs (bed type, specialist or general doctor, nurse,
equipment types), plus every resource with its type, availability and current
holder. Search never re-derives needs; it only decides who gets what.

Formulation
-----------
Needs are cut into **slots**, one per resource a patient needs:

    P1: bed(ICU)  doctor(Cardiology)  nurse  equipment(PatientMonitor) ...

Waiting patients are ordered by priority (Critical first, then High, Medium,
Low), and slots are filled in that fixed order. A fixed order turns the
problem into a tree of choices ("which resource fills this slot?") instead of
every permutation of the same assignments.

State          (pos, used, freed, moved)
               pos   = index of the next slot to fill
               used  = resources taken in this allocation
               freed = beds emptied by moving a patient under treatment
               moved = patients under treatment already moved
Initial state  (0, ∅, ∅, ∅)
Actions        ReserveBed(P, B), AssignDoctor(P, D), AssignNurse(P, N),
               AllocateEquipment(P, E)        fill the current slot
               Relocate(Q, B → B')            move a patient under treatment who
                                              no longer needs bed B (e.g. ICU) to a
                                              free bed of the type they now need,
                                              freeing B for the current slot
               LeaveWaiting(P)                Medium/Low only, before their first
                                              slot: skip that patient this round
Transition     pos + 1 (LeaveWaiting jumps to the next patient, Relocate keeps
               pos), the chosen resource joins ``used``
Goal test      every slot is decided, which means every High/Critical patient
               has every need covered (they have no LeaveWaiting action), and
               no resource is used twice (a used resource is never offered
               again). ``check_allocation`` re-verifies both on the result.

Cost function g(n): the sum of the action costs below. It reads as "how much
the allocation hurts patients and wastes scarce resources":

    every resource assigned                       1   using a resource is not free
    specialist given to a general-doctor patient  +2  specialists are scarce
    emergency bed for an exam-room patient        +2  emergency beds are scarce
    relocating a patient under treatment          4   disrupts a patient's care
    non-specialist for a specialist need          +6  fallback doctor (announced)
    emergency bed for an ICU need                 +10 fallback bed (announced)
    leaving a Medium patient waiting              8   over its 60-minute max wait
    leaving a Low patient waiting                 3   Low tolerates 120 minutes

The order of the numbers is what matters: serving a patient (2 to 4) is
cheaper than leaving them waiting (3 or 8), so free resources go to waiting
patients, Medium before Low. Relocating someone (4) is cheaper than putting a
Critical patient in an emergency bed (10), so an ICU bed is freed whenever
possible, as the team asked; the fallback is used only when no ICU bed can be
freed, and it is announced.

Heuristic h(n) for A*: a lower bound of the cost still to pay.

    each remaining slot of a High/Critical patient       ≥ 1
    each Medium/Low patient not started                  ≥ min(slots, waiting cost)
    ICU slots left beyond free ICU beds: each one costs  ≥ 4 (relocation) while
        movable patients remain, else ≥ 10 (fallback)
    specialist slots left beyond free specialists of
        that specialty: each one costs                   ≥ 6 (fallback doctor)
    Medium/Low patients beyond the beds or doctors that
        remain once High/Critical needs are covered:     at least that many wait,
        so add the cheapest of their extra waiting costs
    needs that must still be filled (High/Critical, or a
        patient already started) exceed every resource
        of that type that could still become free:       h = ∞, a dead end

Every term counts something the remaining actions must pay at least once, and
no cost is counted twice, so h never overestimates (admissible): A* returns an
optimal allocation, the same cost as Uniform Cost Search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

INF = float("inf")

COST_ASSIGN = 1
COST_SPECIALIST_FOR_GENERAL = 2
COST_EMERGENCY_FOR_EXAM = 2
COST_RELOCATE = 4
COST_FALLBACK_DOCTOR = 6
COST_FALLBACK_BED = 10
COST_WAIT = {"Medium": 8, "Low": 3}
# Only used when no valid allocation exists (relaxed search, result flagged invalid)
COST_WAIT_RELAXED = {"Critical": 100, "High": 40, "Medium": 8, "Low": 3}
MANDATORY = ("Critical", "High")

COST_TABLE = [
    ("every resource assigned", COST_ASSIGN),
    ("specialist given to a general-doctor patient", f"+{COST_SPECIALIST_FOR_GENERAL}"),
    ("emergency bed for an exam-room patient", f"+{COST_EMERGENCY_FOR_EXAM}"),
    ("relocating a patient under treatment", COST_RELOCATE),
    ("non-specialist for a specialist need (fallback)", f"+{COST_FALLBACK_DOCTOR}"),
    ("emergency bed for an ICU need (fallback)", f"+{COST_FALLBACK_BED}"),
    ("leaving a Medium patient waiting", COST_WAIT["Medium"]),
    ("leaving a Low patient waiting", COST_WAIT["Low"]),
]

BED_TYPES = {"ICU": "icu", "Emergency": "emergency", "ExamRoom": "exam_room"}


@dataclass(frozen=True)
class Slot:
    pid: str
    kind: str            # bed / doctor / nurse / equipment
    need: str | None     # bed type, specialty ("General" = any doctor), equipment type


@dataclass(frozen=True)
class Resource:
    rid: str
    kind: str            # bed / doctor / nurse / equipment
    type: str | None     # bed type, specialty or equipment type
    available: bool
    holder: str | None   # patient currently assigned to it


@dataclass(frozen=True)
class Action:
    name: str
    pid: str
    resource: str | None
    cost: int
    category: str                 # for the cost breakdown
    note: str = ""                # announcement (fallback, relocation, waiting)
    target: str | None = None     # Relocate: the bed the patient moves to

    def __str__(self) -> str:
        if self.name == "Relocate":
            return f"Relocate({self.pid}, {self.resource} → {self.target})"
        if self.name == "LeaveWaiting":
            return f"LeaveWaiting({self.pid})"
        return f"{self.name}({self.pid}, {self.resource})"


@dataclass(frozen=True)
class State:
    pos: int
    used: frozenset = frozenset()
    freed: frozenset = frozenset()
    moved: frozenset = frozenset()


@dataclass
class AllocationProblem:
    patients: dict[str, dict[str, Any]]
    resources: dict[str, Resource]
    relaxed: bool = False    # True: High/Critical may also be left waiting (no valid allocation exists)
    symmetry: bool = True    # offer one of several interchangeable resources (False: every one)
    slots: list[Slot] = field(init=False)
    first_slot: dict[str, int] = field(init=False)
    next_patient: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        waiting = [pid for pid, p in self.patients.items() if p["waiting"] and p["risk"]]
        waiting.sort(key=lambda pid: (self.patients[pid]["priority"], _pid_key(pid)))
        self.order = waiting
        self.slots, self.first_slot, self.next_patient = [], {}, {}
        for pid in waiting:
            self.first_slot[pid] = len(self.slots)
            self.slots += patient_slots(pid, self.patients[pid])
            self.next_patient[pid] = len(self.slots)
        # patients under treatment who hold a bed of a type they no longer need
        self.movable = {}
        for r in self.resources.values():
            if r.kind == "bed" and r.holder and r.holder in self.patients:
                p = self.patients[r.holder]
                needed = BED_TYPES.get(p.get("bed") or "")
                if not p["waiting"] and needed and needed != r.type and p["risk"] != "Critical":
                    self.movable[r.rid] = (r.holder, needed)
        self._precompute()

    # ---------------------------------------------------------------- problem API

    @property
    def initial(self) -> State:
        return State(0)

    def is_goal(self, s: State) -> bool:
        return s.pos == len(self.slots)

    def mandatory(self, pid: str) -> bool:
        return self.patients[pid]["risk"] in MANDATORY

    def wait_cost(self, pid: str) -> int:
        table = COST_WAIT_RELAXED if self.relaxed else COST_WAIT
        return table[self.patients[pid]["risk"]]

    def free(self, rid: str, s: State) -> bool:
        r = self.resources[rid]
        return (r.available or rid in s.freed) and rid not in s.used

    def free_of(self, kind: str, rtype: str | None, s: State) -> list[Resource]:
        return [r for r in self.resources.values()
                if r.kind == kind and (rtype is None or r.type == rtype) and self.free(r.rid, s)]

    def successors(self, s: State) -> Iterator[tuple[Action, State]]:
        if self.is_goal(s):
            return
        slot = self.slots[s.pos]
        pid = slot.pid
        if s.pos == self.first_slot[pid] and (self.relaxed or not self.mandatory(pid)):
            risk = self.patients[pid]["risk"]
            note = (f"{pid} ({risk}) CANNOT BE SERVED: left waiting, escalate"
                    if self.mandatory(pid) else f"{pid} ({risk}) left waiting: not enough resources this round")
            yield (Action("LeaveWaiting", pid, None, self.wait_cost(pid), "waiting", note),
                   State(self.next_patient[pid], s.used, s.freed, s.moved))
        for action in self._fill(slot, s):
            yield action, State(s.pos + 1, s.used | {action.resource}, s.freed, s.moved)
        if slot.kind == "bed":
            yield from self._relocations(slot, s)

    def _fill(self, slot: Slot, s: State) -> list[Action]:
        options: list[Action] = []
        pid, need = slot.pid, slot.need
        if slot.kind == "bed":
            options += [Action("ReserveBed", pid, r.rid, COST_ASSIGN, "assignment")
                        for r in self.free_of("bed", need, s)]
            if need == "icu":
                options += [Action("ReserveBed", pid, r.rid, COST_ASSIGN + COST_FALLBACK_BED, "fallback bed",
                                   f"{pid} needs an ICU bed but none can be freed: emergency bed {r.rid} instead")
                            for r in self.free_of("bed", "emergency", s)]
            if need == "exam_room":
                options += [Action("ReserveBed", pid, r.rid, COST_ASSIGN + COST_EMERGENCY_FOR_EXAM, "bed upgrade")
                            for r in self.free_of("bed", "emergency", s)]
        elif slot.kind == "doctor":
            for r in self.free_of("doctor", None, s):
                if need == "general":
                    general = r.type in ("general", None)
                    options.append(Action("AssignDoctor", pid, r.rid,
                                          COST_ASSIGN + (0 if general else COST_SPECIALIST_FOR_GENERAL),
                                          "assignment" if general else "specialist time"))
                elif r.type == need:
                    options.append(Action("AssignDoctor", pid, r.rid, COST_ASSIGN, "assignment"))
                else:
                    options.append(Action("AssignDoctor", pid, r.rid, COST_ASSIGN + COST_FALLBACK_DOCTOR,
                                          "fallback doctor",
                                          f"{pid} needs {_pretty(need)} but none is free: "
                                          f"{r.rid} ({_pretty(r.type)}) instead"))
        elif slot.kind == "nurse":
            options += [Action("AssignNurse", pid, r.rid, COST_ASSIGN, "assignment")
                        for r in self.free_of("nurse", None, s)]
        else:
            options += [Action("AllocateEquipment", pid, r.rid, COST_ASSIGN, "assignment")
                        for r in self.free_of("equipment", need, s)]
        return _one_per_kind(options, self.resources) if self.symmetry else options

    def _relocations(self, slot: Slot, s: State) -> Iterator[tuple[Action, State]]:
        """Free a bed of the needed type by moving a patient who no longer needs it.
        Only offered when no bed of that type is free."""
        if self.free_of("bed", slot.need, s):
            return
        seen_targets = set()
        for bed, (holder, needed) in sorted(self.movable.items(), key=lambda kv: _pid_key(kv[0])):
            if self.resources[bed].type != slot.need or holder in s.moved:
                continue
            targets = self.free_of("bed", needed, s)
            if not targets:
                continue
            target = min(targets, key=lambda r: _pid_key(r.rid)).rid
            if (needed, target) in seen_targets:
                continue
            seen_targets.add((needed, target))
            action = Action("Relocate", holder, bed, COST_RELOCATE, "relocation",
                            f"{holder} no longer needs {bed}: moved to {target} to free it for {slot.pid}",
                            target=target)
            yield action, State(s.pos, s.used | {target}, s.freed | {bed}, s.moved | {holder})

    # ---------------------------------------------------------------- heuristic

    def h(self, s: State) -> float:
        """Admissible lower bound of the remaining cost (see the module docstring).

        Patients after the current one are untouched, so their part is
        precomputed (``_suffix``); only the current patient and the free
        resource counts depend on the state.
        """
        if self.is_goal(s):
            return 0
        pid = self.slots[s.pos].pid
        i = self.index[pid]
        if s.pos == self.first_slot[pid]:
            agg = self._suffix[i]
        else:                                    # current patient started: add the slots left
            agg = self._suffix[i + 1].copy()
            for slot in self.slots[s.pos:self.next_patient[pid]]:
                agg.add_slot(slot, count_shortage=self.mandatory(pid) and not self.relaxed)
        total = agg.total
        if self.relaxed:
            return total          # shortage terms assume every High/Critical patient is served

        free = dict(self._base_free)              # free resources by (kind, type)
        for rid in s.freed:
            _bump(free, self.resources[rid], +1)
        for rid in s.used:
            _bump(free, self.resources[rid], -1)
        reserved = dict(agg.reserved)

        # High/Critical shortages: relocations, fallback beds, fallback doctors
        icu_excess = max(0, agg.icu - free.get(("bed", "icu"), 0))
        if icu_excess:
            movable = sum(1 for bed, (holder, _) in self.movable.items()
                          if holder not in s.moved and self.resources[bed].type == "icu")
            relocations = min(icu_excess, movable)
            fallbacks = icu_excess - relocations
            total += relocations * COST_RELOCATE + fallbacks * COST_FALLBACK_BED
            reserved[("bed", "emergency")] = reserved.get(("bed", "emergency"), 0) + fallbacks
        for spec, demand in agg.spec.items():
            total += max(0, demand - free.get(("doctor", spec), 0)) * COST_FALLBACK_DOCTOR

        # Dead end: needs that must still be filled exceed what could ever be free,
        # counting beds that relocations could empty. No completion exists.
        for key, need in reserved.items():
            if key == ("bed", "icu"):
                continue                          # covered above by relocations and fallbacks
            supply = free.get(key, 0)
            if key[0] == "bed":
                supply += sum(1 for bed, (holder, _) in self.movable.items()
                              if holder not in s.moved and self.resources[bed].type == key[1])
            if need > supply:
                return INF

        # Medium/Low contention: beds or doctors left after the patients who must be
        # served are fewer than the Medium/Low patients still to decide, so at least
        # that many wait. Count the cheapest such waits (an exam-room patient may
        # also take an emergency bed).
        def left(key: tuple) -> int:
            return max(0, free.get(key, 0) - reserved.get(key, 0))

        left_em, left_ex, left_doc = left(("bed", "emergency")), left(("bed", "exam_room")), left(("doctor", None))
        total += max(_cheapest(agg.opt_emergency, len(agg.opt_emergency) - left_em),
                     _cheapest(agg.opt_beds, len(agg.opt_beds) - left_em - left_ex),
                     _cheapest(agg.opt_all, len(agg.opt_all) - left_doc))
        return total

    def _precompute(self) -> None:
        """Heuristic parts that do not depend on the state."""
        self.index = {pid: i for i, pid in enumerate(self.order)}
        self._suffix = [_Aggregate()] * (len(self.order) + 1)
        for i in range(len(self.order) - 1, -1, -1):
            pid = self.order[i]
            agg = self._suffix[i + 1].copy()
            slots = self.slots[self.first_slot[pid]:self.next_patient[pid]]
            if self.relaxed or not self.mandatory(pid):
                serve, wait = len(slots), self.wait_cost(pid)
                bed = next((sl.need for sl in slots if sl.kind == "bed"), None)
                agg.add_optional(min(serve, wait), wait - min(serve, wait), bed)
            else:
                for slot in slots:
                    agg.add_slot(slot, count_shortage=True)
            self._suffix[i] = agg
        self._base_free: dict[tuple, int] = {}
        for r in self.resources.values():
            if r.available:
                _bump(self._base_free, r, +1)


class _Aggregate:
    """Lower-bound ingredients for a group of patients."""

    def __init__(self) -> None:
        self.total = 0
        self.icu = 0
        self.spec: dict[str, int] = {}
        self.reserved: dict[tuple, int] = {}
        self.opt_emergency: list[int] = []   # extra cost if left waiting, per Medium/Low patient
        self.opt_beds: list[int] = []
        self.opt_all: list[int] = []

    def copy(self) -> "_Aggregate":
        a = _Aggregate()
        a.total, a.icu = self.total, self.icu
        a.spec, a.reserved = dict(self.spec), dict(self.reserved)
        a.opt_emergency, a.opt_beds, a.opt_all = list(self.opt_emergency), list(self.opt_beds), list(self.opt_all)
        return a

    def add_slot(self, slot: Slot, count_shortage: bool) -> None:
        """A slot that will be filled: costs at least one assignment and takes a resource."""
        self.total += COST_ASSIGN
        key = ("doctor", None) if slot.kind == "doctor" else (slot.kind, slot.need)
        self.reserved[key] = self.reserved.get(key, 0) + 1
        if count_shortage and slot.kind == "bed" and slot.need == "icu":
            self.icu += 1
        elif count_shortage and slot.kind == "doctor" and slot.need != "general":
            self.spec[slot.need] = self.spec.get(slot.need, 0) + 1

    def add_optional(self, lower_bound: int, extra_if_waiting: int, bed: str | None) -> None:
        self.total += lower_bound
        self.opt_all.append(extra_if_waiting)
        if bed in ("emergency", "exam_room"):
            self.opt_beds.append(extra_if_waiting)
        if bed == "emergency":
            self.opt_emergency.append(extra_if_waiting)


def _bump(counts: dict[tuple, int], r: Resource, delta: int) -> None:
    counts[(r.kind, r.type)] = counts.get((r.kind, r.type), 0) + delta
    if r.kind == "doctor":
        counts[("doctor", None)] = counts.get(("doctor", None), 0) + delta


# -------------------------------------------------------------------- building

def patient_slots(pid: str, p: dict[str, Any]) -> list[Slot]:
    slots = []
    if p.get("bed"):
        slots.append(Slot(pid, "bed", BED_TYPES[p["bed"]]))
    if p.get("specialist"):
        slots.append(Slot(pid, "doctor", _atom(p["specialist"])))
    else:
        slots.append(Slot(pid, "doctor", "general"))
    if p.get("nurse"):
        slots.append(Slot(pid, "nurse", None))
    for eq in p.get("equipment", []):
        slots.append(Slot(pid, "equipment", _atom(eq)))
    return slots


def from_requirements(req: dict[str, Any], relaxed: bool = False, symmetry: bool = True) -> AllocationProblem:
    """Build the search problem from ``KnowledgeBase.requirements()``."""
    resources: dict[str, Resource] = {}
    for d in req["resources"]["doctors"]:
        resources[d["id"]] = Resource(d["id"], "doctor", _atom(d["specialty"]) if d["specialty"] else None,
                                      d["available"], d.get("assigned_to"))
    for n in req["resources"]["nurses"]:
        resources[n["id"]] = Resource(n["id"], "nurse", None, n["available"], n.get("assigned_to"))
    for b in req["resources"]["beds"]:
        resources[b["id"]] = Resource(b["id"], "bed", BED_TYPES[b["type"]], b["available"], b.get("assigned_to"))
    for e in req["resources"]["equipment"]:
        resources[e["id"]] = Resource(e["id"], "equipment", _atom(e["type"]), e["available"], e.get("assigned_to"))
    # a resource held by a patient is never free, whatever its availability flag says
    resources = {k: (r if not r.holder else Resource(r.rid, r.kind, r.type, False, r.holder))
                 for k, r in resources.items()}
    return AllocationProblem(req["patients"], resources, relaxed, symmetry)


# -------------------------------------------------------------------- helpers

def _cheapest(increments: list[int], k: int) -> int:
    """Sum of the k smallest values (0 when k <= 0)."""
    return sum(sorted(increments)[:k]) if k > 0 else 0


def _one_per_kind(options: list[Action], resources: dict[str, Resource]) -> list[Action]:
    """Interchangeable resources (same kind, same type, same cost) lead to the
    same future: offer only the lowest id of each group (symmetry breaking)."""
    best: dict[tuple, Action] = {}
    for a in options:
        r = resources[a.resource]
        key = (r.kind, r.type, a.cost, a.category)
        if key not in best or _pid_key(a.resource) < _pid_key(best[key].resource):
            best[key] = a
    return sorted(best.values(), key=lambda a: (a.cost, _pid_key(a.resource)))


def _atom(name: str) -> str:
    """PDF-style name from the contract ("GeneralSurgery") -> type key ("general_surgery")."""
    from reasoning.printer import to_atom

    return to_atom(name)


def _pretty(atom: str | None) -> str:
    from reasoning.printer import constant_to_pdf

    return constant_to_pdf(atom) if atom else "general"


def _pid_key(rid: str):
    head = rid.rstrip("0123456789")
    tail = rid[len(head):]
    return (head, int(tail) if tail else 0)
