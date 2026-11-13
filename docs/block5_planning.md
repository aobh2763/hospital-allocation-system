# Block 5: planning

Search (Block 4) chooses **which** bed, doctor, nurse and equipment each patient
gets. Planning turns that configuration into an **ordered sequence of actions**
that reaches it from the current world, never using an action whose
preconditions do not hold. A validator replays every plan to prove it.

## Search vs planning

| | Search (Block 4) | Planning (Block 5) |
| --- | --- | --- |
| Question | which configuration among the alternatives? | in what order to reach it? |
| State | which slots are filled, which resources are used | the world's facts: `Waiting(P1)`, `Available(ICU1)`, `BedReserved(P1, ICU1)` … |
| Actions | choose a resource for a need | do something in the department, with preconditions and effects |
| Result | `P1: ICU1, D1, N1, M1 …` | `AssignDoctor(P1, D1)`, `ReserveICUBed(P1, ICU1)`, … `TransferPatient(P1, ICU1)` |

## Actions (STRIPS)

A state is a set of facts; anything not listed is false. An action applies when
all its preconditions are in the state; it adds and deletes facts.

| Action | Preconditions | Adds | Deletes |
| --- | --- | --- | --- |
| `AssignDoctor(P, D)` | `Waiting(P)`, `Available(D)` | `DoctorAssigned(P, D)` | `Available(D)` |
| `ReserveICUBed(P, B)` | `Waiting(P)`, `CriticalRisk(P)`, `Type(B, ICU)`, `Available(B)` | `BedReserved(P, B)` | `Available(B)` |
| `ReserveBed(P, B)` | `Waiting(P)`, `Available(B)` | `BedReserved(P, B)` | `Available(B)` |
| `AssignNurse(P, N)` | `Waiting(P)`, `Available(N)` | `NurseAssigned(P, N)` | `Available(N)` |
| `AllocateEquipment(P, E)` | `Waiting(P)`, `Available(E)` | `EquipmentAllocated(P, E)` | `Available(E)` |
| `Relocate(Q, B, B2)` | `UnderTreatment(Q)`, `In(Q, B)`, `Available(B2)` | `In(Q, B2)`, `Available(B)` | `In(Q, B)`, `Available(B2)` |
| `PerformUrgentEvaluation(P, D)` | `DoctorAssigned(P, D)`, `RequiresUrgentEvaluation(P)` | `Evaluated(P)` | |
| `TransferPatient(P, B)` | `Waiting(P)`, `BedReserved(P, B)`, `DoctorAssigned(P, D)`, the nurse if any, `Evaluated(P)` if urgent | `In(P, B)`, `UnderTreatment(P)` | `Waiting(P)`, `BedReserved(P, B)` |
| `StartMonitoring(P, M)` | `In(P, B)`, `EquipmentAllocated(P, M)` | `Monitored(P)` | |
| `Reevaluate(P, D)` | `In(P, B)`, `DoctorAssigned(P, D)`, `Monitored(P)` if monitored | `Reassessed(P)` | |

`AssignDoctor` and `ReserveICUBed` are the PDF's own examples. The PDF's
constraints hold by construction:
- **No transfer before the bed is reserved:** `TransferPatient` needs `BedReserved`.
- **No busy doctor:** `AssignDoctor` deleted `Available(D1)`, so a second assignment of D1 is not applicable.
- **An occupied ICU bed is used only after relocation:** it becomes `Available` once its occupant is moved.

**Where the facts come from.** The initial state is built from Block 2's
contract (`Available`, `Waiting`, `UnderTreatment`, `CriticalRisk`,
`RequiresUrgentEvaluation`, what each patient under treatment holds) plus the
agent's memory of what it has already done (`Evaluated`, `Monitored`,
`Reassessed`).

**Goals.** For each patient Search serves: `In(P, B)`, the doctor, the nurse and
each piece of equipment assigned; `Evaluated(P)` if urgent; `Monitored(P)` if a
monitor was allocated; `Reassessed(P)` if Critical or flagged for reassessment.
For each relocation: `In(Q, B2)`. Patients already under treatment keep their
unfinished follow-up (monitoring, re-evaluation) as goals.

## The planner

Forward state-space search from the initial state, guided by **h_add**, the
additive heuristic of the HSP planner (Bonet & Geffner). Ignoring delete
effects, each fact costs 0 if it holds, otherwise 1 plus the cheapest sum of
the preconditions of an action that adds it; h_add is the sum of the goals'
costs. Goals are weighted by patient priority (Critical ×1000, High ×100,
Medium ×10, Low ×1), so the most urgent patient's chain comes first. Ties
follow the natural order of the actions.

Greedy search with this weighted heuristic does not promise the shortest
plan. Here every plan has the same length, because each ground action is needed
exactly once, so only the order matters. The **validator** replays the plan
from the initial state and checks every precondition and every goal; the agent
never executes a plan the validator rejects.

If no ordering can reach the goals (for example a needed resource is not
available and nothing frees it), the planner says so and lists the goals that
can never be reached.

## Example: the PDF scenario

The agent's first plan, matching the order the PDF describes ("assigning a
doctor to P1, reserving an ICU bed, performing an urgent evaluation,
transferring the patient, starting monitoring, and then performing a new
evaluation"):

```text
 1. AssignDoctor(P1, D1)              8. TransferPatient(P1, ICU1)
 2. AssignNurse(P1, N1)               9. StartMonitoring(P1, M1)
 3. PerformUrgentEvaluation(P1, D1)  10. Reevaluate(P1, D1)
 4. ReserveICUBed(P1, ICU1)          11. AssignDoctor(P2, D2)
 5. AllocateEquipment(P1, M1)        12. ReserveBed(P2, EB1)
 6. AllocateEquipment(P1, DF1)       13. TransferPatient(P2, EB1)
 7. AllocateEquipment(P1, IP1)
Plan valid: every precondition holds in order and every goal is reached.
```

P3 (Low) is not in the plan: both doctors are taken, so Search left P3 waiting.

In the ICU reallocation scenario, `Relocate(P5, ICU1, EB1)` is planned before
`ReserveICUBed(P1, ICU1)`, because only the relocation makes `Available(ICU1)` true.

## Files

| File | Role |
| --- | --- |
| `src/planning/strips.py` | actions, h_add, the planner, the validator |
| `src/planning/domain.py` | the hospital actions, initial state and goals from Blocks 2 and 4 |
| `tests/test_planning_agent.py` | planner, validator and ordering tests |
