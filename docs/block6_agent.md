# Block 6: the integrated agent

One agent chains all the blocks, carries out its plan step by step, and
decides again whenever the world changes, instead of executing an obsolete plan.

## Run it

**Option 1: the complete scenario, end to end**

```bash
poetry run python src/agent_demo.py            # ML and DL computed by the trained models
poetry run python src/agent_demo.py --given    # the PDF's predictions (ML 0.87, DL 0.91)
poetry run python src/agent_demo.py --pause    # wait for Enter between stages, for a live demo
poetry run python src/agent_demo.py --scenario scenarios/icu_reallocation.json --given
```

**Option 2: add patients yourself and watch the agent adapt**

```bash
poetry run python src/agent_cli.py             # the PDF department and patients
poetry run python src/agent_cli.py --empty     # the same department, no patients yet
```

| Key | What it does |
| --- | --- |
| `n` | type a new patient (age, symptoms, history, vitals, earlier hourly vitals) |
| `f` | load a new patient from a JSON file (`scenarios/patient_example.json`) |
| `s` / `r` / `a` | execute the next action / several actions / the whole plan |
| `v` | new vital signs for a patient (the models run again) |
| `o` / `b` | put a resource out of service / back in service |
| `d` | discharge a patient (frees what they held) |
| `w` / `p` | who holds what / the remaining plan |
| `e` | ask the knowledge base: a proof (`RequiresICU(P1)`) or a query (`RequiresEquipment(P1, t)`) |

For a new patient, the agent first shows the ML and DL probability of each risk
level, then deliberates and shows what changed: risk, needs, the new
allocation (who gained or lost which resource) and the new plan.

## One deliberation

```text
observe ─▶ Block 3: ML_Risk / DL_Risk for new or changed patients
        ─▶ Block 2: facts + rules → risk level, RequiresX (forward chaining)
        ─▶ Block 4: A* chooses the allocation (UCS alongside, for comparison)
        ─▶ Block 5: the planner orders the actions; the validator replays them
        ─▶ execute, one action at a time
```

No block decides alone:
- **Learning** gives evidence (`DL_Risk(P1, Critical, 0.61)`).
- **Reasoning** interprets it with the explicit rules (`CriticalRisk(P1)` → `RequiresICU(P1)`).
- **Search** picks the configuration (ICU1 and D1 for P1).
- **Planning** decides how to get there (reserve before transfer).

## Acting and adapting

- **Before each action**, the agent checks whether anything changed since it
  last decided. If so, it looks at two things: whether the rest of the plan
  can still run (the first step whose preconditions no longer hold), and what
  the change did to the patients' requirements (Block 2's diff). Then it
  deliberates again.
- **Reservations are tentative until the transfer.** At each new deliberation the
  reservations of patients still waiting are released, so Search may give a bed
  or doctor to a more urgent newcomer. Reservations the new allocation keeps are
  not redone.
- **The transfer is the commitment.** A patient under treatment keeps their
  resources. Search can only move them down (out of an ICU bed they no longer
  need) or up (from a fallback bed into the ICU when one frees).
- **Memory.** Evaluations, monitoring and re-evaluations already done are not repeated.

## The end-to-end run (option 1)

The PDF's end-to-end example, with predictions computed by the trained models:

| # | What happened | How the agent noticed | What it decided |
| --- | --- | --- | --- |
| 1 | Start: P1 (72, chest pain, SpO2 88), P2 (fever), P3 (abdominal pain); 2 doctors, 2 emergency beds, 1 ICU bed | | ML Critical 0.49 (below 0.60) and DL Critical 0.61 → P1 Critical. P1 gets ICU1, the cardiologist D1, a nurse, a monitor, a pump and a defibrillator; P2 gets EB1 and D2; P3 waits (no doctor left). 13-step plan |
| 2 | ICU1 becomes unavailable just before P1's transfer | the next step `TransferPatient(P1, ICU1)` needs `BedReserved(P1, ICU1)`, which is gone | P1 → EB1 (announced fallback), P2 → EB2. P1 keeps D1, N1 and the equipment; only 7 actions left |
| 3 | A second Critical patient, P4, arrives | Block 2 rates P4 Critical (shock rule: low MAP, SpO2 85) | P4 gets EB2 and D2 (fallback: no neurologist); P2 is now left waiting |
| 4 | P2's vitals worsen | the DL model on the new timeline says High 0.89; Block 2: P2 Medium → High | no valid allocation: 2 High/Critical patients, 1 bed, 1 doctor, 1 nurse left. P4 is served, P2 is announced as "cannot be served: escalate" |

Every plan was validated before execution; 21 actions executed in total; A*
and UCS chose the same allocation at every decision.

## Files

| File | Role |
| --- | --- |
| `src/agent/agent.py` | the agent: observe, deliberate, step, run; the execution model |
| `src/agent/display.py` | what the demo and the interactive mode print |
| `src/agent_demo.py` | option 1, end to end |
| `src/agent_cli.py` | option 2, interactive |
| `tests/test_planning_agent.py` | end-to-end and adaptation tests (with and without the models) |
