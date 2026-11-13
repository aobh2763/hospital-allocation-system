# HTTP API

`src/web_server.py` exposes the integrated agent (Blocks 1–6) over HTTP and
serves the web frontend from `frontend/` at `/`.

```bash
poetry run python src/web_server.py              # http://127.0.0.1:8000, ML/DL computed by the models
poetry run python src/web_server.py --given      # the scenario's written predictions (PDF: 0.87, 0.91)
poetry run python src/web_server.py --port 8080 --scenario icu_reallocation.json
```

- **Format:** JSON in, JSON out. Errors are `{"error": "message"}` with status 400
  (bad input), 404 (unknown patient, resource or path), 405 (wrong method) or
  409 (no scripted event left).
- **One agent per server.** Requests are handled one at a time, because
  SWI-Prolog runs once per process. A request that changes the world returns
  the new deliberation and the full new state, so the UI never needs a second
  call.
- **Notation.** IDs are as in the PDF: patients `P1`, doctors `D1`, nurses `N1`,
  beds `ICU1` / `EB1` / `EX1`, equipment `M1` (monitor), `IP1` (infusion pump),
  `DF1` (defibrillator), `V1` (ventilator). Risk levels are `Low`, `Medium`, `High`, `Critical`.

## Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/meta` | | vocabularies, vital ranges, thresholds, model availability |
| GET | `/api/scenarios` | | scenario files you can reset to |
| GET | `/api/state` | | everything the screen shows (below) |
| POST | `/api/reset` | `{scenario, use_models, empty}` | `{cycle, state}`: a fresh agent and its first deliberation |
| POST | `/api/step` | | `{kind, action, cycle, state}`: execute the next action (or deliberate if something changed) |
| POST | `/api/run` | `{steps: n}` or `{until: "transfer"}` or `{all: true}` | `{executed, cycles, done, state}` |
| GET | `/api/demo` | | the scenario's scripted events and which were played |
| POST | `/api/demo/next` | | `{event, changes, cycle, state}`: play the next scripted event |
| POST | `/api/patients` | patient (below) | `{pid, probabilities, changes, cycle, state}`: admit a patient |
| GET | `/api/patients/{pid}` | | one patient in detail: probabilities, deduction tree, proof, derived facts |
| POST | `/api/patients/{pid}/vitals` | `{hr, map, spo2, temp, rr, lactate}` (any subset) | `{changes, cycle, state}` |
| POST | `/api/patients/{pid}/discharge` | | `{changes, cycle, state}` |
| POST | `/api/resources/{rid}/out` | | `{changes, cycle, state}`: out of service |
| POST | `/api/resources/{rid}/in` | | `{changes, cycle, state}`: back in service |
| POST | `/api/event` | `{type, ...}` | `{changes, cycle, state}`: the generic form of the four above |
| GET | `/api/cycles` | | `{cycles: [summary]}` |
| GET | `/api/cycles/{n}` | | one deliberation in full |
| POST | `/api/query` | `{text}` | a resolution proof, or the rows of a pattern |

`reset`: `scenario` is a file name from `/api/scenarios` (default
`pdf_example.json`); `use_models` (default true) computes ML and DL predictions
with the trained models; `empty: true` keeps the department but no patients and
no scripted events.

## State (`GET /api/state`, and `state` in every change response)

```json
{
  "scenario": {"file": "pdf_example.json", "name": "PDF end-to-end example", "notes": ["..."]},
  "use_models": true,
  "models": {"ml": true, "dl": true},
  "step": 7,                                   // actions executed so far
  "patients": [{
      "pid": "P1", "age": 72, "sex": "m",
      "symptoms": ["chest_pain", "shortness_of_breath"], "conditions": [],
      "vitals": {"hr": 125.0, "sbp": 84.0, "dbp": 50.0, "spo2": 88.0, "temp": 39.0, "rr": 26.0, "lactate": 3.1, "map": 61.3},
      "history": [{"hr": 105, "map": 82, "spo2": 95, "temp": 38.2, "rr": 20, "lactate": 1.8}, "..."],   // oldest first
      "source": "form",                        // "form", "typed in" or "synthea"
      "status": "waiting",                     // or "under_treatment"
      "risk": "Critical", "priority": 1, "max_wait": 0,
      "holds": ["D1", "ICU1", "N1"],           // reserved (waiting) or in use (under treatment)
      "needs": {"bed": "ICU", "doctor": "Cardiology", "specialist": true, "nurse": true,
                "equipment": ["Defibrillator", "InfusionPump", "PatientMonitor"], "monitoring": "Continuous"},
      "flags": {"urgent_evaluation": true, "reassess": true, "insufficient_data": false},
      "predictions": {"ml": ["Critical", 0.4922], "dl": ["Critical", 0.6066]}   // either may be missing
  }],                                          // sorted by priority, then id
  "resources": [
      {"id": "D1", "kind": "doctor", "type": "cardiology", "type_label": "Cardiology", "status": "reserved", "held_by": "P1"},
      {"id": "ICU1", "kind": "bed", "type": "icu", "type_label": "ICU", "status": "free", "held_by": null},
      {"id": "M1", "kind": "equipment", "type": "patient_monitor", "type_label": "PatientMonitor", "status": "in_use", "held_by": "P1"}
  ],                                           // status: free | reserved | in_use | out_of_service; kind: doctor | nurse | bed | equipment
  "plan": [action, "..."],                     // remaining actions, next first
  "next_action": "TransferPatient(P1, ICU1)",  // null when the plan is finished
  "executed": [{"cycle": 1, ...action}],       // in order
  "allocation": allocation,                    // the latest decision
  "valid": true,                               // false: not every High/Critical patient could be served
  "announcements": ["P3 (Low) left waiting: not enough resources this round"],
  "cycles": [summary, "..."],
  "pending_changes": [],                       // observed but not yet deliberated (normally empty)
  "demo": {"events": 3, "next_index": 0}
}
```

**action**

```json
{"label": "ReserveICUBed(P1, ICU1)", "name": "ReserveICUBed", "args": ["P1", "ICU1"], "patient": "P1", "priority": 1,
 "pre": ["Available(ICU1)", "CriticalRisk(P1)", "Type(ICU1, ICU)", "Waiting(P1)"],
 "add": ["BedReserved(P1, ICU1)"], "delete": ["Available(ICU1)"]}
```

Action names: `AssignDoctor`, `AssignNurse`, `ReserveICUBed`, `ReserveBed`,
`AllocateEquipment`, `Relocate` (args `[patient, from, to]`),
`PerformUrgentEvaluation`, `TransferPatient`, `StartMonitoring`, `Reevaluate`.

**allocation**

```json
{"patients": {"P1": {"risk": "Critical", "bed": "ICU1", "doctor": "D1", "nurse": "N1",
                     "equipment": {"defibrillator": "DF1", "infusion_pump": "IP1", "patient_monitor": "M1"}}},
 "relocations": [{"patient": "P5", "from": "ICU1", "to": "EB1"}],
 "waiting": ["P3"],
 "announcements": ["..."]}
```

**summary** (one per deliberation)

```json
{"number": 2, "reasons": ["ICU1 becomes unavailable before P1's transfer"], "valid": true, "cost": 23,
 "astar_expanded": 9, "ucs_expanded": 15, "plan_length": 7, "plan_valid": true}
```

## One deliberation (`cycle`, and `GET /api/cycles/{n}`)

The summary fields, plus:

```json
{
  "invalid_step": {"index": 0, "action": "TransferPatient(P1, ICU1)", "missing": ["BedReserved(P1, ICU1)"]},
                                               // null when the old plan could still run
  "changes": {"P4": {"added": ["RequiresICU(P4)", "..."], "removed": [], "risk": [null, "Critical"]}},
                                               // what the observed change did to Block 2's requirements; risk is [before, after] or null
  "new_predictions": {"P4": {"ml": ["Critical", 0.4944], "dl": ["Critical", 0.5183]}},
  "requirements": { "...": "Block 2's full contract, see docs/predicates.md" },
  "search": {
    "astar": {"algorithm": "A*", "valid": true, "found": true, "cost": 23, "expanded": 9, "generated": 12,
              "max_frontier": 3, "ms": 0.21, "h_start": 23,
              "breakdown": {"assignment": 7, "fallback bed": 11, "waiting": 5},
              "actions": ["ReserveBed(P1, EB1)", "..."],
              "announcements": ["P1 needs an ICU bed but none can be freed: emergency bed EB1 instead"],
              "diagnosis": []},              // when valid is false: why no valid allocation exists
    "ucs": { "...": "same fields" }
  },
  "allocation": allocation,
  "allocation_changes": ["P1: bed ICU1 -> EB1", "P2: bed EB1 -> EB2"],
  "plan": {"found": true, "steps": [action, "..."], "expanded": 7, "ms": 0.2, "errors": [], "unreachable": []},
                                               // errors empty = the validator replayed the plan successfully
  "timings_ms": {"reasoning": 6.2, "search": 0.28, "planning": 0.17}
}
```

## Admitting a patient (`POST /api/patients`)

```json
{"age": 78, "sex": "f",
 "symptoms": ["shortness_of_breath", "confusion"],      // from /api/meta symptoms
 "conditions": ["copd", "diabetes"],                    // from /api/meta conditions
 "vitals": {"hr": 124, "sbp": 92, "dbp": 54, "spo2": 89, "temp": 38.9, "rr": 28, "lactate": 3.4},   // any subset
 "low_bp": false,                                       // "blood pressure low" when sbp/dbp unknown
 "history": [{"hr": 98, "map": 88, "spo2": 95, "temp": 37.9, "rr": 20, "lactate": 1.6}]}            // earlier hours, oldest first, up to 9
```

Only `age` is required. The current vitals become the last hour of the
timeline. The DL model needs at least one earlier hour (two steps in all);
without it there is no DL prediction. The response adds:

```json
{"pid": "P4",
 "probabilities": {"ml": {"Low": 0.308, "Medium": 0.033, "High": 0.017, "Critical": 0.641},
                   "dl": {"Low": 0.002, "Medium": 0.022, "High": 0.815, "Critical": 0.161},
                   "ml_features_known": 0.327}}       // share of the ML model's 55 features a form provides
```

`scenarios/patient_example.json` holds this example patient in the same format.

## Patient detail (`GET /api/patients/{pid}`)

The patient's state fields, plus `probabilities` (as above), `explanation` (the
deduction tree of its risk level, multi-line text), `proof` (the resolution
trace, multi-line text), `deductions` (the derived facts about this patient:
`seq`, `fact_pdf`, `rule`, `layer`, `theta_pdf`), `allocation` (what the
latest decision gives it, or null) and `left_waiting`.

## Query (`POST /api/query`)

`{"text": "RequiresICU(P1)"}` → `{"kind": "proof", "query": "...", "proved": true, "text": "Query: ...\n  1. ← ...\nProved: ..."}`

`{"text": "RequiresEquipment(P1, t)"}` → `{"kind": "rows", "variables": ["t"], "rows": [{"t": "PatientMonitor"}, ...]}`

Single lowercase letters are variables. One predicate per query.

## Reference data (`GET /api/meta`)

```json
{"risk_levels": ["Low", "Medium", "High", "Critical"],
 "symptoms": ["chest_pain", "shortness_of_breath", "abdominal_pain", "fever", "confusion", "headache", "bleeding", "trauma"],
 "conditions": ["hypertension", "diabetes", "heart_disease", "copd", "obesity", "smoker", "stroke", "ckd", "cancer", "asthma"],
 "vitals": [{"key": "hr", "label": "Heart rate", "unit": "bpm", "min": 20, "max": 250}, "..."],
 "history_columns": [{"key": "hr", "min": 20, "max": 250}, "..."],
 "history_max_rows": 9,
 "thresholds": {"hr_high": 110, "hr_very_high": 130, "map_low": 65, "sbp_low": 90, "spo2_low": 92, "spo2_very_low": 88,
                "temp_fever": 38.0, "rr_high": 22, "rr_very_high": 30, "lactate_high": 2.0, "lactate_very_high": 4.0,
                "age_elderly": 65, "model_confidence": 0.6},
 "models": {"ml": true, "dl": true},
 "disclaimer": "Teaching simulation. Not a medical device; no real diagnosis or treatment advice."}
```
