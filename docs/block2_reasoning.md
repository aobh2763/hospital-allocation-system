# Block 2: logical reasoning, deduction, unification, resolution

Block 2 turns patient data, ML/DL predictions and hospital resources into
facts, derives one risk level per patient and the `RequiresX` predicates that
Search and Planning use. Every conclusion comes from a rule and can be traced.
The full list of predicates is in [predicates.md](predicates.md).

## Setup

1. SWI-Prolog 9 or later.
   - Arch: `sudo pacman -Syu swi-prolog` (`-Syu` refreshes the package list
     first; a stale list gives a 404 on the mirror).
   - Windows/macOS: https://www.swi-prolog.org/download/stable
2. Python packages, inside the project environment:
   - `pip install janus-swi` (the official SWI-Prolog bridge; it compiles
     against the installed SWI-Prolog, so install that first)
   - PyTorch is **not** needed to run the agent. The trained GRU is read
     from `prediction/dl/models/gru.pt` and run with NumPy
     (`src/reasoning/numpy_rnn.py`); on the 300 held-out patients it gives
     the same macro-F1 as PyTorch (0.9933). Only retraining needs PyTorch.
3. Only to (re)train the DL model, with PyTorch installed (about a minute on CPU), which saves
   `prediction/dl/models/gru.pt`:
   `python prediction/dl/export_model.py`

## Run

```bash
python src/block2_demo.py                     # PDF example, predictions as in the PDF
python src/block2_demo.py --models            # PDF example, predictions from the real models
python src/block2_demo.py --hospital FAC0001 --date 2023-01-02 --on-call cardiology
python -m unittest discover -s tests -v       # 28 tests
```

The demo prints, in order: the scenario, the ML/DL evidence, the forward-chaining
log, the risk table, a deduction tree, two resolution traces (one proved, one
not), the unification examples, the JSON for Search, then three environment
changes and how the requirements change.

### Try your own patient

```bash
poetry run python src/patient_cli.py                    # asks for the patient
poetry run python src/patient_cli.py --save me.json     # same, and keeps what you typed
poetry run python src/patient_cli.py --file scenarios/patient_example.json
```

It shows the probability of each risk level from the ML model (current state)
and the DL model (the earlier hourly measurements plus the current vitals), the
facts derived in Prolog layer by layer, the final risk and needs, the deduction
tree and a resolution proof of the risk. Then you can type queries such as
`RequiresICU(P1)` (shows its proof) or `RequiresEquipment(P1, t)` (lists answers).
With no earlier measurements there is no evolution to analyse, so no `DL_Risk`.

## Where the data comes from

| File | Built by | One row is | Used for |
| --- | --- | --- | --- |
| `data/patients.csv` | `prediction/build_dataset.py` (Synthea) | one encounter | facts: age, sex, vitals, conditions |
| `data/patient_preprocessed.csv` | `prediction/ml/preprocessing.ipynb` | the same encounter, encoded | ML input (55 features of `hgb_model.joblib`) |
| `data/synthetic_emergency_timeseries_2000_patients.csv` | downloaded | one hour of one patient | DL input (10 hourly steps) |
| `data/hospital/hospital_resources.csv` | HLT-010 sample | facility-day | free ICU and general beds, surge, diversion |
| `data/hospital/staffing.csv` | HLT-010 sample | staff shift | doctors and nurses on shift |
| `data/hospital/equipment.csv` | HLT-010 sample | asset-day | equipment and its availability |
| `scenarios/pdf_example.json` | written by hand | | the PDF's end-to-end example plus three events |

Rows *i* of `patients.csv` and `patient_preprocessed.csv` are the same encounter.

**Known gaps, handled explicitly:**

- The two patient datasets share no patient. `--hospital` pairs a Synthea
  encounter with a time-series patient of the same risk label
  (`environment/datasets.combined_patient`), so one patient has both a history
  for the ML model and a timeline for the DL model. The pairing is simulated.
- The hospital staffing data is perioperative: surgeons, anesthesiologists,
  residents, PA/NP, nurses. There are no cardiologists, pulmonologists or
  neurologists. A patient who needs one gets `FallbackDoctor` options, unless
  `--on-call cardiology` (or a scenario file) adds an on-call specialist.
- There are no ED exam rooms in the data; two are assumed (`caps["exam_room"]`).
- A real facility has hundreds of free beds; caps keep the search problem small
  (`environment/hospital.DEFAULT_CAPS`).
- Patients typed into the interface only have about a third of the ML model's
  features; the rest are training medians, so their ML evidence is weaker and
  often falls under the 0.60 confidence threshold, which triggers
  `RequiresReassessment`.
- The DL dataset is almost separable from its first time step (GRU test
  macro-F1 0.993), so it shows the integration more than the value of temporal
  modelling.

## How it works

```
Patient record ──► facts.py ───────────────┐
ML adapter ──────► ML_Risk(P, L, p) ───────┤
DL adapter ──────► DL_Risk(P, L, p) ───────┼─► SWI-Prolog KB ──► saturate (forward chaining)
Scenario/hospital ► resources, Available ──┘        │
                                                     ├─► export.requirements() ─► Search, Planning
                                                     ├─► prove(goal)   resolution trace
                                                     └─► unify(a, b)   unification demo
```

| File | Role |
| --- | --- |
| `src/patient.py` | one patient whatever the source: vitals, timeline, symptoms, conditions |
| `src/environment/scenario.py` | resources, assignments, predictions, events; JSON in and out |
| `src/environment/datasets.py` | which dataset holds what; rows to `Patient` |
| `src/environment/hospital.py` | scenario from the HLT-010 hospital data |
| `src/reasoning/kb.py` | `KnowledgeBase`: the Python API |
| `src/reasoning/facts.py` | `Patient` and resources to base facts |
| `src/reasoning/adapters.py` | runs the ML (gradient boosting) and DL (GRU) models |
| `src/reasoning/numpy_rnn.py` | reads `gru.pt` without PyTorch and runs the GRU/LSTM in NumPy |
| `src/reasoning/events.py` | applies environment changes (Block 6) |
| `src/reasoning/export.py` | JSON contract for Search, and the diff between updates |
| `src/reasoning/printer.py` | `high_risk(p1)` and `HighRisk(P1)` conversion |
| `src/reasoning/prolog/kb.pl` | base predicates, rule loader, shared solver |
| `src/reasoning/prolog/forward.pl` | deduction: stratified forward chaining with a derivation log |
| `src/reasoning/prolog/prove.pl` | resolution (SLD/SLDNF) with trace, failure explanations, unification demo |
| `src/reasoning/prolog/thresholds.pl` | every threshold and lookup table |
| `src/reasoning/prolog/{findings,evidence,risk,requirements}.pl` | the four rule layers |

**Rules are data.** Each rule is a term `r(Id, Stratum, (Head :- Body))`. The
loader keeps the author's variable names, so traces print `HeartRate(x, v) ∧ v > t`.

**Deduction.** `saturate/0` runs strata 1 to 12 in order, firing every rule
until nothing new appears. Negation (`\+`) only looks at lower strata, so the
result is unique. Each derived fact stores its rule, substitution θ and premises.

**Resolution.** `kb_prove/2` does not use the derived facts. It resolves the
goal clause against rules and base facts (SLD resolution, with negation as
failure for `\+`), records each step's clause, θ and resolvent, and stops at
the empty clause □. The test `test_deduction_and_resolution_agree` checks that
both methods reach the same conclusions.

**Combining evidence.** The final level is the highest level supported by any
source: rules, ML (probability ≥ 0.60) or DL (probability ≥ 0.60). When two
sources are two or more levels apart, `Disagreement(P)` adds
`RequiresReassessment(P)`.

**Dynamic environment.** Base facts change (`set_available`, `add_patient`,
new measurements and predictions), then `update()` wipes the derived facts,
re-derives everything and returns the per-patient difference.

## Design choices (from the Block 2 plan)

| Decision | Choice |
| --- | --- |
| Engine | SWI-Prolog via `janus-swi` |
| Combining rules, ML, DL | highest credible level wins; disagreement flagged |
| Minimum model confidence | 0.60 (`threshold(model_confidence, 0.60)`) |
| Several specialists match | one primary, by precedence |
| No data | `InsufficientData(P)`, still `LowRisk(P)` |
| ICU full | `FallbackBed` with a penalty for Search |
