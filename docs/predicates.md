# Block 2 predicate contract

The predicates Block 2 reads (base facts) and produces (derived facts).
Search (Block 4) and Planning (Block 5) may rely on every derived predicate
below. Changing one needs a yes from the Block 4 and Block 5 owners.

Notation: the PDF style (`RequiresICU(P1)`). In Prolog the same fact is
`requires_icu(p1)`; `reasoning/printer.py` converts between the two.
`P` is a patient, `D` a doctor, `N` a nurse, `B` a bed, `E` equipment.

All thresholds are educational approximations (NEWS2, qSOFA). This is a
simulation, not medical advice. Thresholds live in
`src/reasoning/prolog/thresholds.pl`.

## 1. Base facts (inputs, written by Python)

| Predicate | Written by | Example |
| --- | --- | --- |
| `Patient(P)` | fact builder | `Patient(P1)` |
| `Age(P, years)`, `Sex(P, M/F)` | fact builder | `Age(P1, 72)` |
| `HeartRate(P, bpm)` | fact builder | `HeartRate(P1, 125)` |
| `SystolicBP(P, mmHg)`, `DiastolicBP(P, mmHg)` | fact builder | `SystolicBP(P1, 84)` |
| `MAP(P, mmHg)` | fact builder (computed (SBP + 2 DBP) / 3 if missing) | `MAP(P1, 61.3)` |
| `OxygenSaturation(P, %)` | fact builder | `OxygenSaturation(P1, 88)` |
| `Temperature(P, °C)` | fact builder | `Temperature(P1, 39)` |
| `RespiratoryRate(P, /min)` | fact builder | `RespiratoryRate(P1, 26)` |
| `Lactate(P, mmol/L)` | fact builder | `Lactate(P1, 3.1)` |
| `HasSymptom(P, S)` | fact builder | `HasSymptom(P1, ChestPain)` |
| `HasCondition(P, C)` | fact builder (Synthea `rf_*` flags) | `HasCondition(P1, HeartDisease)` |
| `LowBloodPressure(P)` | fact builder (form field "BP = Low") | `LowBloodPressure(P1)` |
| `Waiting(P)` / `UnderTreatment(P)` | environment | `Waiting(P1)` |
| `ML_Risk(P, Level, prob)` | ML adapter (Block 3) | `ML_Risk(P1, High, 0.87)` |
| `DL_Risk(P, Level, prob)` | DL adapter (Block 3) | `DL_Risk(P1, Critical, 0.91)` |
| `Doctor(D)`, `Specialty(D, S)` | environment | `Specialty(D1, Cardiology)` |
| `Nurse(N)` | environment | `Nurse(N1)` |
| `Bed(B)`, `Type(B, ICU / Emergency / ExamRoom)` | environment | `Type(ICU1, ICU)` |
| `Equipment(E)`, `EquipmentType(E, T)` | environment | `EquipmentType(V1, Ventilator)` |
| `Available(R)` | environment | `Available(ICU1)` |
| `Assigned(P, R)` | environment (executed actions) | `Assigned(P2, EB1)` |
| `SurgeDay`, `Diversion` | environment (hospital data) | `SurgeDay` |

Vocabularies. Symptoms: `ChestPain`, `ShortnessOfBreath`, `AbdominalPain`,
`Fever`, `Confusion`, `Headache`, `Bleeding`, `Trauma`. Conditions:
`Hypertension`, `Diabetes`, `HeartDisease`, `COPD`, `Obesity`, `Smoker`,
`Stroke`, `CKD`, `Cancer`, `Asthma`. Specialties: `Cardiology`, `Pulmonology`,
`GeneralSurgery`, `Neurology`, `Anesthesiology`, `General`. Equipment:
`Ventilator`, `PatientMonitor`, `InfusionPump`, `Defibrillator`, `CTScanner`,
`PortableXray`, `Ultrasound`.

Only the latest value of each vital is a fact. The hourly history stays in the
Python `Patient` record and reaches the KB through `DL_Risk`.

## 2. Patient status (outputs)

| Predicate | Meaning | Derived when (rule id) | Used by |
| --- | --- | --- | --- |
| `LowRisk(P)` / `MediumRisk(P)` / `HighRisk(P)` / `CriticalRisk(P)` | Final risk, exactly one per patient | highest level any source supports (`k_*`) | Search ordering; Planning preconditions (`ReserveICUBed` needs `CriticalRisk`) |
| `RiskLevel(P, Level)` | Same, as one predicate | `k_level_*` | convenience |
| `Priority(P, n)` | 1 Critical, 2 High, 3 Medium, 4 Low | `k_priority` | Search cost and A* heuristic weight |
| `MaxWait(P, minutes)` | 0 / 10 / 60 / 120 | `k_max_wait` | Search waiting-time penalty |
| `RequiresUrgentEvaluation(P)` | Must be seen by a doctor first | High or Critical (`q_urgent_evaluation`) | Planning: `PerformUrgentEvaluation` |
| `RequiresReassessment(P)` | Evidence uncertain or contradictory | `Disagreement(P)` or `LowConfidence(P, s)` | Planning: `Reevaluate`; Block 6 re-runs models |
| `InsufficientData(P)` | No core vitals and no prediction | `f_insufficient_data` | interface asks for data |
| `RequiresMonitoring(P, Continuous / Hourly / Routine)` | Monitoring level | from the risk level (`q_monitoring`) | Planning: `StartMonitoring` |

## 3. Resource needs (outputs)

| Predicate | Derived when (rule id) |
| --- | --- |
| `RequiresICU(P)` | `CriticalRisk(P)` (`q_icu`) |
| `RequiresEmergencyBed(P)` | `HighRisk(P)` or `MediumRisk(P)` (`q_emergency_bed`) |
| `RequiresExamRoom(P)` | `LowRisk(P)` (`q_exam_room`) |
| `NeedsBedType(P, ICU / Emergency / ExamRoom)` | one of the three above (`q_bed_*`) |
| `RequiresSpecialist(P, S)` | one primary specialist among the candidates, by precedence Cardiology > Neurology > GeneralSurgery > Pulmonology (`q_primary_specialist`) |
| `CandidateSpecialist(P, Cardiology)` | chest pain and High/Critical, or heart disease and Critical |
| `CandidateSpecialist(P, Pulmonology)` | hypoxemia and (shortness of breath, COPD or asthma) |
| `CandidateSpecialist(P, GeneralSurgery)` | abdominal pain and High/Critical |
| `CandidateSpecialist(P, Neurology)` | confusion and High/Critical |
| `RequiresDoctor(P, General)` | no specialist required (`q_general_doctor`) |
| `RequiresNurse(P)` | High or Critical (`q_nurse`) |
| `RequiresEquipment(P, Ventilator)` | Critical and SpO2 < 88 % |
| `RequiresEquipment(P, PatientMonitor)` | High or Critical |
| `RequiresEquipment(P, InfusionPump)` | hypotension |
| `RequiresEquipment(P, Defibrillator)` | chest pain and Critical |

## 4. Matching helpers (outputs)

They list the legal options so Search does not explore impossible states.
They never choose. Only `Waiting` patients get them.

| Predicate | Derived when | Search action it allows |
| --- | --- | --- |
| `CanTreat(D, P)` | `Available(D)` and specialty matches `RequiresSpecialist(P, S)`, or `RequiresDoctor(P, General)` | `AssignDoctor(P, D)` |
| `SuitableBed(B, P)` | `Available(B)` and `Type(B, T)` with `NeedsBedType(P, T)` | `ReserveBed(P, B)` |
| `SuitableEquipment(E, P)` | `Available(E)` and type matches `RequiresEquipment(P, T)` | `AllocateEquipment(P, E)` |
| `CanNurse(N, P)` | `Available(N)` and `RequiresNurse(P)` | `AssignNurse(P, N)` |
| `FallbackDoctor(D, P)` | specialist needed, none can treat, `D` available | `AssignDoctor(P, D)` with a penalty |
| `FallbackBed(B, P)` | `RequiresICU(P)`, no ICU bed suitable, `B` a free emergency bed | `ReserveBed(P, B)` with a penalty |

## 5. Internal predicates (not part of the contract)

`Tachycardia`, `SevereTachycardia`, `Hypotension`, `Hypoxemia`,
`SevereHypoxemia`, `HasFever`, `Tachypnea`, `SevereTachypnea`, `HighLactate`,
`VeryHighLactate`, `Elderly`, `AlteredMentalState` (layer 1);
`Evidence(P, Level, Source)`, `LowConfidence(P, Source)` (layer 2);
`SourceOpinion(P, Source, Level)`, `Disagreement(P)` (layer 3). They appear in
traces and may change without notice.

## 6. JSON export

`KnowledgeBase.requirements()` returns the same information as plain JSON, so
Search never calls Prolog:

```json
{
  "patients": {
    "P1": {
      "risk": "Critical", "priority": 1, "max_wait": 0, "waiting": true,
      "urgent_evaluation": true, "reassess": false, "insufficient_data": false,
      "bed": "ICU", "specialist": "Cardiology", "doctor": null, "nurse": true,
      "equipment": ["Defibrillator", "InfusionPump", "PatientMonitor"],
      "monitoring": "Continuous", "assigned": [],
      "candidates": {
        "doctors": ["D1"], "beds": ["ICU1"], "nurses": ["N1", "N2"],
        "equipment": {"Defibrillator": ["DF1"], "InfusionPump": ["IP1"], "PatientMonitor": ["M1", "M2"]},
        "fallback_doctors": [], "fallback_beds": []
      }
    }
  },
  "resources": {"doctors": [{"id": "D1", "specialty": "Cardiology", "available": true, "assigned_to": null}, ...],
                "nurses": [...], "beds": [...], "equipment": [...]},
  "shortages": [{"resource": "ICU bed", "patients": ["P1", "P4"], "available": 0}],
  "contract": {"P1": ["CanTreat(D1, P1)", "Priority(P1, 1)", "RequiresICU(P1)", "..."]}
}
```

`assigned` lists the resources a patient under treatment holds (`Assigned(P, R)`),
and `assigned_to` names the holder of each resource. Search uses them to move a
patient out of a bed they no longer need (see `docs/block4_search.md`).

`KnowledgeBase.update()` also returns the difference with the previous update,
per patient (`added`, `removed`, `risk`). Block 6 re-runs Search and Planning
when it is not empty.
