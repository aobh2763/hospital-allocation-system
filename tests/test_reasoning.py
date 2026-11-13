"""Block 2 tests.   Run:  python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from environment.scenario import Resource, Scenario  # noqa: E402
from patient import Patient  # noqa: E402
from reasoning import printer  # noqa: E402
from reasoning.events import apply_event  # noqa: E402

try:
    from reasoning.kb import KnowledgeBase

    KnowledgeBase()
    HAVE_PROLOG = True
except Exception as e:  # SWI-Prolog or janus-swi missing
    HAVE_PROLOG, PROLOG_ERROR = False, e

PDF = ROOT / "scenarios" / "pdf_example.json"


def pdf_kb():
    kb = KnowledgeBase()
    kb.load_scenario(Scenario.from_json(PDF))
    kb.update()
    return kb


def one_patient_kb(**kw):
    """A knowledge base with a single waiting patient P1."""
    kb = KnowledgeBase()
    kb.load_scenario(Scenario("test", patients=[Patient("P1", **kw)]))
    kb.update()
    return kb


class PrinterTest(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(printer.to_prolog("RequiresSpecialist(P1, Cardiology)"),
                         "requires_specialist(p1,cardiology)")
        self.assertEqual(printer.to_pdf("requires_icu(p1)"), "RequiresICU(P1)")
        self.assertEqual(printer.to_pdf("ml_risk(p1,high,0.87)"), "ML_Risk(P1, High, 0.87)")
        self.assertEqual(printer.to_pdf("evidence(p1,critical,dl)"), "Evidence(P1, Critical, dl)")
        self.assertEqual(printer.to_prolog("ML_Risk(x, l, 0.5)"), "ml_risk(V_x,V_l,0.5)")
        self.assertEqual(printer.to_atom("ChestPain"), "chest_pain")
        self.assertEqual(printer.to_atom("ICU1"), "icu1")
        self.assertEqual(printer.fact("age", "P1", 72), "age(p1,72)")


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class PdfScenarioTest(unittest.TestCase):
    def setUp(self):
        self.kb = pdf_kb()

    def test_risk_levels(self):
        self.assertEqual(self.kb.risk("P1"), "Critical")
        self.assertEqual(self.kb.risk("P2"), "Medium")
        self.assertEqual(self.kb.risk("P3"), "Low")

    def test_p1_requirements(self):
        kb = self.kb
        for goal in ("RequiresICU(P1)", "RequiresUrgentEvaluation(P1)", "RequiresNurse(P1)",
                     "RequiresSpecialist(P1, Cardiology)", "Priority(P1, 1)", "MaxWait(P1, 0)"):
            self.assertTrue(kb.holds(goal), goal)
        # P1 also matches pulmonology, but only one primary specialist is kept
        self.assertTrue(kb.holds("CandidateSpecialist(P1, Pulmonology)"))
        self.assertFalse(kb.holds("RequiresSpecialist(P1, Pulmonology)"))
        eq = {r["t"] for r in kb.query("RequiresEquipment(P1, t)")}
        self.assertEqual(eq, {"PatientMonitor", "InfusionPump", "Defibrillator"})  # SpO2 88 is not < 88

    def test_p2_p3_requirements(self):
        self.assertTrue(self.kb.holds("RequiresEmergencyBed(P2)"))
        self.assertTrue(self.kb.holds("RequiresDoctor(P2, General)"))
        self.assertTrue(self.kb.holds("RequiresExamRoom(P3)"))
        self.assertFalse(self.kb.holds("RequiresICU(P3)"))

    def test_p2_is_medium_only_because_of_ml(self):
        kb = self.kb
        kb.remove("ml_risk(p2,_,_)")
        kb.update()
        self.assertEqual(kb.risk("P2"), "Low")

    def test_pdf_chain_from_dl(self):
        tree = self.kb.explain("RequiresICU(P1)")
        self.assertEqual(tree["rule"], "q_icu")
        self.assertEqual(tree["premises"][0]["rule"], "k_critical")
        self.assertEqual(tree["premises"][0]["premises"][0]["fact"], "evidence(p1, critical, dl)")

    def test_matching_helpers(self):
        reqs = self.kb.requirements()["patients"]
        self.assertEqual(reqs["P1"]["candidates"]["beds"], ["ICU1"])
        self.assertEqual(reqs["P1"]["candidates"]["doctors"], ["D1"])
        self.assertEqual(reqs["P1"]["candidates"]["equipment"]["PatientMonitor"], ["M1", "M2", "M3"])
        self.assertEqual(reqs["P3"]["candidates"]["beds"], ["EX1"])

    def test_contract_schema(self):
        reqs = self.kb.requirements()
        self.assertEqual(set(reqs), {"patients", "resources", "shortages", "contract"})
        keys = {"risk", "priority", "max_wait", "waiting", "urgent_evaluation", "reassess",
                "insufficient_data", "bed", "specialist", "doctor", "nurse", "equipment",
                "monitoring", "assigned", "candidates"}
        for p in reqs["patients"].values():
            self.assertEqual(set(p), keys)

    def test_deduction_and_resolution_agree(self):
        """Every contract fact derived by forward chaining is also proved by resolution."""
        for facts in self.kb.requirements()["contract"].values():
            for f in facts:
                self.assertTrue(self.kb.prove(f)["proved"], f)

    def test_resolution_trace(self):
        r = self.kb.prove("RequiresICU(P1)")
        self.assertTrue(r["proved"])
        self.assertEqual(r["steps"][0]["id"], "q_icu")
        self.assertEqual(r["steps"][-1]["resolvent"], [])            # empty clause
        self.assertIn("□", self.kb.format_proof("RequiresICU(P1)"))

    def test_failed_proof_explains_why(self):
        r = self.kb.prove("RequiresICU(P3)")
        self.assertFalse(r["proved"])
        self.assertEqual(r["why"]["rules"][0]["rule"], "q_icu")

    def test_negation_as_failure_in_trace(self):
        r = self.kb.prove("LowRisk(P3)")
        self.assertTrue(r["proved"])
        self.assertEqual(sum(s["kind"] == "naf" for s in r["steps"]), 3)


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class UnificationTest(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()

    def test_cases(self):
        u = self.kb.unify("RequiresSpecialist(x, Cardiology)", "RequiresSpecialist(P1, Cardiology)")
        self.assertEqual((u["unified"], u["theta"]), (True, "{x/P1}"))
        u = self.kb.unify("ML_Risk(x, l, p)", "ML_Risk(P1, High, 0.87)")
        self.assertEqual(u["theta"], "{x/P1, l/High, p/0.87}")
        u = self.kb.unify("Type(b, ICU)", "Type(EB1, Emergency)")
        self.assertFalse(u["unified"])
        self.assertIn("ICU vs Emergency", u["reason"])
        self.assertFalse(self.kb.unify("Assigned(x, x)", "Assigned(P1, ICU1)")["unified"])
        u = self.kb.unify("x", "f(x)")
        self.assertFalse(u["unified"])
        self.assertIn("occurs check", u["reason"])


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class ThresholdTest(unittest.TestCase):
    CASES = [  # (vitals or field, finding, value that does not fire, value that fires)
        ("hr", "Tachycardia", 110, 111),
        ("hr", "SevereTachycardia", 130, 131),
        ("map", "Hypotension", 65, 64.9),
        ("sbp", "Hypotension", 90, 89),
        ("spo2", "Hypoxemia", 92, 91),
        ("spo2", "SevereHypoxemia", 88, 87),
        ("temp", "HasFever", 37.9, 38.0),
        ("rr", "Tachypnea", 21, 22),
        ("rr", "SevereTachypnea", 29, 30),
        ("lactate", "HighLactate", 1.9, 2.0),
        ("lactate", "VeryHighLactate", 3.9, 4.0),
    ]

    def test_boundaries(self):
        for key, finding, below, above in self.CASES:
            for value, expected in ((below, False), (above, True)):
                kb = one_patient_kb(vitals={key: value})
                self.assertEqual(kb.holds(f"{finding}(P1)"), expected, f"{finding} at {key}={value}")

    def test_age(self):
        self.assertFalse(one_patient_kb(age=64).holds("Elderly(P1)"))
        self.assertTrue(one_patient_kb(age=65).holds("Elderly(P1)"))


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class EvidenceTest(unittest.TestCase):
    def test_disagreement_triggers_reassessment(self):
        kb = one_patient_kb(vitals={"hr": 80, "spo2": 98})
        kb.set_prediction("P1", "ml", "Low", 0.90)
        kb.set_prediction("P1", "dl", "Critical", 0.85)
        kb.update()
        self.assertEqual(kb.risk("P1"), "Critical")
        self.assertTrue(kb.holds("Disagreement(P1)"))
        self.assertTrue(kb.holds("RequiresReassessment(P1)"))

    def test_low_confidence_is_not_evidence(self):
        kb = one_patient_kb(vitals={"hr": 80, "spo2": 98})
        kb.set_prediction("P1", "dl", "Critical", 0.40)
        kb.update()
        self.assertEqual(kb.risk("P1"), "Low")
        self.assertTrue(kb.holds("RequiresReassessment(P1)"))

    def test_insufficient_data(self):
        kb = one_patient_kb(age=40)
        self.assertTrue(kb.holds("InsufficientData(P1)"))
        self.assertEqual(kb.risk("P1"), "Low")

    def test_shock_rule_is_critical_without_models(self):
        kb = one_patient_kb(vitals={"map": 60, "spo2": 85})
        self.assertEqual(kb.risk("P1"), "Critical")
        self.assertTrue(kb.holds("RequiresEquipment(P1, Ventilator)"))

    def test_exactly_one_risk_level_on_synthea(self):
        from environment import datasets

        raw = datasets.synthea_raw().sample(n=1000, random_state=1)
        patients = [Patient.from_synthea(f"P{i}", row) for i, (_, row) in enumerate(raw.iterrows(), 1)]
        kb = KnowledgeBase()
        kb.load_scenario(Scenario("synthea", patients=patients))
        kb.update()
        levels = {}
        for r in kb.query("RiskLevel(x, l)"):
            levels.setdefault(r["x"], []).append(r["l"])
        self.assertEqual(len(levels), 1000)
        self.assertTrue(all(len(v) == 1 for v in levels.values()))


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class DynamicTest(unittest.TestCase):
    def setUp(self):
        self.kb = pdf_kb()
        self.events = Scenario.from_json(PDF).events

    def test_icu_unavailable(self):
        res = apply_event(self.kb, self.events[0])
        self.assertIn("SuitableBed(ICU1, P1)", res.changes["P1"]["removed"])
        self.assertIn("FallbackBed(EB1, P1)", res.changes["P1"]["added"])
        self.assertTrue(self.kb.holds("RequiresICU(P1)"))

    def test_second_critical_patient_creates_shortage(self):
        res = apply_event(self.kb, self.events[1])
        self.assertEqual(res.changes["P4"]["risk"], (None, "Critical"))
        shortage = self.kb.requirements()["shortages"][0]
        self.assertEqual((shortage["resource"], shortage["patients"]), ("ICU bed", ["P1", "P4"]))

    def test_deterioration_raises_risk(self):
        res = apply_event(self.kb, self.events[2])
        self.assertEqual(res.changes["P2"]["risk"], ("Medium", "High"))
        self.assertIn("RequiresUrgentEvaluation(P2)", res.changes["P2"]["added"])

    def test_assignment_removes_patient_from_allocation(self):
        apply_event(self.kb, {"type": "assign", "pid": "P1", "id": "ICU1"})
        self.assertFalse(self.kb.holds("Waiting(P1)"))
        self.assertEqual(self.kb.requirements()["patients"]["P1"]["candidates"]["beds"], [])
        self.assertTrue(self.kb.holds("RequiresICU(P1)"))

    def test_specialist_missing_gives_fallback_doctor(self):
        self.kb.set_available("D1", False)
        self.kb.update()
        self.assertEqual(self.kb.requirements()["patients"]["P1"]["candidates"]["fallback_doctors"], ["D2"])


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class AdapterTest(unittest.TestCase):
    def test_ml_on_synthea_patient(self):
        from environment import datasets
        from reasoning.adapters import MLAdapter

        ml = MLAdapter()
        p = datasets.synthea_patient("P1", 0)
        level, prob = ml.predict(p)
        self.assertIn(level, ("Low", "Medium", "High", "Critical"))
        self.assertGreater(prob, 0.25)
        self.assertEqual(ml.coverage["P1"], 1.0)

    def test_dl_numpy_matches_training_score(self):
        """The NumPy GRU reproduces the PyTorch model's test macro-F1 (no PyTorch needed)."""
        import json

        from sklearn.metrics import f1_score

        from environment import datasets
        from reasoning.adapters import DL_PREP, LEVELS, DLAdapter, dl_unavailable_reason

        reason = dl_unavailable_reason()
        if reason:
            self.skipTest(reason)
        dl = DLAdapter()
        ts = datasets.timeseries()
        y, pred = [], []
        for pid, g in ts[ts["Split"] == "test"].groupby("Patient_ID"):
            level, _ = dl.predict(datasets.timeseries_patient("P1", pid))
            pred.append(LEVELS.index(level))
            y.append(LEVELS.index(g["Risk"].iloc[-1]))
        f1 = f1_score(y, pred, average="macro")
        self.assertAlmostEqual(f1, json.loads(DL_PREP.read_text())["f1_test"], places=3)

    def test_weight_reader_refuses_unknown_classes(self):
        import io
        import pickle
        import zipfile

        from reasoning.numpy_rnn import load_state_dict

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("m/data.pkl", pickle.dumps(print))   # a pickle that names builtins.print
        path = ROOT / "tests" / "_bad_model.pt"
        path.write_bytes(buf.getvalue())
        try:
            with self.assertRaises(pickle.UnpicklingError):
                load_state_dict(path)
        finally:
            path.unlink()

if __name__ == "__main__":
    unittest.main()
