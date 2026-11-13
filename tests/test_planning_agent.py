"""Blocks 5 and 6 tests.   Run:  python -m unittest discover -s tests -v"""

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planning.strips import Action, plan, validate  # noqa: E402

try:
    from reasoning.kb import KnowledgeBase

    KnowledgeBase()
    HAVE_PROLOG = True
except Exception:
    HAVE_PROLOG = False


def A(name, args, pre, add, delete=()):
    return Action(name, tuple(args), frozenset(pre), frozenset(add), frozenset(delete))


class StripsTest(unittest.TestCase):
    """The planner core on a tiny hand-written domain (the PDF's two actions)."""

    def setUp(self):
        self.assign = A("AssignDoctor", ("P1", "D1"), ["PatientWaiting(P1)", "Available(D1)"],
                        ["DoctorAssigned(P1, D1)"], ["Available(D1)"])
        self.reserve = A("ReserveICUBed", ("P1", "ICU1"), ["CriticalRisk(P1)", "Available(ICU1)"],
                         ["PatientAssignedToICU(P1)"], ["Available(ICU1)"])
        self.init = frozenset({"PatientWaiting(P1)", "Available(D1)", "CriticalRisk(P1)", "Available(ICU1)"})
        self.goals = frozenset({"DoctorAssigned(P1, D1)", "PatientAssignedToICU(P1)"})

    def test_plan_reaches_goals_and_is_valid(self):
        r = plan(self.init, self.goals, [self.assign, self.reserve])
        self.assertTrue(r.found)
        self.assertEqual(validate(self.init, r.steps, self.goals), [])

    def test_no_plan_when_a_precondition_can_never_hold(self):
        init = self.init - {"Available(D1)"}          # D1 busy with another patient
        r = plan(init, self.goals, [self.assign, self.reserve])
        self.assertFalse(r.found)
        self.assertIn("DoctorAssigned(P1, D1)", r.unreachable)

    def test_validator_rejects_bad_order(self):
        transfer = A("TransferPatient", ("P1", "ICU1"), ["PatientAssignedToICU(P1)"], ["In(P1, ICU1)"])
        errors = validate(self.init, [transfer, self.reserve], frozenset({"In(P1, ICU1)"}))
        self.assertTrue(errors and "TransferPatient(P1, ICU1)" in errors[0])


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class HospitalPlanningTest(unittest.TestCase):
    def plan_for(self, name):
        from environment.scenario import Scenario
        from planning import Memory, build
        from search import allocate

        kb = KnowledgeBase()
        kb.load_scenario(Scenario.from_json(ROOT / "scenarios" / name))
        kb.update()
        req = kb.requirements()
        problem = build(req, allocate(req).allocation, Memory())
        result = problem.solve()
        return problem, result, [str(a) for a in result.steps]

    def test_pdf_plan_is_valid_and_ordered(self):
        problem, result, steps = self.plan_for("pdf_example.json")
        self.assertEqual(validate(problem.init, result.steps, problem.goals), [])
        order = steps.index
        # the PDF's order: assign doctor, reserve ICU, urgent evaluation, transfer, monitoring, new evaluation
        self.assertLess(order("AssignDoctor(P1, D1)"), order("PerformUrgentEvaluation(P1, D1)"))
        self.assertLess(order("ReserveICUBed(P1, ICU1)"), order("TransferPatient(P1, ICU1)"))
        self.assertLess(order("PerformUrgentEvaluation(P1, D1)"), order("TransferPatient(P1, ICU1)"))
        self.assertLess(order("TransferPatient(P1, ICU1)"), order("StartMonitoring(P1, M1)"))
        self.assertLess(order("StartMonitoring(P1, M1)"), order("Reevaluate(P1, D1)"))
        self.assertLess(order("Reevaluate(P1, D1)"), order("AssignDoctor(P2, D2)"))   # Critical first

    def test_relocation_comes_before_the_icu_reservation(self):
        problem, result, steps = self.plan_for("icu_reallocation.json")
        self.assertEqual(validate(problem.init, result.steps, problem.goals), [])
        self.assertLess(steps.index("Relocate(P5, ICU1, EB1)"), steps.index("ReserveICUBed(P1, ICU1)"))


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class AgentTest(unittest.TestCase):
    def agent(self, name="pdf_example.json", empty=False):
        from agent import Agent
        from environment.scenario import Scenario

        sc = Scenario.from_json(ROOT / "scenarios" / name)
        self.events, sc.events = sc.events, []
        if empty:
            sc.patients, sc.predictions = [], {}
        ag = Agent(sc, use_models=False, compare_ucs=False)
        ag.step()
        return ag

    def assert_consistent(self, ag):
        """Every plan was valid and no resource is held by two patients."""
        for c in ag.cycles:
            self.assertEqual(c.plan_errors, [], f"cycle {c.number}")
        holders = Counter(r["R"] for r in ag.kb.solutions("assigned(X,R)"))
        self.assertTrue(all(n == 1 for n in holders.values()), holders)

    def test_end_to_end_with_the_pdf_events(self):
        ag = self.agent()
        ag.run(until=lambda a: str(a) == "TransferPatient(P1, ICU1)")
        ag.observe(self.events[0])                                  # ICU1 lost before the transfer
        c = ag.step().cycle
        self.assertEqual(str(c.invalid_step[1]), "TransferPatient(P1, ICU1)")
        self.assertIn("BedReserved(P1, ICU1)", c.invalid_step[2])
        self.assertNotIn("AssignDoctor(P1, D1)", [str(a) for a in ag.plan])   # kept, not redone
        ag.run(max_steps=3)
        ag.observe(self.events[1])                                  # P4 arrives
        c = ag.step().cycle
        self.assertEqual(c.changes["P4"]["risk"], (None, "Critical"))
        ag.run(max_steps=3)
        ag.observe(self.events[2])                                  # P2 deteriorates
        c = ag.step().cycle
        self.assertFalse(c.decision.valid)
        ag.run()
        self.assertTrue(ag.kb.holds("In(P1, EB1)") or ag.kb.holds("UnderTreatment(P1)"))
        self.assertTrue(ag.kb.holds("UnderTreatment(P4)"))
        self.assertTrue(ag.kb.holds("Waiting(P2)"))
        self.assert_consistent(ag)

    def test_manual_patient_is_planned_for(self):
        from patient import Patient

        ag = self.agent(empty=True)
        self.assertEqual(ag.plan, [])
        p = Patient("P1", age=70, vitals={"hr": 128, "sbp": 80, "dbp": 48, "spo2": 86, "rr": 30},
                    symptoms=["chest_pain"])
        ag.observe({"type": "new_patient", "patient": p.to_dict()})
        c = ag.step().cycle
        self.assertEqual(c.requirements["patients"]["P1"]["risk"], "Critical")    # shock rule
        self.assertIn("TransferPatient(P1, ICU1)", [str(a) for a in ag.plan])
        ag.run()
        self.assertTrue(ag.kb.holds("In(P1, ICU1)") or ag.kb.holds("UnderTreatment(P1)"))
        self.assert_consistent(ag)

    def test_reserved_bed_goes_to_the_more_urgent_newcomer(self):
        from patient import Patient

        ag = self.agent()
        ag.run(until=lambda a: a.name == "TransferPatient")      # P2 not served yet, EB1 planned for P2
        newcomer = Patient("P4", age=60, vitals={"hr": 125, "sbp": 85, "dbp": 50, "spo2": 90, "rr": 26},
                           symptoms=["shortness_of_breath"])
        ag.observe({"type": "new_patient", "patient": newcomer.to_dict()})
        c = ag.step().cycle
        self.assertIn("P4", c.decision.allocation["patients"])
        self.assert_consistent(ag)

    def test_resource_back_in_service_and_discharge(self):
        ag = self.agent()
        ag.observe({"type": "resource_unavailable", "id": "ICU1"})
        ag.step()
        ag.run()
        self.assertTrue(ag.kb.holds("UnderTreatment(P1)"))
        ag.observe({"type": "resource_available", "id": "ICU1"})
        c = ag.step().cycle
        self.assertIn({"patient": "P1", "from": "EB1", "to": "ICU1"}, c.decision.allocation["relocations"])
        ag.run()
        self.assertTrue(ag.kb.holds("In(P1, ICU1)") or "ICU1" in
                        [r["R"].upper() for r in ag.kb.solutions("assigned(p1,R)")])
        ag.observe({"type": "discharge", "pid": "P1"})
        c = ag.step().cycle
        self.assertIn("P3", c.decision.allocation["patients"])               # D1 is free again
        ag.run()
        self.assert_consistent(ag)

    def test_models_drive_the_agent(self):
        from agent import Agent
        from environment.scenario import Scenario
        from reasoning.adapters import dl_unavailable_reason

        if dl_unavailable_reason():
            self.skipTest(dl_unavailable_reason())
        sc = Scenario.from_json(ROOT / "scenarios" / "pdf_example.json")
        events, sc.events = sc.events, []
        ag = Agent(sc, use_models=True, compare_ucs=False)
        c = ag.step().cycle
        self.assertIn("P1", c.predictions)                                   # computed, not given
        self.assertNotEqual(c.predictions["P1"]["dl"][1], 0.91)
        for ev in events:
            ag.observe(ev)
            ag.step()
            ag.run(max_steps=3)
        ag.run()
        self.assert_consistent(ag)


if __name__ == "__main__":
    unittest.main()
