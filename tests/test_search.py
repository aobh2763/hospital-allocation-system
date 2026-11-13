"""Block 4 tests.   Run:  python -m unittest discover -s tests -v"""

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from search import allocate, check_allocation, from_requirements, search  # noqa: E402
from search.problem import COST_FALLBACK_BED, COST_RELOCATE  # noqa: E402

PRIORITY = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
BED = {"Critical": "ICU", "High": "Emergency", "Medium": "Emergency", "Low": "ExamRoom"}


def patient(risk, specialist=None, nurse=None, equipment=(), waiting=True, assigned=(), bed=None):
    return {"risk": risk, "priority": PRIORITY[risk], "waiting": waiting, "bed": bed or BED[risk],
            "specialist": specialist, "doctor": None if specialist else "General",
            "nurse": risk in ("Critical", "High") if nurse is None else nurse,
            "equipment": list(equipment), "assigned": list(assigned)}


def requirements(patients, doctors=(), nurses=0, beds=(), equipment=(), holders=None):
    holders = holders or {}

    def res(rid, **kw):
        return {"id": rid, "available": rid not in holders, "assigned_to": holders.get(rid), **kw}

    return {"patients": patients, "resources": {
        "doctors": [res(f"D{i}", specialty=s) for i, s in enumerate(doctors, 1)],
        "nurses": [res(f"N{i}") for i in range(1, nurses + 1)],
        "beds": [res(rid, type=t) for rid, t in beds],
        "equipment": [res(rid, type=t) for rid, t in equipment]}}


def random_requirements(rng: random.Random):
    n = rng.randint(2, 6)
    pats = {}
    for i in range(1, n + 1):
        risk = rng.choice(["Critical", "High", "Medium", "Low"])
        spec = rng.choice([None, None, "Cardiology", "Neurology"]) if risk in ("Critical", "High") else None
        eq = rng.sample(["PatientMonitor", "InfusionPump", "Ventilator"], rng.randint(0, 2)) \
            if risk in ("Critical", "High") else []
        pats[f"P{i}"] = patient(risk, spec, equipment=eq)
    doctors = [rng.choice(["General", "Cardiology", "Neurology", "General"]) for _ in range(rng.randint(1, 4))]
    beds = ([(f"ICU{i}", "ICU") for i in range(1, rng.randint(0, 2) + 1)]
            + [(f"EB{i}", "Emergency") for i in range(1, rng.randint(1, 4) + 1)]
            + [(f"EX{i}", "ExamRoom") for i in range(1, rng.randint(0, 2) + 1)])
    equipment = [(f"{t[:2].upper()}{i}", t) for t in ("PatientMonitor", "InfusionPump", "Ventilator")
                 for i in range(1, rng.randint(1, 3) + 1)]
    holders = {}
    if rng.random() < 0.4 and any(b[1] == "ICU" for b in beds):       # someone recovering in ICU1
        pats["P9"] = patient("Medium", waiting=False, assigned=["ICU1"])
        holders["ICU1"] = "P9"
    return requirements(pats, doctors, rng.randint(1, 4), beds, equipment, holders)


class RandomProblemsTest(unittest.TestCase):
    """Properties checked on 150 random departments."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(42)
        cls.cases = [random_requirements(rng) for _ in range(150)]

    def test_ucs_and_astar_agree(self):
        found = 0
        for req in self.cases:
            problem = from_requirements(req)
            u, a = search(problem, "ucs"), search(problem, "astar")
            self.assertEqual(u.found, a.found)
            if u.found:
                found += 1
                self.assertEqual(u.cost, a.cost)
                self.assertLessEqual(a.expanded, u.expanded)
                self.assertEqual(check_allocation(problem, a), [])
                self.assertEqual(check_allocation(problem, u), [])
        self.assertGreater(found, 30)               # the generator makes plenty of solvable cases

    def test_heuristic_never_overestimates(self):
        for req in self.cases:
            problem = from_requirements(req)
            r = search(problem, "ucs")
            if not r.found:
                continue
            state, remaining = problem.initial, r.cost
            for action in r.actions:                # h at every state of an optimal path
                self.assertLessEqual(problem.h(state), remaining)
                state = next(s2 for a2, s2 in problem.successors(state) if a2 == action)
                remaining -= action.cost
            self.assertEqual(problem.h(state), 0)

    def test_dead_end_means_no_solution(self):
        for req in self.cases:
            problem = from_requirements(req)
            if problem.h(problem.initial) == float("inf"):
                self.assertFalse(search(problem, "ucs").found)

    def test_symmetry_breaking_keeps_the_optimum(self):
        for req in self.cases[:40]:
            a = search(from_requirements(req), "astar")
            b = search(from_requirements(req, symmetry=False), "astar", limit=100_000)
            if a.found and b.found:
                self.assertEqual(a.cost, b.cost)


class RulesTest(unittest.TestCase):
    def test_no_resource_shared_and_mandatory_served(self):
        req = requirements({"P1": patient("Critical", equipment=["PatientMonitor"]),
                            "P2": patient("High", equipment=["PatientMonitor"])},
                           doctors=["General", "General"], nurses=2,
                           beds=[("ICU1", "ICU"), ("EB1", "Emergency")],
                           equipment=[("M1", "PatientMonitor"), ("M2", "PatientMonitor")])
        d = allocate(req)
        self.assertTrue(d.valid)
        used = [a.resource for a in d.result.actions]
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual(d.allocation["patients"]["P1"]["bed"], "ICU1")

    def test_relocation_preferred_over_fallback(self):
        req = requirements({"P1": patient("Critical"), "P9": patient("Medium", waiting=False, assigned=["ICU1"])},
                           doctors=["General"], nurses=1,
                           beds=[("ICU1", "ICU"), ("EB1", "Emergency"), ("EB2", "Emergency")],
                           holders={"ICU1": "P9"})
        d = allocate(req)
        names = [str(a) for a in d.result.actions]
        self.assertIn("Relocate(P9, ICU1 → EB1)", names)
        self.assertIn("ReserveBed(P1, ICU1)", names)
        self.assertEqual(d.result.breakdown["relocation"], COST_RELOCATE)
        self.assertTrue(any("moved to EB1" in n for n in d.result.announcements))

    def test_fallback_announced_when_nothing_can_be_freed(self):
        req = requirements({"P1": patient("Critical"), "P9": patient("Critical", waiting=False, assigned=["ICU1"])},
                           doctors=["General"], nurses=1,
                           beds=[("ICU1", "ICU"), ("EB1", "Emergency")], holders={"ICU1": "P9"})
        d = allocate(req)
        self.assertTrue(d.valid)
        self.assertEqual(d.allocation["patients"]["P1"]["bed"], "EB1")
        self.assertEqual(d.result.breakdown["fallback bed"], 1 + COST_FALLBACK_BED)
        self.assertTrue(any("needs an ICU bed" in n for n in d.result.announcements))

    def test_medium_served_before_low(self):
        req = requirements({"P1": patient("Medium"), "P2": patient("Low", bed="Emergency")},
                           doctors=["General", "General"], beds=[("EB1", "Emergency")])
        d = allocate(req)
        self.assertEqual(d.allocation["waiting"], ["P2"])

    def test_no_valid_allocation_is_reported(self):
        req = requirements({"P1": patient("High"), "P2": patient("High")},
                           doctors=["General", "General"], nurses=1,
                           beds=[("EB1", "Emergency"), ("EB2", "Emergency")])
        d = allocate(req)
        self.assertFalse(d.valid)
        self.assertTrue(any("nurse" in m for m in d.diagnosis))
        self.assertEqual(search(d.problem, "astar").found, True)           # relaxed search still acts
        self.assertTrue(any("CANNOT BE SERVED" in n for n in d.result.announcements))
        self.assertEqual(search(from_requirements(req), "astar").expanded, 0)  # A* proves it at once


try:
    from reasoning.kb import KnowledgeBase

    KnowledgeBase()
    HAVE_PROLOG = True
except Exception:
    HAVE_PROLOG = False


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class AgentScenariosTest(unittest.TestCase):
    """Block 2 requirements -> Block 4 allocation on the scenario files."""

    def load(self, name):
        from environment.scenario import Scenario

        scenario = Scenario.from_json(ROOT / "scenarios" / name)
        kb = KnowledgeBase()
        kb.load_scenario(scenario)
        kb.update()
        return kb, scenario

    def test_pdf_scenario_and_events(self):
        from reasoning.events import apply_event

        kb, scenario = self.load("pdf_example.json")
        d = allocate(kb.requirements())
        self.assertTrue(d.valid)
        self.assertEqual(d.allocation["patients"]["P1"]["bed"], "ICU1")
        self.assertEqual(d.allocation["patients"]["P1"]["doctor"], "D1")        # the cardiologist

        apply_event(kb, scenario.events[0])                                       # ICU1 unavailable
        d = allocate(kb.requirements())
        self.assertTrue(d.valid)
        self.assertEqual(d.allocation["patients"]["P1"]["bed"], "EB1")
        self.assertTrue(any("needs an ICU bed" in n for n in d.result.announcements))

        apply_event(kb, scenario.events[1])                                       # P4 arrives
        self.assertTrue(allocate(kb.requirements()).valid)

        apply_event(kb, scenario.events[2])                                       # P2 becomes High
        d = allocate(kb.requirements())
        self.assertFalse(d.valid)
        self.assertIn("P2", d.allocation["waiting"])

    def test_icu_reallocation(self):
        kb, _ = self.load("icu_reallocation.json")
        d = allocate(kb.requirements())
        self.assertTrue(d.valid)
        self.assertEqual(d.allocation["relocations"], [{"patient": "P5", "from": "ICU1", "to": "EB1"}])
        self.assertEqual(d.allocation["patients"]["P1"]["bed"], "ICU1")

    def test_free_resources_match_block2_candidates(self):
        """Search's ideal options at the start are exactly Block 2's matching predicates."""
        kb, _ = self.load("pdf_example.json")
        req = kb.requirements()
        problem = from_requirements(req, symmetry=False)
        s = problem.initial
        for pid in problem.order:
            cand = req["patients"][pid]["candidates"]
            for slot in (sl for sl in problem.slots if sl.pid == pid):
                ideal = {a.resource for a in problem._fill(slot, s) if a.category == "assignment"
                         or (slot.kind == "doctor" and a.category == "specialist time")}
                if slot.kind == "bed":
                    self.assertEqual(ideal, set(cand["beds"]), pid)
                elif slot.kind == "doctor":
                    self.assertEqual(ideal, set(cand["doctors"]), pid)
                elif slot.kind == "nurse":
                    self.assertEqual(ideal, set(cand["nurses"]), pid)


if __name__ == "__main__":
    unittest.main()
