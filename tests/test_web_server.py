"""HTTP API tests.   Run:  python -m unittest discover -s tests -v"""

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from reasoning.kb import KnowledgeBase

    KnowledgeBase()
    HAVE_PROLOG = True
except Exception:
    HAVE_PROLOG = False

EXAMPLE = {"age": 78, "sex": "f", "symptoms": ["shortness_of_breath", "confusion"], "conditions": ["copd"],
           "vitals": {"hr": 124, "sbp": 92, "dbp": 54, "spo2": 89, "temp": 38.9, "rr": 28, "lactate": 3.4},
           "history": [{"hr": 98, "map": 88, "spo2": 95, "temp": 37.9, "rr": 20, "lactate": 1.6},
                       {"hr": 110, "map": 80, "spo2": 92, "temp": 38.4, "rr": 24, "lactate": 2.3}]}


@unittest.skipUnless(HAVE_PROLOG, "SWI-Prolog / janus-swi not available")
class WebServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from web_server import make_server

        cls.server, cls.api = make_server(port=0, use_models=False, quiet=True)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.call("POST", "/api/reset", {"scenario": "pdf_example.json", "use_models": False})

    def call(self, method, path, body=None, expect=200):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                status, payload = r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                status, payload = e.code, json.loads(e.read())
        self.assertEqual(status, expect, payload)
        return payload

    def test_meta_and_scenarios(self):
        meta = self.call("GET", "/api/meta")
        self.assertEqual(meta["risk_levels"], ["Low", "Medium", "High", "Critical"])
        self.assertIn("chest_pain", meta["symptoms"])
        self.assertEqual(meta["thresholds"]["spo2_low"], 92)
        files = [s["file"] for s in self.call("GET", "/api/scenarios")["scenarios"]]
        self.assertIn("pdf_example.json", files)
        self.assertNotIn("patient_example.json", files)

    def test_state_shape(self):
        st = self.call("GET", "/api/state")
        for key in ("scenario", "patients", "resources", "plan", "next_action", "executed", "allocation",
                    "cycles", "demo", "step", "valid", "announcements"):
            self.assertIn(key, st)
        p1 = next(p for p in st["patients"] if p["pid"] == "P1")
        self.assertEqual(p1["risk"], "Critical")
        self.assertEqual(p1["predictions"]["dl"], ["Critical", 0.91])
        self.assertEqual(st["next_action"], "AssignDoctor(P1, D1)")
        self.assertEqual({r["status"] for r in st["resources"]}, {"free"})

    def test_scripted_scenario(self):
        run = self.call("POST", "/api/run", {"until": "transfer"})
        self.assertEqual(run["state"]["next_action"], "TransferPatient(P1, ICU1)")
        icu = next(r for r in run["state"]["resources"] if r["id"] == "ICU1")
        self.assertEqual((icu["status"], icu["held_by"]), ("reserved", "P1"))
        r = self.call("POST", "/api/demo/next")
        self.assertEqual(r["cycle"]["invalid_step"]["action"], "TransferPatient(P1, ICU1)")
        self.assertIn("P1: bed ICU1 -> EB1", r["cycle"]["allocation_changes"])
        self.call("POST", "/api/run", {"steps": 3})
        r = self.call("POST", "/api/demo/next")
        self.assertEqual(r["changes"]["P4"]["risk"], [None, "Critical"])
        r = self.call("POST", "/api/demo/next")
        self.assertFalse(r["cycle"]["valid"])
        self.call("POST", "/api/demo/next", expect=409)
        done = self.call("POST", "/api/run", {"all": True})
        self.assertTrue(done["done"])

    def test_admit_patient(self):
        self.call("POST", "/api/run", {"steps": 4})
        r = self.call("POST", "/api/patients", EXAMPLE)
        self.assertEqual(r["pid"], "P4")
        self.assertIsNone(r["probabilities"]["ml"])            # models off in this test
        self.assertIn("P4", r["changes"])
        self.assertTrue(r["cycle"]["plan"]["steps"])
        detail = self.call("GET", "/api/patients/P4")
        self.assertIn("explanation", detail)
        self.assertTrue(detail["deductions"])

    def test_scripted_arrival_after_manual_admission_gets_a_new_id(self):
        self.call("POST", "/api/patients", EXAMPLE)               # takes P4
        self.call("POST", "/api/demo/next")
        r = self.call("POST", "/api/demo/next")                   # the script's "P4 arrives"
        self.assertIn("P5", r["changes"])
        self.assertIn("admitted as P5", r["event"]["description"])

    def test_resources_vitals_discharge(self):
        r = self.call("POST", "/api/resources/ICU1/out")
        self.assertEqual(next(x for x in r["state"]["resources"] if x["id"] == "ICU1")["status"], "out_of_service")
        r = self.call("POST", "/api/resources/ICU1/in")
        self.assertEqual(next(x for x in r["state"]["resources"] if x["id"] == "ICU1")["status"], "free")
        r = self.call("POST", "/api/patients/P2/vitals", {"hr": 116, "map": 70, "spo2": 93})
        self.assertEqual(len(next(p for p in r["state"]["patients"] if p["pid"] == "P2")["history"]), 4)
        r = self.call("POST", "/api/patients/P3/discharge")
        self.assertNotIn("P3", [p["pid"] for p in r["state"]["patients"]])

    def test_cycles_and_query(self):
        self.call("POST", "/api/resources/ICU1/out")
        summaries = self.call("GET", "/api/cycles")["cycles"]
        self.assertEqual(len(summaries), 2)
        c = self.call("GET", "/api/cycles/2")
        for key in ("search", "plan", "allocation", "requirements", "changes", "invalid_step"):
            self.assertIn(key, c)
        self.assertEqual(c["search"]["astar"]["cost"], c["search"]["ucs"]["cost"])
        q = self.call("POST", "/api/query", {"text": "RequiresICU(P1)"})
        self.assertTrue(q["proved"])
        self.assertIn("□", q["text"])
        q = self.call("POST", "/api/query", {"text": "RequiresEquipment(P1, t)"})
        self.assertEqual(q["kind"], "rows")
        self.assertIn({"t": "PatientMonitor"}, q["rows"])

    def test_errors(self):
        self.call("GET", "/api/nothing", expect=404)
        self.call("POST", "/api/state", {}, expect=405)
        self.call("POST", "/api/patients", {"age": 300}, expect=400)
        self.call("POST", "/api/patients", {"age": 40, "symptoms": ["flu"]}, expect=400)
        self.call("POST", "/api/patients", {"age": 40, "vitals": {"spo2": 140}}, expect=400)
        self.call("GET", "/api/patients/P99", expect=404)
        self.call("POST", "/api/resources/ZZ9/out", expect=404)
        self.call("POST", "/api/query", {"text": "x; halt"}, expect=400)
        self.call("POST", "/api/reset", {"scenario": "../pyproject.toml"}, expect=400)
        self.call("POST", "/api/run", {"steps": 0}, expect=400)
        self.call("GET", "/api/cycles/99", expect=404)

    def test_empty_department(self):
        r = self.call("POST", "/api/reset", {"scenario": "pdf_example.json", "use_models": False, "empty": True})
        self.assertEqual(r["state"]["patients"], [])
        self.assertEqual(r["state"]["demo"]["events"], 0)
        r = self.call("POST", "/api/patients", EXAMPLE)
        self.assertEqual(r["pid"], "P1")


if __name__ == "__main__":
    unittest.main()
