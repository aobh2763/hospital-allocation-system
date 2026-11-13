"""HTTP API for the agent, and static hosting for the web frontend.

    poetry run python src/web_server.py                   # http://127.0.0.1:8000
    poetry run python src/web_server.py --port 8080 --given

Standard library only. SWI-Prolog runs once per process and the agent is a
single shared object, so the server is single-threaded and every request runs
under one lock. The web UI (``front/frontend/``, or ``--frontend DIR``) is served
at ``/``; the API lives
under ``/api``. Every response is JSON; errors are ``{"error": "..."}`` with a
4xx/5xx status. The full contract is in docs/api.md.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent import Agent
from environment.scenario import Scenario
from patient import SYMPTOMS, SYNTHEA_CONDITIONS
from patient_cli import CURRENT_VITALS, HISTORY_COLUMNS, build_patient
from reasoning.printer import to_pdf

ROOT = Path(__file__).resolve().parents[1]
# where the web UI lives: the first of these folders that has an index.html
FRONTEND_CANDIDATES = [ROOT / "front" / "frontend", ROOT / "frontend"]
FRONTEND = next((d for d in FRONTEND_CANDIDATES if (d / "index.html").is_file()), FRONTEND_CANDIDATES[0])
SCENARIOS = ROOT / "scenarios"
DEFAULT_SCENARIO = "pdf_example.json"
LEVELS = ["Low", "Medium", "High", "Critical"]
HISTORY_RANGES = {"hr": (20, 250), "map": (20, 200), "spo2": (50, 100), "temp": (30, 43),
                  "rr": (4, 60), "lactate": (0, 20)}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# ====================================================================== serializers

def action_json(a) -> dict[str, Any]:
    return {"label": str(a), "name": a.name, "args": list(a.args), "patient": a.patient,
            "priority": a.priority, "pre": sorted(a.pre), "add": sorted(a.add), "delete": sorted(a.delete)}


def search_json(dec) -> dict[str, Any] | None:
    if dec is None:
        return None
    r = dec.result
    return {"algorithm": r.algorithm, "valid": dec.valid, "found": r.found, "cost": r.cost,
            "expanded": r.expanded, "generated": r.generated, "max_frontier": r.max_frontier,
            "ms": round(r.seconds * 1000, 3), "h_start": r.h_start if r.h_start != float("inf") else None,
            "breakdown": r.breakdown, "actions": [str(a) for a in r.actions],
            "announcements": r.announcements, "diagnosis": dec.diagnosis}


def cycle_summary(c) -> dict[str, Any]:
    return {"number": c.number, "reasons": c.reasons, "valid": c.decision.valid,
            "cost": c.decision.result.cost, "astar_expanded": c.decision.result.expanded,
            "ucs_expanded": c.ucs.result.expanded if c.ucs else None,
            "plan_length": len(c.plan.steps), "plan_valid": not c.plan_errors}


def cycle_json(c) -> dict[str, Any]:
    inv = None
    if c.invalid_step:
        i, action, missing = c.invalid_step
        inv = {"index": i, "action": str(action), "missing": missing}
    return {
        **cycle_summary(c),
        "invalid_step": inv,
        "changes": c.changes,
        "new_predictions": {pid: {src: list(v) for src, v in by.items()} for pid, by in c.predictions.items()},
        "requirements": c.requirements,
        "search": {"astar": search_json(c.decision), "ucs": search_json(c.ucs)},
        "allocation": c.decision.allocation,
        "allocation_changes": c.allocation_changes,
        "plan": {"found": c.plan.found, "steps": [action_json(a) for a in c.plan.steps],
                 "expanded": c.plan.expanded, "ms": round(c.plan.seconds * 1000, 3),
                 "errors": c.plan_errors, "unreachable": c.plan.unreachable},
        "timings_ms": {k: round(v * 1000, 3) for k, v in c.seconds.items()},
    }


# ====================================================================== the API

class Api:
    """All endpoints. ``handle`` is called with the lock held."""

    def __init__(self, scenario: str = DEFAULT_SCENARIO, use_models: bool = True) -> None:
        self.reset({"scenario": scenario, "use_models": use_models})

    # ------------------------------------------------------------------ routing

    def handle(self, method: str, path: str, body: dict[str, Any]) -> Any:
        routes = [
            ("GET", r"/api/meta", self.meta),
            ("GET", r"/api/scenarios", self.scenarios),
            ("GET", r"/api/state", self.state),
            ("POST", r"/api/reset", self.reset),
            ("POST", r"/api/step", self.step),
            ("POST", r"/api/run", self.run),
            ("POST", r"/api/event", self.event),
            ("GET", r"/api/demo", self.demo),
            ("POST", r"/api/demo/next", self.demo_next),
            ("POST", r"/api/patients", self.admit),
            ("GET", r"/api/patients/(?P<pid>P\d+)", self.patient),
            ("POST", r"/api/patients/(?P<pid>P\d+)/vitals", self.vitals),
            ("POST", r"/api/patients/(?P<pid>P\d+)/discharge", self.discharge),
            ("POST", r"/api/resources/(?P<rid>[A-Z]+\d+)/(?P<state>out|in)", self.resource),
            ("GET", r"/api/cycles", self.cycles),
            ("GET", r"/api/cycles/(?P<n>\d+)", self.cycle),
            ("POST", r"/api/query", self.query),
        ]
        known_path = False
        for m, pattern, fn in routes:
            match = re.fullmatch(pattern, path)
            if match:
                known_path = True
                if m == method:
                    return fn(body or {}, **match.groupdict())
        if known_path:
            raise ApiError(405, f"{method} is not allowed on {path}")
        raise ApiError(404, f"{path} not found")

    # ------------------------------------------------------------------ reference data

    def meta(self, body: dict) -> dict:
        kb = self.agent.kb
        thresholds = {str(r["N"]): r["V"] for r in kb.solutions("threshold(N,V)")}
        return {
            "risk_levels": LEVELS,
            "symptoms": list(SYMPTOMS),
            "conditions": list(SYNTHEA_CONDITIONS.values()),
            "vitals": [{"key": k, "label": label, "unit": unit, "min": lo, "max": hi}
                       for k, label, unit, (lo, hi) in CURRENT_VITALS],
            "history_columns": [{"key": k, "min": HISTORY_RANGES[k][0], "max": HISTORY_RANGES[k][1]}
                                for k in HISTORY_COLUMNS],
            "history_max_rows": 9,
            "thresholds": thresholds,
            "models": {"ml": self.agent.ml is not None, "dl": self.agent.dl is not None},
            "disclaimer": "Teaching simulation. Not a medical device; no real diagnosis or treatment advice.",
        }

    def scenarios(self, body: dict) -> dict:
        out = []
        for f in sorted(SCENARIOS.glob("*.json")):
            d = json.loads(f.read_text())
            if "resources" in d:
                out.append({"file": f.name, "name": d.get("name", f.stem), "notes": d.get("notes", []),
                            "patients": len(d.get("patients", [])), "events": len(d.get("events", []))})
        return {"scenarios": out, "current": self.scenario_file}

    # ------------------------------------------------------------------ state

    def state(self, body: dict | None = None) -> dict:
        ag = self.agent
        req = ag.kb.requirements()
        preds: dict[str, dict] = {}
        for src in ("ML", "DL"):
            for r in ag.kb.query(f"{src}_Risk(x, l, p)"):
                preds.setdefault(r["x"], {})[src.lower()] = [r["l"], float(r["p"])]
        patients = []
        for p in ag.scenario.patients:
            q = req["patients"].get(p.pid, {})
            patients.append({
                "pid": p.pid, "age": p.age, "sex": p.sex, "symptoms": p.symptoms, "conditions": p.conditions,
                "vitals": p.vitals, "history": p.history, "source": p.source.get("dataset", "form"),
                "status": "waiting" if q.get("waiting") else "under_treatment",
                "risk": q.get("risk"), "priority": q.get("priority"), "max_wait": q.get("max_wait"),
                "holds": q.get("assigned", []),
                "needs": {"bed": q.get("bed"), "doctor": q.get("specialist") or q.get("doctor"),
                          "specialist": q.get("specialist") is not None, "nurse": q.get("nurse"),
                          "equipment": q.get("equipment", []), "monitoring": q.get("monitoring")},
                "flags": {"urgent_evaluation": q.get("urgent_evaluation", False),
                          "reassess": q.get("reassess", False),
                          "insufficient_data": q.get("insufficient_data", False)},
                "predictions": preds.get(p.pid, {}),
            })
        patients.sort(key=lambda x: (x["priority"] or 9, int(x["pid"][1:])))
        holders = {r["id"]: r.get("assigned_to") for k in ("doctors", "nurses", "beds", "equipment")
                   for r in req["resources"][k]}
        waiting = {p["pid"] for p in patients if p["status"] == "waiting"}
        resources = []
        for r in ag.scenario.resources:
            held = holders.get(r.rid)
            if r.rid in ag.out_of_service:
                status = "out_of_service"
            elif held:
                status = "reserved" if held in waiting else "in_use"
            else:
                status = "free"
            resources.append({"id": r.rid, "kind": r.kind, "type": r.subtype,
                              "type_label": to_pdf(r.subtype) if r.subtype else None,
                              "status": status, "held_by": held})
        latest = ag.cycles[-1] if ag.cycles else None
        return {
            "scenario": {"file": self.scenario_file, "name": ag.scenario.name, "notes": ag.scenario.notes},
            "use_models": ag.use_models,
            "models": {"ml": ag.ml is not None, "dl": ag.dl is not None},
            "step": len(ag.executed),
            "patients": patients,
            "resources": resources,
            "plan": [action_json(a) for a in ag.plan],
            "next_action": str(ag.plan[0]) if ag.plan else None,
            "executed": [{"cycle": n, **action_json(a)} for n, a in ag.executed],
            "allocation": latest.decision.allocation if latest else None,
            "valid": latest.decision.valid if latest else None,
            "announcements": latest.decision.result.announcements if latest else [],
            "cycles": [cycle_summary(c) for c in ag.cycles],
            "pending_changes": list(ag._pending),
            "demo": {"events": len(self.events), "next_index": self.event_index},
        }

    # ------------------------------------------------------------------ lifecycle

    def reset(self, body: dict) -> dict:
        file = body.get("scenario", getattr(self, "scenario_file", DEFAULT_SCENARIO))
        path = (SCENARIOS / file).resolve()
        if path.parent != SCENARIOS.resolve() or not path.is_file():
            raise ApiError(400, f"unknown scenario {file!r}")
        if "resources" not in json.loads(path.read_text()):
            raise ApiError(400, f"{file} is a patient file, not a scenario")
        scenario = Scenario.from_json(path)
        self.events, scenario.events = scenario.events, []
        if body.get("empty"):
            scenario.patients, scenario.predictions, scenario.assignments = [], {}, []
            self.events = []
        self.event_index = 0
        self.scenario_file = path.name
        self.agent = Agent(scenario, use_models=bool(body.get("use_models", True)))
        first = self.agent.step().cycle
        return {"cycle": cycle_json(first), "state": self.state()}

    def step(self, body: dict) -> dict:
        r = self.agent.step()
        return {"kind": r.kind, "action": action_json(r.action) if r.action else None,
                "cycle": cycle_json(r.cycle) if r.cycle else None, "state": self.state()}

    def run(self, body: dict) -> dict:
        ag = self.agent
        if body.get("until") == "transfer":
            results = ag.run(until=lambda a: a.name == "TransferPatient")
        elif body.get("all"):
            results = ag.run()
        else:
            steps = body.get("steps", 1)
            if not isinstance(steps, int) or not 1 <= steps <= 1000:
                raise ApiError(400, "steps must be an integer between 1 and 1000")
            results = ag.run(max_steps=steps)
        return {"executed": [action_json(r.action) for r in results if r.kind == "executed"],
                "cycles": [cycle_json(r.cycle) for r in results if r.cycle],
                "done": not ag.plan, "state": self.state()}

    # ------------------------------------------------------------------ changes

    def _observe_and_deliberate(self, event: dict) -> dict:
        changes = self.agent.observe(event)
        r = self.agent.step()
        return {"changes": changes, "cycle": cycle_json(r.cycle), "state": self.state()}

    def event(self, body: dict) -> dict:
        kind = body.get("type")
        if kind not in ("resource_unavailable", "resource_available", "new_measurement", "discharge"):
            raise ApiError(400, "type must be resource_unavailable, resource_available, new_measurement "
                                "or discharge (use POST /api/patients for a new patient)")
        if kind.startswith("resource"):
            self._check_resource(body.get("id"))
        else:
            self._check_patient(body.get("pid"))
        if kind == "new_measurement":
            body = {**body, "measurement": self._check_measurement(body.get("measurement"))}
        return self._observe_and_deliberate(body)

    def resource(self, body: dict, rid: str, state: str) -> dict:
        self._check_resource(rid)
        return self._observe_and_deliberate(
            {"type": "resource_unavailable" if state == "out" else "resource_available", "id": rid})

    def vitals(self, body: dict, pid: str) -> dict:
        self._check_patient(pid)
        return self._observe_and_deliberate({"type": "new_measurement", "pid": pid,
                                             "measurement": self._check_measurement(body),
                                             "description": f"new vital signs for {pid}"})

    def discharge(self, body: dict, pid: str) -> dict:
        self._check_patient(pid)
        return self._observe_and_deliberate({"type": "discharge", "pid": pid})

    def admit(self, body: dict) -> dict:
        ag = self.agent
        age = body.get("age")
        if not isinstance(age, (int, float)) or not 0 <= age <= 120:
            raise ApiError(400, "age is required, between 0 and 120")
        sex = body.get("sex")
        if sex not in (None, "m", "f"):
            raise ApiError(400, "sex must be 'm', 'f' or null")
        symptoms = self._check_list(body.get("symptoms", []), SYMPTOMS, "symptom")
        conditions = self._check_list(body.get("conditions", []), tuple(SYNTHEA_CONDITIONS.values()), "condition")
        vitals = {}
        for key, label, unit, (lo, hi) in CURRENT_VITALS:
            v = (body.get("vitals") or {}).get(key)
            if v is None:
                continue
            if not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ApiError(400, f"{label} must be between {lo} and {hi} {unit}")
            vitals[key] = float(v)
        history = body.get("history", [])
        if not isinstance(history, list) or len(history) > 9:
            raise ApiError(400, "history must be a list of at most 9 hourly rows")
        history = [self._check_measurement(row, partial=True) for row in history]
        pid = ag.scenario.next_patient_id()
        p = build_patient(age, sex, symptoms, conditions, vitals, bool(body.get("low_bp", False)), history, pid)
        probabilities = {
            "ml": ag.ml.probabilities(p) if ag.ml else None,
            "dl": ag.dl.probabilities(p) if ag.dl and len(p.history) >= 2 else None,
            "ml_features_known": None,
        }
        if ag.ml:
            probabilities["ml_features_known"] = round(ag.ml.coverage[pid], 3)
        result = self._observe_and_deliberate({"type": "new_patient", "patient": p.to_dict(),
                                               "description": f"new patient {pid} arrives"})
        return {"pid": pid, "probabilities": probabilities, **result}

    # ------------------------------------------------------------------ scripted scenario

    def demo(self, body: dict) -> dict:
        return {"events": [{"index": i, "description": e.get("description", e["type"]), "type": e["type"],
                            "played": i < self.event_index} for i, e in enumerate(self.events)],
                "next_index": self.event_index}

    def demo_next(self, body: dict) -> dict:
        if self.event_index >= len(self.events):
            raise ApiError(409, "no scripted event left; POST /api/reset to start again")
        event = dict(self.events[self.event_index])
        self.event_index += 1
        description = event.get("description", event["type"])
        if event["type"] == "new_patient":
            # a patient admitted by hand may already use the scripted id
            pid = event["patient"]["pid"]
            if any(p.pid == pid for p in self.agent.scenario.patients):
                new = self.agent.scenario.next_patient_id()
                event["patient"] = {**event["patient"], "pid": new}
                description = f"{description} (admitted as {new}: {pid} is taken)"
                event["description"] = description
        return {"event": {"index": self.event_index - 1, "description": description},
                **self._observe_and_deliberate(event)}

    # ------------------------------------------------------------------ inspection

    def patient(self, body: dict, pid: str) -> dict:
        ag = self.agent
        p = self._check_patient(pid)
        state = next(x for x in self.state()["patients"] if x["pid"] == pid)
        probabilities = {"ml": ag.ml.probabilities(p) if ag.ml else None,
                         "dl": ag.dl.probabilities(p) if ag.dl and len(p.history) >= 2 else None}
        risk = state["risk"]
        latest = ag.cycles[-1] if ag.cycles else None
        return {
            **state,
            "probabilities": probabilities,
            "explanation": ag.kb.format_explanation(f"{risk}Risk({pid})") if risk else None,
            "proof": ag.kb.format_proof(f"{risk}Risk({pid})") if risk else None,
            "deductions": [s for s in ag.kb.deductions() if re.search(rf"\b{pid}\b", s["fact_pdf"])],
            "allocation": (latest.decision.allocation.get("patients", {}).get(pid) if latest else None),
            "left_waiting": bool(latest and pid in latest.decision.allocation.get("waiting", [])),
        }

    def cycles(self, body: dict) -> dict:
        return {"cycles": [cycle_summary(c) for c in self.agent.cycles]}

    def cycle(self, body: dict, n: str) -> dict:
        i = int(n)
        if not 1 <= i <= len(self.agent.cycles):
            raise ApiError(404, f"no deliberation {i}")
        return cycle_json(self.agent.cycles[i - 1])

    def query(self, body: dict) -> dict:
        text = (body.get("text") or "").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\([A-Za-z0-9_., ]*\))?", text):
            raise ApiError(400, "write one predicate, e.g. RequiresICU(P1) or RequiresEquipment(P1, t)")
        kb = self.agent.kb
        args = text[text.find("(") + 1:text.rfind(")")].split(",") if "(" in text else []
        variables = [a.strip() for a in args if re.fullmatch(r"[a-z]\d*", a.strip())]
        try:
            if variables:
                return {"kind": "rows", "query": text, "variables": variables, "rows": kb.query(text)}
            proof = kb.prove(text)
            return {"kind": "proof", "query": text, "proved": proof["proved"], "text": kb.format_proof(text)}
        except Exception as e:
            raise ApiError(400, f"could not run {text}: {e}") from None

    # ------------------------------------------------------------------ validation helpers

    def _check_patient(self, pid):
        try:
            return self.agent.scenario.patient(pid)
        except StopIteration:
            raise ApiError(404, f"no patient {pid}") from None

    def _check_resource(self, rid):
        try:
            return self.agent.scenario.resource(rid)
        except StopIteration:
            raise ApiError(404, f"no resource {rid}") from None

    @staticmethod
    def _check_list(values, allowed, what):
        if not isinstance(values, list) or any(v not in allowed for v in values):
            raise ApiError(400, f"each {what} must be one of: {', '.join(allowed)}")
        return list(dict.fromkeys(values))

    @staticmethod
    def _check_measurement(row, partial: bool = False) -> dict[str, float]:
        if not isinstance(row, dict):
            raise ApiError(400, "a measurement is an object such as {\"hr\": 120, \"map\": 70}")
        out = {}
        for k, v in row.items():
            if v is None:
                continue
            if k not in HISTORY_RANGES:
                raise ApiError(400, f"unknown measurement {k!r}; use {', '.join(HISTORY_COLUMNS)}")
            lo, hi = HISTORY_RANGES[k]
            if not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ApiError(400, f"{k} must be between {lo} and {hi}")
            out[k] = float(v)
        if not out:
            raise ApiError(400, "a measurement needs at least one value")
        return out


# ====================================================================== HTTP

def make_handler(api: Api, lock: threading.Lock, frontend: Path = FRONTEND, quiet: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EDAgent/1.0"

        def log_message(self, fmt, *args):          # one short line per request
            if not quiet:
                sys.stderr.write(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")

        def _send(self, status: int, payload: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, default=_default, allow_nan=False).encode(),
                       "application/json; charset=utf-8")

        def do_OPTIONS(self):
            self._send(204, b"", "text/plain")

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def _dispatch(self, method: str) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if not path.startswith("/api"):
                return self._static(path) if method == "GET" else self._json(405, {"error": "method not allowed"})
            try:
                body = {}
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    if n:
                        body = json.loads(self.rfile.read(n))
                        if not isinstance(body, dict):
                            raise ApiError(400, "the request body must be a JSON object")
                with lock:
                    self._json(200, api.handle(method, path, body))
            except ApiError as e:
                self._json(e.status, {"error": str(e)})
            except json.JSONDecodeError:
                self._json(400, {"error": "the request body is not valid JSON"})
            except Exception as e:  # keep serving; report the bug
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def _static(self, path: str) -> None:
            if not frontend.is_dir():
                return self._json(200, {"message": f"API running, but {frontend} has no index.html. "
                                                   "Pass --frontend DIR. API reference: docs/api.md"})
            target = (frontend / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if frontend.resolve() not in target.parents and target != frontend.resolve() / "index.html":
                return self._json(403, {"error": "forbidden"})
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                return self._json(404, {"error": f"{path} not found"})
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

    return Handler


def _default(o):
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, float) and o == float("inf"):
        return None
    raise TypeError(f"cannot serialise {type(o).__name__}")


def make_server(host: str = "127.0.0.1", port: int = 8000, scenario: str = DEFAULT_SCENARIO,
                use_models: bool = True, quiet: bool = False,
                frontend: Path = FRONTEND) -> tuple[HTTPServer, Api]:
    api = Api(scenario, use_models)
    return HTTPServer((host, port), make_handler(api, threading.Lock(), frontend, quiet)), api


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--given", action="store_true", help="use the scenario's written predictions")
    ap.add_argument("--frontend", type=Path, default=FRONTEND, help=f"web UI folder (default: {FRONTEND})")
    args = ap.parse_args(argv)
    try:
        server, _ = make_server(args.host, args.port, args.scenario, not args.given, frontend=args.frontend.resolve())
    except OSError as e:
        if e.errno != 98:                         # EADDRINUSE
            raise
        print(f"Port {args.port} is already in use (another web_server.py still running?). "
              f"Stop it, or pick another port: --port {args.port + 1}", file=sys.stderr)
        return 1
    print(f"Agent ready. Open http://{args.host}:{args.port}")
    print(f"  UI: {args.frontend.resolve()}   API: /api (docs/api.md)   Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
