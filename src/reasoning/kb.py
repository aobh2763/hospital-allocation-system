"""The Knowledge Base: SWI-Prolog (through janus-swi) behind a small Python API.

Typical use::

    kb = KnowledgeBase()
    kb.load_scenario(scenario)              # base facts: patients, resources
    kb.set_prediction("P1", "dl", "Critical", 0.91)
    result = kb.update()                    # deduction (forward chaining)
    kb.risk("P1")                           # "Critical"
    kb.requirements()                       # contract for Search (docs/predicates.md)
    print(kb.format_proof("RequiresICU(P1)"))   # resolution trace

Goals and patterns are written in the PDF notation (``RequiresICU(P1)``,
``RequiresEquipment(P1, t)``); see reasoning/printer.py.

SWI-Prolog runs once per Python process, so there is one knowledge base per
process: creating a new KnowledgeBase resets it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from environment.scenario import Scenario
from patient import Patient
from reasoning import export
from reasoning.facts import patient_facts, prediction_fact, resource_facts, scenario_facts
from reasoning.printer import fact, to_atom, to_pdf, to_prolog

PROLOG_DIR = Path(__file__).resolve().parent / "prolog"
ENGINE_FILES = ("kb.pl", "forward.pl", "prove.pl")
RULE_FILES = ("findings.pl", "evidence.pl", "risk.pl", "requirements.pl")

LAYERS = {"f": "L1 findings", "e": "L2 evidence", "k": "L3 risk", "q": "L4 needs", "m": "L4 matching"}


def _janus():
    try:
        import janus_swi
    except ImportError as e:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "janus-swi is not installed. Install SWI-Prolog (Arch: `sudo pacman -S swi-prolog`, "
            "others: https://www.swi-prolog.org/download/stable) then `pip install janus-swi`."
        ) from e
    return janus_swi


class PrologError(RuntimeError):
    pass


@dataclass
class UpdateResult:
    derived: int                       # number of derived facts
    changes: dict[str, dict[str, Any]]  # requirement diff per patient, see export.diff


class KnowledgeBase:
    _engine_loaded = False

    def __init__(self) -> None:
        self._j = _janus()
        if not KnowledgeBase._engine_loaded:
            for f in ENGINE_FILES:
                self._j.consult(str(PROLOG_DIR / f))
            KnowledgeBase._engine_loaded = True
        self._j.consult(str(PROLOG_DIR / "thresholds.pl"))  # (re)load default thresholds
        self._call("clear_rules")
        for f in RULE_FILES:
            self._call("load_rules(F)", F=str(PROLOG_DIR / f))
        self._call("kb_clear")
        self.scenario: Scenario | None = None
        self._last_requirements: dict[str, Any] = {}

    # ------------------------------------------------------------ low level

    def _call(self, goal: str, **inputs: Any) -> dict[str, Any]:
        try:
            r = self._j.query_once(goal, inputs)
        except self._j.PrologError as e:
            raise PrologError(f"{goal} {inputs}: {e}") from None
        if not r.get("truth"):
            raise PrologError(f"Prolog goal failed: {goal} {inputs}")
        return r

    def add(self, fact_string: str) -> None:
        """Assert one base fact, written in Prolog syntax (see reasoning/facts.py)."""
        self._call("kb_add(S)", S=fact_string)

    def add_all(self, facts: list[str]) -> None:
        for f in facts:
            self.add(f)

    def remove(self, pattern: str) -> None:
        """Retract every base fact matching a Prolog pattern, e.g. ``available(icu1)``."""
        self._call("kb_remove(S)", S=pattern)

    def solutions(self, pattern: str) -> list[dict[str, Any]]:
        """All bindings of a Prolog pattern over base and derived facts."""
        rows = self._call("kb_solutions(S, Rows)", S=pattern)["Rows"]
        return [{name: value for name, value in row} for row in rows]

    def holds(self, goal: str) -> bool:
        """True if a goal (PDF notation) is a base or derived fact."""
        return bool(self.solutions(to_prolog(goal)))

    def query(self, pattern: str) -> list[dict[str, str]]:
        """Pattern in PDF notation, e.g. ``RequiresEquipment(P1, t)`` -> [{'t': 'PatientMonitor'}, ...]."""
        rows = self.solutions(to_prolog(pattern))
        return [{k[2:] if k.startswith("V_") else k: to_pdf(str(v)) for k, v in row.items()} for row in rows]

    # ------------------------------------------------------------ environment

    def load_scenario(self, scenario: Scenario) -> None:
        self._call("kb_clear")
        self.scenario = scenario
        self.add_all(scenario_facts(scenario))

    def add_patient(self, p: Patient) -> None:
        self.add_all(patient_facts(p))
        if self.scenario is not None and all(x.pid != p.pid for x in self.scenario.patients):
            self.scenario.patients.append(p)

    def remove_patient(self, pid: str) -> None:
        self._call("kb_remove_patient(P)", P=to_atom(pid))

    def refresh_patient(self, p: Patient) -> None:
        """Replace a patient's facts after new measurements; predictions are kept."""
        preds = [(src, r) for src in ("ml", "dl") for r in self.solutions(f"{src}_risk({to_atom(p.pid)},L,P)")]
        self.remove_patient(p.pid)
        self.add_all(patient_facts(p))
        for src, r in preds:
            self.add(f"{src}_risk({to_atom(p.pid)},{r['L']},{r['P']})")

    def set_prediction(self, pid: str, source: str, level: str, prob: float) -> None:
        """Insert (or replace) ``ML_Risk(P, Level, Prob)`` / ``DL_Risk(P, Level, Prob)``."""
        self.remove(f"{source}_risk({to_atom(pid)},_,_)")
        self.add(prediction_fact(pid, source, level, prob))

    def set_available(self, rid: str, available: bool) -> None:
        self.remove(fact("available", rid))
        if available:
            self.add(fact("available", rid))
        if self.scenario is not None:
            try:
                self.scenario.resource(rid).available = available
            except StopIteration:
                pass

    def add_resource(self, r) -> None:
        self.add_all(resource_facts(r))
        if self.scenario is not None:
            self.scenario.resources.append(r)

    def assign(self, pid: str, rid: str) -> None:
        """Record an executed assignment: the patient leaves the waiting list."""
        self.add(fact("assigned", pid, rid))
        self.set_available(rid, False)
        self.remove(fact("waiting", pid))
        self.add(fact("under_treatment", pid))

    def set_threshold(self, name: str, value: float) -> None:
        self._call("kb_set_threshold(N, V)", N=name, V=value)

    # ------------------------------------------------------------ reasoning

    def update(self) -> UpdateResult:
        """Re-derive everything from the current base facts (deduction) and
        report how each patient's requirements changed since the last update."""
        n = self._call("kb_saturate(N)")["N"]
        reqs = self.requirements()
        changes = export.diff(self._last_requirements, reqs)
        self._last_requirements = reqs
        return UpdateResult(derived=n, changes=changes)

    def patients(self) -> list[str]:
        return sorted((to_pdf(r["X"]) for r in self.solutions("patient(X)")), key=_pid_key)

    def risk(self, pid: str) -> str | None:
        rows = self.solutions(f"risk_level({to_atom(pid)},L)")
        return to_pdf(rows[0]["L"]) if rows else None

    def requirements(self) -> dict[str, Any]:
        """The contract for Search and Planning (docs/predicates.md)."""
        return export.requirements(self)

    def base_facts(self) -> list[str]:
        return [to_pdf(s) for s in self._call("kb_base_facts(L)")["L"]]

    def rules(self) -> list[dict[str, Any]]:
        return self._call("kb_rules(L)")["L"]

    # ------------------------------------------------------------ traces

    def deductions(self) -> list[dict[str, Any]]:
        """Forward-chaining log: every derived fact with its rule and substitution."""
        steps = self._call("kb_deductions(L)")["L"]
        for s in steps:
            s["layer"] = LAYERS[s["rule"][0]]
            s["fact_pdf"] = to_pdf(s["fact"])
            s["theta_pdf"] = _theta(s["theta"])
        return steps

    def format_deductions(self, patient: str | None = None) -> str:
        lines = []
        for s in self.deductions():
            if patient and not re.search(rf"\b{re.escape(patient)}\b", s["fact_pdf"]):
                continue
            lines.append(f"{s['seq']:>4}  [{s['layer']}]  {s['fact_pdf']:<40} by {s['rule']:<27} θ = {s['theta_pdf']}")
        return "\n".join(lines)

    def explain(self, goal: str) -> dict[str, Any]:
        """How a fact was deduced, as a tree of rules and premises."""
        return self._call("kb_explain(S, T)", S=to_prolog(goal))["T"]

    def format_explanation(self, goal: str) -> str:
        lines: list[str] = []

        def walk(node: dict[str, Any], prefix: str, last: bool, root: bool) -> None:
            branch = "" if root else ("└─ " if last else "├─ ")
            text = to_pdf(node["fact"])
            if node["kind"] == "derived":
                text += f"   [{node['rule']}]  θ = {_theta(node['theta'])}"
            elif node["kind"] == "fact":
                text += "   (fact)"
            elif node["kind"] == "builtin":
                text += "   (built-in test)"
            elif node["kind"] == "naf":
                text = "¬" + text + "   (not provable: negation as failure)"
            else:
                text += "   (not in the knowledge base)"
            lines.append(prefix + branch + text)
            kids = node.get("premises", [])
            child_prefix = prefix + ("" if root else ("   " if last else "│  "))
            for i, k in enumerate(kids):
                walk(k, child_prefix, i == len(kids) - 1, False)

        walk(self.explain(goal), "", True, True)
        return "\n".join(lines)

    def prove(self, goal: str) -> dict[str, Any]:
        """Resolution (SLD refutation) of a query, e.g. ``RequiresICU(P1)``."""
        r = self._call("kb_prove(S, R)", S=to_prolog(goal))["R"]
        r["proved"] = _bool(r["proved"])
        return r

    def format_proof(self, goal: str) -> str:
        r = self.prove(goal)
        g = to_pdf(to_prolog(goal))
        lines = [f"Query: {g}?",
                 f"Refutation: add the goal clause ← {g} (that is ¬{g}) and try to derive the empty clause □."]
        for i, s in enumerate(r["steps"], 1):
            goal_s = to_pdf(s["goal"])
            res = _resolvent(s["resolvent"])
            if s["kind"] == "rule":
                body = " ∧ ".join(to_pdf(b) for b in s["body"])
                rule = f"{body} → {to_pdf(s['head'])}" if body else to_pdf(s["head"])
                lines.append(f"{i:>3}. ← {goal_s}\n       resolve with rule {s['id']}: {rule}\n"
                             f"       θ = {_theta(s['theta'])}   ⇒ {res}")
            elif s["kind"] == "fact":
                lines.append(f"{i:>3}. ← {goal_s}\n       resolve with fact {to_pdf(s['clause'])}"
                             f"   θ = {_theta(s['theta'])}   ⇒ {res}")
            elif s["kind"] == "builtin":
                lines.append(f"{i:>3}. ← {goal_s}\n       built-in test {to_pdf(s['result'])} is true   ⇒ {res}")
            else:
                lines.append(f"{i:>3}. ← ¬{goal_s}\n       {goal_s} has no proof (negation as failure)   ⇒ {res}")
        if r["proved"]:
            lines.append(f"Proved: {g} follows from the knowledge base.")
        else:
            lines.append(f"Not proved: {g} does not follow from the knowledge base.")
            lines += _why_lines(r["why"], "  ")
        return "\n".join(lines)

    def unify(self, a: str, b: str) -> dict[str, Any]:
        """Unify two expressions in PDF notation, e.g. ``RequiresSpecialist(x, Cardiology)``
        and ``RequiresSpecialist(P1, Cardiology)`` -> θ = {x/P1}."""
        r = self._call("kb_unify(A, B, R)", A=to_prolog(a), B=to_prolog(b))["R"]
        out = {"unified": _bool(r["unified"])}
        if out["unified"]:
            out["theta"] = _theta(r["theta"])
            out["result"] = to_pdf(r["result"])
        else:
            a, b = to_pdf(r["a"]), to_pdf(r["b"])
            out["reason"] = {
                "occurs_check": f"occurs check: {a} would have to contain itself ({a} = {b})",
                "constant_clash": f"constant clash: {a} vs {b}",
                "constant_vs_compound": f"a constant cannot match a compound term: {a} vs {b}",
                "predicate_clash": f"predicate clash: {a} vs {b}",
            }[r["reason"]]
        return out


# ---------------------------------------------------------------- helpers

def _bool(v: Any) -> bool:
    """Prolog atoms true/false reach Python as the strings "true"/"false"."""
    return v is True or v == "true"


def _pid_key(pid: str):
    return (pid[:1], int(pid[1:])) if pid[1:].isdigit() else (pid, 0)


def _theta(pairs: list[list[str]]) -> str:
    if not pairs:
        return "{}"
    return "{" + ", ".join(f"{to_pdf(str(n))}/{to_pdf(str(v))}" for n, v in pairs) + "}"


def _resolvent(goals: list[str]) -> str:
    return "□" if not goals else "← " + ", ".join(to_pdf(g) for g in goals)


def _why_lines(why: dict[str, Any], indent: str) -> list[str]:
    lines = [f"{indent}{to_pdf(why['goal'])}: {why['reason']}"]
    for r in why.get("rules", []):
        lines.append(f"{indent}  rule {r['rule']} stops at {to_pdf(r['failed_at'])}")
        sub = r.get("because")
        if sub and sub.get("rules"):
            lines += _why_lines(sub, indent + "    ")
        elif sub:
            lines.append(f"{indent}    {to_pdf(sub['goal'])}: {sub['reason']}")
    return lines
