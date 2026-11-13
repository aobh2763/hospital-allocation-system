"""Uniform Cost Search and A* on the allocation problem, plus result helpers.

Both use the same best-first loop; only the priority differs:

    UCS   f(n) = g(n)            cost paid so far
    A*    f(n) = g(n) + h(n)     plus the heuristic's lower bound of what is left

The goal test is applied when a state is taken off the frontier (not when it
is generated), which is what makes both return a cheapest solution. A state
reached again with a lower g is put back on the frontier, so A* stays optimal
with an admissible heuristic even where it is not consistent.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from search.problem import MANDATORY, Action, AllocationProblem, State


@dataclass
class SearchResult:
    algorithm: str
    found: bool
    actions: list[Action] = field(default_factory=list)
    cost: float = 0
    expanded: int = 0          # states taken off the frontier and expanded
    generated: int = 0         # states put on the frontier
    max_frontier: int = 0
    seconds: float = 0.0
    limit_reached: bool = False
    h_start: float = 0

    @property
    def breakdown(self) -> dict[str, int]:
        c: Counter = Counter()
        for a in self.actions:
            c[a.category] += a.cost
        return dict(c)

    @property
    def announcements(self) -> list[str]:
        return [a.note for a in self.actions if a.note]


def search(problem: AllocationProblem, algorithm: str = "astar", limit: int = 500_000) -> SearchResult:
    if algorithm not in ("ucs", "astar"):
        raise ValueError("algorithm must be 'ucs' or 'astar'")
    use_h = algorithm == "astar"
    h = problem.h if use_h else (lambda s: 0)
    name = "A*" if use_h else "UCS"

    t0 = time.perf_counter()
    start = problem.initial
    h0 = h(start)
    if h0 == float("inf"):                            # the heuristic proves there is no solution
        return SearchResult(name, False, [], 0, 0, 1, 1, time.perf_counter() - t0, h_start=h0)
    tie = itertools.count()
    frontier: list[tuple[float, int, int, State]] = [(h0, 0, next(tie), start)]
    best_g: dict[State, float] = {start: 0}
    parent: dict[State, tuple[State | None, Action | None]] = {start: (None, None)}
    expanded, generated, max_frontier = 0, 1, 1

    while frontier:
        f, g, _, s = heapq.heappop(frontier)
        if g > best_g[s]:
            continue                                  # a cheaper copy was already handled
        if problem.is_goal(s):
            return SearchResult(name, True, _path(parent, s), g, expanded, generated, max_frontier,
                                time.perf_counter() - t0, h_start=h0)
        expanded += 1
        if expanded > limit:
            return SearchResult(name, False, [], 0, expanded, generated, max_frontier,
                                time.perf_counter() - t0, limit_reached=True, h_start=h0)
        for action, s2 in problem.successors(s):
            g2 = g + action.cost
            if g2 < best_g.get(s2, float("inf")):
                h2 = h(s2)
                if h2 == float("inf"):
                    continue                          # dead end: no completion exists
                best_g[s2] = g2
                parent[s2] = (s, action)
                heapq.heappush(frontier, (g2 + h2, g2, next(tie), s2))
                generated += 1
        max_frontier = max(max_frontier, len(frontier))

    return SearchResult(name, False, [], 0, expanded, generated, max_frontier,
                        time.perf_counter() - t0, h_start=h0)


def _path(parent, s: State) -> list[Action]:
    actions = []
    while parent[s][0] is not None:
        s, a = parent[s]
        actions.append(a)
    return actions[::-1]


# ------------------------------------------------------------------ results

def allocation(problem: AllocationProblem, result: SearchResult) -> dict[str, Any]:
    """The selected configuration, per patient, for Planning (Block 5)."""
    out: dict[str, Any] = {"patients": {}, "relocations": [], "waiting": [], "announcements": result.announcements}
    for pid in problem.order:
        out["patients"][pid] = {"risk": problem.patients[pid]["risk"], "bed": None, "doctor": None,
                                "nurse": None, "equipment": {}}
    for a in result.actions:
        if a.name == "Relocate":
            out["relocations"].append({"patient": a.pid, "from": a.resource, "to": a.target})
        elif a.name == "LeaveWaiting":
            out["waiting"].append(a.pid)
            out["patients"].pop(a.pid, None)
        elif a.name == "ReserveBed":
            out["patients"][a.pid]["bed"] = a.resource
        elif a.name == "AssignDoctor":
            out["patients"][a.pid]["doctor"] = a.resource
        elif a.name == "AssignNurse":
            out["patients"][a.pid]["nurse"] = a.resource
        else:
            etype = problem.resources[a.resource].type
            out["patients"][a.pid]["equipment"][etype] = a.resource
    return out


def check_allocation(problem: AllocationProblem, result: SearchResult) -> list[str]:
    """Independent check of the team's validity rules, replaying the actions in
    order. Returns the problems found (an empty list means valid)."""
    errors: list[str] = []
    freed: set[str] = set()
    taken: Counter = Counter()
    for a in result.actions:
        if a.name == "LeaveWaiting":
            continue
        rid = a.target if a.name == "Relocate" else a.resource
        r = problem.resources[rid]
        if not (r.available or rid in freed):
            errors.append(f"{rid} is not free when {a} uses it")
        taken[rid] += 1
        if a.name == "Relocate":
            freed.add(a.resource)
    errors += [f"{rid} is given to {n} patients at the same time" for rid, n in taken.items() if n > 1]

    filled = Counter((a.pid, problem.resources[a.resource].kind)
                     for a in result.actions if a.name not in ("Relocate", "LeaveWaiting"))
    for pid in problem.order:
        if problem.patients[pid]["risk"] not in MANDATORY:
            continue
        for (_, kind), n in Counter((s.pid, s.kind) for s in problem.slots if s.pid == pid).items():
            if filled[(pid, kind)] < n:
                errors.append(f"{pid} ({problem.patients[pid]['risk']}) has no {kind}")
    return errors


def diagnose(problem: AllocationProblem) -> list[str]:
    """Why no valid allocation exists: needs of High/Critical patients that the
    available resources cannot cover, even with fallbacks and relocations."""
    demand: Counter = Counter()
    for s in problem.slots:
        if problem.mandatory(s.pid) and s.kind in ("nurse", "equipment"):
            demand[(s.kind, s.need)] += 1
        elif problem.mandatory(s.pid) and s.kind == "bed":
            demand[("bed", "icu or emergency" if s.need in ("icu", "emergency") else s.need)] += 1
        elif problem.mandatory(s.pid):
            demand[("doctor", "any")] += 1
    supply: Counter = Counter()
    for r in problem.resources.values():
        if not r.available and r.rid not in problem.movable:
            continue
        if r.kind == "bed":
            supply[("bed", "icu or emergency" if r.type in ("icu", "emergency") else r.type)] += 1
        elif r.kind == "doctor":
            supply[("doctor", "any")] += 1
        else:
            supply[(r.kind, r.type)] += 1
    msgs = []
    for (kind, need), n in sorted(demand.items(), key=str):
        if supply[(kind, need)] < n:
            what = f"{kind} ({need})" if need else kind
            msgs.append(f"{n} High/Critical patients need a {what}, only {supply[(kind, need)]} available")
    return msgs or ["no combination satisfies every High/Critical patient at once"]


# ------------------------------------------------------------------ agent entry point

@dataclass
class Decision:
    valid: bool                     # True: the team's two rules hold
    result: SearchResult
    problem: AllocationProblem
    allocation: dict[str, Any]
    diagnosis: list[str]            # why no valid allocation exists (empty when valid)


def allocate(requirements: dict[str, Any], algorithm: str = "astar", limit: int = 500_000) -> Decision:
    """Search for a valid allocation; if none exists, explain why and return the
    best relaxed one (High/Critical may wait), flagged as not valid."""
    from search.problem import from_requirements

    problem = from_requirements(requirements)
    result = search(problem, algorithm, limit)
    if result.found:
        return Decision(True, result, problem, allocation(problem, result), [])
    diagnosis = (["search stopped at the node limit before finding a solution"]
                 if result.limit_reached else diagnose(problem))
    relaxed = from_requirements(requirements, relaxed=True)
    best = search(relaxed, algorithm, limit)
    return Decision(False, best, relaxed, allocation(relaxed, best) if best.found else {}, diagnosis)
