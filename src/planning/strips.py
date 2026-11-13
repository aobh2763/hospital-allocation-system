"""A small STRIPS planner: states, actions with preconditions and effects,
forward state-space search, and a plan validator.

A state is a set of ground atoms written in the PDF notation, e.g.
``{"Waiting(P1)", "Available(D1)", "CriticalRisk(P1)"}``. Anything not in the
set is false (closed world).

An action ``a`` is applicable in state ``s`` when ``pre(a) ⊆ s``; applying it
gives ``(s − del(a)) ∪ add(a)``. A plan is valid when, replayed from the initial
state, every action is applicable at its turn and the final state contains
every goal (``validate``).

``plan`` searches forward from the initial state with greedy best-first search
guided by h_add (Bonet & Geffner's additive heuristic, as in the HSP planner):
the cost of an atom is 0 if it holds, otherwise 1 + the cheapest sum of the
costs of the preconditions of an action that adds it, ignoring delete effects;
h_add(s) is the sum of the goals' costs, each goal weighted by its patient's
priority (Critical goals weigh most), so a Critical patient's actions come
first. Ties are broken by the natural order of actions (assign, reserve,
evaluate, transfer, monitor, re-evaluate). Greedy search with a weighted,
inadmissible heuristic does not promise the shortest plan; here every plan has
the same length (each ground action is needed exactly once), so only the order
matters, and ``validate`` proves each plan correct.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

INF = float("inf")


@dataclass(frozen=True)
class Action:
    name: str
    args: tuple[str, ...]
    pre: frozenset[str]
    add: frozenset[str]
    delete: frozenset[str] = frozenset()
    patient: str = ""          # patient the action serves (for ordering and display)
    priority: int = 9          # 1 = Critical ... 4 = Low
    rank: int = 0              # natural order of action types

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    def applicable(self, state: frozenset[str]) -> bool:
        return self.pre <= state

    def apply(self, state: frozenset[str]) -> frozenset[str]:
        return (state - self.delete) | self.add


@dataclass
class PlanResult:
    found: bool
    steps: list[Action] = field(default_factory=list)
    expanded: int = 0
    seconds: float = 0.0
    unreachable: list[str] = field(default_factory=list)   # goals no action sequence can reach


def h_add(state: frozenset[str], goals: frozenset[str], actions: list[Action],
          weights: dict[str, float] | None = None) -> float:
    cost: dict[str, float] = {a: 0 for a in state}
    changed = True
    while changed:
        changed = False
        for act in actions:
            c = 0.0
            for p in act.pre:
                pc = cost.get(p)
                if pc is None:
                    break
                c += pc
            else:
                c += 1
                for q in act.add:
                    if c < cost.get(q, INF):
                        cost[q] = c
                        changed = True
    w = weights or {}
    return sum(w.get(g, 1) * cost.get(g, INF) for g in goals)


def plan(init: frozenset[str], goals: frozenset[str], actions: list[Action], limit: int = 50_000,
         weights: dict[str, float] | None = None) -> PlanResult:
    """Greedy best-first search on h_add. ``weights`` scale each goal's cost in
    the heuristic (the hospital domain weights goals by patient priority, so a
    Critical patient's goals are pursued first)."""
    t0 = time.perf_counter()
    if h_add(init, goals, actions) == INF:
        reach = _relaxed_reachable(init, actions)
        return PlanResult(False, unreachable=sorted(goals - reach), seconds=time.perf_counter() - t0)
    tie = itertools.count()
    frontier = [(h_add(init, goals, actions, weights), 0, 0, next(tie), init)]
    parent: dict[frozenset, tuple[frozenset | None, Action | None]] = {init: (None, None)}
    expanded = 0
    while frontier:
        _, _, _, _, s = heapq.heappop(frontier)
        if goals <= s:
            return PlanResult(True, _path(parent, s), expanded, time.perf_counter() - t0)
        expanded += 1
        if expanded > limit:
            break
        for a in actions:
            if not a.applicable(s):
                continue
            s2 = a.apply(s)
            if s2 in parent:
                continue
            h = h_add(s2, goals, actions, weights)
            if h == INF:
                continue
            parent[s2] = (s, a)
            heapq.heappush(frontier, (h, a.priority, a.rank, next(tie), s2))
    return PlanResult(False, expanded=expanded, seconds=time.perf_counter() - t0)


def validate(init: frozenset[str], steps: list[Action], goals: frozenset[str]) -> list[str]:
    """Replay the plan. Returns the problems found (empty list = valid plan)."""
    errors = []
    s = init
    for i, a in enumerate(steps, 1):
        missing = a.pre - s
        if missing:
            errors.append(f"step {i} {a}: precondition(s) not satisfied: {', '.join(sorted(missing))}")
        s = a.apply(s)
    unmet = goals - s
    if unmet:
        errors.append(f"goals not reached: {', '.join(sorted(unmet))}")
    return errors


def first_invalid(state: frozenset[str], steps: list[Action]) -> tuple[int, Action, list[str]] | None:
    """The first remaining step whose preconditions fail when replayed from ``state``."""
    s = state
    for i, a in enumerate(steps):
        missing = sorted(a.pre - s)
        if missing:
            return i, a, missing
        s = a.apply(s)
    return None


def _path(parent, s) -> list[Action]:
    steps = []
    while parent[s][0] is not None:
        s, a = parent[s]
        steps.append(a)
    return steps[::-1]


def _relaxed_reachable(init: frozenset[str], actions: list[Action]) -> frozenset[str]:
    reach = set(init)
    changed = True
    while changed:
        changed = False
        for a in actions:
            if a.pre <= reach and not a.add <= reach:
                reach |= a.add
                changed = True
    return frozenset(reach)
