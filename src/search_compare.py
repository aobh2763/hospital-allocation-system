"""Block 4 experiment: Uniform Cost Search vs A* on departments of growing size.

    poetry run python src/search_compare.py              # full run (a minute or two)
    poetry run python src/search_compare.py --quick      # 2 seeds, up to 12 patients

For each size n (4 to 20 patients), departments are built from the
HLT-010 hospital data (FAC0003, 2023-01-02, day shift) with n Synthea patients
whose ML and DL predictions are computed by the real models, until 5 of them
have a valid allocation (the others are reported separately). Block 2 derives the
needs, then both algorithms solve the same problem. Measured, as the PDF asks:
execution time, states explored, solution cost and allocation quality
(fallbacks, relocations, patients left waiting).

The department is sized so a valid allocation exists but resources are contested:
beds, nurses and monitors for every High/Critical patient, one ICU bed fewer than
Critical patients, and Medium/Low patients compete for what is left.

A second series turns symmetry breaking off (every interchangeable resource is
offered) on the small sizes, to show how fast the state space grows without it.

Writes docs/results/search_comparison.csv and docs/figures/search_comparison.png.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

from environment.hospital import from_hospital_data
from reasoning.adapters import DLAdapter, MLAdapter, run_models
from reasoning.kb import KnowledgeBase
from search import check_allocation, from_requirements, search

ROOT = Path(__file__).resolve().parents[1]
CSV_OUT = ROOT / "docs" / "results" / "search_comparison.csv"
FIG_OUT = ROOT / "docs" / "figures" / "search_comparison.png"
MIX_WEIGHTS = {"Critical": 2, "High": 3, "Medium": 3, "Low": 2}
LIMIT = 200_000

UCS_COLOR, ASTAR_COLOR = "#2a78d6", "#eb6834"   # default categorical slots 1 and 2


def scaled_scenario(n: int, seed: int):
    rng = random.Random(1000 * n + seed)
    mix = rng.choices(list(MIX_WEIGHTS), weights=list(MIX_WEIGHTS.values()), k=n)
    crit = mix.count("Critical")
    must = crit + mix.count("High")
    caps = {"icu": max(1, crit - 1), "emergency": must + (n - must) // 3, "exam_room": 2,
            "doctors_per_specialty": max(2, (must + 2) // 3), "nurses": must, "equipment_per_type": must}
    return from_hospital_data("FAC0003", "2023-01-02", caps=caps, patient_mix=mix,
                              extra_specialists=["cardiology"], seed=1000 * n + seed)


def requirements_for(scenario, ml, dl):
    kb = KnowledgeBase()
    kb.load_scenario(scenario)
    run_models(kb, scenario.patients, ml, dl)
    kb.update()
    return kb.requirements()


def run_one(req, n, seed, symmetry: bool) -> list[dict]:
    problem = from_requirements(req, symmetry=symmetry)
    rows = []
    for alg in ("ucs", "astar"):
        r = search(problem, alg, LIMIT)
        if r.found and check_allocation(problem, r):
            raise AssertionError(f"invalid allocation n={n} seed={seed} {alg}")
        cats = [a.category for a in r.actions]
        rows.append({
            "patients": n, "seed": seed, "symmetry": symmetry, "slots": len(problem.slots),
            "must_serve": sum(problem.mandatory(p) for p in problem.order),
            "algorithm": r.algorithm, "found": r.found, "limit_reached": r.limit_reached,
            "cost": r.cost if r.found else "", "h_start": r.h_start,
            "expanded": r.expanded, "generated": r.generated, "max_frontier": r.max_frontier,
            "ms": round(r.seconds * 1000, 2),
            "fallback_beds": cats.count("fallback bed"), "fallback_doctors": cats.count("fallback doctor"),
            "relocations": cats.count("relocation"), "left_waiting": cats.count("waiting"),
        })
    ucs, astar = rows
    if ucs["found"] and astar["found"] and ucs["cost"] != astar["cost"]:
        raise AssertionError(f"UCS and A* disagree on n={n} seed={seed}: {ucs['cost']} vs {astar['cost']}")
    return rows


def feasible(rows: list[dict]) -> list[dict]:
    """Runs of problems where a valid allocation exists (UCS found one, or neither hit the limit)."""
    ok = {(r["patients"], r["seed"]) for r in rows if r["symmetry"] and r["algorithm"] == "UCS" and r["found"]}
    return [r for r in rows if (r["patients"], r["seed"]) in ok]


def summarise(all_rows: list[dict]) -> None:
    rows = feasible(all_rows)
    print("\nProblems with a valid allocation (means over seeds):")
    print(f"\n{'n':>3} {'sym':>4} {'slots':>5} {'cost':>6} | {'UCS expanded':>12} {'A* expanded':>12} "
          f"{'saved':>6} | {'UCS ms':>7} {'A* ms':>7} | fallbacks  waiting")
    keys = sorted({(r["patients"], r["symmetry"]) for r in rows}, key=lambda k: (not k[1], k[0]))
    for n, sym in keys:
        grp = [r for r in rows if r["patients"] == n and r["symmetry"] == sym]
        u = [r for r in grp if r["algorithm"] == "UCS"]
        a = [r for r in grp if r["algorithm"] == "A*"]
        mean = lambda rs, k: statistics.mean(r[k] for r in rs)  # noqa: E731
        hit = any(r["limit_reached"] for r in grp)
        costs = [r["cost"] for r in a if r["found"]]
        saved = 1 - mean(a, "expanded") / mean(u, "expanded")
        print(f"{n:>3} {'on' if sym else 'off':>4} {mean(u, 'slots'):>5.0f} "
              f"{(statistics.mean(costs) if costs else float('nan')):>6.1f} | "
              f"{mean(u, 'expanded'):>12,.0f} {mean(a, 'expanded'):>12,.0f} {saved:>6.0%} | "
              f"{mean(u, 'ms'):>7.1f} {mean(a, 'ms'):>7.1f} | "
              f"{mean(a, 'fallback_beds') + mean(a, 'fallback_doctors'):>9.1f} {mean(a, 'left_waiting'):>8.1f}"
              + ("   (UCS hit the node limit: its counts are lower bounds)" if hit else ""))

    bad = [r for r in all_rows if r["symmetry"] and (r["patients"], r["seed"]) not in
           {(x["patients"], x["seed"]) for x in rows}]
    if bad:
        print("\nProblems with no valid allocation (the agent must report them):")
        for n, seed in sorted({(r["patients"], r["seed"]) for r in bad}):
            u = next(r for r in bad if (r["patients"], r["seed"], r["algorithm"]) == (n, seed, "UCS"))
            a = next(r for r in bad if (r["patients"], r["seed"], r["algorithm"]) == (n, seed, "A*"))
            print(f"  n={n} seed={seed}: UCS explores {u['expanded']:,} states in {u['ms']:.1f} ms to find "
                  f"none; A* expands {a['expanded']} (h = ∞ at the start) in {a['ms']:.1f} ms")


def plot(all_rows: list[dict]) -> None:
    rows = feasible(all_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surface)
    panels = [("expanded", "States expanded (mean over seeds)"), ("ms", "Execution time, ms (mean over seeds)")]
    sizes = sorted({r["patients"] for r in rows if r["symmetry"]})
    for ax, (key, label) in zip(axes, panels):
        ax.set_facecolor(surface)
        for alg, color in (("UCS", UCS_COLOR), ("A*", ASTAR_COLOR)):
            ys = [statistics.mean(r[key] for r in rows
                                  if r["symmetry"] and r["algorithm"] == alg and r["patients"] == n) for n in sizes]
            ax.plot(sizes, ys, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=surface, markeredgewidth=1.5, label=alg)
            ax.annotate(alg, (sizes[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=10, color=ink)
        ax.set_title(label, loc="left", fontsize=11, color=ink)
        ax.set_xlabel("Patients waiting", color=muted)
        ax.set_xticks(sizes)
        ax.grid(axis="y", color=grid, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
        ax.set_ylim(bottom=0)
        ax.set_xlim(sizes[0] - 0.5, sizes[-1] + 1.8)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("A* reaches the same optimal allocation as UCS while expanding fewer states",
                 x=0.01, ha="left", fontsize=12.5, color=ink, fontweight="bold")
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=160, facecolor=surface)
    print(f"\nFigure: {FIG_OUT.relative_to(ROOT)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    sizes = range(4, 13, 2) if args.quick else range(4, 21, 2)
    target = 2 if args.quick else 5           # feasible departments per size
    no_symmetry_sizes = range(4, 9, 2) if args.quick else range(4, 11, 2)

    ml, dl = MLAdapter(), DLAdapter()
    rows: list[dict] = []
    for n in sizes:
        ok = seed = 0
        while ok < target and seed < 4 * target:
            req = requirements_for(scaled_scenario(n, seed), ml, dl)
            pair = run_one(req, n, seed, symmetry=True)
            rows += pair
            if pair[0]["found"]:
                ok += 1
                if n in no_symmetry_sizes:
                    rows += run_one(req, n, seed, symmetry=False)
            seed += 1
        print(f"  n={n}: {ok} feasible departments, {seed - ok} infeasible", flush=True)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Results: {CSV_OUT.relative_to(ROOT)} ({len(rows)} runs)")
    summarise(rows)
    plot(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
