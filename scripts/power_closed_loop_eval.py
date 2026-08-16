#!/usr/bin/env python
"""How large an effect can 20 paired seeds actually resolve? Exact, not simulated.

WHY THIS EXISTS. PR-08's generation leg proposes ~10 050 restyled clips and two Recipe B retrains
so that an augmented arm can be compared against an unaugmented one. The comparison is scored by
``../vla-training/eval/run_apple_eval.py``: **closed-loop success over a fixed seed set**, paired
(both arms see the same seeds), analysed with **McNemar's exact test** — that repo's own choice,
recorded in ``eu-hub/RUNS.md``, not one invented here.

That instrument has already been run several times and reports ``p = 0.500`` for every pairing it
has ever scored. Before spending the clips it is worth knowing what it *could* have detected. This
script answers that from the test's definition, so the answer does not depend on a simulation seed.

    .venv/bin/python scripts/power_closed_loop_eval.py

NOTHING HERE IS A RESULT ABOUT AUGMENTATION. It is a property of the measuring device: how many
successes an arm must gain before this test can call the gain real. Feeding it a hoped-for effect
size does not make that effect exist.

McNemar, exactly. Only the discordant seeds count — the ones where the two arms disagree. With
``b`` seeds where only the baseline succeeded and ``c`` where only the treatment did, the null says
each discordant seed is a fair coin, so ``b ~ Binomial(b + c, 0.5)`` and the two-sided exact
p-value is ``2 * P(X <= min(b, c))``, capped at 1. Concordant seeds — both succeed, both fail —
carry no information and do not enter the test at all. **This is the fact that governs everything
below: 20 seeds are not 20 observations, they are however many disagreements the two arms happen to
produce, and at a 1/20 baseline that is a very small number.**
"""

from __future__ import annotations

import argparse
from math import comb

# vla-training's measured in-house scores, for reference. Sources, all in ../vla-training:
#   docs/isaaclab-arena-eval.md — GR00T Recipe A 1/20, Recipe B 1/20, vendor checkpoint 5/20
#   eu-hub/RUNS.md              — MuJoCo 0/10 and 2/20; McNemar exact p = 0.500 on every pairing
MEASURED = {
    "GR00T Recipe A (Arena, 20 seeds)": (1, 20),
    "GR00T Recipe B (Arena, 20 seeds)": (1, 20),
    "vendor checkpoint (Arena, 20 seeds)": (5, 20),
    "GR00T Recipe B (MuJoCo, 20 seeds)": (2, 20),
}
ALPHA = 0.05


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact p for ``b`` vs ``c`` discordant pairs. 1.0 when there are none."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def min_discordant_for_significance(alpha: float = ALPHA) -> int:
    """Fewest one-sided discordant pairs that can reach ``alpha``. The instrument's floor.

    With every discordant seed favouring one arm (``b = 0``), the p-value is ``2 * 0.5**n``. No
    amount of *concordant* agreement helps: a treatment that wins on 4 seeds the baseline lost and
    loses none cannot reach 0.05 however many seeds are run, if only 4 ever disagree.
    """
    n = 1
    while mcnemar_exact_p(0, n) > alpha:
        n += 1
    return n


def _max_minority(d: int, alpha: float) -> int:
    """Largest ``k`` with ``2 * P(X <= k) <= alpha`` for ``X ~ Binomial(d, 0.5)``; -1 if none.

    The whole test collapses to this: among ``d`` discordant seeds, it is significant exactly when
    the smaller side is at most ``_max_minority(d)``. Precomputing it per ``d`` is what turns the
    power calculation from a triple loop into a double one.
    """
    best = -1
    for k in range(d // 2 + 1):
        if mcnemar_exact_p(k, d - k) <= alpha:
            best = k
        else:
            break
    return best


def power(p_base: float, p_treat: float, n_seeds: int, alpha: float = ALPHA) -> float:
    """P(the test calls it) under independence between arms, exactly.

    Factored: the number of discordant seeds is Binomial(n, p_c + p_b), and given ``d`` of them the
    split is Binomial(d, p_c / (p_c + p_b)). Concordant seeds never enter — which is the whole point.

    Independence is the OPTIMISTIC assumption and is stated as such. Two policies on the same task
    and the same seeds fail on the same hard seeds; that positive correlation makes seeds concordant,
    and concordant seeds are invisible to McNemar. Real power is therefore at or below these numbers.
    """
    p_c = p_treat * (1.0 - p_base)      # only the treatment succeeds
    p_b = p_base * (1.0 - p_treat)      # only the baseline succeeds
    p_dis = p_c + p_b
    if p_dis <= 0.0:
        return 0.0
    q = p_c / p_dis

    total = 0.0
    for d in range(2, n_seeds + 1):
        k_max = _max_minority(d, alpha)
        if k_max < 0:
            continue                     # d discordant seeds cannot reach alpha however they split
        p_d = comb(n_seeds, d) * (p_dis**d) * ((1.0 - p_dis) ** (n_seeds - d))
        if not p_d:
            continue
        # min(c, d - c) <= k_max  <=>  c <= k_max or c >= d - k_max. Disjoint because 2*k_max < d.
        split = sum(comb(d, c) * (q**c) * ((1.0 - q) ** (d - c)) for c in range(k_max + 1))
        split += sum(
            comb(d, c) * (q**c) * ((1.0 - q) ** (d - c)) for c in range(d - k_max, d + 1)
        )
        total += p_d * split
    return total


def seeds_for_power(p_base: float, p_treat: float, target: float = 0.8, cap: int = 400) -> int | None:
    """Fewest paired seeds reaching ``target`` power, or None if ``cap`` is not enough."""
    for n in range(2, cap + 1):
        if power(p_base, p_treat, n) >= target:
            return n
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-successes", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args(argv)

    p_base = args.baseline_successes / args.seeds
    floor = min_discordant_for_significance(args.alpha)

    print("=== The instrument: McNemar exact, paired seeds, two-sided")
    print(f"    alpha {args.alpha}  |  seeds {args.seeds}  |  baseline {args.baseline_successes}"
          f"/{args.seeds} = {p_base:.2f}")
    print()
    print("    Measured in-house scores this is calibrated against (../vla-training):")
    for name, (s, n) in MEASURED.items():
        print(f"      {name:44s} {s}/{n}")
    print()
    print(f"--- FLOOR: {floor} discordant seeds, ALL favouring one arm, are needed to reach "
          f"{args.alpha}")
    print(f"    (p = 2 * 0.5^{floor} = {mcnemar_exact_p(0, floor):.4f};"
          f" at {floor - 1} it is {mcnemar_exact_p(0, floor - 1):.4f}, which does not)")
    print(f"    So with a {args.baseline_successes}/{args.seeds} baseline the treatment must reach")
    print(f"    at least {args.baseline_successes + floor}/{args.seeds} — and only if it keeps every")
    print("    success the baseline already had. That is the best case, not the requirement.")
    print()

    print(f"--- POWER at n = {args.seeds}, by true treatment success rate")
    print("    treat/20   p_treat   power    (P that the test calls a real effect)")
    for successes in range(0, args.seeds + 1, 2):
        p_treat = successes / args.seeds
        if p_treat < p_base:
            continue
        pw = power(p_base, p_treat, args.seeds, args.alpha)
        bar = "#" * int(pw * 40)
        print(f"    {successes:2d}/{args.seeds}      {p_treat:.2f}     {pw:5.1%}  {bar}")
    print()

    print("--- SEEDS NEEDED for 80% power")
    print("    treat/20   p_treat   paired seeds required")
    for successes in range(2, args.seeds + 1, 2):
        p_treat = successes / args.seeds
        if p_treat <= p_base:
            continue
        n = seeds_for_power(p_base, p_treat)
        print(f"    {successes:2d}/{args.seeds}      {p_treat:.2f}     {n if n else '> 400'}")
    print()
    print("Independence between arms is assumed, which is optimistic: correlated failures make")
    print("seeds concordant, and concordant seeds are invisible to this test. Real power is lower.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
