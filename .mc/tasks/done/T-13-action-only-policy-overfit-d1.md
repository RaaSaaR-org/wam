---
id: T-13
aliases:
- T-13
title: "Action-only policy — overfit D1"
slug: action-only-policy-overfit-d1
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m2
- data
- backbone
- eval
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-26
updated: 2026-07-26
---

# Action-only policy — overfit D1

## Description

Action-only policy on open backbone; overfit D1 successfully — *gate passed on synthetic D1: loss →
0.09 % of initial (`scripts/overfit_d1.py`); passed again on real D1-scale data (10 GR00T G1
apple-to-plate episodes, `--dataset datasets/gr00t-apple`): loss → 0.01 % of initial, E1 holdout mse
1.8e-5 / mae 0.0028*

---

Migrated from `TASKS.md` (milestone M2) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
