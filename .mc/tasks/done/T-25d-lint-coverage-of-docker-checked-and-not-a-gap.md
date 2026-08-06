---
id: T-25d
aliases:
- T-25d
title: "Lint coverage of `docker/` — checked, and not a gap"
slug: lint-coverage-of-docker-checked-and-not-a-gap
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m4
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-29
updated: 2026-07-29
---

# Lint coverage of `docker/` — checked, and not a gap

## Description

Lint coverage of `docker/`: checked and **not** a gap. `[tool.ruff]` in `pyproject.toml` sets only
`line-length = 100`, so a bare `ruff check .` from the repo root already walks every Python file in
the repo, `docker/dds/conformance.py` included, and passes (`All checks passed!`). The narrower
`ruff check src tests scripts` habit misses `docker/`; use `ruff check .`. **Correction 2026-08-06:
`ruff check .` no longer passes — 10 errors, all in `docs/anim/` (5 UP031, 2 EXE001, 2 RUF100, 1
SIM115), none in `src/`, `tests/`, `scripts/` or `docker/`.** Nothing in those files changed; **ruff
did** (0.16.0 widened the default rule set beyond the `E4/E7/E9/F` this repo's bare `[tool.ruff]
line-length = 100` relied on). The finding this leaves standing is the original one — the repo-root
invocation is the right habit — plus a new one: **a lint claim with no pinned linter version is a
claim with an expiry date.** Pin the version or select the rules; until one of those happens, `ruff
check .` is expected to report `docs/anim/`.

**Correction 2026-08-06 (second, closing):** both were done, not one. `ruff==0.16.0` is pinned in
the `dev` and `local` extras (`pyproject.toml:62`, `:105`) and the rule set is named in
`[tool.ruff.lint]` (`pyproject.toml:147-149`). `.venv/bin/ruff check .` → `All checks passed!`
again, with **no** edit to `docs/anim/`, no `# noqa` and no per-file-ignore added. One claim above
is itself wrong and is left visible rather than rewritten: `E4/E7/E9/F` was **not** the set this
repo was clean under. Under ruff 0.16.0, selecting it reports **221** findings (F405 158×, E402
58×, E731 2×, E741 2×, E702 1×), none of them the 10 — 0.16.0's default contains neither E402 nor
F405. Those five codes are listed with counts and reasons in the `ignore` beside the `select`;
E731/E741/E702 (5 sites) are the only source-fixable ones. The meta-lesson gains its second half:
*a lint claim with no **named rule set** has an expiry date too* — the version pin alone would have
frozen today's 10 errors in place instead of explaining them.

---

Migrated from `TASKS.md` (milestone M4) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
