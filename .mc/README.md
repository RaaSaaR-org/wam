# `.mc/` — tasks as files

MissionControl (`mc`) in embedded mode. One task per file, YAML frontmatter + Markdown body,
git as the database. Same layout as `robot-management-system/.mc/`.

```
.mc/tasks/todo/T-39-….md    active: backlog | todo | in-progress | review
.mc/tasks/done/T-16-….md    finished: done | cancelled
.mc/data/*.json             generated index (gitignored) — rebuild with `mc index`
```

`mc task move` moves the file between `todo/` and `done/` itself; do not move it by hand and
leave `status:` behind.

## Commands worth knowing

```bash
mc task next                 # highest-priority actionable task, blocked ones filtered out
mc task board                # kanban across backlog / todo / in-progress / review / done
mc show T-16                 # one task's full record
mc list tasks --status todo  # filter
mc new task "…" --priority 2 --depends-on T-39
mc task move T-39 in-progress
mc index                     # after editing files by hand
mc validate                  # see the caveat below
```

## Conventions specific to this repo

- **IDs stay `T-NN`.** `T-16`, `T-25c` and friends are cited in `README.md`, `docs/`,
  `cluster/discoverer/*.sbatch`, `runs/` artifacts and git commit subjects. Renumbering them to
  `TASK-016` would break every one of those references for no gain, so `.mc/config.yml` sets
  `id_prefixes.task: T` and the existing IDs were carried over verbatim. `mc new task` mints
  `T-040` onward — mc pads to three digits, and its `^T-(\d+)` scan reads the short and
  letter-suffixed IDs correctly when picking the next number.
- **`mc validate` reports 44 `task-filename` issues, and that is expected.** Its filename check is
  hard-wired to `PREFIX-NNN-slug.md` (exactly three digits), which `T-01` and `T-25c` cannot
  satisfy. Nothing else fails, and every other command — `index`, `board`, `next`, `show`, `list`
  — works on these files. Treat a *new* issue class in `mc validate` as a real failure; the
  `task-filename` block is noise.
- **`tags:` carry the milestone** (`m0`…`m4`) plus topical tags, so `mc list tasks --tag m3`
  reproduces a milestone view. mc has no milestone concept of its own.
- **`depends_on:` is load-bearing.** T-32 depends on T-39, which is why `mc task next` returned
  T-39 and not the bigger job behind it. That ordering is pre-registered (PR-07), not a
  preference — keep it in the frontmatter, not only in prose. **T-39 reported 2026-08-16
  (`VOID (labels)`); its dependents do not auto-unblock, because the verdict's own premise was
  withdrawn by PR-12/PR-13 and what follows is the project owner's call, not a state transition.**
- **`TASKS.md` stays** as the milestone index with the M0–M4 exit criteria, the M3 narrative and
  the OD table. It links here; it no longer holds task detail.

## Where the content came from

Migrated 2026-08-06 from `TASKS.md`, 44 entries, each one's prose carried over unchanged (only
re-wrapped to 100 columns). `created`/`updated` are real dates: the first and last commit whose
`TASKS.md` diff touched that ID (`git log -G`). **T-37, T-38 and T-39 have no commit history at
all** — they exist only in the uncommitted working tree — so their dates come from the entry text
instead, and are recorded here as inferred rather than presented as measured.
