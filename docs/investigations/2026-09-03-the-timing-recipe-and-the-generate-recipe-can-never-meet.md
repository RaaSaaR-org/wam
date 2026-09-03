# The timing recipe and the generate recipe resolve THROUGHPUT.json to two different paths

**Found by the first licensed generation submission, which died at that gate in three seconds.**
Job `192122`, `2026-09-03T03:33:08`–`03:33:11` on `dgx2`, `FAILED 1:0`:

```
FATAL: no .../runs/t040-transfer25-restyle-2026-09-02/THROUGHPUT.json.
       PR-08 §8 item 3 makes the measurement a gate, not a warm-up
```

The gate is right and it fired correctly. **The measurement exists, it qualifies, and the signed
ceiling is derived from it — it was simply in another directory, and no recipe in the file puts it
where the generation path looks.**

---

## 1. The defect

`97_transfer25_restyle.sbatch`:

```
:603    OUT=${PROJ}/runs/${RUN_ID}
:1389   THROUGHPUT=${OUT}/THROUGHPUT.json      # not overridable from the environment
```

Both the TIMING writer and the GENERATE reader resolve that one expression. So they meet **only if
both run under the same `RUN_ID`.** The file's own two recipes make sure they do not:

| recipe | header line | `RUN_ID` | resolves `THROUGHPUT.json` to |
|---|---|---|---|
| TIMING | `:7-9` | `t040-transfer25-restyle-timing-<YYYY-MM-DD>` **(explicit, "part of the recipe")** | `runs/…-timing-<date>/` |
| GENERATE | `:31-34` | **passed none** → the default `t040-transfer25-restyle` | `runs/t040-transfer25-restyle/` |

**Two different directories, by construction, in the same file.** Following both recipes exactly, in
order, produces a generation submission that cannot see the measurement the timing submission just
made.

### How it got here, which matters because the change that caused it was a fix

The dated timing `RUN_ID` was added 2026-08-27 for a real and serious reason, and the header states
it at `:11-18`: the generate recipe's default `RUN_ID` was where **job 189142's poisoned artifact**
sat — `0.2 s/frame`, written by a run whose own log says *"0 success, 1 error"* — and a fresh timing
submission was silently reading it instead of measuring. Dating the timing run fixed that.

It also moved the artifact out of the only path the generation branch can read, and nothing in the
file was changed to bridge the two. **A fix on one side of a shared expression, with no reader on
the other side updated.** The generate recipe was never re-tested end to end after it, because no
generation run had ever been licensed until now.

## 2. What was done, and why it is not the obvious thing

**The FATAL's own advice is to re-measure**, and re-measuring was refused as the *worse* option.

`PARTITION_CEILING_GPU_H = 2013.75`, signed by the project owner on 2026-09-01, is **arithmetic over
this exact artifact**: `gpu_hours_per_variant = 80.55 × 25 style-instances = 2013.75`. The
generation gate then multiplies the artifact's `seconds_per_frame` by `whole_partition_frames` and
checks the projection against that signed figure.

So a fresh measurement would hand the gate a `seconds_per_frame` **the signed ceiling was not
derived from**. The authorisation and the check it is enforced by would silently stop being about
the same number — and nothing in either file would say so. Re-measuring is the right advice for an
operator who has no measurement; it is the wrong move for one whose ceiling already descends from
one.

**So the measured artifact was copied, byte-identical, into the generation run directory:**

```
runs/t040-transfer25-restyle-timing-2026-08-28/THROUGHPUT.json   cfad31f50011bf26…
runs/t040-transfer25-restyle-2026-09-02/THROUGHPUT.json          cfad31f50011bf26…   (copy)
```

`cp`, never `mv` — the timing run still holds the artifact it produced, because that directory *is*
the record of that measurement.

### Why this is a copy and not a waiver, and the difference is checkable

`throughput_qualification()` (`:1337-1381`) is called by **both** paths — deliberately, in its own
words: *"those two must never be able to answer differently about the same file."* It reads the
artifact's **content** and never its path: schema `wam.transfer25_throughput/1`, the three
disqualification shapes, and `units_succeeded >= 1`. A copied artifact therefore passes **only if
the measurement itself qualifies**, and the poisoned 189142 artifact would still be refused wherever
it were placed. Nothing about the gate was loosened, and no override exists to loosen it — the file
says so: *"There is no override: a ceiling is not a thing to be overridden on a submit line."*

### The copy says so, in the run directory, beside the artifact

`runs/t040-transfer25-restyle-2026-09-02/THROUGHPUT.provenance.json`
(`wam.transfer25_throughput_provenance/1`) records that this run did **not** measure the number, the
source path and digest, byte-identity, who copied it and when, the four-point reasoning above, and
`T40_RULE_V20` §5's mandatory disclosure. **`THROUGHPUT.json` itself was not touched** — annotating a
measurement in place is how a measurement stops being one.

## 3. What this does not do

* **It changes no measured value.** `1.6896 s/frame`, `80.55 GPU-h/variant`, `2013.75` are what they
  were; the artifact is byte-identical to the one the ceiling was signed against.
* **It lifts no gate and edits no rule.** `T40_RULE_V1` §1, `T40_RULE_V2` §3, PR-08 §8 unchanged.
* **It does not fix the sbatch.** See §4 — deliberately.
* **It does not re-open the chunk shape.** `STAGE=1 STYLE_SET=train CHUNK_INDEX=1 CHUNK_TOTAL=4`
  stands as released by `PR-08-DET-2026-09-02`, and the resubmission passes the identical line.

## 4. The fix that was NOT made, and why not yet

The file should not be able to do this to the next operator. Two candidate repairs:

* make the generate recipe name the same `RUN_ID` the timing recipe used; or
* let the generation path read a `THROUGHPUT` that is qualified but elsewhere, recording where.

**Neither was made now.** Editing the gate file *between a failed submission and its resubmission*
is the one moment when a change to it cannot be distinguished, by a later reader, from making the
gate stop complaining. The measurement was placed where the unmodified file already looks, and the
file is untouched. The repair belongs in its own change, with its own tests, when nothing is
pending on it.

**Recorded as owed.** A recipe that cannot be followed end to end is a defect even when a careful
operator can work around it once.

## 5. What the three seconds also proved

The run reached the throughput gate, which means everything before it passed on real inputs for the
first time: the T-39 attestation and its artifact digest (`92b6eaea…`), the allocation ceiling
(1 GPU / 26 threads / 192 GB), the full `check_style_partition.py` verification (16 styles, hash
`9334fd01…`, sidecar match, `PASS`), the generator pin
(`nvidia/Cosmos-Transfer2.5-2B@ce844032…`), the **corpus binding this repository spent 2026-09-02
building** — and the work list itself: **101 episodes, 4 styles, 404 units, 172 888 frames**,
`work.jsonl` sha `e558577a9f1967c5`.

That work list is on disk, stamped, with no `DONE` and no `passes.jsonl`. The resubmission resumes
onto it rather than rebuilding it.
