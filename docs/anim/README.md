# docs/anim — the receding-horizon explainer

An animated version of the one idea in `../architecture.md` §3: **every cycle predicts a whole
action chunk, but only its first `prefix_steps` run — the tail is discarded, not queued** (FR-05,
`src/wam/runtime/executor.py`). Five beats, looping: OBSERVE → PREDICT → FILTER → EXECUTE → RE-PLAN.

| File | What |
|------|------|
| `receding-horizon.json` | the animation (Lottie, 660×464, 10 s, ~92 KB) |
| `receding-horizon.html` | a local player — open it in a browser to watch the loop |
| `receding-horizon-poster.png` | the still fallback (GitHub, no-JS, `prefers-reduced-motion`) |
| `make-receding-horizon.py` | generates the JSON |
| `make-poster.py` → `receding-horizon-poster.svg` | generates the poster, from the same geometry |

## Regenerate

```bash
python3 make-receding-horizon.py                                   # -> receding-horizon.json
python3 "$EMAI_LOTTIE_KIT/check_seam.py" receding-horizon.json     # the loop must wrap invisibly
python3 make-poster.py                                             # -> ...-poster.svg
```

`make-receding-horizon.py` needs the EmAI Lottie kit; it defaults to
`~/develop/emai/headquarter/.claude/skills/emai-animation`, override with `EMAI_LOTTIE_KIT`.
Rendering the poster SVG to PNG needs cairo and the brand fonts — see the header of `make-poster.py`.

## Embed it

The animation is a plain Lottie: load it with lottie-web (or `lottie-react`) into a mount that is
**laid out and visible before `loadAnimation`** — loading into a `display:none` container collapses
every text label. `receding-horizon.html` is a working copy of that snippet, including the poster
fallback and the `prefers-reduced-motion` guard. LinkedIn and GitHub cannot play Lottie; post or
render the poster PNG there instead.
