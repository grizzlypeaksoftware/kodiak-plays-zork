# Kodiak Plays Zork

[Kodiak](https://github.com/grizzlypeaksoftware/kodiak) is a 152M-parameter, encoder-only **decision model**. It never
generates text: it picks one of the labels you give it, with a calibrated probability, or it abstains. This demo lets it
play a text adventure **by choosing**: [Jericho](https://github.com/microsoft/jericho) lists the commands the game accepts,
and Kodiak picks one.

What the demo shows:

1. **Kodiak can never issue an invalid command.** Every move comes from the game's valid-action list, whoever chose it.
   The run summary counts invalid moves; the count is 0 by construction.
2. **It decides in milliseconds on a CPU.** About 200 ms per decision on this machine's CPU (see [Latency](#latency)).
3. **A System 1 / System 2 cascade.** Kodiak takes the routine moves it's confident about. When it abstains or is unsure,
   the turn goes to a fallback: exploration, or a local LLM. The summary reports what share of moves each part made.

**Set expectations:** Kodiak is a *public research preview* and it plays badly. It doesn't solve puzzles, it loops, and it
is drawn to commands that sound meaningful ("say fee", "push keys to ground"). Zork is hard even for large LLMs. The goal
here isn't to beat the game; it's to show the cascade and the "never off the menu" property, and to measure both honestly.

## Contents

| File | What it is |
|---|---|
| `server.py` | Kodiak as a local HTTP service on `127.0.0.1:8765` (`POST /decide`, `GET /health`) |
| `play.py` | The agent loop: state → candidates → Kodiak → cascade → move, with a terminal view, JSONL transcript and summary |
| `repro_kodiak.py` | Minimal reproduction of the Kodiak behaviors noted below (no game needed) |
| `requirements.txt` | Pinned dependencies |
| `games/` | Where game files go (not committed) |
| `runs/` | Transcripts and summaries; `advent-100-cpu*` is the reference run |

## Setup

Python 3.12. CPU is enough.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm        # Jericho requires it
```

Jericho builds the Frotz interpreter when it is installed, so you need a C compiler (it installed cleanly on Linux aarch64).
The first run downloads the model (~600 MB) from
[cortex-agent-llc/kodiak-small-r1-preview](https://huggingface.co/cortex-agent-llc/kodiak-small-r1-preview).

## Game files

Jericho does not ship game files, and this repo doesn't either. **Jericho only lists valid actions for the exact builds it
knows (matched by MD5)**, so the file has to be the right one:

| Game | File | MD5 | Source |
|---|---|---|---|
| Zork I | `games/zork1.z3` | `b732a93a6244ddd92a9b9a3e3a46c687` | Your own legitimately obtained copy (e.g. from a purchased Infocom collection). Don't download it from unofficial sites. |
| Zork II | `games/zork2.z3` | `5bcd91ee055e9bd42812617571be227b` | Same |
| Zork III | `games/zork3.z3` | `ffda9ee2d428fa2fa8e75a1914ff6959` | Same |
| Adventure (Inform port by Graham Nelson) | `games/advent.z5` | `ee2242e155fd8910921b0f8e04019a3a` | Freely distributable: `https://ifarchive.org/if-archive/games/zcode/Advent.z5` |
| 9:05 (Adam Cadre) | `games/905.z5` | `4c5067169b834d247a30bb08d1039896` | Freely distributable: `https://ifarchive.org/if-archive/games/zcode/905.z5` |

Microsoft's 2025 MIT-licensed release of the Zork source doesn't help here: a build from that source has a different MD5,
so Jericho treats it as unsupported and can't list valid actions. `play.py` refuses to start on an unsupported file.

**The extension doesn't matter; the MD5 does.** Jericho's docs call the Zork I file `zork1.z5`, but the supported build is a
version-3 story file that Infocom collections usually ship as `zork1.z3`. Other Infocom files from the same collection
(Zork Zero, Beyond Zork, Planetfall) are builds Jericho doesn't know, so they won't work. Check a file with:

```bash
python -c "import hashlib,sys; from jericho import defines; h=hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest(); print(h, defines.BINDINGS_DICT.get(h, {}).get('name', 'NOT SUPPORTED'))" games/zork1.z3
```

```bash
mkdir -p games
curl -L -o games/advent.z5 https://ifarchive.org/if-archive/games/zcode/Advent.z5
md5sum games/*.z5
```

## Hosting Kodiak

**Option B: local HTTP service (recommended).** The model loads once at startup; the server binds to localhost only.

```bash
python server.py                   # CPU, 8 torch threads, http://127.0.0.1:8765
python server.py --device cuda     # a GPU if you have one to spare
python server.py --threads 4       # tune for your CPU (more threads is not always faster)
```

```bash
curl -s 127.0.0.1:8765/decide -H 'content-type: application/json' -d '{
  "state": "West of House. You are standing in an open field west of a white house. There is a small mailbox here.",
  "questions": [{"type": "choice", "id": "action", "text": "Which command best makes progress in this adventure?",
                 "labels": ["open mailbox", "north", "south", "west"]}]}'
```

The body is Kodiak's own Request format (`state`, `questions`, optional `options` such as `null_threshold` and
`min_confidence`). The `{"inputs": {...}}` wrapper used by the model repo's `handler.py` is accepted too. The response is
Kodiak's Response: `{"model", "latency_ms", "answers": {id: {"answer", "confidence", "probs", "p_null", "abstain_reason"}}}`.

**Option A: in-process.** `python play.py ... --in-process` loads the model inside the game loop instead.

## Running

```bash
python server.py &
python play.py --game games/advent.z5 --max-moves 100
python play.py --game games/zork1.z3  --max-moves 100
python play.py --game games/zork1.z3  --llm-model qwen3:30b-a3b     # System 2 = a local LLM via Ollama
```

| Flag | Default | Meaning |
|---|---|---|
| `--threshold` | 0.5 | Take Kodiak's move when its confidence is ≥ this; otherwise it's a System 2 turn |
| `--null-threshold` | model's (0.75) | Kodiak abstains ("unanswerable") when p_null ≥ this |
| `--max-moves` | 100 | Stop after this many moves |
| `--max-repeats` | 2 | Skip a command tried this many times in the same room, while alternatives exist |
| `--llm-model` | none | Ollama model for System 2; without it, System 2 is exploration |
| `--ollama-url` | `$OLLAMA_HOST` or `http://127.0.0.1:11434` | Ollama endpoint (read from the environment, no secrets in code) |
| `--kodiak-url` | `$KODIAK_URL` or `http://127.0.0.1:8765` | The server from `server.py` |
| `--in-process` / `--device` / `--threads` | off / cpu / 8 | Load Kodiak in the game process instead |
| `--seed` | 0 | Game RNG and exploration seed |
| `--no-danger` | off | Don't ask the second question ("Is the player in danger?") |
| `--stop-on-death` | off | Stop at game over (default: restart and keep counting moves) |
| `--transcript`, `--out` | `runs/<game>-<time>.jsonl` | Where the transcript and `-summary.json` go |
| `--quiet`, `--text-lines`, `--no-color` | | Terminal view |

## How a turn works

1. **State** (kept under 450 tokens, counted with Kodiak's own tokenizer; Kodiak was trained on states up to 512): the
   room description, the last 3 commands with what they produced, the inventory, and commands already tried in this room.
   The room and inventory come from running `look` and `inventory` and rolling the game back, so they cost no move.
2. **Candidates:** `get_valid_actions()`, deduplicated (case, spacing, direction abbreviations; Jericho already merges
   commands with the same effect), then the anti-loop filter (step 5).
3. **Ask Kodiak** two questions in one forward pass: "Which command best makes progress in this adventure?" (choice over
   the candidates) and "Is the player in danger?" (yes/no). Kodiak takes at most 32 labels per question, so more candidates
   go through a **tournament**: chunks of ≤ 32 become separate questions in the same request, and each chunk's top label goes
   to a final question over the winners.
4. **Cascade.** If Kodiak answered with confidence ≥ `--threshold`, it makes the move. Otherwise (abstained, or unsure)
   System 2 does: exploration (an untried command in this room, else random), or the LLM, which gets the same state and
   must pick from the same list (Ollama JSON-schema `enum`). If the LLM fails, exploration takes over. A turn with a single
   candidate is **forced** and doesn't ask anyone.
5. **Anti-loop.** Two counters: (room, command) and (exact world state, command), using Jericho's world-state hash. The
   second catches undo loops like take/drop and walking back and forth. Commands already done from this exact state, or
   tried `--max-repeats` times in this room, are skipped while at least two alternatives remain; in a room where
   everything has been tried, the least-tried commands stay in.

Each transcript line holds the turn's state, valid actions, candidates, skipped repeats, Kodiak's full answers (every round
of a tournament, both questions), the move, who chose it and why, the LLM's answer if asked, the observation, reward and
score.

## Results

Both runs: Kodiak served by `server.py` on CPU (8 threads), System 2 = exploration, threshold 0.5, seed 0.

### Zork I, 100 moves, CPU (`runs/zork1-100-cpu*`)

| | |
|---|---|
| Moves | 100 (0 game overs) |
| Score | 0 → **5** (max 350) |
| Rooms visited | 9 (West/North/South of House, Behind House, Forest, Forest Path, Clearing, Canyon View, Up a Tree) |
| Decided by | **Kodiak 40 (40%)**, exploration 58 (58%), LLM 0, forced 2 (2%) |
| Kodiak | asked 98 times; abstained 0; below the 0.5 threshold 58 |
| Kodiak latency | 204 ms average model time, 208 ms round trip (p50 203 ms) |
| Danger question | "yes" 41 times, "no" 57 |
| States | at most 304 tokens; at most 14 candidates per turn (no tournament was needed) |
| **Invalid moves** | **0** |

Kodiak's opening move was "west" (0.80), not "open mailbox". It then spent most of the game circling the forest around
the house ("go around forest", "go around trees"). The only points came from **exploration**: it climbed the tree and took
the jewel-encrusted egg (+5). After that, both deciders fiddled with the egg, nest and canary for ~40 moves. Exploration
also opened the kitchen window but closed it again on the next move, so the run never got into the house, and never
reached the lamp, the trapdoor or the underground. That's the expected level for this preview. The point of the run is the
last row.

### Adventure, 100 moves, CPU (`runs/advent-100-cpu*`)

| | |
|---|---|
| Moves | 100 (0 game overs) |
| Score | 36 → 36 (max 350); Adventure starts you at 36 |
| Rooms visited | 7 |
| Decided by | **Kodiak 43 (43%)**, exploration 57 (57%), LLM 0, forced 0 |
| Kodiak | asked 100 times; abstained 2 (unanswerable); below the 0.5 threshold 55 |
| Kodiak latency | 199 ms average model time, 204 ms round trip (p50 187 ms) |
| Danger question | "yes" 16 times, "no" 84 |
| States | 94 tokens on average, 175 at most; at most 20 candidates per turn |
| **Invalid moves** | **0** |

It wandered the surface: road, valley, streambed, grate, forest, building. It never picked up the lamp or unlocked the grate,
so it never reached the caves. Before the world-state anti-loop was added, a run spent about 20 moves toggling
"take bottle" / "put bottle down" at confidence 0.6–0.98 and still credited 61% of moves to Kodiak: **a high Kodiak share
isn't the same as good play**.

A 12-move check with `--llm-model qwen3:30b-a3b` split the moves 5 Kodiak / 7 LLM, with 0 invalid moves. The LLM took
~0.9 s per move once loaded, against ~0.2 s for Kodiak on CPU.


## Latency

Measured on an NVIDIA DGX Spark (GB10: 20-core Arm CPU), 200-token state, one question with 20 labels:

| Setup | Median |
|---|---|
| CPU, 4 threads | 289 ms |
| CPU, 8 threads | 252 ms |
| CPU, 20 threads | 463 ms (oversubscribed) |
| GPU (GB10, while a training job was running) | 17 ms |

Kodiak's docs quote ~80 ms on 8 CPU cores and ~8 ms on a GPU; that's presumably a different (x86) CPU. On this Arm CPU it's
about 3× slower, still well under a second, and small next to an LLM call. In-process mode was slower here (~370 ms) than
the server, most likely because torch shares the CPU with Jericho's parallel valid-action search; use the server.

## Notes on Kodiak (research preview)

The Kodiak code wasn't modified. What I observed, with a minimal reproduction in `repro_kodiak.py`:

```
$ python repro_kodiak.py
action -> take keys conf 0.24, p_null 0.00, top: take keys 0.24, take bottle 0.19, push keys to ground 0.17, light lantern with bottle 0.11
danger -> no conf 0.86, p_null 0.03, top: no 0.86, yes 0.12
```

1. **Weak action ranking, spread-out confidence.** In the well house, the useful move ("take lantern") isn't in the top 4,
   while "push keys to ground" and "light lantern with bottle" are. With 10–20 candidates the top confidence is usually
   0.2–0.4, so most such turns go to System 2 at a 0.5 threshold. With the full 20-candidate list for that room, its pick was
   "light lantern with bottle" (0.20).
2. **Confident about loops.** It gave 0.6–0.98 to "take bottle" / "put bottle down" alternately, and it keeps choosing
   "say fee" / "say fie" / "say foe" (magic words that change the game state, so Jericho lists them as valid). Confidence is
   calibrated on Kodiak's own eval tasks, not on "does this move make progress in a game", so it isn't a quality signal here.
   The demo handles loops outside the model; the magic-word loop survives because each "say" really changes the state.
3. **Rarely abstains.** p_null was near 0 on almost every turn (2 abstentions in 100 on Adventure, 0 in 98 on Zork). Low-confidence turns, not abstentions,
   drive the cascade.
4. **The danger question has false positives.** It said "yes" 16 times in 100 on Adventure's harmless surface, and 41 in 98 in Zork's forest,
   where nothing can hurt you.
5. **API constraints to design around** (behavior, not bugs): a choice question needs 2–32 labels, so a single valid
   command is a "forced" move and more than 32 need a tournament; `confidence` is `(1 − p_null) × P(label)`; in a
   tournament, the final confidence is relative to the finalists only, so it's higher than a single-round confidence would be.

## Licensing

- **This demo:** Apache-2.0, like Kodiak. It's a separate folder/repo, not part of the Kodiak repo.
- **Kodiak** (code and weights): Apache-2.0, Cortex Agent LLC. Installed via pip.
- **Jericho:** GPL-2.0 (Microsoft Research), and it bundles Frotz (GPL). The demo depends on it as an installed package
  only and copies none of its code. If you distribute a bundle that includes Jericho, GPL terms apply to that bundle.
- **Game files:** not included. Zork I is a commercial Infocom title (now owned by Microsoft/Activision); use your own copy.
  Adventure (Inform port) and 9:05 are freely distributable from the IF Archive under their authors' terms.
- **LLM fallback:** whatever model you run in Ollama, under its own license.
