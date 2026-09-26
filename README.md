# Kodiak Plays Zork

[Kodiak](https://github.com/grizzlypeaksoftware/kodiak) is a 152M-parameter, encoder-only **decision model**. It never
generates text: it picks one of the labels you give it, with a calibrated probability, or it abstains. This demo lets it
play text adventures **by choosing**: [Jericho](https://github.com/microsoft/jericho) lists the commands the game accepts,
and Kodiak picks one.

What the demo shows:

1. **Kodiak can never issue an invalid command.** Every move comes from the game's valid-action list, whoever chose it.
   Across 100 benchmark games (10,000 moves) the invalid-move count is 0.
2. **It decides fast on a CPU:** 250–350 ms per decision on an Arm CPU, no GPU needed (see [Latency](#latency)).
3. **A System 1 / System 2 cascade.** Kodiak takes the moves it's confident about. When it abstains or is unsure, the turn
   goes to a fallback: exploration, or a local LLM. Every run reports what share of moves each part made.
4. **An honest benchmark.** The same games and seeds, with and without the model, so a new Kodiak version can be compared
   against this one and against a no-model baseline.
5. **A head-to-head with Jev**, TypeSafe AI's closed System One model, through its API on the same harness
   (see [Kodiak vs Jev](#kodiak-vs-jev-zork-i)): Jev picks far better moves on Zork I, but walks into the grue.

**Set expectations.** Kodiak is a *public research preview*, and the benchmark says plainly how it plays. With harness v1,
the r1 preview clearly helps on one game (Detective), roughly ties on three, and **hurts on Zork I** compared with
exploration alone. The v2 preview scores best on Zork I, but its own moves earn almost nothing and it gets stuck on
Detective (see [r1 vs v2](#model-comparison-r1-preview-vs-v2-preview)).
Zork is hard even for large LLMs. The goal here isn't to beat the game; it's to show the cascade and the "never off the
menu" property, and to measure the model honestly.

## Contents

| File | What it is |
|---|---|
| `server.py` | Kodiak as a local HTTP service on `127.0.0.1:8765` (`POST /decide`, `GET /health`) |
| `play.py` | The agent loop: state → candidates → Kodiak → cascade → move, with a terminal view, JSONL transcript and summary |
| `benchmark.py` | Runs models and the no-model baseline on the same games and seeds, and prints a comparison table |
| `repro_kodiak.py` | Minimal reproduction of the Kodiak behaviors noted below (no game needed) |
| `requirements.txt` | Pinned dependencies |
| `games/` | Where game files go (not committed) |
| `runs/` | Transcripts, summaries and benchmark results (local only, not committed) |

## Setup

Python 3.12 on Linux or macOS (on Windows, use WSL). CPU is enough.

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
knows, matched by MD5**, so the file has to be the right one. The extension doesn't matter.

| Game | File | MD5 | Source |
|---|---|---|---|
| Zork I | `games/zork1.z3` | `b732a93a6244ddd92a9b9a3e3a46c687` | Your own legitimately obtained copy (e.g. from a purchased Infocom collection). Don't download it from unofficial sites. |
| Zork II | `games/zork2.z3` | `5bcd91ee055e9bd42812617571be227b` | Same |
| Zork III | `games/zork3.z3` | `ffda9ee2d428fa2fa8e75a1914ff6959` | Same |
| Adventure (Inform port, Graham Nelson) | `games/advent.z5` | `ee2242e155fd8910921b0f8e04019a3a` | IF Archive, freely distributable |
| Balances (Graham Nelson) | `games/balances.z5` | from Jericho | IF Archive, freely distributable |
| Detective (Matt Barringer) | `games/detective.z5` | from Jericho | IF Archive, freely distributable |
| All Quiet on the Library Front | `games/library.z5` | from Jericho | IF Archive, freely distributable |
| 9:05 (Adam Cadre) | `games/905.z5` | `4c5067169b834d247a30bb08d1039896` | IF Archive, freely distributable |

```bash
mkdir -p games
for f in Advent Balances detective library 905; do
  curl -sSfL -o "games/$(echo $f | tr A-Z a-z).z5" "https://ifarchive.org/if-archive/games/zcode/$f.z5"
done
# check that a file is a build Jericho supports:
python -c "import hashlib,sys; from jericho import defines; h=hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest(); print(h, defines.BINDINGS_DICT.get(h, {}).get('name', 'NOT SUPPORTED'))" games/zork1.z3
```

Jericho's docs call the Zork I file `zork1.z5`, but the supported build is a version-3 story file that Infocom collections
usually ship as `zork1.z3`. Other files from the same collections (Zork Zero, Beyond Zork, Planetfall) are builds Jericho
doesn't know, and a build from Microsoft's 2025 MIT-licensed Zork source has a different MD5 too. `play.py` refuses to
start on an unsupported file.

## Hosting Kodiak

**Option B: local HTTP service (recommended).** The model loads once at startup; the server binds to localhost only.

```bash
python server.py                          # CPU, 8 torch threads, http://127.0.0.1:8765
python server.py --model ./my-kodiak-v2   # any Hugging Face repo id or local model folder
python server.py --device cuda            # a GPU if you have one to spare
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

**Option A: in-process.** `python play.py ... --in-process` loads the model inside the game loop instead (slower here,
because torch then shares the CPU with the game).

## Running

```bash
python server.py &
python play.py --game games/zork1.z3 --max-moves 100
python play.py --game games/zork1.z3 --baseline                      # no model: exploration only
python play.py --game games/zork1.z3 --llm-model qwen3:30b-a3b       # System 2 = a local LLM via Ollama
```

| Flag | Default | Meaning |
|---|---|---|
| `--threshold` | 0.5 | Take Kodiak's move when its confidence is ≥ this; otherwise it's a System 2 turn |
| `--question` | `explore` | Wording of the action question (`explore`, `progress`, `next`; see [Tuning](#tuning-practice-games-only)) |
| `--baseline` | off | Never ask Kodiak: the no-model baseline |
| `--null-threshold` | model's (0.75) | Kodiak abstains ("unanswerable") when p_null ≥ this |
| `--max-moves` | 100 | Stop after this many moves |
| `--max-repeats` | 2 | Skip a command tried this many times in the same room, while alternatives exist |
| `--llm-model` | none | Ollama model for System 2; without it, System 2 is exploration |
| `--ollama-url` | `$OLLAMA_HOST` or `http://127.0.0.1:11434` | Ollama endpoint (from the environment; no secrets in code) |
| `--kodiak-url` | `$KODIAK_URL` or `http://127.0.0.1:8765` | The server from `server.py` |
| `--in-process` / `--model` / `--device` / `--threads` | off / preview / cpu / 8 | Load Kodiak in the game process instead |
| `--backend jev` / `--jev-model` / `--env-file` | kodiak / jev-latest / none | Use TypeSafe's Jev API as System 1; key from `$TYPESAFE_API_KEY` or `$JEV-KEY`, optionally loaded from an env file (never printed) |
| `--seed` | 0 | Game RNG and exploration seed |
| `--no-danger` | off | Don't ask the second question ("Is the player in danger?") |
| `--stop-on-death` | off | Stop at game over (default: restart and keep counting moves) |
| `--transcript`, `--out` | `runs/<game>-<time>.jsonl` | Where the transcript and `-summary.json` go |
| `--quiet`, `--text-lines`, `--no-color` | | Terminal view |

## How a turn works (harness v1)

1. **State**, kept under 450 tokens (counted with Kodiak's own tokenizer; Kodiak was trained on states up to 512): the room
   description, the last 3 commands with what they produced (and any points they earned), the inventory, the score and when
   points were last earned, exits not yet tried from this room, and commands already tried here. The room and inventory
   come from running `look` and `inventory` and rolling the game back, so they cost no move.
2. **Candidates:** Jericho's valid actions, deduplicated (case, spacing, direction abbreviations), then filtered:
   - **anti-loop:** two counters, (room, command) and (exact world state, command) via Jericho's world-state hash. Commands
     already done from this exact state, or tried `--max-repeats` times in this room, are skipped while at least two
     alternatives remain;
   - **no immediate undo:** a command that just reverses the previous one (open → close, take → drop, put in → take out) is
     skipped while alternatives remain.
3. **Ask Kodiak** two questions in one forward pass: "Which command best makes progress in this text adventure: exploring
   new places, getting useful items or solving a puzzle?" (choice over the candidates) and "Is the player in danger?"
   (yes/no). Kodiak takes at most 32 labels per question, so more candidates go through a **tournament**: chunks of ≤ 32
   become separate questions in the same request, and each chunk's top label goes to a final question over the winners.
4. **Cascade.** If Kodiak answered with confidence ≥ `--threshold`, it makes the move. Otherwise (abstained, or unsure)
   System 2 does. **Exploration** tries untried commands in this room in this order: useful-looking interactions (take,
   open, read, examine…), then exits, then anything else, then odd ones ("throw X at Y"); then a random command. The **LLM**
   gets the same state and must pick from the same list (Ollama JSON-schema `enum`); if it fails, exploration takes over. A
   turn with a single candidate is **forced** and doesn't ask anyone.

**No lookahead.** The harness never tries moves in the emulator to see their rewards, deaths or where exits lead. Every
signal it uses is something a player could observe, plus Jericho's valid-action list (which Jericho computes by trying
commands; that's the one concession, and it's the same for every configuration).

**Crash safety.** In some Zork I states (the nest inside the egg you carry), merely *testing* "drop all down nest" or
"put all in nest" crashes the Frotz emulator, in every Jericho search mode; with the parallel search it can also hang
forever. So each turn's valid-action search runs in a forked child process. If the child dies, a traced rerun finds the
command, which is blocked for the rest of the game (a command that crashes the game isn't a valid move). Runs list blocked
commands in their summary.

Each transcript line holds the turn's state, valid actions, candidates, skipped commands, Kodiak's full answers (every round
of a tournament, both questions), the move, who chose it and why, the LLM's answer if asked, the observation, reward and
score.

## Benchmarking and comparing models

```bash
G=games
# the reference protocol: 5 games x 5 seeds x 100 moves, baseline + the preview model
python benchmark.py --baseline --model cortex-agent-llc/kodiak-small-r1-preview \
  --games $G/advent.z5 $G/balances.z5 $G/detective.z5 $G/library.z5 $G/zork1.z3 --seeds 0 1 2 3 4

# Jev on Zork I (API key from an env file; each move is one API call)
python benchmark.py --jev --env-file path/to/.env --games $G/zork1.z3 --seeds 0 1 2 3 4

# a new model against the preview, same protocol
python benchmark.py --baseline --model cortex-agent-llc/kodiak-small-r1-preview --model ./kodiak-v2 \
  --games $G/advent.z5 $G/balances.z5 $G/detective.z5 $G/library.z5 $G/zork1.z3 --seeds 0 1 2 3 4
```

Each model gets its own `server.py` (CPU by default) while its games run; runs are sequential so latency isn't distorted.
A game that crashes or exceeds `--run-timeout` (900 s) is reported and left out of the table instead of stalling the
benchmark. Results go to `runs/bench-<time>/results.md` and `results.json`.

**Rules for a fair comparison:**

- **Freeze the harness.** Compare models only on the same harness version (git tags `harness-v0`, `harness-v1`).
- **Practice vs test games.** Adventure, Balances, Detective and Library are practice games; anything tuned (question
  wording, threshold) is chosen on them, with seeds 10–12. **Zork I and seeds 0–4 are never used for tuning.**
- **Report the spread.** One 100-move game is noisy; look at mean ± standard deviation over the 5 seeds, and treat
  differences smaller than the spread as ties.
- **Look past the Kodiak share.** A high share of moves isn't good play: in early testing Kodiak made 61% of the moves in a
  run where it spent ~20 of them picking up and dropping a bottle. Score gained, rooms visited, and points per move by
  chooser are the quality signals.

## Results

All numbers: CPU (NVIDIA DGX Spark, 20-core Arm), Kodiak served by `server.py` with 8 threads, System 2 = exploration,
threshold 0.5, 100 moves per game, seeds 0–4, mean ± sd over the 5 seeds. **0 invalid moves in all 100 benchmark games below (and in every tuning game).**

### Harness v1 (current)

| Game (max score) | Score gained, baseline | Score gained, Kodiak cascade | Rooms, baseline | Rooms, Kodiak | Kodiak's share | Kodiak ms |
|---|---|---|---|---|---|---|
| **Zork I** (350) | **13.0 ± 2.7** | 6.0 ± 5.5 | 10.2 ± 0.8 | 10.0 ± 1.0 | 18% | 275 |
| Detective (360) | 92 ± 28 | **178 ± 113** | 13.2 ± 1.6 | 14.4 ± 5.0 | 31% | 305 |
| Library (30) | 12.4 ± 0.9 | 12.8 ± 1.8 | 6.8 ± 0.8 | 7.2 ± 0.4 | 34% | 351 |
| Balances (51) | 10.0 ± 0.0 | 9.0 ± 2.2 | 3.0 | 3.0 | 28% | 286 |
| Adventure (350) | 0 | 0 | 6.8 ± 0.4 | 6.8 ± 0.4 | 16% | 246 |

(Adventure starts at 36 points and 100 moves isn't enough to earn more; Detective starts at 10.)

### Model comparison: r1-preview vs v2-preview

Same frozen harness (v1), games and seeds; only the model differs.
[`kodiak-small-v2-preview`](https://huggingface.co/cortex-agent-llc/kodiak-small-v2-preview) was benchmarked while a
training job was running on the same machine, so don't read anything into latency differences (the model times were
similar: 240–360 ms).

| Points gained | Baseline | r1-preview | v2-preview |
|---|---|---|---|
| **Zork I** | 13.0 ± 2.7 | 6.0 ± 5.5 | **21.4 ± 18.0** (per seed: 10, 42, 10, 5, 40) |
| **Detective** | 92 ± 28 | **178 ± 113** | 10 ± 0 (stuck in all 5 seeds) |
| Library | 12.4 ± 0.9 | **12.8 ± 1.8** | 9.0 ± 2.7 |
| Balances | 10.0 | 9.0 ± 2.2 | 10.0 |
| Adventure | 0 | 0 | 0 |

| Model behavior | r1-preview | v2-preview |
|---|---|---|
| Share of moves made by the model | 25% | **53%** |
| Points per 100 of the model's own moves, all games | **68.4** | 1.7 |
| Points per 100 of the model's own moves, Zork I | 0.0 (88 moves) | 1.3 (157 moves) |
| Points per 100 exploration moves, Zork I | 7.4 | **32.4** |
| Abstentions | 3 | 0 |
| Danger question says "yes" | 27% of turns | **75%** of turns |
| Rooms visited, Zork I | 10.0 | **11.2** |

**Reading it.**

- **v2 is more confident** and makes twice the share of moves, but its own moves earn almost nothing (2 points on Zork I, from
  putting the egg in the trophy case).
- **Its Zork I lead comes from two seeds** where the exploration fallback got into the house and down to the cellar (+25). The
  median seed scored 10, below the baseline's 15. v2's moves did steer the game somewhere exploration could score (32
  points per 100 exploration moves, against 7 with r1), but with 5 seeds and this spread, it's not a clear win.
- **Detective shows the risk of confidence without judgment.** After taking the paper (+10), v2 walks outside and then
  cycles "north" (which fails), "put paper down" and "take paper" for 95 moves at confidence 0.5–0.8. The only real exit
  ("west") is in the candidates on every turn and listed in the state as an untried exit; v2 never picks it, and because
  it's confident, the fallback never gets a turn. The model is deterministic, so all 5 seeds end the same way.
- **The danger question got worse** ("yes" on three-quarters of turns).

### Kodiak vs Jev (Zork I)

[Jev](https://typesafe.ai) is TypeSafe AI's closed "System One" decision model, the model that inspired Kodiak. `play.py
--backend jev` calls its API (`POST /v1/systemone`) with the same states, candidates and question wording; the yes/no
danger question uses Jev's native yes/no type. Jev has no abstention, so its cascade runs on confidence alone, against the
same 0.5 threshold. Same frozen harness (v1), Zork I, seeds 0–4, 100 moves.

| Zork I, 5 seeds | Baseline | Kodiak r1-preview | Kodiak v2-preview | Jev (`jev-latest`) |
|---|---|---|---|---|
| Points gained (per seed) | 15, 15, 15, 10, 10 | 10, 10, 0, 0, 10 | 10, 42, 10, 5, 40 | 10, 15, 10, 10, 10 |
| **Points gained, mean ± sd** | 13.0 ± 2.7 | 6.0 ± 5.5 | **21.4 ± 18.0** | 11.0 ± 2.2 |
| Rooms visited | 10.2 | 10.0 | 11.2 | **12.2** |
| Deaths (5 games) | 0 | 1 | 1 | 3 |
| Share of moves made by the model | – | 18% | 31% | **50%** |
| Points earned by the model's own moves | – | 0 (in 88 moves) | 2 (in 157) | **45 (in 252)** |
| Median confidence of the model's pick | – | 0.32 | 0.37 | 0.51 |
| Danger question says "yes" | – | 13% | 60% | 4% |
| Latency per decision (p50) | – | 275 ms, local CPU | 301 ms, local CPU | 133 ms, network API |
| Cost | free | free (local) | free (local) | 499 calls, 272k input + 59k output tokens |
| **Invalid moves** | 0 | 0 | 0 | **0** |

**Same position, different picks.** Every one of Jev's 498 decision positions was replayed offline through both Kodiak
models (identical state, candidates and question):

| | Agrees with Jev's pick | Confident (≥ 0.5) |
|---|---|---|
| Jev | – | 51% of positions |
| Kodiak r1-preview | 23% | 32% |
| Kodiak v2-preview | 19% | 46% |
| (r1 vs v2) | 60% with each other | |

| Position | Jev | Kodiak r1 | Kodiak v2 |
|---|---|---|---|
| West of House, move 1 | **open mailbox** (0.78) | west (0.66) | west (0.42) |
| Behind House, window open (5 games) | **west**, into the kitchen, 0.43–0.66 | close window, 0.30–0.40 (all 5) | west twice, "put down …" 3× (0.24–0.33) |
| Dark attic, "likely to be eaten by a grue" | north (the fatal move) | north, 0.09–0.13 | north once, other moves twice (≤ 0.24) |

**Reading it.**

- **Jev plays Zork more sensibly move by move.** It opens the mailbox, climbs in through the kitchen window on its own
  (4 of 5 games), explores the most rooms, and its own moves earned 45 points where Kodiak's earned 0–2. Its danger answer
  is also far better calibrated here (4% "yes").
- **But it didn't score more than exploration alone** (11.0 vs 13.0), because it went up into the dark attic and walked
  into the grue in 3 of 5 games: each death costs 10 points and restarts the game. In every one of those deaths, **Jev's
  own danger answer said "yes" at 0.94–0.95** in the same forward pass that chose "north". The harness logs the danger
  answer but doesn't act on it; using it as a veto ("in danger → don't move into the dark, escalate") is the obvious next
  harness change, for every model.
- **Kodiak r1 knows less but hurts less on Zork:** it rarely commits (32% confident), so exploration carries most turns. Its
  pick behind the house is "close window" every time, the opposite of the key move.
- **Kodiak v2's best score comes from exploration**, as noted above; head to head it agrees with Jev least (19%).
- **Speed and cost:** Jev answered in ~130 ms over the network; Kodiak takes ~300 ms on this Arm CPU (~17 ms on a GPU),
  runs locally, and costs nothing per call.

### Harness v0 (the first version, for reference)

| Game | Score gained, baseline | Score gained, Kodiak cascade | Kodiak's share |
|---|---|---|---|
| Zork I | **6.0 ± 6.5** | 1.0 ± 2.2 | 45% |
| Detective | **96 ± 27** | 46 ± 25 | 68% |
| Library | **13.0 ± 2.2** | 5.6 ± 4.7 | 69% |
| Balances | 10 | 10 | 48% |
| Adventure | 0 | 0 | 48% |

### How good are Kodiak's own moves?

Points earned per 100 moves, split by who chose the move (Kodiak configuration, all 25 games per harness):

| | Kodiak's moves, v0 | Exploration's moves, v0 | Kodiak's moves, v1 | Exploration's moves, v1 |
|---|---|---|---|---|
| All games | 14.4 | 35.0 | **68.4** | 52.8 |
| Detective | 55.6 | 196.2 | **275.6** | 244.2 |
| Library | 0.0 | 18.5 | 1.2 | 19.5 |
| Balances | 4.1 | 15.6 | 0.0 | 12.5 |
| Zork I | 0.0 | 1.9 | 0.0 | 7.4 |

**Reading it.**

- **Harness v1 changed the picture more than anything else.** Under v0 the cascade lost to exploration on three games. Under
  v1 it roughly ties on three, wins big on Detective, and still loses on Zork I.
- **The gain is concentrated in one game.** On Detective (a linear game with frequent rewards, where the obvious move is usually the
  right one) Kodiak's confident picks beat exploration's. On the other games its moves earned almost nothing; its
  contribution there is to cost as little as possible.
- **On Zork I, the cascade scores about half the baseline.** The 88 Kodiak-chosen Zork moves earned 0 points. All points
  came from exploration: climbing in through the kitchen window (+10) in 3 of 5 games. Exploration alone did that in all 5
  games and also took the egg from the tree (+5) in 3.
- **The v1 changes did three things**: the "explore" question wording (the largest effect; see Tuning), the undo filter, and a
  state that shows progress and untried exits. With the new wording Kodiak is less often confident, so it hands about
  three-quarters of the moves to exploration, and the moves it keeps are better.

### Tuning (practice games only)

Harness v1, Adventure + Balances + Detective + Library, seeds 10–12. "Normalized gain" = score gained ÷ max score, averaged
over the four games.

| Configuration | Normalized gain | Rooms | Kodiak's share | Points per 100 Kodiak moves |
|---|---|---|---|---|
| **`explore` wording, threshold 0.5 (chosen)** | **22.9%** | 7.3 | 30% | 42.3 |
| exploration only (baseline) | 21.4% | 7.4 | 0% | – |
| `explore`, threshold 0.7 | 17.9% | 6.8 | 6% | 27.0 |
| `next` wording ("Which command should the player try next?") | 13.5% | 5.1 | 68% | 26.8 |
| `explore`, threshold 0.3 | 13.4% | 4.3 | 79% | 33.6 |
| `progress` wording (the original, "…best makes progress in this adventure?") | 12.4% | 4.2 | 70% | 1.2 |

**The question wording is the biggest lever found so far:** the same model on the same states earns 1.2 points per 100 of
its moves with the original wording and 42.3 with the `explore` wording. The threshold 0.7 row is almost pure exploration
yet scored 3.5 points below the baseline, which shows the noise level of a 3-seed comparison.

### What other agents score (context, not a like-for-like comparison)

| Source | Setup | Zork I | Detective | Library | Balances |
|---|---|---|---|---|---|
| [Jericho paper](https://arxiv.org/abs/1909.05398): random agent | 12 generic commands | 0 | 113.7 | 0 | 0 |
| Jericho paper: NAIL | no training, one episode, hand-built heuristics | 10.3 | 136.9 | 0.9 | 10 |
| Jericho paper: DRRN | RL trained on each game, avg of last 100 episodes | 32.6 | 197.8 | 17 | 10 |
| LLM agents on Jericho (ReAct, Reflexion, [MC-DML](https://arxiv.org/html/2504.16855v1)) | large LLMs, many steps/attempts | ~48–52 | | | |
| [Affine Layer leaderboard](https://affinelayer.com/zork/) | frontier LLMs, 200 moves, **given the manual, a map and a walkthrough** | o1 78, o3-pro 204, human 239 | | | |
| **This demo, baseline** | valid-action exploration, no training, 100 moves | 13.0 | 102 (total) | 12.4 | 10 |
| **This demo, Kodiak preview** | 152M encoder, never saw these games, 100 moves | 6.0 | 188 (total) | 12.8 | 9 |

Protocols differ (step budgets, training, walkthroughs, and our agents choose from Jericho's valid actions, which is a big
help), so treat this as orientation only. The comparison that's fair is inside this repo: models against each other and
against the baseline, on the same harness.

### Other checks

- **LLM fallback:** a 12-move check with `--llm-model qwen3:30b-a3b` split the moves 5 Kodiak / 7 LLM, 0 invalid; the LLM
  took ~0.9 s per move once loaded, Kodiak ~0.2 s.
- **Danger question** (asked every turn, same forward pass): "yes" on 27% of turns in v1 and 38% in v0, mostly in places where
  nothing can hurt you. Not a useful signal yet.
- **Abstention:** Kodiak abstained ("unanswerable") on 3 of ~2,480 decisions in v1. The cascade is driven by low confidence,
  not by abstention.

## Latency

Measured on an NVIDIA DGX Spark (GB10, 20-core Arm CPU), 200-token state, one question with 20 labels:

| Setup | Median |
|---|---|
| CPU, 4 threads | 289 ms |
| CPU, 8 threads | 252 ms |
| CPU, 20 threads | 463 ms (oversubscribed) |
| GPU (GB10, while a training job was running) | 17 ms |

In the benchmark, Kodiak's average model time was 200–300 ms per decision (v0) and 250–350 ms (v1, whose states are
longer). Kodiak's docs quote ~80 ms on 8 CPU cores, presumably on a different (x86) CPU; on this Arm CPU it's about 3×
slower, still well under a second and small next to an LLM call.

## Notes on Kodiak (research preview)

The Kodiak code wasn't modified. What I observed, with a minimal reproduction in `repro_kodiak.py`:

```
$ python repro_kodiak.py
action -> take keys conf 0.24, p_null 0.00, top: take keys 0.24, take bottle 0.19, push keys to ground 0.17, light lantern with bottle 0.11
danger -> no conf 0.86, p_null 0.03, top: no 0.86, yes 0.12
```

1. **Very sensitive to the question wording.** See Tuning: a 35× difference in points per move between two reasonable
   phrasings of the same question.
2. **Weak action ranking.** In Adventure's well house, the useful move ("take lantern") isn't in the top 4, while
   "push keys to ground" and "light lantern with bottle" are.
3. **Confident about loops.** Before the harness filtered them, it gave 0.6–0.98 to "take bottle" / "put bottle down"
   alternately, and it keeps choosing "say fee" / "say fie" / "say foe" (magic words that change the game state, so Jericho
   lists them as valid). Confidence is calibrated on Kodiak's own eval tasks, not on "does this move help in a game".
4. **Rarely abstains** (3 in ~2,480 decisions). Low confidence, not abstention, drives the cascade.
5. **The danger question has many false positives** (27–38% "yes" on mostly harmless turns).
6. **API constraints to design around** (behavior, not bugs): a choice question needs 2–32 labels, so a single valid
   command is a "forced" move and more than 32 need a tournament; `confidence` is `(1 − p_null) × P(label)`; in a
   tournament, the final confidence is relative to the finalists only.

## Notes on Jericho

Found while building this (Jericho 3.3.1, Linux aarch64):

- **Emulator crash on some candidate commands.** In Zork I, with the bird's nest inside the egg the player carries,
  testing "drop all down nest" (or "put all in nest") during `get_valid_actions()` crashes Frotz with a segfault, in the
  ctypes, pure-Python and parallel search modes alike. To reproduce: replay a game that ends with "take egg", "put nest in
  egg" while in the forest, then call `get_valid_actions()`.
- **Hang in the parallel search.** With the default `use_parallel=True`, the same situation can leave `Pool.map` waiting
  forever (a worker never returns). This demo uses the serial search in a forked child instead; see Crash safety.

## Licensing

- **This demo:** Apache-2.0, like Kodiak. It's a separate repo, not part of the Kodiak repo.
- **Kodiak** (code and weights): Apache-2.0, Cortex Agent LLC. Installed via pip.
- **Jericho:** GPL-2.0 (Microsoft Research), and it bundles Frotz (GPL). The demo depends on it as an installed package
  only and copies none of its code. If you distribute a bundle that includes Jericho, GPL terms apply to that bundle.
- **Game files:** not included. Zork I–III are commercial Infocom titles (now owned by Microsoft/Activision); use your own
  copies. Adventure, Balances, Detective, Library and 9:05 are freely distributable from the IF Archive under their
  authors' terms.
- **Run data** (transcripts contain game text) stays local and isn't committed.
- **LLM fallback:** whatever model you run in Ollama, under its own license.
