"""Kodiak plays a text adventure by *choosing* among the game's valid commands.

Each turn: build a short state (room, recent turns, inventory, what was tried here), ask Kodiak which valid command
makes progress, and take it if Kodiak is confident enough (System 1). Otherwise the turn goes to a fallback (System 2):
exploration, or a local LLM through Ollama. Every command comes from Jericho's valid-action list, so no move is ever
invalid, whoever chose it.

    python server.py &                                   # Kodiak on http://127.0.0.1:8765
    python play.py --game games/zork1.z5 --max-moves 100
    python play.py --game games/advent.z5 --in-process   # no server: load Kodiak inside this process
    python play.py --game games/zork1.z5 --llm-model qwen3:30b-a3b   # System 2 = an LLM (Ollama)

Writes a JSONL transcript (one line per turn) and a summary JSON next to it (default: runs/).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import requests
from jericho import FrotzEnv

QUESTIONS = {
    "progress": "Which command best makes progress in this adventure?",
    "explore": "Which command best makes progress in this text adventure: exploring new places, getting useful items "
               "or solving a puzzle?",
    "next": "Which command should the player try next?",
}
DANGER_Q = "Is the player in danger?"
HARNESS_VERSION = "v1"
MAX_LABELS = 32  # Kodiak's per-question label limit
STATE_TOKENS = 450  # Kodiak was trained on states up to 512 tokens
DIRECTIONS = {"n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
              "se": "southeast", "sw": "southwest", "u": "up", "d": "down"}


# ---------------------------------------------------------------------------
# Kodiak clients
# ---------------------------------------------------------------------------


class KodiakHTTP:
    """Talks to server.py."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        try:
            health = self.session.get(f"{self.url}/health", timeout=5).json()
        except requests.RequestException as e:
            sys.exit(f"Kodiak server not reachable at {self.url} ({e}).\nStart it with: python server.py "
                     f"(or run play.py with --in-process).")
        self.name = f"{health['model']} @ {self.url} ({health['device']})"

    def answer(self, state, questions: list[dict], options: dict) -> dict:
        r = self.session.post(f"{self.url}/decide", json={"state": state, "questions": questions, "options": options},
                              timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Kodiak server error {r.status_code}: {r.text[:500]}")
        return r.json()


class KodiakLocal:
    """Loads the model inside this process (Option A)."""

    def __init__(self, model: str, device: str, threads: int):
        import torch

        from kodiak_s1.hub import Kodiak

        if threads > 0:
            torch.set_num_threads(threads)
        self.kodiak = Kodiak.from_pretrained(model, device=device)
        self.name = f"{model} (in-process, {next(self.kodiak.model.parameters()).device})"

    def answer(self, state, questions: list[dict], options: dict) -> dict:
        return self.kodiak.answer([{"state": state, "questions": questions, "options": options}])[0]


def token_counter():
    """Kodiak's own tokenizer when available (it is installed with kodiak-s1), else ~4 characters per token."""
    try:
        from kodiak_s1.tokenizer import count_tokens

        count_tokens("warm up")
        return count_tokens
    except Exception:
        return lambda text: math.ceil(len(text) / 4)


# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text.replace("\r", ""))
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def peek(env: FrotzEnv, command: str) -> str:
    """Run a command and roll the game back, so looking costs no move."""
    saved = env.get_state()
    obs = env.step(command)[0]
    env.set_state(saved)
    return clean(obs)


def location(env: FrotzEnv) -> tuple[int, str]:
    loc = env.get_player_location()
    return (loc.num, loc.name) if loc is not None else (-1, "(unknown)")


def normalize(command: str) -> str:
    c = " ".join(command.lower().split())
    return DIRECTIONS.get(c, c)


def candidates_from(valid: list[str]) -> list[str]:
    """Deduplicate equivalent commands (case, spacing, direction abbreviations), keeping first-seen order."""
    seen, out = set(), []
    for v in valid:
        c = normalize(v)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def anti_loop(cands: list[str], tried_here: Counter, tried_in_situation: Counter, max_repeats: int) -> list[str]:
    """Skip repeats while alternatives exist. `tried_here` counts (room, command); `tried_in_situation` counts
    (exact world state, command), which catches undo loops such as take/drop and walking back and forth.

    Takes the first tier that leaves at least two commands: (1) never done from this exact state and tried fewer than
    `max_repeats` times in this room; (2) never done from this exact state; (3) the two least-tried, so the filter
    never switches itself off in a room visited often.
    """
    if len(cands) < 2:
        return cands
    for tier in ([c for c in cands if tried_in_situation[c] == 0 and tried_here[c] < max_repeats],
                 [c for c in cands if tried_in_situation[c] == 0]):
        if len(tier) >= 2:
            return tier
    key = {c: (tried_in_situation[c], tried_here[c]) for c in cands}
    cutoff = sorted(key.values())[1]
    return [c for c in cands if key[c] <= cutoff]


EXITS = set(DIRECTIONS.values()) | {"in", "out", "enter", "exit"}
TAKE_VERBS = ("take", "get", "pick up")
DROP_VERBS = ("drop", "put down")
INVERSE_VERBS = [(t, d) for t in TAKE_VERBS for d in DROP_VERBS] + [
    ("open", "close"), ("lock", "unlock"), ("turn on", "turn off"), ("switch on", "switch off"), ("wear", "remove"),
    ("wear", "take off"), ("put on", "take off"), ("light", "extinguish"), ("light", "douse"), ("light", "turn off")]


def undo_of(prev: str) -> set[str]:
    """Commands that would just reverse `prev` (open/close, take/drop, put in/take from...), judged from its text."""
    m = re.fullmatch(r"(?:put|insert|place) (.+?) (?:in|into|inside|on|onto) (.+)", prev)
    if m:
        x, y = m.groups()
        return {f"{v} {x}" for v in TAKE_VERBS} | {f"take {x} from {y}", f"get {x} from {y}", f"take {x} out of {y}"}
    m = re.fullmatch(r"(?:take|get|remove) (.+?) (?:from|out of|off) (.+)", prev)
    if m:
        x, y = m.groups()
        return {f"put {x} in {y}", f"put {x} into {y}", f"put {x} on {y}", f"insert {x} in {y}", f"drop {x}"}
    out = set()
    for a, b in INVERSE_VERBS:
        for x, y in ((a, b), (b, a)):
            if prev.startswith(x + " "):
                out.add(f"{y} {prev[len(x) + 1:]}")
    m = re.fullmatch(r"put (.+) down", prev)
    if m:
        out |= {f"{v} {m.group(1)}" for v in TAKE_VERBS}
    for v in TAKE_VERBS:
        if prev.startswith(v + " "):
            out.add(f"put {prev[len(v) + 1:]} down")
    return out


def chunks(items: list[str], size: int) -> list[list[str]]:
    """Split into the fewest chunks of at most `size`, balanced so no chunk ends up with a single label."""
    n = math.ceil(len(items) / size)
    k, r = divmod(len(items), n)
    out, i = [], 0
    for j in range(n):
        step = k + (1 if j < r else 0)
        out.append(items[i:i + step])
        i += step
    return out


def build_state(look: str, inventory: str, history: deque, tried_here: list[str], count_tokens,
                progress: str = "", new_exits: list[str] | None = None, budget: int = STATE_TOKENS) -> tuple[str, int]:
    """Room + recent turns + inventory + progress + what's untried here, trimmed to the token budget (oldest turns go
    first)."""
    turns = list(history)
    look_chars, obs_chars = 700, 300
    while True:
        parts = [f"Where you are:\n{look[:look_chars]}"]
        if turns:
            parts.append("Recent turns:\n" + "\n".join(f"> {c}\n{o[:obs_chars]}" for c, o in turns))
        parts.append(f"Inventory: {inventory}")
        if progress:
            parts.append(progress)
        if new_exits:
            parts.append("Exits not tried from here: " + ", ".join(new_exits))
        if tried_here:
            parts.append("Already tried here: " + ", ".join(tried_here[-8:]))
        text = "\n\n".join(parts)
        n = count_tokens(text)
        if n <= budget:
            return text, n
        if turns:
            turns = turns[1:]
        elif obs_chars > 100:
            obs_chars -= 50
        elif look_chars > 150:
            look_chars -= 100
        else:
            return text, n  # still over; Kodiak itself truncates at 2048 tokens


def inventory_text(env: FrotzEnv) -> str:
    inv = peek(env, "inventory")
    # "You are carrying:\n  a lamp\n  a sword" -> "a lamp, a sword"; "You are empty-handed." stays as it is.
    lines = [line.strip() for line in inv.split("\n") if line.strip()]
    if len(lines) > 1 and lines[0].endswith(":"):
        return ", ".join(lines[1:])
    return " ".join(lines) or "nothing"


# ---------------------------------------------------------------------------
# Deciders
# ---------------------------------------------------------------------------


def ask_kodiak(kodiak, state: str, cands: list[str], options: dict, danger: bool,
               question: str = QUESTIONS["explore"]) -> dict:
    """One decision over any number of candidates: a single question if they fit, else a tournament.

    Round 1 puts every chunk of <= 32 candidates in one request (one question per chunk, one forward pass) and takes
    each chunk's most probable label; round 2 asks once more over the winners. The final answer's confidence is relative
    to the finalists, so a tournament win is not directly comparable to a single-round confidence.
    """
    rounds, model_ms, t0 = [], 0.0, time.perf_counter()
    groups = chunks(cands, MAX_LABELS)
    danger_q = [{"type": "choice", "id": "danger", "text": DANGER_Q, "labels": ["yes", "no"]}] if danger else []
    if len(groups) == 1:
        resp = kodiak.answer(state, [{"type": "choice", "id": "action", "text": question, "labels": cands}] + danger_q,
                             options)
        rounds.append(resp["answers"])
        model_ms += resp["latency_ms"]
        final = resp["answers"]["action"]
    else:
        qs = [{"type": "choice", "id": f"action_{i}", "text": question, "labels": g} for i, g in enumerate(groups)]
        resp = kodiak.answer(state, qs + danger_q, options)
        rounds.append(resp["answers"])
        model_ms += resp["latency_ms"]
        winners = [max(a["probs"], key=a["probs"].get) for q, a in resp["answers"].items() if q.startswith("action_")]
        resp2 = kodiak.answer(state, [{"type": "choice", "id": "action", "text": question, "labels": winners}], options)
        rounds.append(resp2["answers"])
        model_ms += resp2["latency_ms"]
        final = resp2["answers"]["action"]
    return {"final": final, "danger": rounds[0].get("danger"), "rounds": rounds, "model_ms": model_ms,
            "roundtrip_ms": (time.perf_counter() - t0) * 1000}


USEFUL = re.compile(r"^(take|get|pick up|open|read|examine|unlock|turn on|switch on|light|wear|enter|climb|search|"
                    r"move|pull|look (in|under|behind))\b")
ODD = re.compile(r"^(throw|eat|drink|kiss|taste|smell|jump|sing|shout)\b|^push .+ to | at ")


def explore(cands: list[str], tried_here: Counter, rng: random.Random) -> tuple[str, str]:
    """Untried commands in this room, in this order: useful-looking interactions (take, open, read...), exits,
    anything else, odd ones (throw X at Y...); then a random command. Uses only what a player could see."""
    untried = [c for c in cands if tried_here[c] == 0]
    for how, tier in (("untried: useful", [c for c in untried if USEFUL.match(c) and not ODD.search(c)]),
                      ("untried: exit", [c for c in untried if c in EXITS]),
                      ("untried", [c for c in untried if not ODD.search(c)]),
                      ("untried: odd", untried)):
        if tier:
            return rng.choice(tier), how
    return rng.choice(cands), "random"


def ask_llm(model: str, host: str, state: str, cands: list[str], timeout: float) -> tuple[str | None, float, str]:
    """Ask a local Ollama model to pick one of the valid commands (constrained to them by a JSON schema enum)."""
    t0 = time.perf_counter()
    prompt = (f"You are playing a text adventure. Game state:\n\n{state}\n\n"
              "Choose the single best next command to make progress (explore, collect useful items, solve puzzles, "
              "avoid repeating what did not work). You must pick exactly one command from this list:\n"
              + "\n".join(f"- {c}" for c in cands)
              + '\n\nAnswer with JSON: {"command": "<one command from the list>"}')
    body = {"model": model, "stream": False, "think": False, "options": {"temperature": 0.2},
            "format": {"type": "object", "properties": {"command": {"type": "string", "enum": cands}},
                       "required": ["command"]},
            "messages": [{"role": "user", "content": prompt}]}
    try:
        r = requests.post(f"{host.rstrip('/')}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        cmd = normalize(json.loads(r.json()["message"]["content"]).get("command", ""))
        ms = (time.perf_counter() - t0) * 1000
        return (cmd, ms, "") if cmd in cands else (None, ms, f"LLM answered off-list: {cmd!r}")
    except Exception as e:  # noqa: BLE001 - any LLM failure falls back to exploration
        return None, (time.perf_counter() - t0) * 1000, f"LLM error: {e}"


# ---------------------------------------------------------------------------
# Terminal view
# ---------------------------------------------------------------------------


class View:
    def __init__(self, color: bool, quiet: bool, text_lines: int):
        self.color, self.quiet, self.text_lines = color, quiet, text_lines

    def c(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def turn(self, rec: dict) -> None:
        if self.quiet:
            return
        who = rec["chooser"]
        tag = {"kodiak": self.c("KODIAK", "1;32"), "explore": self.c("EXPLORE", "1;33"),
               "llm": self.c("LLM", "1;35"), "forced": self.c("FORCED", "1;36")}.get(who, who.upper())
        k = rec.get("kodiak")
        kinfo = ""
        if k:
            f = k["final"]
            conf = f"conf {f['confidence']:.2f}" if f["answer"] is not None else f"abstained ({f['abstain_reason']})"
            kinfo = f"  kodiak: {conf}, pick {f['answer'] or '-'!s}, {k['model_ms']:.0f} ms model / {k['roundtrip_ms']:.0f} ms total"
            if k["rounds"] and len(k["rounds"]) > 1:
                kinfo += f", tournament of {rec['n_candidates']}"
            d = k.get("danger")
            if d:
                kinfo += f", danger: {d['answer'] or 'abstain'}"
        print(self.c(f"── move {rec['turn']:>3} ── {rec['location']} ── score {rec['score_before']} ──", "2"))
        print(f"{self.c('>', '1')} {self.c(rec['command'], '1')}   [{tag}] {self.c(rec['reason'], '2')}")
        if kinfo:
            print(self.c(kinfo, "2"))
        lines = rec["observation"].split("\n")
        shown = lines[: self.text_lines]
        print("\n".join("  " + line for line in shown) + ("\n  ..." if len(lines) > len(shown) else ""))
        if rec["reward"]:
            print(self.c(f"  +{rec['reward']} points", "1;32"))
        if rec["done"]:
            print(self.c("  *** game over ***", "1;31"))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, help="Z-machine game file supported by Jericho (e.g. games/zork1.z5)")
    ap.add_argument("--max-moves", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="take Kodiak's move when its confidence is >= this (default 0.5); otherwise System 2")
    ap.add_argument("--null-threshold", type=float, default=None,
                    help="Kodiak abstains when p_null >= this (default: the model's tuned value, 0.75)")
    ap.add_argument("--seed", type=int, default=0, help="game and exploration seed (default 0)")
    ap.add_argument("--max-repeats", type=int, default=2,
                    help="skip a command already tried this many times in the same room, when alternatives exist")
    ap.add_argument("--kodiak-url", default=os.environ.get("KODIAK_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--in-process", action="store_true", help="load Kodiak in this process instead of calling server.py")
    ap.add_argument("--model", default="cortex-agent-llc/kodiak-small-r1-preview", help="for --in-process")
    ap.add_argument("--device", default="cpu", help="for --in-process: cpu, cuda or auto (default cpu)")
    ap.add_argument("--threads", type=int, default=8, help="for --in-process: torch CPU threads")
    ap.add_argument("--baseline", action="store_true",
                    help="exploration only: never ask Kodiak (the no-model baseline for benchmarks)")
    ap.add_argument("--question", choices=sorted(QUESTIONS), default="explore",
                    help="wording of the action question: " + "; ".join(f"{k} = {v!r}" for k, v in QUESTIONS.items()))
    ap.add_argument("--no-danger", action="store_true", help="don't ask the second question ('Is the player in danger?')")
    ap.add_argument("--llm-model", default=None, help="Ollama model for System 2 (default: exploration only)")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                    help="Ollama endpoint (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
    ap.add_argument("--llm-timeout", type=float, default=120.0)
    ap.add_argument("--stop-on-death", action="store_true", help="stop at game over instead of restarting")
    ap.add_argument("--out", default="runs", help="folder for the transcript and summary (default runs/)")
    ap.add_argument("--transcript", default=None, help="transcript path (default: <out>/<game>-<time>.jsonl)")
    ap.add_argument("--text-lines", type=int, default=8, help="game text lines shown per move")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    ap.add_argument("--no-color", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    warnings.filterwarnings("ignore", module="jericho")
    view = View(color=sys.stdout.isatty() and not a.no_color, quiet=a.quiet, text_lines=a.text_lines)
    rng = random.Random(a.seed)
    count_tokens = token_counter()

    if a.baseline:
        kodiak = None
    else:
        kodiak = KodiakLocal(a.model, a.device, a.threads) if a.in_process else KodiakHTTP(a.kodiak_url)
    model_name = "none (baseline: exploration only)" if kodiak is None else kodiak.name
    options = {} if a.null_threshold is None else {"null_threshold": a.null_threshold}

    env = FrotzEnv(a.game, seed=a.seed)
    if not env.is_fully_supported or not env.bindings:
        sys.exit(f"{a.game} is not a Jericho-supported build (unknown md5), so there is no valid-action list. "
                 "See the README's 'Game files' section.")
    obs, info = env.reset()
    game = Path(a.game).stem
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    transcript = Path(a.transcript or Path(a.out) / f"{game}-{stamp}.jsonl")
    transcript.parent.mkdir(parents=True, exist_ok=True)

    if not a.quiet:
        print(view.c(f"Kodiak plays {game}  |  {model_name}  |  threshold {a.threshold}"
                     f"  |  System 2: {'LLM ' + a.llm_model if a.llm_model else 'exploration'}", "1"))
        print(clean(obs))

    history: deque = deque(maxlen=3)
    tried: dict[int, Counter] = {}  # room -> command counts
    tried_state: dict[str, Counter] = {}  # exact world state -> command counts
    rooms: set[str] = set()
    stats = Counter()
    k_model_ms, k_rt_ms, llm_ms = [], [], []
    start_score, best_score, score = info["score"], info["score"], info["score"]
    episodes = 1
    max_score = env.get_max_score()
    prev_cmd, last_points = None, None  # last_points: (points, command, turn)

    with transcript.open("w") as out:
        for turn in range(1, a.max_moves + 1):
            loc_id, loc_name = location(env)
            rooms.add(loc_name)
            here = tried.setdefault(loc_id, Counter())
            situation = tried_state.setdefault(env.get_world_state_hash(), Counter())
            look = peek(env, "look")
            inv = inventory_text(env)
            valid = env.get_valid_actions()
            all_cands = candidates_from(valid)
            cands = anti_loop(all_cands, here, situation, a.max_repeats)
            undo = undo_of(prev_cmd) if prev_cmd else set()
            if len([c for c in cands if c not in undo]) >= 2:  # don't immediately reverse the last move
                cands = [c for c in cands if c not in undo]
            tried_here = [c for c, _ in here.most_common()]
            progress = f"Score: {score} of {max_score}. " + (
                f"Last points: +{last_points[0]} for '{last_points[1]}', {turn - last_points[2]} moves ago."
                if last_points else "No points scored yet.")
            new_exits = [c for c in cands if c in EXITS and here[c] == 0]
            state, n_tokens = build_state(look, inv, history, tried_here, count_tokens, progress, new_exits)

            rec: dict = {"turn": turn, "episode": episodes, "location": loc_name, "location_id": loc_id,
                         "score_before": score, "state": state, "state_tokens": n_tokens,
                         "valid_actions": valid, "candidates": cands, "n_candidates": len(cands),
                         "skipped": [c for c in all_cands if c not in cands]}

            if not cands:  # no valid action found at all (rare): "look" is always accepted by the parser
                cmd, chooser, reason = "look", "forced", "no valid actions listed"
            elif len(cands) == 1:
                cmd, chooser, reason = cands[0], "forced", "only one valid action"
            elif kodiak is None:
                cmd, how = explore(cands, here, rng)
                chooser, reason = "explore", f"baseline; {how}"
            else:
                k = ask_kodiak(kodiak, state, cands, options, danger=not a.no_danger, question=QUESTIONS[a.question])
                rec["kodiak"] = k
                k_model_ms.append(k["model_ms"])
                k_rt_ms.append(k["roundtrip_ms"])
                f = k["final"]
                if f["answer"] is None:
                    stats["abstain"] += 1
                    stats[f"abstain_{f['abstain_reason']}"] += 1
                if f["answer"] is not None and f["confidence"] >= a.threshold:
                    cmd, chooser, reason = f["answer"], "kodiak", f"confidence {f['confidence']:.2f} >= {a.threshold}"
                else:
                    why = (f"kodiak abstained ({f['abstain_reason']})" if f["answer"] is None
                           else f"kodiak unsure ({f['confidence']:.2f} < {a.threshold})")
                    cmd = None
                    if a.llm_model:
                        cmd, ms, err = ask_llm(a.llm_model, a.ollama_url, state, cands, a.llm_timeout)
                        llm_ms.append(ms)
                        rec["llm"] = {"model": a.llm_model, "command": cmd, "latency_ms": ms, "error": err or None}
                        if cmd:
                            chooser, reason = "llm", f"{why}; LLM {ms:.0f} ms"
                        else:
                            stats["llm_errors"] += 1
                    if not cmd:
                        cmd, how = explore(cands, here, rng)
                        chooser, reason = "explore", f"{why}; {how}"

            if cands and cmd not in cands:  # by construction this never happens; counted to prove it
                stats["invalid"] += 1
            here[cmd] += 1
            situation[cmd] += 1
            stats[chooser] += 1
            obs, reward, done, info = env.step(cmd)
            obs = clean(obs)
            score = info["score"]
            best_score = max(best_score, score)
            history.append((cmd, obs + (f" [+{reward} points]" if reward > 0 else "")))
            if reward > 0:
                last_points = (reward, cmd, turn)
            prev_cmd = cmd
            rec.update({"command": cmd, "chooser": chooser, "reason": reason, "observation": obs, "reward": reward,
                        "score": score, "game_moves": info["moves"], "done": done})
            out.write(json.dumps(rec) + "\n")
            out.flush()
            view.turn(rec)

            if done:
                stats["game_overs"] += 1
                if a.stop_on_death:
                    break
                obs, info = env.reset()
                episodes += 1
                history.clear()
                prev_cmd = None
                score = info["score"]
                if not a.quiet:
                    print(view.c("  (restarting the game)", "2"))

    moves = sum(stats[k] for k in ("kodiak", "explore", "llm", "forced"))
    asked = len(k_model_ms)

    def pct(n: int) -> float:
        return round(100 * n / moves, 1) if moves else 0.0

    summary = {
        "game": game, "model": model_name, "threshold": a.threshold, "seed": a.seed,
        "system2": f"llm:{a.llm_model}" if a.llm_model else "exploration",
        "harness": HARNESS_VERSION, "question": a.question, "max_repeats": a.max_repeats,
        "moves": moves, "episodes": episodes, "game_overs": stats["game_overs"],
        "score": {"start": start_score, "final": score, "best": best_score, "max_possible": max_score},
        "rooms_visited": len(rooms), "rooms": sorted(rooms),
        "decided_by": {k: {"moves": stats[k], "pct": pct(stats[k])} for k in ("kodiak", "explore", "llm", "forced")},
        "kodiak_asked": asked,
        "kodiak_abstentions": {"total": stats["abstain"], "unanswerable": stats["abstain_unanswerable"],
                               "low_confidence": stats["abstain_low_confidence"]},
        "kodiak_below_threshold": asked - stats["kodiak"] - stats["abstain"],
        "kodiak_latency_ms": {"model_avg": round(sum(k_model_ms) / asked, 1) if asked else None,
                              "roundtrip_avg": round(sum(k_rt_ms) / asked, 1) if asked else None,
                              "roundtrip_p50": round(sorted(k_rt_ms)[asked // 2], 1) if asked else None},
        "llm_latency_ms_avg": round(sum(llm_ms) / len(llm_ms), 1) if llm_ms else None,
        "llm_latency_ms_p50": round(sorted(llm_ms)[len(llm_ms) // 2], 1) if llm_ms else None,
        "llm_errors": stats["llm_errors"],
        "invalid_commands": stats["invalid"],
        "transcript": str(transcript),
    }
    summary_path = transcript.with_name(transcript.stem + "-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary, view)
    print(f"transcript: {transcript}\nsummary:    {summary_path}")
    return summary


def print_summary(s: dict, view: View) -> None:
    d = s["decided_by"]
    lat = s["kodiak_latency_ms"]
    print(view.c("\n══════════ summary ══════════", "1"))
    print(f"game            {s['game']}  (seed {s['seed']}, threshold {s['threshold']}, System 2: {s['system2']})")
    print(f"moves           {s['moves']}  ({s['episodes']} episode(s), {s['game_overs']} game over(s))")
    sc = s["score"]
    print(f"score           {sc['final']} final, {sc['best']} best, started at {sc['start']} (max {sc['max_possible']})")
    print(f"rooms visited   {s['rooms_visited']}")
    print(f"decided by      Kodiak {d['kodiak']['moves']} ({d['kodiak']['pct']}%)  |  exploration {d['explore']['moves']} "
          f"({d['explore']['pct']}%)  |  LLM {d['llm']['moves']} ({d['llm']['pct']}%)  |  forced {d['forced']['moves']} "
          f"({d['forced']['pct']}%)")
    ab = s["kodiak_abstentions"]
    print(f"kodiak          asked {s['kodiak_asked']}x, abstained {ab['total']}x (unanswerable {ab['unanswerable']}), "
          f"below threshold {s['kodiak_below_threshold']}x")
    if lat["model_avg"] is not None:
        print(f"kodiak latency  {lat['model_avg']} ms model avg, {lat['roundtrip_avg']} ms round trip avg "
              f"(p50 {lat['roundtrip_p50']} ms)")
    if s["llm_latency_ms_avg"] is not None:
        print(f"llm latency     {s['llm_latency_ms_avg']} ms avg (p50 {s['llm_latency_ms_p50']} ms), "
              f"{s['llm_errors']} error(s)")
    print(f"invalid moves   {s['invalid_commands']}")


if __name__ == "__main__":
    main()
