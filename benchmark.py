"""Compare models (and the no-model baseline) on the same games, seeds and harness.

    python benchmark.py --baseline --model cortex-agent-llc/kodiak-small-r1-preview \\
        --games games/advent.z5 games/balances.z5 games/detective.z5 games/library.z5 --seeds 0 1 2 3 4

    python benchmark.py --model cortex-agent-llc/kodiak-small-r1-preview --model /path/to/new-model \\
        --games games/zork1.z3 --seeds 0 1 2 3 4

Each model gets its own server.py (CPU by default) for the duration of its runs; play.py runs once per game and seed.
Transcripts, summaries and the results table go to runs/bench-<time>/. Runs are sequential so latency isn't distorted.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


def start_server(model: str, port: int, device: str, threads: int, log: Path) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, str(HERE / "server.py"), "--model", model, "--port", str(port),
                             "--device", device, "--threads", str(threads)],
                            stdout=log.open("w"), stderr=subprocess.STDOUT)
    for _ in range(300):
        if proc.poll() is not None:
            sys.exit(f"server for {model} exited; see {log}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).ok:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.terminate()
    sys.exit(f"server for {model} didn't come up in 300 s; see {log}")


def play(game: str, seed: int, out: Path, a: argparse.Namespace, url: str | None) -> dict:
    transcript = out / f"{Path(game).stem}-s{seed}.jsonl"
    cmd = [sys.executable, str(HERE / "play.py"), "--game", game, "--seed", str(seed), "--max-moves", str(a.max_moves),
           "--threshold", str(a.threshold), "--quiet", "--no-color", "--transcript", str(transcript)]
    cmd += ["--kodiak-url", url] if url else ["--baseline"]
    cmd += a.play_args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"play.py failed on {game} seed {seed}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(transcript.with_name(transcript.stem + "-summary.json").read_text())


def fmt(xs: list[float], digits: int = 1) -> str:
    if not xs:
        return "–"
    m = statistics.mean(xs)
    return f"{m:.{digits}f} ± {statistics.stdev(xs):.{digits}f}" if len(xs) > 1 else f"{m:.{digits}f}"


def table(results: dict[str, dict[str, list[dict]]]) -> str:
    rows = ["| Config | Game | Score gain | Best score | Rooms | Kodiak share | Kodiak ms (avg) | Game overs | Invalid |",
            "|---|---|---|---|---|---|---|---|---|"]
    for config, games in results.items():
        for game, runs in games.items():
            gain = [r["score"]["best"] - r["score"]["start"] for r in runs]
            best = [r["score"]["best"] for r in runs]
            rooms = [r["rooms_visited"] for r in runs]
            share = [r["decided_by"]["kodiak"]["pct"] for r in runs]
            lat = [r["kodiak_latency_ms"]["model_avg"] for r in runs if r["kodiak_latency_ms"]["model_avg"] is not None]
            overs = [r["game_overs"] for r in runs]
            invalid = sum(r["invalid_commands"] for r in runs)
            rows.append(f"| {config} | {game} (max {runs[0]['score']['max_possible']}) | {fmt(gain)} | {fmt(best)} | "
                        f"{fmt(rooms)} | {fmt(share) + '%' if lat else '–'} | {fmt(lat, 0) if lat else '–'} | "
                        f"{fmt(overs)} | {invalid} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--model", action="append", default=[], help="Hugging Face repo id or local folder; repeatable")
    ap.add_argument("--kodiak-url", default=None, help="use an already-running server instead of starting one per model")
    ap.add_argument("--baseline", action="store_true", help="also run the exploration-only baseline")
    ap.add_argument("--max-moves", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out", default=None, help="default runs/bench-<time>")
    ap.add_argument("--play-args", nargs=argparse.REMAINDER, default=[],
                    help="anything after this is passed to every play.py run (e.g. --play-args --no-danger)")
    a = ap.parse_args()
    if not (a.model or a.kodiak_url or a.baseline):
        ap.error("give at least one --model, --kodiak-url or --baseline")

    out = Path(a.out or HERE / "runs" / f"bench-{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    configs: list[tuple[str, str | None, str | None]] = []  # (label, model to serve, url)
    if a.baseline:
        configs.append(("baseline (no model)", None, None))
    if a.kodiak_url:
        configs.append((f"server {a.kodiak_url}", None, a.kodiak_url))
    configs += [(Path(m).name, m, None) for m in a.model]

    results: dict[str, dict[str, list[dict]]] = {}
    for label, model, url in configs:
        folder = out / label.split(" ")[0]
        folder.mkdir(exist_ok=True)
        server = None
        if model:
            print(f"[{label}] starting server on {a.device}...", flush=True)
            server = start_server(model, a.port, a.device, a.threads, folder / "server.log")
            url = f"http://127.0.0.1:{a.port}"
        try:
            for game in a.games:
                for seed in a.seeds:
                    t0 = time.perf_counter()
                    s = play(game, seed, folder, a, url)
                    results.setdefault(label, {}).setdefault(Path(game).stem, []).append(s)
                    print(f"[{label}] {Path(game).stem} seed {seed}: score {s['score']['start']}→{s['score']['best']}, "
                          f"rooms {s['rooms_visited']}, kodiak {s['decided_by']['kodiak']['pct']}%, "
                          f"invalid {s['invalid_commands']} ({time.perf_counter() - t0:.0f} s)", flush=True)
        finally:
            if server:
                server.terminate()
                server.wait(timeout=30)

    md = table(results)
    header = (f"{len(a.seeds)} seeds × {a.max_moves} moves, threshold {a.threshold}, "
              f"extra play.py args: {' '.join(a.play_args) or 'none'}")
    (out / "results.md").write_text(f"{header}\n\n{md}\n")
    (out / "results.json").write_text(json.dumps({"args": vars(a), "results": results}, indent=1) + "\n")
    print(f"\n{header}\n\n{md}\n\nsaved to {out}/results.md")


if __name__ == "__main__":
    main()
