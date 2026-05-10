"""
Evaluate the closed-loop executor across skills × env-ids × seeds.

Usage:
    python test/evaluate.py
    python test/evaluate.py --skills mpc ppo --seeds 5 --results-dir ./eval_results
    python test/evaluate.py --offline

Metrics tracked per (skill, env_id):
  - Success rate            — primary measure of task completion
  - Avg / min / max time    — wall-clock seconds per run
  - Avg replans used        — planning efficiency (fewer = better)
  - Skill failure breakdown — which sub-skill (reach/pick/place/push_disk) fails most
"""
import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "high_level_planner"))

from executor import run, _resolve_checkpoint  # noqa: E402

ENV_IDS = [
    "PushO-WallObstacles-v1",
    "PushO-Scattered",
    "PushO-TrappedDisk",
]
SKILLS = ["mpc", "ppo", "auto"]
N_SEEDS = 50


class _Tee:
    """Write to stdout and a log file simultaneously."""
    def __init__(self, log_path: Path):
        self._file = open(log_path, "w", buffering=1)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # needed so sys.stdout replacement doesn't break things
    def fileno(self):
        return self._stdout.fileno()


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _print_table(results: dict, times: dict, replans: dict, skill_fails: dict):
    """
    results[skill][env_id]    = [bool, ...]
    times[skill][env_id]      = [float seconds, ...]
    replans[skill][env_id]    = [int, ...]
    skill_fails[skill][env_id]= {skill_name: count}
    """
    skills = sorted(results)
    envs = ENV_IDS
    sep = "=" * 110

    print(f"\n{sep}")
    print(f"{'METRIC':<22}" + "".join(f"{e:<30}" for e in envs))
    print("-" * 110)

    for skill in skills:
        print(f"\n  skill = {skill}")
        for label, getter in [
            ("success rate",   lambda sk, ev: _success_cell(results, sk, ev)),
            ("avg time",       lambda sk, ev: _time_cell(times, sk, ev)),
            ("avg replans",    lambda sk, ev: _replans_cell(replans, sk, ev)),
            ("top fail skill", lambda sk, ev: _fail_cell(skill_fails, sk, ev)),
        ]:
            row = f"  {label:<20}" + "".join(f"{getter(skill, ev):<30}" for ev in envs)
            print(row)

    print(sep)


def _success_cell(results, skill, env_id):
    outcomes = results[skill].get(env_id, [])
    if not outcomes:
        return "—"
    s, n = sum(outcomes), len(outcomes)
    return f"{s}/{n} ({100*s/n:.0f}%)"


def _time_cell(times, skill, env_id):
    ts = times[skill].get(env_id, [])
    if not ts:
        return "—"
    avg = sum(ts) / len(ts)
    return f"avg {_fmt_time(avg)} [{_fmt_time(min(ts))}–{_fmt_time(max(ts))}]"


def _replans_cell(replans, skill, env_id):
    rs = replans[skill].get(env_id, [])
    if not rs:
        return "—"
    return f"avg {sum(rs)/len(rs):.1f} [{min(rs)}–{max(rs)}]"


def _fail_cell(skill_fails, skill, env_id):
    agg: dict[str, int] = defaultdict(int)
    for run_fails in skill_fails[skill].get(env_id, []):
        for k, v in run_fails.items():
            agg[k] += v
    if not agg:
        return "none"
    top = max(agg, key=agg.get)
    return f"{top} ({agg[top]}x)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills",        nargs="+", default=SKILLS, choices=SKILLS)
    ap.add_argument("--env-ids",       nargs="+", default=ENV_IDS, dest="env_ids", choices=ENV_IDS)
    ap.add_argument("--seeds",         type=int,  default=N_SEEDS,
                    help="Number of seeds (0 … seeds-1)")
    ap.add_argument("--max-replans",   type=int,  default=5, dest="max_replans")
    ap.add_argument("--offline",       action="store_true")
    ap.add_argument("--model",         default="gemini-2.5-flash")
    ap.add_argument("--reach-checkpoint",  default="Reach",      dest="reach_checkpoint")
    ap.add_argument("--pick-checkpoint",   default="PickSkill",  dest="pick_checkpoint")
    ap.add_argument("--place-checkpoint",  default="PlaceSkill", dest="place_checkpoint")
    ap.add_argument("--push-o-checkpoint", default="PushO",      dest="push_o_checkpoint")
    ap.add_argument("--reach-device",      default=None,         dest="reach_device")
    ap.add_argument("--results-dir",       default="./eval_results", dest="results_dir")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"eval_{timestamp}.txt"
    results_file = results_dir / "results.json"

    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Evaluation log: {log_path}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Config: skills={args.skills}  env_ids={args.env_ids}  seeds={args.seeds}  max_replans={args.max_replans}\n")

    # load prior results for resuming
    if results_file.exists():
        with open(results_file) as f:
            saved = json.load(f)
        print(f"[resume] loaded prior results from {results_file}")
    else:
        saved = {}

    def _load_nested(key):
        return {sk: {ev: saved.get(sk, {}).get(ev, {}).get(key, [])
                     for ev in ENV_IDS} for sk in SKILLS}

    results     = _load_nested("success")
    times_store = _load_nested("elapsed_s")
    replans_store = _load_nested("n_replans")
    fails_store = _load_nested("skill_fail_counts")

    reach_ckpt  = _resolve_checkpoint(args.reach_checkpoint)
    pick_ckpt   = _resolve_checkpoint(args.pick_checkpoint)
    place_ckpt  = _resolve_checkpoint(args.place_checkpoint)
    push_o_ckpt = _resolve_checkpoint(args.push_o_checkpoint)

    total = len(args.skills) * len(args.env_ids) * args.seeds
    done = 0

    for skill in args.skills:
        for env_id in args.env_ids:
            completed = len(results[skill][env_id])
            for seed in range(args.seeds):
                done += 1
                if seed < completed:
                    print(f"[{done}/{total}] skip  skill={skill}  env={env_id}  seed={seed} (cached)")
                    continue

                print(f"\n[{done}/{total}] skill={skill}  env={env_id}  seed={seed}")
                t0 = time.perf_counter()
                try:
                    ret = run(
                        seed=seed,
                        max_replans=args.max_replans,
                        offline=args.offline,
                        model=args.model,
                        skill=skill,
                        env_id=env_id,
                        checkpoint=reach_ckpt  if skill in ("ppo", "auto") else None,
                        pick_checkpoint=pick_ckpt   if skill in ("ppo", "auto") else None,
                        place_checkpoint=place_ckpt  if skill in ("ppo", "auto") else None,
                        push_o_checkpoint=push_o_ckpt if skill in ("ppo", "auto") else None,
                        reach_device=args.reach_device,
                    )
                    success       = ret["success"]
                    n_replans     = ret["n_replans"]
                    skill_fail_c  = ret["skill_fail_counts"]
                except Exception:
                    traceback.print_exc()
                    success, n_replans, skill_fail_c = False, args.max_replans, {}

                elapsed = time.perf_counter() - t0
                results[skill][env_id].append(bool(success))
                times_store[skill][env_id].append(round(elapsed, 2))
                replans_store[skill][env_id].append(n_replans)
                fails_store[skill][env_id].append(skill_fail_c)

                print(f"  → {'SUCCESS' if success else 'FAIL'}  |  time={_fmt_time(elapsed)}  |  replans={n_replans}  |  skill_fails={skill_fail_c}")

                # merge back into saved dict and persist
                for sk in SKILLS:
                    saved.setdefault(sk, {})
                    for ev in ENV_IDS:
                        saved[sk].setdefault(ev, {})
                        saved[sk][ev]["success"]          = results[sk][ev]
                        saved[sk][ev]["elapsed_s"]        = times_store[sk][ev]
                        saved[sk][ev]["n_replans"]        = replans_store[sk][ev]
                        saved[sk][ev]["skill_fail_counts"] = fails_store[sk][ev]
                with open(results_file, "w") as f:
                    json.dump(saved, f, indent=2)

    print(f"\n\nFinished: {datetime.now().isoformat()}")
    _print_table(results, times_store, replans_store, fails_store)
    print(f"\nFull results JSON : {results_file}")
    print(f"Full console log  : {log_path}")

    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
