#!/usr/bin/env python3
"""Recalculate the published E1 figures from measurement extracts in data/.

    python3 recalculate.py                               print the figures
    python3 recalculate.py --json results.json           also write them as JSON
    python3 recalculate.py --check expected/results.json compare with the published values

Standard library only, Python 3.9 or later. It recomputes the analysis from the
recorded measurements; it does not rerun the training.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
TEST_IMAGES = 3669          # the Oxford-IIIT Pet test split, evaluated in full
TOLERANCE = 1e-9            # for --check


class RecordError(Exception):
    """A record is missing, incomplete or not what the calculation expects."""


def load(rel):
    path = DATA / rel
    if not path.is_file():
        raise RecordError(f"missing record: data/{rel}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def round_ends(rel):
    """round number -> seconds of its round_end event; every configured round must be there."""
    events = load(rel)
    start = [e for e in events if e["event"] == "run_start"]
    if len(start) != 1:
        raise RecordError(f"data/{rel}: expected one run_start, found {len(start)}")
    configured = start[0]["config"]["num_rounds"]
    measured = [e for e in events if e["event"] == "round_end"]
    ends = {e["round"]: e["seconds"] for e in measured}
    if len(ends) != len(measured):
        raise RecordError(f"data/{rel}: duplicate round_end")
    if sorted(ends) != list(range(1, configured + 1)):
        raise RecordError(f"data/{rel}: round_end for rounds {sorted(ends)}, configured 1..{configured}")
    return ends, events


def round_time(rel, one_measured_round=False):
    """Seconds per round. Round 1 is a warm-up. Multi-GPU records: the median of
    rounds 2-4. The same-site one-GPU baseline has a single measured round (round 2)."""
    ends, _ = round_ends(rel)
    if one_measured_round:
        if sorted(ends) != [1, 2]:
            raise RecordError(f"data/{rel}: expected a warm-up and one measured round")
        return ends[2]
    if sorted(ends) != [1, 2, 3, 4]:
        raise RecordError(f"data/{rel}: expected a warm-up and three measured rounds")
    return statistics.median([ends[2], ends[3], ends[4]])


def series(baseline, points):
    out = []
    for gpus, seconds in points:
        speedup = baseline / seconds
        out.append({"gpus": gpus, "round_seconds": seconds, "speedup": speedup, "efficiency": speedup / gpus})
    return out


def final_accuracy(rel):
    """Top-1 accuracy (percent) of the last complete evaluation, on the full test split."""
    _, events = round_ends(rel)
    done = [e for e in events if e["event"] == "evaluation" and e.get("status") == "complete"]
    if not done:
        raise RecordError(f"data/{rel}: no complete final evaluation")
    last = done[-1]
    if last.get("samples") != TEST_IMAGES or last["metrics"].get("samples") != TEST_IMAGES:
        raise RecordError(f"data/{rel}: final evaluation covers {last.get('samples')} images, not {TEST_IMAGES}")
    return last["metrics"]["accuracy"] * 100


def calculate():
    same = "same-datacenter/"
    ger = "coordinator-germany/"
    t1_same = round_time(same + "a40x01.jsonl", one_measured_round=True)
    t1_night1 = round_time(ger + "night1-a40x01.jsonl")
    same_dc = series(t1_same, [(1, t1_same)] + [(n, round_time(f"{same}a40x{n:02d}.jsonl")) for n in (8, 16)])
    night1 = series(t1_night1, [(n, round_time(f"{ger}night1-a40x{n:02d}.jsonl")) for n in (1, 2, 4, 8, 16)])
    night2 = series(t1_night1, [(n, round_time(f"{ger}night2-a40x{n:02d}.jsonl")) for n in (8, 16)])
    cross = series(t1_night1, [(16, same_dc[-1]["round_seconds"])])[0]

    a = final_accuracy("mixed-vendor/control-a.jsonl")
    b = final_accuracy("mixed-vendor/control-b.jsonl")
    mixed = final_accuracy("mixed-vendor/mixed.jsonl")
    mean = (a + b) / 2
    gap = abs(mixed - mean)
    tolerance = max(0.5, abs(a - b))

    return {
        "scaling": {
            "same_datacenter": {"baseline_seconds": t1_same, "points": same_dc},
            "coordinator_germany": {"baseline_seconds": t1_night1, "night1": night1, "night2": night2},
            "same_datacenter_16_vs_night1_baseline": cross,
        },
        "mixed_vendor_accuracy": {
            "control_a_percent": a,
            "control_b_percent": b,
            "mixed_percent": mixed,
            "controls_mean_percent": mean,
            "gap_pp": gap,
            "tolerance_pp": tolerance,
            "within_tolerance": gap <= tolerance,
            "controls_at_least_90_percent": a >= 90 and b >= 90,
        },
    }


def report(r):
    s = r["scaling"]
    lines = ["E1 recalculation: ViT-B/16 on Oxford-IIIT Pet, preliminary", ""]

    def table(title, points, note=""):
        lines.append(title)
        lines.append("  GPUs   round, s   speed-up   efficiency")
        for p in points:
            lines.append(f"  {p['gpus']:>4}  {p['round_seconds']:>9.3f}   {p['speedup']:>7.2f}x   {p['efficiency'] * 100:>8.1f}%")
        if note:
            lines.append("  " + note)
        lines.append("")

    table("Scaling, coordinator in the GPUs' datacenter (16-17 Oct 2026)", s["same_datacenter"]["points"],
          "The 1-GPU time is one measured round; the others are the median of rounds 2-4.")
    table("Scaling, coordinator in Germany, night 1 (13-14 Oct 2026)", s["coordinator_germany"]["night1"])
    table("Scaling, coordinator in Germany, night 2 (16-17 Oct 2026), against night 1's 1-GPU time",
          s["coordinator_germany"]["night2"])
    c = s["same_datacenter_16_vs_night1_baseline"]
    lines.append(f"16 GPUs in the GPUs' datacenter against night 1's 1-GPU time: {c['speedup']:.2f}x, "
                 f"{c['efficiency'] * 100:.1f}%")
    lines.append("")
    m = r["mixed_vendor_accuracy"]
    lines.append(f"Mixed-vendor accuracy, final top-1 on {TEST_IMAGES} test images (14 Oct 2026)")
    lines.append(f"  NVIDIA-only control A   {m['control_a_percent']:.4f}%")
    lines.append(f"  NVIDIA-only control B   {m['control_b_percent']:.4f}%")
    lines.append(f"  A40 + RTX 4090 + MI300X {m['mixed_percent']:.4f}%")
    lines.append(f"  Gap from the controls' mean {m['gap_pp']:.4f} percentage points; "
                 f"tolerance {m['tolerance_pp']:.4f} pp; within: {m['within_tolerance']}; "
                 f"both controls at least 90%: {m['controls_at_least_90_percent']}")
    return "\n".join(lines)


def compare(got, want, path="", diffs=None):
    diffs = [] if diffs is None else diffs
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            diffs.append(f"{path or '/'}: keys differ")
        else:
            for k in want:
                compare(got[k], want[k], f"{path}/{k}", diffs)
    elif isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            diffs.append(f"{path}: length differs")
        else:
            for i, (g, w) in enumerate(zip(got, want)):
                compare(g, w, f"{path}[{i}]", diffs)
    elif isinstance(want, bool) or not isinstance(want, (int, float)):
        if got != want:
            diffs.append(f"{path}: {got!r} != {want!r}")
    elif abs(got - want) > TOLERANCE * max(1.0, abs(want)):
        diffs.append(f"{path}: {got!r} != {want!r}")
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", metavar="FILE", help="write the results as JSON")
    ap.add_argument("--check", metavar="FILE", help="compare with expected results; exit 1 on any difference")
    args = ap.parse_args(argv)
    try:
        results = calculate()
    except RecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(report(results))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    if args.check:
        diffs = compare(results, json.loads(Path(args.check).read_text()))
        if diffs:
            print("\nCHECK FAILED:\n  " + "\n  ".join(diffs), file=sys.stderr)
            return 1
        print(f"\nCheck passed: every value matches {args.check}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
