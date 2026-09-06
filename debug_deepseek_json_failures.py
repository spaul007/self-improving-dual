"""Independently debug DeepSeek's malformed tool-call JSON, using the
verbose artifacts already captured on disk (round_NNN/verbose/
editor_attempt_N_response.json, written by agent_editor.py when
verbose: true) -- no live API calls, no dependency on the run still being
alive.

For each captured _raw_arguments string (a tool call whose JSON failed to
json.loads live), this:
  1. Re-parses it to get the EXACT error type/position from Python's own
     json module.
  2. Classifies whether the failure looks like TRUNCATION (error very near
     the end of the string -- consistent with an output-token-budget cutoff)
     vs a genuine mid-string ESCAPING bug (error well before the end).
  3. Tries a cheap truncation-repair (append closing brackets/braces) to see
     how many are recoverable that way alone, without a real JSON-repair
     library.
  4. Prints a few representative snippets around each error so a human can
     eyeball the actual character(s) at fault.

Usage:
    python3 debug_deepseek_json.py <run_dir>
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path


def try_bracket_repair(raw: str) -> tuple[bool, str]:
    """Cheapest possible truncation repair: try appending 0-6 closing
    brackets/braces (in a few plausible combinations) and see if any
    combination parses. Returns (recovered, which_suffix)."""
    candidates = [
        "", '"', '"}', '"}]}', '"}]', "}", "]}", '"}]}', "}]}", "]}]}",
    ]
    for suffix in candidates:
        try:
            json.loads(raw + suffix)
            return True, suffix
        except (json.JSONDecodeError, ValueError):
            continue
    return False, ""


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 debug_deepseek_json.py <run_dir>")
        sys.exit(1)
    run_dir = sys.argv[1]

    files = sorted(glob.glob(f"{run_dir}/round_*/verbose/editor_attempt_*_response.json"))
    print(f"Scanning {len(files)} verbose response files under {run_dir}\n")

    results = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        for tc in d.get("tool_calls", []):
            args = tc.get("arguments")
            if not (isinstance(args, dict) and "_raw_arguments" in args):
                continue
            raw = args["_raw_arguments"]
            if not isinstance(raw, str) or not raw:
                continue
            try:
                json.loads(raw)
                continue  # parses fine now -- not a real failure instance
            except json.JSONDecodeError as e:
                length = len(raw)
                frac = e.pos / length if length else 0.0
                recovered, suffix = try_bracket_repair(raw)
                results.append({
                    "file": f,
                    "error_type": e.msg,
                    "pos": e.pos,
                    "length": length,
                    "frac_through": round(frac, 4),
                    "near_end": frac > 0.95,
                    "snippet": raw[max(0, e.pos - 80):e.pos + 80],
                    "bracket_recoverable": recovered,
                    "repair_suffix": suffix,
                })

    print(f"Total malformed-JSON instances found: {len(results)}\n")
    if not results:
        return

    print("=== Error type breakdown ===")
    for et, cnt in Counter(r["error_type"] for r in results).most_common():
        print(f"  {cnt:3d}  {et}")
    print()

    near_end = [r for r in results if r["near_end"]]
    mid_string = [r for r in results if not r["near_end"]]
    print(f"Error position within the string:")
    print(f"  near the end (>95% through, truncation-like):  {len(near_end)} / {len(results)}")
    print(f"  earlier (<=95% through, escaping-bug-like):     {len(mid_string)} / {len(results)}")
    print()

    recoverable = [r for r in results if r["bracket_recoverable"]]
    print(f"Recoverable by a cheap append-closing-brackets repair alone: "
          f"{len(recoverable)} / {len(results)} "
          f"({100 * len(recoverable) / len(results):.0f}%)")
    print()

    print("=== Position distribution (fraction through the string) ===")
    buckets = Counter()
    for r in results:
        bucket = int(r["frac_through"] * 10) * 10
        buckets[bucket] += 1
    for pct in sorted(buckets):
        print(f"  {pct:3d}-{pct+10:3d}%: {'#' * buckets[pct]} ({buckets[pct]})")
    print()

    print("=== Sample snippets around the failure point (first 8) ===")
    for r in results[:8]:
        print(f"--- {r['file']} ---")
        print(f"  error: {r['error_type']} @ pos {r['pos']}/{r['length']} "
              f"({r['frac_through'] * 100:.1f}% through) "
              f"bracket-recoverable={r['bracket_recoverable']}")
        print(f"  snippet: ...{r['snippet']!r}...")
        print()


if __name__ == "__main__":
    main()
