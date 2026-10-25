"""Render agent trajectories from a predictions / failed-cases file as plain text.

Keeps only what is needed to follow the agent: question, gold, prediction, and
per step Thought / Action / Observation, plus a one-line summary of each Collect
fan-out. Drops the bulky internals (prob_propose candidates, ranked lists,
gnn_top scores, ...) that make the raw file hard to read.

Entity ids in Observations and Actions are marked against the gold set:
`id*` is a gold answer, `id**` is the hard answer.

Writes `<input stem>_traces.txt` next to the input unless --out / --stdout.

Usage:
    python view_traces.py results/.../xxx_failed_cases.json
    python view_traces.py results/.../xxx_predictions.jsonl --index 93 --short_relations
    python view_traces.py results/.../xxx_failed_cases.json --limit 10 --out traces.txt
    python view_traces.py results/.../xxx_failed_cases.json --index 93 --stdout
"""

import argparse
import json
import re
from pathlib import Path

ENTITY_RE = re.compile(r"(?<![\w/.])(\d+)(?![\w/.])")
RELATION_RE = re.compile(r"/[\w./]+")


def load_cases(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return data if isinstance(data, list) else [data]


def as_set(values):
    return {str(v) for v in (values or [])}


def mark_entities(text, gold, hard):
    """Append * to gold-answer ids and ** to hard-answer ids."""
    def sub(match):
        entity = match.group(1)
        if entity in hard:
            return entity + "**"
        if entity in gold:
            return entity + "*"
        return entity
    return ENTITY_RE.sub(sub, text)


def shorten_relations(text):
    """Keep the last two segments of Freebase-style relation paths."""
    def sub(match):
        parts = [p for p in re.split(r"[/.]", match.group(0)) if p]
        return ".".join(parts[-2:]) if len(parts) > 2 else match.group(0)
    return RELATION_RE.sub(sub, text)


def summarize_collect(logs, short):
    lines = []
    for log in logs or []:
        entity = log.get("entity")
        if not log.get("fan_out"):
            lines.append(f"    {entity}: no fan-out ({log.get('reason', 'n/a')})")
            continue
        path = " -> ".join(f"{rel} ({d})" for rel, d in log.get("path", []))
        if short:
            path = shorten_relations(path)
        lines.append(
            f"    {entity}: root {log.get('root')} | {path} | "
            f"hops {log.get('hop_sizes')} | collected {log.get('n_collected')}"
        )
    return lines


def render_case(case, args):
    gold = as_set(case.get("answers"))
    hard = as_set(case.get("hard_answer"))
    prediction = as_set(case.get("prediction"))
    hits = sorted(prediction & (gold | hard))

    def clean(text):
        text = str(text or "").strip("\n")
        if args.short_relations:
            text = shorten_relations(text)
        return mark_entities(text, gold, hard) if not args.no_mark else text

    out = [
        "=" * 80,
        f"#{case.get('index', case.get('id', '?'))}  {case.get('question', '')}",
        f"hard answer: {sorted(hard)}   gold: {len(gold)} answers"
        + (f" {sorted(gold)}" if len(gold) <= args.max_gold else ""),
        f"prediction ({len(prediction)}): {sorted(prediction)[:args.max_gold]}"
        + ("" if len(prediction) <= args.max_gold else " ...")
        + f"   hits: {hits if hits else 'none'}",
    ]
    if case.get("error"):
        out.append(f"ERROR: {case['error']}")
    flags = [k for k in ("representative_only", "include_score", "type_check") if case.get(k)]
    if flags:
        out.append(f"flags: {', '.join(flags)}")

    for record in case.get("records", []):
        i = record.get("i")
        out.append("")
        if record.get("thought") and record.get("thought") != record.get("action"):
            out.append(f"Thought {i}: {clean(record['thought'])}")
        out.append(f"Action {i}: {clean(record.get('action'))}")
        observation = clean(record.get("observation"))
        if observation:
            out.append(f"Observation {i}:")
            out.extend("  " + line if line else "" for line in observation.splitlines())
        if "collect" in record and not args.no_collect:
            out.append("  [collect fan-out]")
            out.extend(summarize_collect(record["collect"], args.short_relations))

    if case.get("llm_output") and not args.no_llm_output:
        out.append("")
        out.append(f"Fallback LLM output: {clean(case['llm_output'])}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="predictions / failed-cases file (.json or .jsonl)")
    parser.add_argument("--index", type=int, nargs="*",
                        help="only these case indices (the `index` field)")
    parser.add_argument("--question", default=None,
                        help="only cases whose question contains this substring")
    parser.add_argument("--limit", type=int, default=None, help="render at most N cases")
    parser.add_argument("--only_errors", action="store_true", help="only cases with an error")
    parser.add_argument("--short_relations", action="store_true",
                        help="shorten /a/b/c.d/e relations to their last two segments")
    parser.add_argument("--no_mark", action="store_true", help="do not mark gold ids with */**")
    parser.add_argument("--no_collect", action="store_true", help="hide collect fan-out summaries")
    parser.add_argument("--no_llm_output", action="store_true", help="hide the fallback LLM output")
    parser.add_argument("--max_gold", type=int, default=10,
                        help="list gold/prediction ids only up to this many")
    parser.add_argument("--out", default=None,
                        help="output file (default: <input stem>_traces.txt next to the input)")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing a file")
    args = parser.parse_args()

    cases = load_cases(args.path)
    if args.index:
        wanted = set(args.index)
        cases = [c for c in cases if c.get("index") in wanted]
    if args.question:
        cases = [c for c in cases if args.question.lower() in str(c.get("question", "")).lower()]
    if args.only_errors:
        cases = [c for c in cases if c.get("error")]
    if args.limit is not None:
        cases = cases[:args.limit]

    text = "\n\n".join(render_case(case, args) for case in cases) + "\n"
    if args.stdout:
        print(text)
        return
    in_path = Path(args.path)
    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_traces.txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {len(cases)} case(s) to {out_path}")


if __name__ == "__main__":
    main()
