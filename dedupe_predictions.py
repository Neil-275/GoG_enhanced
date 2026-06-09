#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def dedupe_jsonl(input_path: Path, output_path: Path, keep: str = "first") -> int:
    seen = {}
    ordered = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            idx = obj.get("index")

            if idx is None:
                raise ValueError(f"Missing 'index' on line {line_num}")

            if keep == "first":
                if idx not in seen:
                    seen[idx] = obj
                    ordered.append(idx)
            elif keep == "last":
                if idx not in seen:
                    ordered.append(idx)
                seen[idx] = obj
            else:
                raise ValueError("keep must be 'first' or 'last'")

    with output_path.open("w", encoding="utf-8") as f:
        for idx in ordered:
            f.write(json.dumps(seen[idx], ensure_ascii=False) + "\n")

    return len(ordered)


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate JSONL objects that share the same index"
    )
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output JSONL file",
        default=None,
    )
    parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="first",
        help="Which duplicate to keep for each index",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_deduped.jsonl"
    )

    count = dedupe_jsonl(input_path, output_path, keep=args.keep)
    print(f"Wrote {count} unique records to {output_path}")


if __name__ == "__main__":
    main()