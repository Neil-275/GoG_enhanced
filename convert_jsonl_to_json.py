#!/usr/bin/env python
"""
Convert JSONL (JSON Lines) format to formatted JSON for better readability.
Each line in JSONL is a separate JSON object. This script combines them into a JSON array.
"""

import json
import argparse
from pathlib import Path


def convert_jsonl_to_json(input_file: str, output_file: str = None, indent: int = 2) -> str:
    """
    Convert JSONL file to formatted JSON.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to save the output JSON file. If None, uses input filename with .json extension
        indent: Number of spaces for indentation (default: 2)
    
    Returns:
        Path to the output file
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Determine output file path
    if output_file is None:
        output_path = input_path.with_suffix('.json')
    else:
        output_path = Path(output_file)
    
    # Read JSONL and parse each line
    data = []
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                continue
    
    # Write formatted JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)
    
    print(f"✓ Converted {len(data)} objects from JSONL to JSON")
    print(f"✓ Output file: {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSONL (JSON Lines) format to formatted JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_jsonl_to_json.py predictions.jsonl
  python convert_jsonl_to_json.py predictions.jsonl -o output.json
  python convert_jsonl_to_json.py predictions.jsonl --indent 4
        """
    )
    parser.add_argument(
        "--input_file",
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to save the output JSON file (default: same name with .json extension)"
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Number of spaces for indentation (default: 2)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = convert_jsonl_to_json(args.input_file, args.output, args.indent)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
