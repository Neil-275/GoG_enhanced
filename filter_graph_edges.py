"""
Script to filter drop_edges from graph files.
Optimized to handle very large knowledge graph files efficiently.
"""

import csv
import argparse
from pathlib import Path
from typing import Set, Tuple
import sys


def load_drop_edges(drop_edges_file: str) -> Set[Tuple[str, str, str]]:
    """
    Load drop_edges from CSV file into a set for O(1) lookup.
    
    Args:
        drop_edges_file: Path to the drop_edges CSV file
        
    Returns:
        Set of tuples (head_id, relation, tail_id) to be dropped
    """
    drop_edges = set()
    
    print(f"Loading drop_edges from {drop_edges_file}...")
    
    try:
        with open(drop_edges_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header row
            next(reader, None)
            
            for row_num, row in enumerate(reader, start=2):
                if len(row) >= 3:
                    head_id, relation, tail_id = row[0], row[1], row[2]
                    drop_edges.add((head_id, relation, tail_id))
                else:
                    print(f"Warning: Skipping malformed row {row_num}: {row}")
    
    except FileNotFoundError:
        print(f"Error: File not found: {drop_edges_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading drop_edges file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(drop_edges)} edges to drop")
    return drop_edges


def filter_graph_edges(
    graph_file: str,
    output_file: str,
    drop_edges: Set[Tuple[str, str, str]],
    chunk_size: int = 100000
) -> None:
    """
    Filter drop_edges from graph file and write to output file.
    Uses line-by-line reading for memory efficiency with large files.
    
    Args:
        graph_file: Path to input graph file (tab-separated)
        output_file: Path to output filtered graph file
        drop_edges: Set of edges to drop
        chunk_size: Number of lines to process before reporting progress
    """
    
    print(f"\nFiltering graph file: {graph_file}")
    print(f"Output will be saved to: {output_file}")
    
    dropped_count = 0
    kept_count = 0
    
    try:
        with open(graph_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, start=1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Parse the edge (tab-separated)
                parts = line.split('\t')
                
                if len(parts) >= 3:
                    head_id, relation, tail_id = parts[0], parts[1], parts[2]
                    
                    # Check if this edge should be dropped
                    if (head_id, relation, tail_id) not in drop_edges:
                        outfile.write(line + '\n')
                        kept_count += 1
                    else:
                        dropped_count += 1
                else:
                    # Keep malformed lines (or could skip them)
                    print(f"Warning: Skipping malformed line {line_num}: {line}")
                
                # Progress reporting for large files
                if (line_num % chunk_size) == 0:
                    print(f"  Processed {line_num} lines... "
                          f"(dropped: {dropped_count}, kept: {kept_count})")
        
        print(f"\nFiltering complete!")
        print(f"  Total dropped edges: {dropped_count}")
        print(f"  Total kept edges: {kept_count}")
        print(f"  Filtered graph saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Graph file not found: {graph_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing graph file: {e}")
        sys.exit(1)


def is_graph_file(filepath: str, sample_size: int = 5) -> bool:
    """
    Check if a file is a graph file by examining its format.
    A graph file should have tab-separated edges with at least 3 columns.
    
    Args:
        filepath: Path to the file to check
        sample_size: Number of non-empty lines to check
        
    Returns:
        True if file appears to be a graph file, False otherwise
    """
    try:
        checked_lines = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    return False
                
                checked_lines += 1
                if checked_lines >= sample_size:
                    return True
        
        return checked_lines >= sample_size
    except Exception:
        return False


def filter_folder(
    input_folder: str,
    output_folder: str,
    drop_edges_file: str,
    chunk_size: int = 100000
) -> None:
    """
    Filter all .txt graph files in input folder and save to output folder.
    Automatically detects which .txt files are graph files based on format.
    
    Args:
        input_folder: Path to input folder containing graph files
        output_folder: Path to output folder for filtered graphs
        drop_edges_file: Path to drop_edges CSV file
        chunk_size: Number of lines to process before reporting progress
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files in input folder
    all_txt_files = sorted(input_path.glob('*.txt'))
    
    if not all_txt_files:
        print(f"Error: No .txt files found in {input_folder}")
        sys.exit(1)
    
    # Filter to only graph files (tab-separated with 3+ columns)
    print("Detecting graph files...")
    graph_files = [f for f in all_txt_files if is_graph_file(str(f))]
    
    if not graph_files:
        print(f"Error: No valid graph files found in {input_folder}")
        print(f"  (Found {len(all_txt_files)} .txt files, but none have 3+ tab-separated columns)")
        sys.exit(1)
    
    print(f"Found {len(graph_files)} graph file(s) to process (skipped {len(all_txt_files) - len(graph_files)} non-graph files)\n")
    
    print(f"Found {len(graph_files)} graph file(s) to process (skipped {len(all_txt_files) - len(graph_files)} non-graph files)\n")
    
    # Load drop edges once
    drop_edges = load_drop_edges(drop_edges_file)
    
    # Process each file
    total_dropped = 0
    total_kept = 0
    
    for idx, graph_file in enumerate(graph_files, 1):
        output_file = output_path / graph_file.name
        
        print(f"\n[{idx}/{len(graph_files)}] Processing: {graph_file.name}")
        
        # Filter the graph
        dropped, kept = filter_graph_edges_and_return_counts(
            str(graph_file),
            str(output_file),
            drop_edges,
            chunk_size,
            suppress_warnings=True
        )
        
        total_dropped += dropped
        total_kept += kept
    
    # Summary
    print(f"\n" + "="*60)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"="*60)
    print(f"Processed files: {len(graph_files)}")
    print(f"Total dropped edges: {total_dropped}")
    print(f"Total kept edges: {total_kept}")
    print(f"Output folder: {output_path}")
    print(f"="*60)


def filter_graph_edges_and_return_counts(
    graph_file: str,
    output_file: str,
    drop_edges: Set[Tuple[str, str, str]],
    chunk_size: int = 100000,
    suppress_warnings: bool = False
) -> Tuple[int, int]:
    """
    Filter drop_edges from graph file and return counts.
    Internal version that returns counts for batch processing.
    """
    dropped_count = 0
    kept_count = 0
    
    try:
        with open(graph_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, start=1):
                line = line.strip()
                
                if not line:
                    continue
                
                parts = line.split('\t')
                
                if len(parts) >= 3:
                    head_id, relation, tail_id = parts[0], parts[1], parts[2]
                    
                    if (head_id, relation, tail_id) not in drop_edges:
                        outfile.write(line + '\n')
                        kept_count += 1
                    else:
                        dropped_count += 1
                else:
                    if not suppress_warnings:
                        print(f"  Warning: Skipping malformed line {line_num}: {line}")
                
                if (line_num % chunk_size) == 0:
                    print(f"  Processed {line_num} lines... "
                          f"(dropped: {dropped_count}, kept: {kept_count})")
        
        print(f"  Complete: dropped {dropped_count}, kept {kept_count}")
        
    except FileNotFoundError:
        print(f"  Error: Graph file not found: {graph_file}")
        sys.exit(1)
    except Exception as e:
        print(f"  Error processing graph file: {e}")
        sys.exit(1)
    
    return dropped_count, kept_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter drop_edges from graph files (handles very large KGs efficiently)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode
  python filter_graph_edges.py all.txt drops.csv -o filtered.txt
  
  # Batch folder mode
  python filter_graph_edges.py data/raw data/filtered drops.csv --batch
  
  # Batch with custom chunk size
  python filter_graph_edges.py input_folder output_folder drops.csv --batch --chunk-size 500000
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Single file mode: path to graph file. Batch mode: path to input folder'
    )
    
    parser.add_argument(
        'drop_edges_file_or_output',
        type=str,
        help='Single file mode: path to drop_edges CSV. Batch mode: path to output folder'
    )
    
    parser.add_argument(
        'drop_edges_file',
        nargs='?',
        type=str,
        help='(Batch mode only) Path to drop_edges CSV file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='(Single file mode) Path to output file (default: input_file with _filtered suffix)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing mode (process all .txt files in input folder)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Number of lines to process before reporting progress (default: 100000)'
    )
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch:
        if args.drop_edges_file is None:
            print("Error: In batch mode, you must provide 3 arguments:")
            print("  python filter_graph_edges.py <input_folder> <output_folder> <drop_edges_file> --batch")
            sys.exit(1)
        
        input_folder = args.input
        output_folder = args.drop_edges_file_or_output
        drop_edges_file = args.drop_edges_file
        
        # Validate paths
        if not Path(input_folder).exists():
            print(f"Error: Input folder not found: {input_folder}")
            sys.exit(1)
        
        if not Path(drop_edges_file).exists():
            print(f"Error: Drop edges file not found: {drop_edges_file}")
            sys.exit(1)
        
        filter_folder(input_folder, output_folder, drop_edges_file, args.chunk_size)
    
    # Single file mode
    else:
        if args.drop_edges_file is not None:
            print("Error: In single file mode, use only 2 positional arguments:")
            print("  python filter_graph_edges.py <graph_file> <drop_edges_file> [-o <output_file>]")
            sys.exit(1)
        
        graph_file = args.input
        drop_edges_file = args.drop_edges_file_or_output
        
        # Determine output file path
        if args.output is None:
            graph_path = Path(graph_file)
            output = str(graph_path.parent / f"{graph_path.stem}_filtered.txt")
        else:
            output = args.output
        
        # Validate input files exist
        if not Path(graph_file).exists():
            print(f"Error: Graph file not found: {graph_file}")
            sys.exit(1)
        
        if not Path(drop_edges_file).exists():
            print(f"Error: Drop edges file not found: {drop_edges_file}")
            sys.exit(1)
        
        # Execute filtering
        drop_edges = load_drop_edges(drop_edges_file)
        filter_graph_edges(graph_file, output, drop_edges, args.chunk_size)


if __name__ == '__main__':
    main()
