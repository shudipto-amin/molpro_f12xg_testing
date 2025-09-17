import subprocess
import os
import argparse
import re

def read_distances(file_path):
    """Read distances from a file with a line like: distances=[1.0,2.5,3.0]"""
    with open(file_path, "r") as f:
        for line in f:
            match = re.match(r"^\s*distances\s*=\s*\[(.*)\]\s*$", line)
            if match:
                values = match.group(1).split(",")
                return [float(v.strip()) for v in values if v.strip()]
    raise ValueError(f"No valid 'distances=[...]' line found in {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate modified input files with different distances.")
    parser.add_argument("input_file", help="Input .inp file (e.g., something.inp)")
    parser.add_argument("distances_file", help="Text file containing 'distances=[...]'")
    parser.add_argument("--dry-run", action="store_true", help="Print output instead of writing files")
    parser.add_argument("-r", "--run", action="store_true",
                        help="Run 'qmolpro -M 45 <file_name>' for each generated file")
    parser.add_argument("--qmolpro-path", default="/home/linux3_i1/amin/q-scripts/qmolpro-generic",
                        help="Full path to qmolpro executable (required for --run)")
    args = parser.parse_args()

    # Read distances from file
    distances = read_distances(args.distances_file)

    # Extract base name (without .inp)
    basename, ext = os.path.splitext(args.input_file)
    if ext.lower() != ".inp":
        parser.error("Input file must have .inp extension")

    # Ensure output directory exists

    generated_files = []

    for d in distances:
        d_str = f"{d:6.3f}"  # format to 6.3f
        output_file = f"{basename}_r_{d:06.3f}.inp"

        # Run sed once, capture output
        result = subprocess.run(
            ["sed", f"s/distance/{d_str}/", args.input_file],
            capture_output=True,
            text=True,
            check=True
        ).stdout

        if args.dry_run:
            print(f"\n[DRY-RUN] Would create: {output_file}")
            print(result.strip())
        else:
            with open(output_file, "w") as f:
                f.write(result)
            print(f"Created {output_file}")
            generated_files.append(output_file)

    # Run qmolpro if requested
    if args.run and not args.dry_run:
        for file in generated_files:
            print(f"Running: {args.qmolpro_path} -M 45 {file}")
            subprocess.run([args.qmolpro_path, "-M", "45", file], check=True)

if __name__ == "__main__":
    main()

