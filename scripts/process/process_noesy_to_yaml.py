#!/usr/bin/env python3

"""
Script to process NOESY restraint files into YAML format.

This script reads NOESY restraints from an input file, filters them based on distance,
disambiguates them by selecting the shortest distance for each peak ID, and then
generates a YAML formatted string which is saved to an output file.
"""

import argparse
import sys
import os

# Assuming the script is in scripts/process/ and src is a sibling of scripts/
# This allows running the script directly for development/testing.
# For production, PYTHONPATH should ideally be set.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.boltz.data.noesy import (
        read_noesy_restraints,
        filter_noesy_restraints,
        disambiguate_shortest_distance,
        generate_noesy_yaml_from_restraints
    )
except ImportError as e:
    print(f"Error importing from src.boltz.data.noesy: {e}")
    print(f"Ensure that the project root ({PROJECT_ROOT}) is in your PYTHONPATH or the script is run from a location where 'src' is accessible.")
    sys.exit(1)


def main():
    """
    Main function to parse arguments and process NOESY restraints.
    """
    parser = argparse.ArgumentParser(
        description="Process NOESY restraint files and convert them to YAML format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_noesy_file",
        type=str,
        required=True,
        help="Path to the input NOESY restraint file (e.g., .txt format)."
    )
    parser.add_argument(
        "--output_yaml_file",
        type=str,
        required=True,
        help="Path to save the generated YAML output file."
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=1.5,
        help="Minimum distance for initial filtering. Default: 1.5 Angstroms."
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=6.0,
        help="Maximum distance for initial filtering. Default: 6.0 Angstroms."
    )
    parser.add_argument(
        "--protein_id",
        type=str,
        default="A",
        help="Protein chain ID to use in YAML output if applicable by the generation function. Default: 'A'."
    )

    args = parser.parse_args()

    print(f"Reading NOESY restraints from: {args.input_noesy_file}")
    raw_restraints = read_noesy_restraints(file_path=args.input_noesy_file)
    if not raw_restraints:
        # read_noesy_restraints prints warnings if file not found or empty.
        # If it returns empty, there's nothing to process.
        print("No restraints found or file was empty. Exiting.")
        # Create an empty YAML file as per typical behavior
        yaml_output = generate_noesy_yaml_from_restraints([], args.protein_id)
        try:
            with open(args.output_yaml_file, 'w') as f:
                f.write(yaml_output)
            print(f"Generated empty YAML constraints file at: {args.output_yaml_file}")
        except IOError as e:
            print(f"Error writing empty YAML to {args.output_yaml_file}: {e}")
        sys.exit(0)


    print(f"Filtering restraints with min_dist={args.min_dist}, max_dist={args.max_dist}...")
    filtered_restraints = filter_noesy_restraints(
        raw_restraints,
        max_distance=args.max_dist,
        min_distance=args.min_dist
    )
    if not filtered_restraints:
        print("No restraints remaining after distance filtering.")
        yaml_output = generate_noesy_yaml_from_restraints([], args.protein_id)
        try:
            with open(args.output_yaml_file, 'w') as f:
                f.write(yaml_output)
            print(f"Generated empty YAML constraints file due to filtering: {args.output_yaml_file}")
        except IOError as e:
            print(f"Error writing empty YAML to {args.output_yaml_file}: {e}")
        sys.exit(0)

    print("Disambiguating restraints by shortest distance for each peak ID...")
    disambiguated_restraints = disambiguate_shortest_distance(filtered_restraints)
    if not disambiguated_restraints:
        # This case might be less common if filter_noesy_restraints already returned data,
        # but good to handle.
        print("No restraints remaining after disambiguation.")
        yaml_output = generate_noesy_yaml_from_restraints([], args.protein_id)
        try:
            with open(args.output_yaml_file, 'w') as f:
                f.write(yaml_output)
            print(f"Generated empty YAML constraints file due to disambiguation: {args.output_yaml_file}")
        except IOError as e:
            print(f"Error writing empty YAML to {args.output_yaml_file}: {e}")
        sys.exit(0)

    print(f"Generating YAML output with protein_id='{args.protein_id}'...")
    yaml_output = generate_noesy_yaml_from_restraints(
        disambiguated_restraints,
        protein_id=args.protein_id
    )

    try:
        with open(args.output_yaml_file, 'w') as f:
            f.write(yaml_output)
        print(f"Successfully processed NOESY restraints and saved YAML to {args.output_yaml_file}")
    except IOError as e:
        print(f"Error writing YAML to {args.output_yaml_file}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
