Please see our [data processing instructions](../../docs/training.md).

---

## `process_noesy_to_yaml.py`

**Purpose:**

This script processes NOESY (Nuclear Overhauser Effect Spectroscopy) restraint files, which are typically in a plain text format (e.g., `.txt`). The primary goal is to filter these restraints and convert them into a YAML format suitable for use as constraints in Boltz.

The processing involves two main steps:
1.  **Initial Distance Filtering:** Restraints are filtered to keep only those within a specified minimum and maximum distance (e.g., H-H distances).
2.  **Disambiguation:** For NOESY peaks that may have multiple possible assignments (ambiguous peaks), this script selects the assignment with the shortest distance for each unique peak ID. If multiple assignments share the same shortest distance, the first one encountered in the input file is kept.

**Command-Line Arguments:**

*   `--input_noesy_file <PATH>`
    *   **Purpose:** Path to the input NOESY restraint file. This file should list restraints, typically with fields for residue indices, atom names, distance, and peak ID.
    *   **Required:** Yes

*   `--output_yaml_file <PATH>`
    *   **Purpose:** Path where the generated YAML output file will be saved.
    *   **Required:** Yes

*   `--min_dist <FLOAT>`
    *   **Purpose:** Minimum distance (in Angstroms) for the initial filtering of restraints. Restraints with distances below this value will be excluded.
    *   **Default:** `1.5`

*   `--max_dist <FLOAT>`
    *   **Purpose:** Maximum distance (in Angstroms) for the initial filtering of restraints. Restraints with distances above this value will be excluded.
    *   **Default:** `6.0`

*   `--protein_id <STRING>`
    *   **Purpose:** Specifies the protein chain ID. This ID is passed to the YAML generation function.
    *   **Default:** `"A"`
    *   **Note:** While this argument is available, the current YAML output structure for 'noesy' constraints does not directly incorporate this `protein_id`. It is included in the function signature of the underlying YAML generation function but not used in the actual NOESY block YAML structure produced by `generate_noesy_yaml_from_restraints`.

**Example Usage:**

Assuming you are running the script from the root directory of the Boltz repository:

```bash
python scripts/process/process_noesy_to_yaml.py \
    --input_noesy_file examples/example_noesy_restraints.txt \
    --output_yaml_file examples/generated_noesy_constraints.yaml \
    --min_dist 2.0 \
    --max_dist 5.5
```

This command will:
1.  Read restraints from `examples/example_noesy_restraints.txt`.
2.  Filter them, keeping only those with distances between 2.0 Å and 5.5 Å.
3.  Disambiguate the filtered restraints.
4.  Generate a YAML file named `generated_noesy_constraints.yaml` in the `examples/` directory.