"""Module for handling NOESY restraint data."""

import numpy as np
from typing import List, Tuple, Dict
import collections
import io
from ruamel.yaml import YAML

ResidueAtomDistanceMap: Dict[str, Dict[str, Tuple[str, float, float]]] = {
    "VAL": {
        "HG1": ("CB", 1.2, 0.8),  # Target: CB, Offset: +1.2A, Tolerance: +/-0.8A
        "HG2": ("CB", 1.2, 0.8),
        "QG": ("CB", 1.8, 1.0),   # For pseudo-atom QG (HG1*+HG2*)
    },
    "LEU": {
        "HD1": ("CG", 1.2, 0.8),  # Target: CG, Offset: +1.2A, Tolerance: +/-0.8A
        "HD2": ("CG", 1.2, 0.8),
        "QD": ("CG", 1.8, 1.0),   # For pseudo-atom QD (HD1*+HD2*)
    },
    "ILE": {
        "HG2": ("CG1", 1.2, 0.8), # Target: CG1, Offset: +1.2A, Tolerance: +/-0.8A
        "HD1": ("CD1", 1.2, 0.8), # Target: CD1, Offset: +1.2A, Tolerance: +/-0.8A
        # Note: ILE has HG1 (CG1), HG2 (CG1), HD1 (CD1), HE (CE), QG (CG1), QD (CD1)
        # Adding more specific mappings if needed.
    },
    "ALA": {
        "HB": ("CB", 0.9, 0.5)   # Target: CB, Offset: +0.9A, Tolerance: +/-0.5A
    },
    "_GENERIC_H_": {  # For backbone amide protons
        "H": ("N", 1.0, 0.5)    # Map backbone H to N, Offset: +1.0A, Tolerance: +/-0.5A
    }
}


def read_noesy_restraints(
    file_path: str = None,
    noesy_constraints_nmr_format: List[str] = None
) -> List[Tuple[int, int, int, float, str, str]]:
    """
    Parses NOESY restraint data from a file or a list of strings.

    The input format per line is expected to be:
    residueFrom(1-idx) residueTo(1-idx) peakID distance atomFrom atomTo
    (space or tab-separated).

    Args:
        file_path: Path to the NOESY restraint file.
        noesy_constraints_nmr_format: A list of strings, each being a restraint line.

    Returns:
        A list of tuples, where each tuple contains:
        (res_idx_1 (0-indexed), res_idx_2 (0-indexed), peak_id,
         distance, atom_name_1_str, atom_name_2_str).
    """
    raw_restraints: List[Tuple[int, int, int, float, str, str]] = []
    lines_to_process: List[str] = []

    if noesy_constraints_nmr_format is not None:
        lines_to_process = [line.strip() for line in noesy_constraints_nmr_format]
    elif file_path:
        try:
            with open(file_path, 'r') as f:
                lines_to_process = [line.strip() for line in f]
        except FileNotFoundError:
            print(f"Warning: NOESY restraint file not found at {file_path}")
            return raw_restraints
    else:
        print("Warning: No input provided for read_noesy_restraints.")
        return raw_restraints

    for i, line in enumerate(lines_to_process):
        if not line or line.startswith("#"):  # Skip empty lines or comments
            continue
        parts = line.split()
        if len(parts) == 6:
            try:
                res_idx_1 = int(parts[0]) - 1  # Convert to 0-indexed
                res_idx_2 = int(parts[1]) - 1  # Convert to 0-indexed
                peak_id = int(parts[2])
                distance = float(parts[3])
                atom_name_1_str = parts[4]
                atom_name_2_str = parts[5]
                raw_restraints.append(
                    (res_idx_1, res_idx_2, peak_id, distance, atom_name_1_str, atom_name_2_str)
                )
            except ValueError as e:
                print(f"Warning: Malformed line {i+1} in NOESY data: '{line}'. Error: {e}. Skipping.")
        else:
            print(f"Warning: Malformed line {i+1} in NOESY data: '{line}'. Expected 6 fields, got {len(parts)}. Skipping.")
    return raw_restraints


def filter_noesy_restraints(
    raw_restraints: List[Tuple[int, int, int, float, str, str]],
    max_distance: float = 6.0,
    min_distance: float = 1.5
) -> List[Tuple[int, int, int, float, str, str]]:
    """
    Filters NOESY restraints based on distance.

    Args:
        raw_restraints: A list of raw NOESY restraints.
        max_distance: Maximum H-H distance to keep.
        min_distance: Minimum H-H distance to keep.

    Returns:
        A filtered list of NOESY restraints.
    """
    filtered_restraints: List[Tuple[int, int, int, float, str, str]] = []
    for restraint in raw_restraints:
        distance = restraint[3]
        if min_distance <= distance <= max_distance:
            filtered_restraints.append(restraint)
    return filtered_restraints


def map_proton_distances_to_backbone(
    filtered_restraints: List[Tuple[int, int, int, float, str, str]],
    residue_types_map: Dict[int, str]
) -> List[Tuple[int, int, int, float, str, str, float]]:
    """
    Maps proton-proton distances to backbone atom distances using ResidueAtomDistanceMap.

    Args:
        filtered_restraints: A list of filtered NOESY restraints (H-H distances).
        residue_types_map: A dictionary mapping 0-indexed residue index to residue type (e.g., "VAL").

    Returns:
        A list of tuples, where each tuple contains:
        (res_idx_1, res_idx_2, peak_id, final_distance,
         final_atom_name_1, final_atom_name_2, distance_tolerance_for_potential).
    """
    mapped_restraints: List[Tuple[int, int, int, float, str, str, float]] = []

    for r1_idx, r2_idx, peak_id, hh_dist, atom1_str, atom2_str in filtered_restraints:
        final_atom1_str = atom1_str
        final_atom2_str = atom2_str
        current_offset = 0.0
        current_tolerance = 0.5  # Default tolerance

        res1_type = residue_types_map.get(r1_idx)
        res2_type = residue_types_map.get(r2_idx)

        if res1_type is None:
            # print(f"Warning: Residue type for index {r1_idx} not found. Skipping restraint {peak_id}.")
            continue
        if res2_type is None:
            # print(f"Warning: Residue type for index {r2_idx} not found. Skipping restraint {peak_id}.")
            continue

        # Atom 1 mapping
        map_key1 = atom1_str
        res_map1 = ResidueAtomDistanceMap.get(res1_type)
        if res_map1 and map_key1 in res_map1:
            target_atom, offset, tolerance = res_map1[map_key1]
            final_atom1_str = target_atom
            current_offset += offset
            current_tolerance = max(current_tolerance, tolerance)
        elif atom1_str == "H" and "_GENERIC_H_" in ResidueAtomDistanceMap and "H" in ResidueAtomDistanceMap["_GENERIC_H_"]:
            target_atom, offset, tolerance = ResidueAtomDistanceMap["_GENERIC_H_"]["H"]
            final_atom1_str = target_atom
            current_offset += offset
            current_tolerance = max(current_tolerance, tolerance)

        # Atom 2 mapping
        map_key2 = atom2_str
        res_map2 = ResidueAtomDistanceMap.get(res2_type)
        if res_map2 and map_key2 in res_map2:
            target_atom, offset, tolerance = res_map2[map_key2]
            final_atom2_str = target_atom
            current_offset += offset
            current_tolerance = max(current_tolerance, tolerance)
        elif atom2_str == "H" and "_GENERIC_H_" in ResidueAtomDistanceMap and "H" in ResidueAtomDistanceMap["_GENERIC_H_"]:
            target_atom, offset, tolerance = ResidueAtomDistanceMap["_GENERIC_H_"]["H"]
            final_atom2_str = target_atom
            current_offset += offset
            current_tolerance = max(current_tolerance, tolerance)

        final_dist = max(0.1, hh_dist + current_offset)  # Ensure positive distance

        mapped_restraints.append(
            (r1_idx, r2_idx, peak_id, final_dist, final_atom1_str, final_atom2_str, current_tolerance)
        )

    return mapped_restraints


def disambiguate_shortest_distance(
    restraints: List[Tuple[int, int, int, float, str, str]]
) -> List[Tuple[int, int, int, float, str, str]]:
    """
    Disambiguates NOESY restraints by selecting the one with the shortest distance for each peakID.

    Args:
        restraints: A list of NOESY restraints. Each restraint is a tuple:
                    (res_idx_1 (0-indexed), res_idx_2 (0-indexed), peak_id,
                     distance, atom_name_1_str, atom_name_2_str).
                     This list is expected to be the output of an initial
                     distance-based filtering (e.g., from filter_noesy_restraints).

    Returns:
        A new list of NOESY restraints where each peak_id is unique.
        For each peak_id that had multiple assignments, only the assignment
        with the shortest distance is kept. If multiple assignments share
        the same shortest distance, the first one encountered in the input list
        is kept.
    """
    if not restraints:
        return []

    # Group restraints by peak_id while preserving order for tie-breaking.
    # peak_id_to_restraints_map: Dict[int, List[Tuple[int, int, int, float, str, str]]] = {}
    # Using defaultdict(list) is cleaner for appending.
    peak_id_to_restraints_map = collections.defaultdict(list)
    for restraint in restraints:
        peak_id = restraint[2]
        peak_id_to_restraints_map[peak_id].append(restraint)

    disambiguated_restraints: List[Tuple[int, int, int, float, str, str]] = []

    # To ensure original order for tie-breaking, we iterate through original restraints' peak_ids
    # as they were first encountered.
    # However, the problem states "the first one encountered in the input list is kept" for ties.
    # The grouping above already preserves this order within each peak_id's list.
    # So, we can iterate through the map's keys and then find the minimum.

    processed_peak_ids = set() # To keep track of peak_ids processed from the map
                               # This is not strictly necessary if we just iterate through map items.

    # Iterate through the grouped restraints to find the shortest distance for each peak_id
    for peak_id, associated_restraints in peak_id_to_restraints_map.items():
        if not associated_restraints: # Should not happen with defaultdict(list) if peak_id existed
            continue

        if len(associated_restraints) == 1:
            disambiguated_restraints.append(associated_restraints[0])
        else:
            # Find the restraint with the minimum distance.
            # If distances are tied, the first one in 'associated_restraints' list is chosen
            # because of the initial population order.
            min_dist_restraint = associated_restraints[0]
            min_dist = associated_restraints[0][3] # distance

            for i in range(1, len(associated_restraints)):
                current_restraint_dist = associated_restraints[i][3]
                if current_restraint_dist < min_dist:
                    min_dist = current_restraint_dist
                    min_dist_restraint = associated_restraints[i]
            disambiguated_restraints.append(min_dist_restraint)
            
    # The order of restraints for different peak_ids in the final list doesn't strictly matter
    # as per the example output. The current approach iterates through peak_ids as they appear
    # in the defaultdict, which depends on insertion order from the input list.

    return disambiguated_restraints


def generate_noesy_yaml_from_restraints(
    restraints: List[Tuple[int, int, int, float, str, str]],
    protein_id: str = "A" # Default protein ID, not used in this specific YAML structure
) -> str:
    """
    Generates a YAML string representing NOESY constraints.

    Args:
        restraints: A list of filtered and disambiguated NOESY restraints.
                    Each restraint is a tuple:
                    (res_idx_1 (0-indexed), res_idx_2 (0-indexed), peak_id,
                     distance, atom_name_1_str, atom_name_2_str).
        protein_id: The chain ID to use for the protein in the YAML output. Defaults to "A".
                    Note: This parameter is included as per signature but not directly
                    used in the 'noesy' block structure based on provided examples.

    Returns:
        A string formatted as YAML, representing the NOESY constraints.
    """
    yaml_constraints = []
    for r_idx1, r_idx2, p_id, dist, a1, a2 in restraints:
        yaml_constraints.append({
            'noesy': {
                'residueFrom': r_idx1 + 1,  # Convert to 1-indexed
                'residueTo': r_idx2 + 1,    # Convert to 1-indexed
                'peakID': p_id,
                'distance': float(dist),    # Ensure it's a float
                'atomFrom': a1,
                'atomTo': a2
            }
        })

    output_data = {'constraints': yaml_constraints}

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2) # For pretty printing
    string_stream = io.StringIO()
    yaml.dump(output_data, string_stream)
    return string_stream.getvalue()


__all__ = [
    "ResidueAtomDistanceMap",
    "read_noesy_restraints",
    "filter_noesy_restraints",
    "map_proton_distances_to_backbone",
    "disambiguate_shortest_distance",
    "generate_noesy_yaml_from_restraints"
]
