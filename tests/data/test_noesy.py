import unittest
import sys
import os
from ruamel.yaml import YAML

# Add project root to sys.path to allow importing from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.boltz.data.noesy import (
        disambiguate_shortest_distance,
        generate_noesy_yaml_from_restraints
    )
except ImportError as e:
    print(f"Error importing from src.boltz.data.noesy: {e}")
    print(f"Ensure that the project root ({PROJECT_ROOT}) is in your PYTHONPATH.")
    # This will cause tests to fail if imports don't work, which is intended.
    raise

class TestNoesyFunctions(unittest.TestCase):

    def test_empty_list_disambiguation(self):
        self.assertEqual(disambiguate_shortest_distance([]), [])

    def test_unique_peak_ids_disambiguation(self):
        restraints = [
            (0, 1, 1, 2.0, 'H', 'H'),
            (2, 3, 2, 3.0, 'HA', 'HB')
        ]
        self.assertEqual(disambiguate_shortest_distance(restraints), restraints)

    def test_ambiguous_clear_shortest_disambiguation(self):
        restraints = [
            (0, 1, 1, 2.0, 'H', 'H'),
            (2, 3, 1, 3.0, 'HA', 'HB'), # Same peak ID 1, longer distance
            (4, 5, 2, 2.5, 'N', 'H')
        ]
        expected = [
            (0, 1, 1, 2.0, 'H', 'H'),
            (4, 5, 2, 2.5, 'N', 'H')
        ]
        # Convert to list of tuples and sort by peak ID for comparison
        # as order of different peak IDs might change.
        result = sorted(disambiguate_shortest_distance(restraints), key=lambda x: x[2])
        expected_sorted = sorted(expected, key=lambda x: x[2])
        self.assertEqual(result, expected_sorted)


    def test_ambiguous_tied_distance_disambiguation(self):
        restraints = [
            (0, 1, 1, 2.0, 'H', 'H'),       # Keep (unique peak ID)
            (2, 3, 2, 3.0, 'HA', 'HB'),     # Peak ID 2, distance 3.0
            (2, 4, 2, 2.5, 'HA', 'HC'),     # Peak ID 2, distance 2.5 (shortest for peak 2) - Keep
            (5, 6, 3, 4.0, 'N', 'H'),       # Peak ID 3, distance 4.0 (first encountered for tie) - Keep
            (5, 7, 3, 4.0, 'N', 'HA')      # Peak ID 3, distance 4.0 (tied, second)
        ]
        expected = [
            (0, 1, 1, 2.0, 'H', 'H'),
            (2, 4, 2, 2.5, 'HA', 'HC'),
            (5, 6, 3, 4.0, 'N', 'H')
        ]
        # The order of restraints with different peak IDs in the output
        # depends on dict iteration order of defaultdict.
        # We sort by peak ID for stable comparison.
        result = sorted(disambiguate_shortest_distance(restraints), key=lambda x: x[2])
        expected_sorted = sorted(expected, key=lambda x: x[2])
        self.assertEqual(result, expected_sorted)

    def test_empty_list_yaml_generation(self):
        yaml_str = generate_noesy_yaml_from_restraints([])
        yaml = YAML()
        data = yaml.load(yaml_str)
        self.assertEqual(data, {'constraints': []})

    def test_simple_restraints_yaml_generation(self):
        restraints = [
            (88, 224, 7, 2.3, 'H', 'H'),
            (23, 26, 2, 4.0, 'HA', 'HB3')
        ]
        # protein_id is default "A", not used in 'noesy' block per current implementation
        yaml_str = generate_noesy_yaml_from_restraints(restraints)
        yaml = YAML()
        data = yaml.load(yaml_str)

        expected_data = {
            'constraints': [
                {
                    'noesy': {
                        'residueFrom': 89,
                        'residueTo': 225,
                        'peakID': 7,
                        'distance': 2.3,
                        'atomFrom': 'H',
                        'atomTo': 'H'
                    }
                },
                {
                    'noesy': {
                        'residueFrom': 24,
                        'residueTo': 27,
                        'peakID': 2,
                        'distance': 4.0,
                        'atomFrom': 'HA',
                        'atomTo': 'HB3'
                    }
                }
            ]
        }
        # Comparing the loaded data structures
        self.assertEqual(data, expected_data)

if __name__ == '__main__':
    unittest.main()
