import os
import json
import unittest
from pathlib import Path

class TestSimilarities(unittest.TestCase):
    """
    Test suite to evaluate the similarity matrices of sentences in neet/evidence_cases/.
    Expected similarities are read from LaSSI/tests/assertions/similarities_neet.json.
    
    "null" values are between 0.0 and 1.0 (exclusive) in the JSON file.
    
    The summary will display:
    - Completely Working: All exact matches and valid "null"s.
    - Partially Working: Some exact matches/valid "null"s, but some mismatches.
    - Not Working At All: No valid matches.
    """
    stats = {
        "completely_working": 0,
        "partially_working": 0,
        "not_working": 0,
        "missing_data": 0
    }
    test_results = []
        
    @classmethod
    def tearDownClass(cls):
        print("\n" + "="*30)
        print("--- Similarity Evaluation Summary ---")
        print(f"Completely Working: {cls.stats['completely_working']}")
        print(f"Partially Working:  {cls.stats['partially_working']}")
        print(f"Not Working At All: {cls.stats['not_working']}")
        print(f"Missing Data/Skipped: {cls.stats['missing_data']}")
        print("="*30 + "\n")
        
    def test_evaluate_similarities(self):
        rel_dir = os.path.dirname(os.path.abspath(__file__))
        expected_filepath = os.path.join(rel_dir, "assertions/similarities_neet.json")
        
        # Read expected similarities file
        if not os.path.exists(expected_filepath):
            self.skipTest(f"Missing expected similarities file at {expected_filepath}")
                
        with open(expected_filepath, 'r') as f:
            expected_data = json.load(f)
            
        project_root = Path(rel_dir).parent.absolute().parent.absolute()
        catabolites_dir = os.path.join(project_root, "catabolites")
        evidence_cases_dir = os.path.join(project_root, "neet", "evidence_cases")
        
        if not os.path.exists(evidence_cases_dir):
            self.skipTest(f"Evidence cases directory not found at {evidence_cases_dir}")
            
        all_tests = sorted([f.replace(".yaml", "") for f in os.listdir(evidence_cases_dir) if f.endswith(".yaml")])
        
        for test_name in all_tests:
            with self.subTest(test_name=test_name):
                if test_name not in expected_data:
                    self.__class__.stats["missing_data"] += 1
                    # Log skip and continue instead of self.skipTest
                    print(f"Skipped: No expected similarities defined for {test_name} in similarities_neet.json")
                    continue
                    
                matrix_file = os.path.join(catabolites_dir, test_name, "confusion_matrices_Logical.json")
                if not os.path.exists(matrix_file):
                    self.__class__.stats["missing_data"] += 1
                    # Fail explicitly inside subtest without killing everything
                    self.fail(f"[{test_name}] Computed matrix not found in catabolites. Did you run the pipeline?")
                    continue
                    
                with open(matrix_file, 'r') as f:
                    computed_matrix = json.load(f)
                    
                expected_matrix = expected_data[test_name]
                
                total_checks = 0
                passed_checks = 0
                failures = []
                
                for i in range(len(expected_matrix)):
                    for j in range(len(expected_matrix[i])):
                        total_checks += 1
                        expected_val = expected_matrix[i][j]
                        
                        if i >= len(computed_matrix) or j >= len(computed_matrix[i]):
                            failures.append(f"Cell ({i},{j}) out of bounds in computed matrix.")
                            continue
                            
                        computed_val = computed_matrix[i][j]
                        
                        if expected_val == "null" or expected_val is None:
                            # User is unsure, meaning value should be > 0 and < 1
                            if 0.0 < computed_val < 1.0:
                                passed_checks += 1
                            else:
                                failures.append(f"Cell ({i},{j}): expected 'null' (0 < val < 1), got {computed_val}")
                        else:
                            # Exact match expected
                            try:
                                exp_float = float(expected_val)
                                if abs(computed_val - exp_float) < 1e-5:
                                    passed_checks += 1
                                else:
                                    failures.append(f"Cell ({i},{j}): expected {exp_float}, got {computed_val}")
                            except ValueError:
                                failures.append(f"Cell ({i},{j}): Invalid expected value format: {expected_val}")
                                
                if total_checks == 0:
                    self.__class__.stats["missing_data"] += 1
                elif passed_checks == total_checks:
                    self.__class__.stats["completely_working"] += 1
                elif passed_checks > 0:
                    self.__class__.stats["partially_working"] += 1
                else:
                    self.__class__.stats["not_working"] += 1
                    
                # Assert no failures so unittest reports them
                if failures:
                    self.fail(f"[{test_name}] Similarity mismatches ({passed_checks}/{total_checks} passed):\n" + "\n".join(failures))

if __name__ == '__main__':
    unittest.main()
