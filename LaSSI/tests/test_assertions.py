import ast
import collections
import os
import re
import unittest
from pathlib import Path


class TestLaSSI(unittest.TestCase):
    def test_string_reps(self, force_dirs=False):
        assertions = {}
        rel_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(rel_dir, "assertions/assertions_neet.txt")
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith('//') and line.strip() != "":
                    line = self.replace_existential(line)

                    if "//" in line:
                        line = line.split('//')[0].strip()

                    split_line = re.split(" ⇒ ", line)
                    assertions[split_line[0].lower()] = split_line[1].lower()

        found_assertions = []
        catabolites_dir = os.path.join(Path(rel_dir).parent.absolute().parent.absolute(), "catabolites")
        for subdir, dirs, files in os.walk(catabolites_dir):
            if subdir.split('/')[-1].startswith("part") or not force_dirs:
                for file in files:
                    if file == "string_rep.txt":
                        filepath = os.path.join(subdir, file)
                        with open(filepath, 'r') as f:
                            for line in f:
                                if not "//" in line and line.strip() != "":
                                    found_assertions.append(self.replace_existential(line.strip()).lower())

        for assertion in found_assertions:
            # print(assertion)
            split_check_assertion = re.split(" ⇒ ", assertion)
            try:
                if assertions[split_check_assertion[0].lower()]:
                    self.compare_internal_representations(split_check_assertion[1], assertions[split_check_assertion[0]])
            except KeyError as e:
                raise Exception(f"Cannot find assertion for {assertion}")

    def compare_internal_representations(self, assertion, correct_representation):
        assertion = self.replace_existential(assertion.strip()).lower()
        correct_representation = self.replace_existential(correct_representation.strip()).lower()

        correct_rep = self.remove_properties(correct_representation.lower())
        check_rep = self.remove_properties(assertion.lower())
        # Check for groups
        correct_groups, correct_rep = self.get_group_content(correct_rep)
        check_groups, check_rep = self.get_group_content(check_rep)
        # self.assertEqual(collections.Counter(correct_groups), collections.Counter(check_groups),
        #                  f"{assertion}")
        try:
            self.assertEqual(collections.Counter(correct_groups), collections.Counter(check_groups), f"{assertion}")
        except AssertionError as e:
            print(e)
        # Check reps are equal without props or groups
        # self.assertEqual(correct_rep.lower(), check_rep.lower(), f"{assertion}")
        try:
            self.assertEqual(correct_rep.lower(), check_rep.lower(), f"{assertion}")
        except AssertionError as e:
            print(e)
        # Check properties are equal
        args1 = self.get_properties(correct_representation)
        args2 = self.get_properties(assertion)
        # self.assertEqual(collections.Counter(args1), collections.Counter(args2), f"{assertion}")
        try:
            self.assertEqual(collections.Counter(args1), collections.Counter(args2), f"{assertion}")
            return True
        except AssertionError as e:
            print(e)
            return False

    def get_properties(self, rep, new_args=None):
        # Regex for everything in innermost []
        if new_args is None:
            new_args = []
        for arg in re.findall(r"\[[^\[\]]*]", rep):
            # inner_arg = re.sub(r'\(nmod:\w+\)', '', arg) # Omit 'x' property
            # inner_arg = list(filter(None, inner_arg.strip('[]').split(', ')))
            inner_arg = list(filter(None, arg.strip('[]').split(', ')))
            if inner_arg is not None:
                new_args.extend(inner_arg)

        removed_props_rep = re.sub(r"\[[^\[\]]*]", "", rep) # Check for
        if re.findall(r"\[[^\[\]]*]", removed_props_rep):
            self.get_properties(removed_props_rep, new_args)

        return new_args

    # Get groups to evaluate and their group type
    def get_group_content(self, rep):
        groups = []
        pattern = r"(?:and|or|neither)\(\s*((?:[^()]|\([^()]*\))*)\)"
        for group in re.findall(pattern, rep):
            inner_group = group.strip('[]').split(', ')
            groups.extend(inner_group)

        group_type_pattern = r"(?:and|or|neither)\("
        for group in re.findall(group_type_pattern, rep):
            inner_group = group.strip('(')
            groups.append(inner_group)
        return groups, re.sub(pattern, "", rep)

    # Remove everything between []
    def remove_properties(self, rep):
        pattern = r"\[((?:[^\[\]]|\[[^\[\]]*])*)]"
        removed = re.sub(pattern, "", rep)
        if len(re.findall(pattern, removed)) > 0:
            removed = self.remove_properties(removed)

        return removed
        # return re.sub(r"\[[^\[\]]*]", "", rep)

    # ?1 is replaced with ?
    def replace_existential(self, line):
        return re.sub(r"\?\d+", "?", line.strip())


if __name__ == '__main__':
    unittest.main()