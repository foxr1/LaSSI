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
            split_check_assertion = re.split(" ⇒ ", assertion)
            sentence = split_check_assertion[0].lower()
            try:
                if assertions[sentence]:
                    with self.subTest(sentence=sentence):
                        self.compare_internal_representations(split_check_assertion[1], assertions[sentence])
            except KeyError as e:
                raise Exception(f"Cannot find assertion for {assertion}")

    def compare_internal_representations(self, assertion, correct_representation):
        assertion = self.replace_existential(assertion.strip()).lower()
        correct_representation = self.replace_existential(correct_representation.strip()).lower()

        # Remove superscript type markers (e.g. ⁽ᵛᵉʳᵇ⁾) introduced in the new format
        assertion = re.sub(r'⁽[^⁾]+⁾', '', assertion)
        correct_representation = re.sub(r'⁽[^⁾]+⁾', '', correct_representation)

        correct_rep = self.remove_properties(correct_representation)
        check_rep = self.remove_properties(assertion)
        
        # Check for groups
        correct_groups, correct_rep = self.get_group_content(correct_rep)
        check_groups, check_rep = self.get_group_content(check_rep)
        
        self.assertEqual(collections.Counter(correct_groups), collections.Counter(check_groups), f"Groups mismatch for {assertion}")
            
        # Check reps are equal without props or groups
        self.assertEqual(correct_rep, check_rep, f"Base rep mismatch for {assertion}")
            
        # Check properties are equal
        args1 = self.get_properties(correct_representation)
        args2 = self.get_properties(assertion)
        self.assertEqual(collections.Counter(args1), collections.Counter(args2), f"Properties mismatch for {assertion}")

    def split_ignoring_nesting(self, s, sep=', '):
        result = []
        current = []
        parens = 0
        brackets = 0
        i = 0
        while i < len(s):
            if s[i] == '(': parens += 1
            elif s[i] == ')': parens -= 1
            elif s[i] == '[': brackets += 1
            elif s[i] == ']': brackets -= 1
            
            if s[i:i+len(sep)] == sep and parens == 0 and brackets == 0:
                result.append("".join(current))
                current = []
                i += len(sep)
                continue
                
            current.append(s[i])
            i += 1
        if current:
            result.append("".join(current))
        return result

    def get_properties(self, rep, new_args=None):
        properties = []
        
        def extract(s):
            stack = 0
            start = -1
            for i, char in enumerate(s):
                if char == '[':
                    if stack == 0:
                        start = i
                    stack += 1
                elif char == ']':
                    stack -= 1
                    if stack == 0 and start != -1:
                        content = s[start+1:i]
                        items = self.split_ignoring_nesting(content)
                        for item in items:
                            clean_item = self.remove_properties(item)
                            if clean_item.strip():
                                properties.append(clean_item.strip())
                            extract(item)
        
        extract(rep)
        # Normalize punct properties (e.g. punct_<dot>: -> punct: and punct_::: -> punct::)
        properties = [re.sub(r'^\(punct_[^:]*:', '(punct:', p) for p in properties]
        return properties

    def get_group_content(self, rep):
        groups = []
        group_types = ["and(", "or(", "neither("]
        
        def extract(s):
            i = 0
            result = []
            while i < len(s):
                found = False
                for g in group_types:
                    if s[i:i+len(g)] == g:
                        stack = 1
                        for j in range(i+len(g), len(s)):
                            if s[j] == '(': stack += 1
                            elif s[j] == ')': stack -= 1
                            if stack == 0:
                                content = s[i+len(g):j]
                                groups.append(g.strip('('))
                                items = self.split_ignoring_nesting(content)
                                groups.extend([self.remove_properties(item).strip() for item in items if self.remove_properties(item).strip()])
                                extract(content)
                                i = j + 1
                                found = True
                                break
                        if found: break
                if not found:
                    result.append(s[i])
                    i += 1
            return "".join(result)
            
        rep = extract(rep)
        return groups, rep

    def remove_properties(self, rep):
        result = []
        stack = 0
        for char in rep:
            if char == '[':
                stack += 1
            elif char == ']':
                stack -= 1
            elif stack == 0:
                result.append(char)
        return "".join(result)

    def replace_existential(self, line):
        return re.sub(r"\?\d+", "?", line.strip())


if __name__ == '__main__':
    unittest.main()