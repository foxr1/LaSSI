__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

import json
import os
from functools import lru_cache


_DEFAULT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "raw_data", "dependency_roles.json"
))


@lru_cache(maxsize=1)
def _load(path: str = _DEFAULT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: frozenset(v) if isinstance(v, list) else v for k, v in data.items()}


def get(group_name: str) -> frozenset:
    """Return the set of labels under `group_name` from dependency_roles.json.

    KeyError surfaces a missing group rather than silently returning empty —
    callers should declare the groups they depend on."""
    groups = _load()
    return groups[group_name]


def copula_complement_pos_tags() -> frozenset:
    return get("copula_complement_pos_tags")


def predicative_subject_pos_tags() -> frozenset:
    return get("predicative_subject_pos_tags")


def nominal_modifier_edges() -> frozenset:
    return get("nominal_modifier_edges")


def possessive_edges() -> frozenset:
    return get("possessive_edges")


def nominal_modifier_edges_no_poss() -> frozenset:
    return nominal_modifier_edges() - possessive_edges()


def preposition_marker_labels() -> frozenset:
    return get("preposition_marker_labels")


def lexical_merge_edges() -> frozenset:
    return get("lexical_merge_edges")


def space_case_marker_keys() -> frozenset:
    return get("space_case_marker_keys")


def verb_disqualifying_incoming_edges() -> frozenset:
    return get("verb_disqualifying_incoming_edges")
