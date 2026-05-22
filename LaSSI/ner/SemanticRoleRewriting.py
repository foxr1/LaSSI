"""Shared semantic-role rule matching for graph and final-kernel rewriting.

This module sits between structural graph construction and final logical
assembly: it interprets HOnK logical-rule premises once, then lets each caller
decide how to preserve the matched role in its own representation.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import networkx as nx

from LaSSI.structures.internal_graph.EntityRelationship import Singleton


DEPENDENCY_CONDITIONS = {"Dependency", "DependencyLabel"}
TARGET_TYPE_CONDITIONS = {"SingletonHasBeenMatchedBy", "TargetSingletonHasBeenMatchedBy"}
SOURCE_TYPE_CONDITIONS = {"SourceSingletonHasBeenMatchedBy"}


@dataclass(frozen=True)
class LogicalClassification:
    construct_name: str
    construct_property: Optional[str]


def normalized_node_type(node, honk=None):
    if node is None:
        return "None"
    if node.type in ("DATE", "TIME"):
        return "SUTime"
    if (
            honk is not None and
            node.type == "IN" and
            bool(set(node.named_entity.lower().split()) & {tn.lower() for tn in honk.getTemporalNouns()})
    ):
        return "SUTime"
    if node.type in {"GPE", "LOC", "FAC", "SUTime", "RB", "IN", "verb"}:
        return str(node.type)
    return "None"


def rule_classifications(rule):
    classifications = [LogicalClassification(rule.logicalConstructName, rule.logicalConstructProperty)]
    classifications.extend(
        LogicalClassification(name, prop)
        for name, prop in (rule.additional_classifications or [])
    )
    return [classification for classification in classifications if classification.construct_name is not None]


def rule_has_condition(rule, condition_names):
    return any(condition.name in condition_names for condition in rule.premises)


def _condition_matches(condition, predicate_matches: Callable[[str, object], bool]):
    return any(predicate_matches(condition.name, value) for value in condition.values)


def rule_matches(rule, predicate_matches: Callable[[str, object], bool]):
    return (
        all(_condition_matches(condition, predicate_matches) for condition in rule.premises)
        and all(not _condition_matches(condition, predicate_matches) for condition in rule.not_premises)
    )


def _rule_preposition_specificity(rule):
    for premise in rule.premises:
        if premise.name == "Preposition":
            return max((len(value) for value in premise.values), default=0)
    return 0


def _rule_condition_specificity(rule):
    return sum(len(condition.values) for condition in rule.premises)


def select_best_matching_rule(
        rules: Iterable,
        predicate_matches: Callable[[str, object], bool],
        rule_filter: Optional[Callable[[object], bool]] = None,
):
    matching_rules = [
        rule for rule in rules
        if (rule_filter is None or rule_filter(rule)) and rule_matches(rule, predicate_matches)
    ]
    if not matching_rules:
        return None
    return max(matching_rules, key=lambda rule: (
        _rule_preposition_specificity(rule),
        _rule_condition_specificity(rule),
    ))


def dependency_rule_predicate(source, target, edge_label, honk):
    def _matches(condition_name, value):
        if condition_name in DEPENDENCY_CONDITIONS:
            return isinstance(edge_label, Singleton) and edge_label.named_entity == value
        if condition_name in SOURCE_TYPE_CONDITIONS:
            return normalized_node_type(source, honk) == value
        if condition_name in TARGET_TYPE_CONDITIONS:
            return normalized_node_type(target, honk) == value
        return False

    return _matches


def select_dependency_role_rule(honk, source, target, edge_label):
    return select_best_matching_rule(
        honk.getLogicalRewritingRules().values(),
        dependency_rule_predicate(source, target, edge_label, honk),
        lambda rule: rule_has_condition(rule, DEPENDENCY_CONDITIONS),
    )


class DependencyRoleRewriter:
    """Preserve dependency-labelled semantic roles before graph contraction."""

    def __init__(self, honk):
        self.honk = honk

    def preserve_dependency_role(self, graph: nx.MultiDiGraph, edge):
        source = graph.nodes[edge[0]]['data']
        target = graph.nodes[edge[1]]['data']
        edge_label = edge[2]['label']
        if not (
                isinstance(source, Singleton) and
                isinstance(target, Singleton) and
                isinstance(edge_label, Singleton)
        ):
            return False

        rule = select_dependency_role_rule(self.honk, source, target, edge_label)
        if rule is None:
            return False

        source_props = dict(source.properties)
        for classification in rule_classifications(rule):
            property_key = classification.construct_name.upper()
            target_props = dict(target.properties)
            if classification.construct_property is not None:
                target_props["type"] = classification.construct_property
            rewritten_target = target.update_node_props(target_props)
            existing_values = list(source_props.get(property_key, []))
            existing_ids = {value.id for value in existing_values if isinstance(value, Singleton)}
            if rewritten_target.id not in existing_ids:
                existing_values.append(rewritten_target)
            source_props[property_key] = existing_values

        nx.set_node_attributes(graph, {edge[0]: source.update_node_props(source_props)}, 'data')
        return True
