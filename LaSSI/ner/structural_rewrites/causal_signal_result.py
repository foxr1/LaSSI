__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

"""Recover binary cause/effect structure for causal-signal result clauses.

Sentences like "the offence ... resulted in a suspect being arrested and
charged" can arrive as a targetless kernel with every participant flattened
into CAUSATION.  The causal signal itself is binary: the offence is the cause,
and the suspect/status clause is the effect.  This rule reconstructs those two
slots while keeping ordinary causation properties unchanged.
"""

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    as_list,
    copy_props,
    replace_kernel,
)
from LaSSI.ner.string_functions import is_position_key
from LaSSI.ner.structural_rewrites.predicates import matches_class, wrap_entities
from LaSSI.structures import DependencyRoles
from LaSSI.structures.internal_graph.EntityRelationship import Grouping, SetOfSingletons, Singleton


class CausalSignalResultRule(StructuralRewriteRule):
    name = "causal_signal_result"
    phase = "post_cleanup"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        edge = kernel.kernel.edgeLabel
        if not matches_class(edge, "CausalSignalVerb", ctx, kernel=kernel):
            return None
        if not self._is_empty_argument(kernel.kernel.source):
            return None
        if not self._is_empty_argument(kernel.kernel.target):
            return None

        props = dict(kernel.properties)
        causation = [v for v in as_list(props.get("CAUSATION")) if isinstance(v, Singleton)]
        if not causation:
            return None

        effect_head = self._effect_head(causation, ctx)
        if effect_head is None:
            return None
        actions = [v for v in causation if self._is_action_verb(v)]
        cause_nodes = [
            v for v in causation
            if v is not effect_head
            and v not in actions
            and self._is_cause_content(v, ctx)
        ]
        cause_nodes = self._fold_classifier_heads(cause_nodes, ctx, kernel)
        if not cause_nodes or not actions:
            return None
        return {
            "cause_nodes": cause_nodes,
            "effect_head": effect_head,
            "actions": actions,
        }

    def apply(self, kernel, bindings, ctx):
        props = copy_props(kernel)
        props.pop("CAUSATION", None)

        cause_nodes = self._sort_nodes(bindings["cause_nodes"])
        if len(cause_nodes) == 1:
            cause = cause_nodes[0]
        else:
            cause = wrap_entities(kernel.id, cause_nodes, Grouping.AND)

        effect = self._build_effect(bindings["effect_head"], bindings["actions"])
        return replace_kernel(kernel, source=cause, target=effect).update_node_props(props)

    @staticmethod
    def _is_empty_argument(value):
        return (
            value is None
            or (isinstance(value, Singleton) and value.type == "existential")
        )

    @staticmethod
    def _is_action_verb(value):
        return isinstance(value, Singleton) and str(value.type or "").lower() == "verb"

    def _effect_head(self, values, ctx):
        status_heads = [
            v for v in values
            if isinstance(v, Singleton) and matches_class(v, "StatusNoun", ctx)
            and not self._is_action_verb(v)
        ]
        if not status_heads:
            return None
        return self._sort_nodes(status_heads)[0]

    def _is_cause_content(self, value, ctx):
        if not isinstance(value, Singleton):
            return False
        if self._is_action_verb(value):
            return False
        if matches_class(value, "StatusNoun", ctx):
            return False
        return matches_class(value, "ContentNode", ctx)

    def _fold_classifier_heads(self, nodes, ctx, kernel):
        nodes = list(nodes)
        remove_indices = set()
        for head_idx, head in enumerate(nodes):
            if not self._is_classifier_head_surface(head, ctx, kernel):
                continue
            child_idx = self._compound_child_index(head, nodes, head_idx, ctx)
            if child_idx is None:
                continue
            child = nodes[child_idx]
            child_props = copy_props(child)
            extras = as_list(child_props.get("extra"))
            if not any(getattr(e, "id", None) == head.id for e in extras):
                extras.append(head)
            child_props["extra"] = extras
            nodes[child_idx] = child.update_node_props(child_props)
            remove_indices.add(head_idx)
        return [node for idx, node in enumerate(nodes) if idx not in remove_indices]

    @staticmethod
    def _is_classifier_head_surface(node, ctx, kernel):
        if not isinstance(node, Singleton):
            return False
        surface_only = node.update_node_props({})
        return matches_class(surface_only, "EventClassifierHeadNoun", ctx, kernel=kernel)

    @staticmethod
    def _compound_child_index(head, nodes, head_idx, ctx):
        graph = getattr(getattr(ctx, "matchers", None), "G", None)
        if graph is not None:
            for child_idx, child in enumerate(nodes):
                if child_idx == head_idx or not isinstance(child, Singleton):
                    continue
                if head.id not in graph or child.id not in graph:
                    continue
                for _src, dst, data in graph.out_edges(head.id, data=True):
                    label = data.get("label")
                    if dst == child.id and getattr(label, "named_entity", None) == "compound":
                        return child_idx
                for _src, dst, data in graph.out_edges(child.id, data=True):
                    label = data.get("label")
                    if dst == head.id and getattr(label, "named_entity", None) == "compound":
                        return child_idx
        for child_idx in range(head_idx + 1, len(nodes)):
            child = nodes[child_idx]
            if isinstance(child, Singleton):
                return child_idx
        for child_idx in range(head_idx - 1, -1, -1):
            child = nodes[child_idx]
            if isinstance(child, Singleton):
                return child_idx
        return None

    def _build_effect(self, head, actions):
        props = {
            k: v for k, v in copy_props(head).items()
            if k not in DependencyRoles.preposition_marker_labels()
            and not self._is_position_key(k)
        }
        actions = self._sort_nodes(actions)
        actioned = as_list(props.get("actioned"))
        extras = as_list(props.get("extra"))

        main_action = actions[0]
        for action in actions:
            action_props = dict(action.properties)
            if action_props.get("actioned"):
                main_action = action
                break

        for value in as_list(dict(main_action.properties).get("actioned")):
            if value not in actioned:
                actioned.append(value)
        if main_action.named_entity and main_action.named_entity not in actioned:
            actioned.append(main_action.named_entity)

        for action in actions:
            if action is main_action:
                continue
            if not any(getattr(existing, "id", None) == action.id for existing in extras):
                extras.append(action)

        if actioned:
            props["actioned"] = actioned
        if extras:
            props["extra"] = extras
        return head.update_node_props(props)

    @staticmethod
    def _is_position_key(key):
        return isinstance(key, str) and is_position_key(key)

    @classmethod
    def _sort_nodes(cls, nodes):
        return sorted(nodes, key=cls._node_pos)

    @staticmethod
    def _node_pos(node):
        if not isinstance(node, Singleton):
            return float("inf")
        raw = dict(node.properties).get("pos")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(getattr(node, "min", 0) or 0)
