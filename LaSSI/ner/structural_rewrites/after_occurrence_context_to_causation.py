__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.string_functions import lemmatize_verb
from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    append_unique_property_value,
    property_values,
)
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class AfterOccurrenceContextToCausationRule(StructuralRewriteRule):
    """Promote occurrence contexts under service state changes to causation.

    `after X occurred` is parsed as a temporal subordinate event and the
    logical rewriter correctly preserves the event as `X[(type:occur)]`.
    For change-of-state predicates such as `close`, that event is normally the
    reason for the state change in transport notices ("closed after storm
    damage occurred"), not just background temporal context.

    The rule is intentionally narrow: it only moves TEMPORAL_CONTEXT entries
    whose rewritten event type is an occurrence verb, and only when the outer
    predicate is a configured change-of-state verb.  This keeps cases like
    `while maintenance work takes place` as TEMPORAL_CONTEXT."""

    name = "after_occurrence_context_to_causation"
    phase = "post_logical_rewrite"

    _OCCURRENCE_TYPES = frozenset({"occur", "happen"})

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        if not self._is_change_of_state_kernel(kernel, ctx):
            return None

        props = dict(kernel.properties)
        temporal_contexts = property_values(props, "TEMPORAL_CONTEXT")
        occurrence_contexts = [
            node for node in temporal_contexts
            if self._is_occurrence_context(node)
        ]
        if not occurrence_contexts:
            return None
        return {
            "props": props,
            "temporal_contexts": temporal_contexts,
            "occurrence_contexts": occurrence_contexts,
        }

    def apply(self, kernel, bindings, ctx):
        props = {
            key: list(value) if isinstance(value, (list, tuple)) else value
            for key, value in bindings["props"].items()
        }
        occurrence_ids = {
            getattr(node, "id", None) for node in bindings["occurrence_contexts"]
        }

        remaining_temporal = [
            node for node in bindings["temporal_contexts"]
            if getattr(node, "id", None) not in occurrence_ids
        ]
        if remaining_temporal:
            props["TEMPORAL_CONTEXT"] = remaining_temporal
        else:
            props.pop("TEMPORAL_CONTEXT", None)

        for node in bindings["occurrence_contexts"]:
            append_unique_property_value(props, "CAUSATION", node)
        return kernel.update_node_props(props)

    def _is_change_of_state_kernel(self, kernel, ctx):
        edge = kernel.kernel.edgeLabel
        if not isinstance(edge, Singleton) or not edge.named_entity:
            return False
        edge_lemma = lemmatize_verb(edge.named_entity).lower()
        change_verbs = {
            str(v).lower() for v in ctx.services.getHOnK().getChangeOfStateVerbs()
        }
        return edge_lemma in change_verbs

    def _is_occurrence_context(self, node):
        if not isinstance(node, Singleton):
            return False
        event_type = dict(node.properties).get("type")
        if not isinstance(event_type, str):
            return False
        return lemmatize_verb(event_type).lower() in self._OCCURRENCE_TYPES
