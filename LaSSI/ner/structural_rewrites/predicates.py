__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

"""Shared class / structure predicates over kernel nodes.

These are the small, reusable building blocks the declarative engine and the
structural/graph primitives both depend on: "does this node match a HOnK
class?", "which node sits in this slot?", "build an existential / an AND
wrapper". Class membership resolves through `KernelOntologyMatchers.matches_class`
(HOnK / raw_data) when a matcher is present, with light structural fallbacks for
the matcher-less unit-test context."""

from LaSSI.ner.structural_rewrites.base import as_list, is_date_like, is_location_like
from LaSSI.ner.structural_rewrites.config import structural_lexical_set
from LaSSI.ner.node_functions import create_props_for_singleton
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    SetOfSingletons,
    Singleton,
)


def kernel_slot(kernel, slot):
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return None
    if slot == "source":
        return kernel.kernel.source
    if slot == "target":
        return kernel.kernel.target
    if slot in {"edge", "edge_label", "edgeLabel"}:
        return kernel.kernel.edgeLabel
    return None


def edge_label(kernel):
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return None
    return kernel.kernel.edgeLabel


def matches_class(value, class_name, ctx, *, kernel=None):
    matcher = getattr(ctx, "matchers", None)
    if matcher is not None and hasattr(matcher, "matches_class"):
        return matcher.matches_class(value, class_name, kernel=kernel)
    if class_name == "LocationLike":
        return is_location_like(value, ctx)
    if class_name == "DateLike":
        return is_date_like(value)
    if class_name == "ContentNode":
        return is_content_node(value, ctx)
    if class_name == "ContextNode":
        return not is_content_node(value, ctx)
    if class_name == "LifecycleHeadPhrase":
        return matches_lifecycle_head_phrase(value)
    return False


def is_content_node(node, ctx):
    matcher = getattr(ctx, "matchers", None)
    if matcher is not None and hasattr(matcher, "is_content_node"):
        return matcher.is_content_node(node)
    if isinstance(node, SetOfSingletons):
        return bool(node.entities)
    if not isinstance(node, Singleton):
        return False
    context_types = structural_lexical_set("context_entity_types") or {
        "DATE", "TIME", "SUTime", "GPE", "LOC", "FAC", "existential",
    }
    return str(getattr(node, "type", "") or "") not in context_types


def matches_lifecycle_head_phrase(value):
    if not isinstance(value, Singleton) or not value.named_entity:
        return False
    try:
        from LaSSI.HOnK.TBox.LifecycleManager import _get_lifecycle_phrases
        phrases = _get_lifecycle_phrases()
    except Exception:
        phrases = {}
    return any(
        part != "descriptive"
        for _, part in phrases.get(value.named_entity.strip().lower(), set())
    )


def is_negated_lifecycle_value(node, ctx):
    if not (isinstance(node, SetOfSingletons) and node.type == Grouping.NOT and ctx is not None):
        return False
    for entity in node.entities:
        if matches_class(entity, "LifecycleHeadPhrase", ctx):
            return True
    return False


def is_promotable_target(node, ctx):
    if is_negated_lifecycle_value(node, ctx):
        return False
    if isinstance(node, SetOfSingletons):
        if node.type in {Grouping.AND, Grouping.OR}:
            if len(node.entities) >= 2:
                return True
            if len(node.entities) == 1:
                return is_promotable_target(node.entities[0], ctx)
            return False
        return True
    return is_content_node(node, ctx)


def extras_in(node):
    ids = set()
    if isinstance(node, SetOfSingletons):
        for entity in node.entities:
            ids.update(extras_in(entity))
        return ids
    if not isinstance(node, Singleton):
        return ids
    for item in as_list(dict(node.properties).get("extra")):
        if isinstance(item, (list, tuple)):
            for sub in item:
                sub_id = getattr(sub, "id", None)
                if sub_id is not None:
                    ids.add(sub_id)
        else:
            item_id = getattr(item, "id", None)
            if item_id is not None:
                ids.add(item_id)
    return ids


def wrap_entities(node_id, entities, group):
    mins = [getattr(c, "min", 0) for c in entities]
    maxs = [getattr(c, "max", 0) for c in entities]
    return SetOfSingletons(
        id=node_id,
        type=group,
        entities=tuple(entities),
        min=min(mins) if mins else 0,
        max=max(maxs) if maxs else 0,
        confidence=1.0,
    )


def create_existential(ctx, kernel):
    try:
        from LaSSI.ner.node_functions import create_existential_node
        return create_existential_node()
    except Exception:
        return Singleton(
            id=-(abs(getattr(kernel, "id", 0)) + 1),
            named_entity="?",
            properties=create_props_for_singleton({}),
            min=getattr(kernel, "min", -1),
            max=getattr(kernel, "max", -1),
            type="existential",
            confidence=1.0,
        )


def node_property_value_matches_class(node, property_name, class_name, ctx):
    if not isinstance(node, Singleton):
        return False
    value = dict(node.properties).get(property_name)
    if not isinstance(value, str):
        return False
    # The class (e.g. OccurrenceVerb) resolves through matches_class -> HOnK, so
    # vocabulary stays in the ontology / raw_data rather than this module.
    return matches_class(value, class_name, ctx)
