__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures import DependencyRoles
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    Relationship,
    SetOfSingletons,
    Singleton,
)
from LaSSI.structures.kernels.SentenceX import get_prepositions


class AndNmodExtraRule(StructuralRewriteRule):
    """Recover nominal-modifier complements omitted from AND entities.

    During kernel construction, a node that survives as an AND conjunct is
    counted as part of the kernel. Its ``nmod`` complement can then be skipped
    as already accounted for, even though the graph still has the dependency
    edge (for example ``chance -nmod-> precipitation[of]``). This rule reads
    that surviving graph structure and restores the complement as an ``extra``
    on the conjunct, matching the shape produced when the same ``nmod`` is
    rewritten through the normal property path.
    """

    name = "and_nmod_extra"
    phase = "post_logical_rewrite"

    _CONTEXT_TYPES = frozenset({
        "DATE", "TIME", "SUTime",
        "GPE", "LOC", "FAC",
        "existential",
    })

    @classmethod
    def _is_and_target(cls, kernel):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return False
        return isinstance(kernel.kernel.target, SetOfSingletons) and kernel.kernel.target.type == Grouping.AND

    @classmethod
    def _is_content_extra(cls, node):
        if not isinstance(node, Singleton):
            return False
        if str(getattr(node, "type", "") or "") in cls._CONTEXT_TYPES:
            return False
        return bool(get_prepositions(node))

    @staticmethod
    def _referenced_ids(node):
        ids = set()
        if isinstance(node, Singleton):
            ids.add(node.id)
            for value in dict(node.properties).values():
                ids.update(AndNmodExtraRule._referenced_ids(value))
            if node.kernel is not None:
                ids.update(AndNmodExtraRule._referenced_ids(node.kernel.source))
                ids.update(AndNmodExtraRule._referenced_ids(node.kernel.target))
                ids.update(AndNmodExtraRule._referenced_ids(node.kernel.edgeLabel))
        elif isinstance(node, SetOfSingletons):
            ids.add(node.id)
            for entity in node.entities:
                ids.update(AndNmodExtraRule._referenced_ids(entity))
        elif isinstance(node, (list, tuple)):
            for item in node:
                ids.update(AndNmodExtraRule._referenced_ids(item))
        return ids

    @staticmethod
    def _has_extra(props, candidate):
        extra = props.get("extra")
        if extra is None:
            return False
        items = extra if isinstance(extra, (list, tuple)) else [extra]
        return any(
            isinstance(item, Singleton) and (
                item.id == candidate.id or item.named_entity == candidate.named_entity
            )
            for item in items
        )

    @classmethod
    def _nmod_children(cls, entity, ctx):
        graph = getattr(getattr(ctx, "matchers", None), "G", None)
        if graph is None or not isinstance(entity, Singleton) or entity.id not in graph:
            return []

        children = []
        nominal_edges = DependencyRoles.nominal_modifier_edges()
        for _, child_id, data in graph.out_edges(entity.id, data=True):
            label = data.get("label")
            label_name = getattr(label, "named_entity", None)
            if label_name not in nominal_edges:
                continue
            child = graph.nodes[child_id].get("data") if child_id in graph else None
            if cls._is_content_extra(child):
                children.append(child)
        return children

    def matches(self, kernel, ctx):
        if not self._is_and_target(kernel):
            return None

        and_target = kernel.kernel.target
        target_ids = self._referenced_ids(and_target)
        actions = []
        for idx, entity in enumerate(and_target.entities):
            if not isinstance(entity, Singleton):
                continue
            props = dict(entity.properties)
            for child in self._nmod_children(entity, ctx):
                if child.id in target_ids or self._has_extra(props, child):
                    continue
                actions.append((idx, child))
        return {"actions": actions} if actions else None

    def apply(self, kernel, bindings, ctx):
        and_target = kernel.kernel.target
        entities = list(and_target.entities)

        for idx, child in bindings["actions"]:
            entity = entities[idx]
            props = dict(entity.properties)
            extras = props.get("extra")
            if extras is None:
                props["extra"] = [child]
            else:
                extras = list(extras) if isinstance(extras, (list, tuple)) else [extras]
                if not self._has_extra(props, child):
                    extras.append(child)
                props["extra"] = extras
            entities[idx] = entity.update_node_props(props)

        new_target = and_target.update_entities(entities)
        return Singleton(
            id=kernel.id,
            named_entity=kernel.named_entity,
            properties=kernel.properties,
            min=kernel.min,
            max=kernel.max,
            type=kernel.type,
            confidence=kernel.confidence,
            kernel=Relationship(
                source=kernel.kernel.source,
                target=new_target,
                edgeLabel=kernel.kernel.edgeLabel,
                isNegated=kernel.kernel.isNegated,
            ),
        )
