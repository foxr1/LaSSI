__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    SetOfSingletons,
    Singleton,
)


class FlattenNestedAndRule(StructuralRewriteRule):
    """Inline ``AND(X, AND(Y, Z))`` into ``AND(X, Y, Z)``.

    The appositive merging in ``NodeMerger`` can produce a nested AND when
    only a subset of the comma-separated items end up sharing a single
    contraction step (e.g. weather_003 sentence 1's "Cloudy, 13.35°C, wind
    3.64 mph, 16% chance" yields ``AND(wind, AND(mph, % chance))`` because
    ``wind`` was lifted out of the ``mph``-compound and rejoined after the
    appositive contraction had already wrapped ``mph + % chance`` in an
    AND).  Same Grouping nested under itself never carries extra meaning,
    so collapse it.

    Applies recursively through:
      * Kernel source / target.
      * Kernel SetOfSingletons entities.
      * Each value in the top-level property dict (e.g. SPECIFICATION /
        SPACE / TIME entries holding sets).

    Only flattens when the parent and child have the *same* Grouping type
    (AND under AND, OR under OR) — mixing types (AND of ORs, NOT wraps,
    GROUPING containers) is left alone because the nesting is meaningful
    there."""

    name = "flatten_nested_and"
    phase = "post_logical_rewrite"

    _FLATTENABLE = frozenset({Grouping.AND, Grouping.OR})

    @classmethod
    def _flatten(cls, node):
        if isinstance(node, SetOfSingletons):
            new_entities = []
            changed = False
            for entity in node.entities:
                flat_entity = cls._flatten(entity)
                if (
                    isinstance(flat_entity, SetOfSingletons)
                    and flat_entity.type == node.type
                    and node.type in cls._FLATTENABLE
                ):
                    new_entities.extend(flat_entity.entities)
                    changed = True
                else:
                    if flat_entity is not entity:
                        changed = True
                    new_entities.append(flat_entity)
            if changed:
                return node.update_entities(new_entities)
            return node
        if isinstance(node, Singleton):
            new_node = node
            if node.kernel is not None:
                src = cls._flatten(node.kernel.source) if node.kernel.source is not None else None
                tgt = cls._flatten(node.kernel.target) if node.kernel.target is not None else None
                if src is not node.kernel.source:
                    new_node = new_node.update_kernel(src, 'source')
                if tgt is not node.kernel.target:
                    new_node = new_node.update_kernel(tgt, 'target')
            props = dict(new_node.properties) if new_node.properties else {}
            changed_props = False
            for key, value in list(props.items()):
                if isinstance(value, (list, tuple)):
                    new_value = [cls._flatten(item) for item in value]
                    if any(a is not b for a, b in zip(new_value, value)):
                        props[key] = new_value
                        changed_props = True
                else:
                    new_value = cls._flatten(value)
                    if new_value is not value:
                        props[key] = new_value
                        changed_props = True
            if changed_props:
                new_node = new_node.update_node_props(props)
            return new_node
        return node

    def matches(self, kernel, ctx):
        # Apply unconditionally; the recursive walker is a no-op when there
        # is nothing to flatten and returns the same object.
        return {}

    def apply(self, kernel, bindings, ctx):
        return self._flatten(kernel)
