__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    Relationship,
    SetOfSingletons,
    Singleton,
)


class SpecificationAndToTargetRule(StructuralRewriteRule):
    """Promote content sitting in a property bucket to the kernel's target
    slot when the target is currently an existential placeholder.

    Pattern:
      * Kernel target is an existential placeholder (e.g. ``?1``).
      * One or more promotable property buckets (SPECIFICATION / TOPIC /
        TOGETHERNESS) hold the kernel's true content. Multiple items are
        folded into a single AND group; a lone item is lifted as-is.

    Rewrite:
      * Drop the consumed property bucket(s).
      * Replace the existential target with the collected content (single
        node or AND-wrapped multi-node group).

    Origin: verbless nominal sentences like
    ``Met Office forecast for Newcastle upon Tyne: Light rain, 12.76°C, …``
    are built as ``be(noun_root, ?existential)`` by
    ``_build_implicit_be_kernel`` in ``SentenceX``.  The post-colon content
    lands in SPECIFICATION (often as an AND group, but sometimes a single
    head noun — e.g. ``be(forecast, ?)[SPECIFICATION:rain[persistent]]``
    for "Newcastle is forecast to have persistent heavy rain").
    Semantically the SPECIFICATION *is* what the kernel asserts, so it
    belongs in the target slot, not in a side bucket — regardless of
    whether one or many items are sitting there.

    Promotion is gated on the candidate being a genuine content node so a
    misrouted date or location (which the construct rules sometimes leave
    in SPECIFICATION before TimeCanonicalisation/SPACE folding rescue it)
    doesn't end up as the asserted target."""

    name = "specification_and_to_target"
    phase = "post_logical_rewrite"

    _PROMOTABLE_KEYS = ("SPECIFICATION", "TOPIC", "TOGETHERNESS")
    _CONJUNCTION_TYPES = frozenset({Grouping.AND, Grouping.OR})
    _NON_CONTENT_TYPES = frozenset({
        "existential",
        "DATE", "TIME", "SUTime",
        "GPE", "LOC", "FAC",
    })

    @classmethod
    def _is_promotable_target(cls, node):
        # AND/OR conjunctions with ≥2 entities are always content; a
        # singleton AND/OR is degenerate so unwrap-and-recheck.
        if isinstance(node, SetOfSingletons):
            if node.type in cls._CONJUNCTION_TYPES:
                if len(node.entities) >= 2:
                    return True
                if len(node.entities) == 1:
                    return cls._is_promotable_target(node.entities[0])
                return False
            return True
        if isinstance(node, Singleton):
            node_type = str(getattr(node, "type", "") or "")
            if node_type in cls._NON_CONTENT_TYPES:
                return False
            return True
        return False

    @staticmethod
    def _extras_in(node):
        # Collect ids referenced via the `extra` property chain (including
        # nested NOT/AND wrappers) so we can drop bare duplicates from the
        # outer conjunction.
        ids = set()
        if isinstance(node, SetOfSingletons):
            for entity in node.entities:
                ids.update(SpecificationAndToTargetRule._extras_in(entity))
            return ids
        if not isinstance(node, Singleton):
            return ids
        props = dict(node.properties)
        extras = props.get('extra')
        if extras is None:
            return ids
        if isinstance(extras, (list, tuple)):
            extras_iter = extras
        else:
            extras_iter = [extras]
        for item in extras_iter:
            if isinstance(item, (list, tuple)):
                for sub in item:
                    sub_id = getattr(sub, 'id', None)
                    if sub_id is not None:
                        ids.add(sub_id)
            else:
                item_id = getattr(item, 'id', None)
                if item_id is not None:
                    ids.add(item_id)
        return ids

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        target = kernel.kernel.target
        if target is None or not (
            isinstance(target, Singleton) and target.type == 'existential'
        ):
            return None

        props = dict(kernel.properties)
        # Collect every candidate across the promotable buckets so a
        # SPECIFICATION+TOGETHERNESS pair (sentence 3) can be folded into
        # one AND target rather than left as two separate buckets.
        collected = []
        consumed_keys = []
        for key in self._PROMOTABLE_KEYS:
            value = props.get(key)
            if value is None:
                continue
            candidates = list(value) if isinstance(value, (list, tuple)) else [value]
            for item in candidates:
                collected.append(item)
            consumed_keys.append(key)

        if not collected:
            return None

        # Single candidate: promote any genuine content node (conjunction
        # or otherwise). Dates / locations / existentials are excluded —
        # they belong in TIME / SPACE / nowhere, not in the target slot.
        if len(collected) == 1:
            only = collected[0]
            if self._is_promotable_target(only):
                return {"property_keys": consumed_keys, "conjunction": only}
            return None

        # Dedupe by id and drop items already referenced via `extra` on a
        # sibling — the GSM rules sometimes emit both `chance[extra:X]` and
        # a bare `X` for the same of-clause.
        seen_ids = set()
        extras_seen = set()
        unique = []
        for item in collected:
            cand_id = getattr(item, 'id', None)
            if cand_id is not None and cand_id in seen_ids:
                continue
            if cand_id is not None:
                seen_ids.add(cand_id)
            extras_seen.update(self._extras_in(item))
            unique.append(item)
        unique = [
            item for item in unique
            if getattr(item, 'id', None) is None
            or getattr(item, 'id', None) not in extras_seen
        ]
        if len(unique) == 0:
            return None
        if len(unique) == 1:
            # After dedupe a single item remains (typically because its
            # sibling was already nested inside its `extra`). Promote it
            # to the target slot directly — no AND wrap needed.
            if not self._is_promotable_target(unique[0]):
                return None
            return {"property_keys": consumed_keys, "conjunction": unique[0]}
        mins = [getattr(c, 'min', 0) for c in unique]
        maxs = [getattr(c, 'max', 0) for c in unique]
        wrapped = SetOfSingletons(
            id=kernel.id,
            type=Grouping.AND,
            entities=tuple(unique),
            min=min(mins) if mins else 0,
            max=max(maxs) if maxs else 0,
            confidence=1.0,
        )
        return {"property_keys": consumed_keys, "conjunction": wrapped}

    def apply(self, kernel, bindings, ctx):
        new_props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }
        for key in bindings["property_keys"]:
            new_props.pop(key, None)
        new_kernel_relation = Relationship(
            source=kernel.kernel.source,
            target=bindings["conjunction"],
            edgeLabel=kernel.kernel.edgeLabel,
            isNegated=kernel.kernel.isNegated,
        )
        return Singleton(
            id=kernel.id,
            named_entity=kernel.named_entity,
            properties=kernel.properties,
            min=kernel.min,
            max=kernel.max,
            type=kernel.type,
            confidence=kernel.confidence,
            kernel=new_kernel_relation,
        ).update_node_props(new_props)
