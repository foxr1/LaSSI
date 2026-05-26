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


class QuantityCompoundMergeRule(StructuralRewriteRule):
    """Reroute quantified nouns sitting in a kernel's ``QUANTITY`` bucket
    into the AND target where they semantically belong.

    Why this exists: a verbless nominal whose colon-list is led by a
    non-noun (e.g. "Met Office forecast for X: **Cloudy**, 12.17°C, wind
    5.5 mph, 9% chance of precipitation.") doesn't get an appos chain
    from Stanford the way a noun-led list does ("Light rain, 12.76°C,
    ..."), because the JJ "Cloudy" can't be an appos anchor.  The graph
    grammar then bundles the colon-list under a ``multipleindobj`` whose
    quantified items (``%``, ``°C``) end up in the predicate-level
    ``QUANTITY`` property rather than as siblings of the AND target.
    Semantically they ARE siblings (each colon-item is a fact in the
    same forecast), and any one that immediately precedes a bare head
    noun in the AND group is a compound modifier of that head
    (``9% chance``).

    Two cases, picked per QUANTITY entry on purely structural grounds:

      * **Compound merge**: the quantified noun ``Q`` sits at the
        position immediately preceding an AND-group head ``H`` (``Q.pos
        + 1 == H.pos``), ``H`` is a bare noun without its own ``nummod``
        / ``measurement``, and ``Q`` carries a ``nummod``.  Rename ``H``
        to ``"{Q.name} {H.name}"`` and copy ``Q.nummod`` onto it; widen
        the surface span.  Everything else on ``H`` — including any
        ``extra`` populated by the earlier nmod-to-extra fold (e.g.
        ``chance[extra:precipitation]``) — is preserved.

      * **Lift to sibling**: the quantified noun has no adjacent AND
        head (``°C`` sits between commas with the closest AND noun two
        positions away).  Add ``Q`` to the AND entities unchanged.

    In both cases the consumed entry is dropped from ``QUANTITY``.

    No lemma matching — adjacency, ``nummod`` presence, and quantified
    surface shape are the only signals.  Compatible with the existing
    ``LifecycleSubjectPromotionRule`` downstream: once
    ``chance[extra:precipitation]`` becomes ``% chance[nummod:N,
    extra:precipitation]``, the lifecycle promotion swaps it to
    ``precipitation[extra:% chance, nummod:N]`` exactly as it would for
    a parser-clean appos-anchored sentence."""

    name = "quantity_compound_merge"
    phase = "post_logical_rewrite"

    _NON_QUANTITY_TYPES = frozenset({
        "DATE", "TIME", "SUTime",
        "GPE", "LOC", "FAC",
        "existential",
    })

    @staticmethod
    def _pos_of(node):
        if not isinstance(node, Singleton):
            return None
        raw = dict(node.properties).get('pos')
        if raw is None:
            return None
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_quantified_noun(node):
        if not isinstance(node, Singleton):
            return False
        if 'nummod' not in dict(node.properties):
            return False
        node_type = str(getattr(node, 'type', '') or '')
        if node_type in QuantityCompoundMergeRule._NON_QUANTITY_TYPES:
            return False
        return True

    @staticmethod
    def _and_target(kernel):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        tgt = kernel.kernel.target
        if isinstance(tgt, SetOfSingletons) and tgt.type == Grouping.AND:
            return tgt
        return None

    def matches(self, kernel, ctx):
        and_target = self._and_target(kernel)
        if and_target is None:
            return None
        quantity = dict(kernel.properties).get('QUANTITY')
        if quantity is None:
            return None
        items = list(quantity) if isinstance(quantity, (list, tuple)) else [quantity]
        candidates = [item for item in items if self._is_quantified_noun(item)]
        if not candidates:
            return None

        and_entities = list(and_target.entities)
        actions = []
        for q in candidates:
            q_pos = self._pos_of(q)
            if q_pos is None:
                continue
            head_idx = None
            for idx, ent in enumerate(and_entities):
                if not isinstance(ent, Singleton):
                    continue
                if getattr(ent, 'type', None) != 'noun':
                    continue
                ent_props = dict(ent.properties)
                if 'nummod' in ent_props or 'measurement' in ent_props:
                    continue
                ent_pos = self._pos_of(ent)
                if ent_pos is None:
                    continue
                if q_pos + 1 == ent_pos:
                    head_idx = idx
                    break
            if head_idx is not None:
                actions.append({'q': q, 'kind': 'merge', 'head_idx': head_idx})
            else:
                actions.append({'q': q, 'kind': 'lift'})
        if not actions:
            return None
        return {'actions': actions}

    def apply(self, kernel, bindings, ctx):
        and_target = self._and_target(kernel)
        and_entities = list(and_target.entities)
        consumed_q_ids = set()

        for action in bindings['actions']:
            q = action['q']
            consumed_q_ids.add(id(q))
            if action['kind'] == 'merge':
                head_idx = action['head_idx']
                head = and_entities[head_idx]
                head_props = dict(head.properties)
                q_props = dict(q.properties)
                if 'nummod' in q_props:
                    head_props['nummod'] = q_props['nummod']
                new_name = f"{q.named_entity} {head.named_entity}".strip()
                merged_props_node = head.update_name(new_name).update_node_props(head_props)
                merged = Singleton(
                    id=merged_props_node.id,
                    named_entity=merged_props_node.named_entity,
                    properties=merged_props_node.properties,
                    min=min(head.min, q.min),
                    max=max(head.max, q.max),
                    type=merged_props_node.type,
                    confidence=merged_props_node.confidence,
                    kernel=merged_props_node.kernel,
                )
                and_entities[head_idx] = merged
            else:  # 'lift'
                and_entities.append(q)

        new_and = and_target.update_entities(and_entities)

        new_props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }
        quantity = new_props.get('QUANTITY')
        items = (
            list(quantity)
            if isinstance(quantity, (list, tuple))
            else ([quantity] if quantity is not None else [])
        )
        remaining = [it for it in items if id(it) not in consumed_q_ids]
        if remaining:
            new_props['QUANTITY'] = remaining
        else:
            new_props.pop('QUANTITY', None)

        new_relation = Relationship(
            source=kernel.kernel.source,
            target=new_and,
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
            kernel=new_relation,
        ).update_node_props(new_props)
