__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import (
    SetOfSingletons,
    Singleton,
)


class LifecycleSubjectPromotionRule(StructuralRewriteRule):
    """Promote a lifecycle-domain noun appearing as an ``extra`` to the
    head position, swapping it with whatever metric / modifier noun is
    currently the head.

    Anchored on ``LifecycleStates.ttl``: the set of "domain head nouns"
    is exactly the assertive (non-descriptive) lifecycle phrases that
    the comparator's partition logic already treats as actual events
    (``rain`` / ``precipitation`` / ``clear skies`` / ``cloudy`` / …).
    No hardcoded weather vocabulary — adding a new partition phrase
    to ``LifecycleStates.ttl`` extends this rule automatically.

    Structural rewrite (illustrated for ``chance of precipitation``):

        chance[(extra:precipitation), (amod:better-than-even)]
                       ↓
        precipitation[(extra:chance), (amod:better-than-even)]

    Why: a forecast's "chance of X" sentence is structurally about X
    (the actual event), with ``chance`` acting as a metric on how
    likely X is.  Promoting the lifecycle noun to the head means
    downstream head-noun matching aligns surfaces that put the same
    event in different syntactic positions ("57% chance of
    precipitation" vs "better-than-even chance of precipitation" vs
    a bare "precipitation").  The swap is name-only — every other
    property (``cop`` / ``amod`` / ``nummod`` magnitude markers,
    positional dependency keys, etc.) stays where it was, since
    those modifiers were originally describing the metric or the
    enclosing predicate, not the head noun.

    The rule skips:
    - Singletons whose own head is already a lifecycle noun (nothing
      to promote — a symmetric swap would loop).
    - Extras that are not Singletons (lists or SetOfSingletons of
      non-lifecycle items are left untouched).
    - Cases where the head's named_entity is empty (e.g. structural
      wrapper Singletons).
    """

    name = "lifecycle_subject_promotion"
    phase = "post_logical_rewrite"

    _LIFECYCLE_HEAD_LEMMAS_CACHE = None

    @classmethod
    def _lifecycle_head_lemmas(cls):
        if cls._LIFECYCLE_HEAD_LEMMAS_CACHE is None:
            try:
                from LaSSI.HOnK.TBox.LifecycleManager import _get_lifecycle_phrases
                phrases = _get_lifecycle_phrases()
                # Exclude the `descriptive` partition — event-facet verbs
                # like 'involve' / 'mention' aren't head-noun candidates
                # and shouldn't trigger a structural promotion.
                cls._LIFECYCLE_HEAD_LEMMAS_CACHE = {
                    label for label, (_, part) in phrases.items()
                    if part != 'descriptive'
                }
            except Exception:
                cls._LIFECYCLE_HEAD_LEMMAS_CACHE = set()
        return cls._LIFECYCLE_HEAD_LEMMAS_CACHE

    def matches(self, kernel, ctx):
        # Pattern-walker: does any Singleton reachable from `kernel`
        # have `extra` pointing at a lifecycle head noun while its own
        # head is *not* one?
        return {"_": True} if self._has_candidate(kernel) else None

    def apply(self, kernel, bindings, ctx):
        return self._rewrite(kernel)

    @classmethod
    def _has_candidate(cls, node):
        if isinstance(node, SetOfSingletons):
            return any(cls._has_candidate(e) for e in node.entities)
        if not isinstance(node, Singleton):
            return False
        if node.kernel is not None:
            if cls._has_candidate(node.kernel.source):
                return True
            if cls._has_candidate(node.kernel.target):
                return True
        if cls._is_candidate(node):
            return True
        for _, v in dict(node.properties).items():
            vs = v if isinstance(v, (list, tuple)) else [v]
            for item in vs:
                if cls._has_candidate(item):
                    return True
        return False

    @classmethod
    def _is_candidate(cls, node):
        if not isinstance(node, Singleton):
            return False
        if not node.named_entity:
            return False
        if node.named_entity.strip().lower() in cls._lifecycle_head_lemmas():
            return False
        extra_value = dict(node.properties).get('extra')
        if extra_value is None:
            return False
        extras = extra_value if isinstance(extra_value, (list, tuple)) else [extra_value]
        for e in extras:
            if (isinstance(e, Singleton)
                    and e.named_entity
                    and e.named_entity.strip().lower() in cls._lifecycle_head_lemmas()):
                return True
        return False

    @classmethod
    def _rewrite(cls, node):
        if isinstance(node, SetOfSingletons):
            new_entities = [cls._rewrite(e) for e in node.entities]
            if any(a is not b for a, b in zip(node.entities, new_entities)):
                node = node.update_entities(new_entities)
            return node
        if not isinstance(node, Singleton):
            return node
        new_node = node
        if node.kernel is not None:
            if node.kernel.source is not None:
                rs = cls._rewrite(node.kernel.source)
                if rs is not node.kernel.source:
                    new_node = new_node.update_kernel(rs, 'source')
            if node.kernel.target is not None:
                rt = cls._rewrite(node.kernel.target)
                if rt is not node.kernel.target:
                    new_node = new_node.update_kernel(rt, 'target')

        # Recurse into properties first so deeply nested candidates
        # get promoted before their containing node is considered.
        props = dict(new_node.properties)
        changed = False
        for k, v in list(props.items()):
            if isinstance(v, (list, tuple)):
                new_items = [cls._rewrite(item) for item in v]
                if any(a is not b for a, b in zip(v, new_items)):
                    props[k] = type(v)(new_items)
                    changed = True
            else:
                nv = cls._rewrite(v)
                if nv is not v:
                    props[k] = nv
                    changed = True
        if changed:
            new_node = new_node.update_node_props(props)

        if cls._is_candidate(new_node):
            new_node = cls._swap_head_and_lifecycle_extra(new_node)
        return new_node

    @classmethod
    def _swap_head_and_lifecycle_extra(cls, node):
        """Swap `node.named_entity` with the first lifecycle-head-noun
        entry inside `node.properties['extra']`.  Everything else is
        preserved: positional keys, amod / cop / nummod magnitude
        markers, and the other extras (if any) are left in place.

        The lifecycle entry's own `named_entity` becomes the new head;
        the original head name takes the lifecycle entry's slot inside
        `extra` (by renaming the Singleton, so its `id` and positional
        info survive). This keeps both the head's and the extra's
        upstream provenance intact while inverting the surface
        relationship.
        """
        props_dict = dict(node.properties)
        extra_value = props_dict.get('extra')
        if extra_value is None:
            return node
        is_list = isinstance(extra_value, list)
        is_tuple = isinstance(extra_value, tuple)
        extras = list(extra_value) if (is_list or is_tuple) else [extra_value]

        lemmas = cls._lifecycle_head_lemmas()
        target_idx = None
        for i, e in enumerate(extras):
            if (isinstance(e, Singleton)
                    and e.named_entity
                    and e.named_entity.strip().lower() in lemmas):
                target_idx = i
                break
        if target_idx is None:
            return node

        promoted = extras[target_idx]
        promoted_props = dict(promoted.properties)
        relation_props = cls._relation_props(promoted_props)

        # The promoted Singleton stays in `extra` but takes the
        # original head's name (so the original head identity survives
        # as the metric of the new head).
        metric_props = {
            key: value for key, value in promoted_props.items()
            if key not in relation_props
        }
        extras[target_idx] = (
            promoted
            .update_name(node.named_entity)
            .update_node_props(metric_props)
        )
        if is_list:
            props_dict['extra'] = extras
        elif is_tuple:
            props_dict['extra'] = tuple(extras)
        else:
            props_dict['extra'] = extras[0]
        props_dict.update(relation_props)
        new_node = node.update_node_props(props_dict).update_type(promoted.type)
        return new_node.update_name(promoted.named_entity)

    @staticmethod
    def _relation_props(props):
        """Properties like ``21.000000: of`` encode the dependency edge that
        introduced the lifecycle extra.  After promotion that relation belongs
        on the promoted semantic head, while the old head becomes the metric
        in ``extra``."""
        relation_props = {}
        for key, value in props.items():
            try:
                float(str(key))
            except (TypeError, ValueError):
                continue
            relation_props[key] = value
        return relation_props
