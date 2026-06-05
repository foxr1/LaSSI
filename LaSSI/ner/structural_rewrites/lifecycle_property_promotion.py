__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

import re

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    append_unique_property_value,
    property_values,
)
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    SetOfSingletons,
    Singleton,
)


class LifecyclePropertyPromotionRule(StructuralRewriteRule):
    """Preserve lifecycle/status facts stranded in syntactic properties.

    Some short label-value fragments, such as header clauses, can surface as a
    targetless predicate with raw syntactic buckets (`noun`, `nmod`, ...).
    Those buckets are later discarded by the semantic-key filter unless their
    contents have been promoted first.  This rule promotes only ontology-backed
    content:

    * HOnK `StatusNoun` values are retained as `TIME_STATUS`.
    * Negated values whose head belongs to `LifecycleStates.ttl` are retained as
      `SPECIFICATION`.

    The match is intentionally structural and vocabulary-driven: adding/removing
    lifecycle phrases or status nouns changes the behaviour without editing
    this rule.
    """

    name = "lifecycle_property_promotion"
    phase = "post_hoist_pre_dedupe"

    _SEMANTIC_TARGET_KEYS = frozenset({"TIME_STATUS", "SPECIFICATION"})
    _RAW_SYNTACTIC_KEYS = frozenset({
        "noun",
        "verb",
        "nmod",
        "obl",
        "dobj",
        "obj",
        "iobj",
        "dep",
        "xcomp",
        "ccomp",
    })
    _STRUCTURAL_CONTAINER_KEYS = frozenset({"SENTENCE"})
    # Logical-context buckets that may house a *negated* lifecycle fact. "with no
    # suspect identified" routes the negated noun into TOGETHERNESS (surface
    # preposition `with`), but a negated existence is a SPECIFICATION, not a
    # togetherness — so pull negated lifecycle values out into SPECIFICATION while
    # leaving any positive ("with a suspect") values untouched.
    _NEGATED_CONTEXT_CONTAINER_KEYS = frozenset({"TOGETHERNESS"})
    _STATUS_VERB_TYPE_BY_LEMMA = {
        "await": "awaiting",
    }

    def matches(self, kernel, ctx):
        return {"_": True} if self._has_promotable(kernel, ctx) else None

    def apply(self, kernel, bindings, ctx):
        return self._rewrite(kernel, ctx)

    @classmethod
    def _rewrite(cls, node, ctx):
        if isinstance(node, SetOfSingletons):
            rewritten_entities = [cls._rewrite(e, ctx) for e in node.entities]
            if any(a is not b for a, b in zip(node.entities, rewritten_entities)):
                return node.update_entities(rewritten_entities)
            return node
        if not isinstance(node, Singleton):
            return node

        rewritten = node
        if node.kernel is not None:
            if node.kernel.source is not None:
                new_source = cls._rewrite(node.kernel.source, ctx)
                if new_source is not node.kernel.source:
                    rewritten = rewritten.update_kernel(new_source, "source")
            if node.kernel.target is not None:
                new_target = cls._rewrite(node.kernel.target, ctx)
                if new_target is not node.kernel.target:
                    rewritten = rewritten.update_kernel(new_target, "target")

        props = dict(rewritten.properties)
        changed = False
        for key, value in list(props.items()):
            values = property_values(props, key)
            rewritten_values = []
            for item in values:
                rewritten_item = cls._rewrite(item, ctx)

                promoted = False
                if key in cls._RAW_SYNTACTIC_KEYS:
                    status_node = cls._status_projection(rewritten_item, ctx)
                    if status_node is not None:
                        append_unique_property_value(props, "TIME_STATUS", status_node)
                        changed = True
                        promoted = True

                    negated_lifecycle = cls._negated_lifecycle_projection(rewritten_item, ctx)
                    if negated_lifecycle is not None:
                        append_unique_property_value(props, "SPECIFICATION", negated_lifecycle)
                        changed = True
                        promoted = True

                if not promoted and key in cls._STRUCTURAL_CONTAINER_KEYS:
                    status_node = cls._sentence_status_projection(rewritten_item, ctx)
                    if status_node is not None:
                        append_unique_property_value(props, "TIME_STATUS", status_node)
                        changed = True
                        promoted = True
                    else:
                        negated_lifecycle = cls._negated_lifecycle_projection(rewritten_item, ctx)
                        if negated_lifecycle is not None:
                            append_unique_property_value(props, "SPECIFICATION", negated_lifecycle)
                            changed = True
                            promoted = True

                if not promoted and key in cls._NEGATED_CONTEXT_CONTAINER_KEYS:
                    negated_lifecycle = cls._negated_lifecycle_projection(rewritten_item, ctx)
                    if negated_lifecycle is not None:
                        append_unique_property_value(props, "SPECIFICATION", negated_lifecycle)
                        changed = True
                        promoted = True

                if not promoted and key == "TIME_STATUS" and cls._is_status_node(rewritten_item, ctx):
                    normalized = cls._status_node_with_state_type(rewritten_item)
                    if normalized is not rewritten_item:
                        rewritten_item = normalized

                if not promoted:
                    rewritten_values.append(rewritten_item)

            if len(rewritten_values) != len(values) or any(a is not b for a, b in zip(values, rewritten_values)):
                if not rewritten_values:
                    props.pop(key, None)
                else:
                    props[key] = rewritten_values if isinstance(value, (list, tuple)) else rewritten_values[0]
                changed = True

        return rewritten.update_node_props(props) if changed else rewritten

    @classmethod
    def _has_promotable(cls, node, ctx):
        if isinstance(node, SetOfSingletons):
            return any(cls._has_promotable(e, ctx) for e in node.entities)
        if not isinstance(node, Singleton):
            return False
        if node.kernel is not None:
            if cls._has_promotable(node.kernel.source, ctx):
                return True
            if cls._has_promotable(node.kernel.target, ctx):
                return True
        for key, value in dict(node.properties).items():
            if key == "TIME_STATUS":
                for item in property_values(dict(node.properties), key):
                    if cls._is_status_node(item, ctx) and cls._needs_type_override(item):
                        return True
                    if cls._has_promotable(item, ctx):
                        return True
                continue
            if key in cls._STRUCTURAL_CONTAINER_KEYS:
                for item in property_values(dict(node.properties), key):
                    if cls._sentence_status_projection(item, ctx) is not None:
                        return True
                    if cls._negated_lifecycle_projection(item, ctx) is not None:
                        return True
                    if cls._has_promotable(item, ctx):
                        return True
                continue
            if key in cls._NEGATED_CONTEXT_CONTAINER_KEYS:
                for item in property_values(dict(node.properties), key):
                    if cls._negated_lifecycle_projection(item, ctx) is not None:
                        return True
                continue
            if key in cls._SEMANTIC_TARGET_KEYS or key not in cls._RAW_SYNTACTIC_KEYS:
                continue
            for item in property_values(dict(node.properties), key):
                if cls._status_projection(item, ctx) is not None:
                    return True
                if cls._negated_lifecycle_projection(item, ctx) is not None:
                    return True
                if cls._has_promotable(item, ctx):
                    return True
        return False

    @classmethod
    def _needs_type_override(cls, value):
        if not isinstance(value, Singleton):
            return False
        props = dict(value.properties)
        qualifier, _, _ = cls._pick_status_qualifier(props)
        if qualifier is None:
            return False
        return props.get("type") != qualifier

    @classmethod
    def _sentence_status_projection(cls, value, ctx):
        """Match a SENTENCE-wrapped status clause and project it to TIME_STATUS.

        Handles copula shapes like "Investigation complete", where the status
        noun is the copula subject and the qualifier appears as a `cop`
        adjective on the existential target, and verbal status shapes like
        "Awaiting court outcome", where `await` supplies the status type and
        the target is a status noun.
        """
        if not isinstance(value, Singleton) or value.kernel is None:
            return None
        edge = value.kernel.edgeLabel
        if edge is None:
            return None
        edge_name = str(getattr(edge, "named_entity", "") or "").strip().lower()
        if edge_name != "be":
            return cls._verbal_status_projection(value, ctx)
        source = value.kernel.source
        target = value.kernel.target
        if not cls._is_status_node(source, ctx):
            return None
        if target is None:
            return None
        cop_items = property_values(dict(target.properties), "cop")
        adj = next(
            (
                c for c in cop_items
                if isinstance(c, Singleton)
                and c.named_entity
                and " " not in str(c.named_entity).strip()
            ),
            None,
        )
        if adj is None:
            return None
        merged_props = dict(source.properties)
        existing_amod = list(property_values(merged_props, "amod"))
        existing_amod.append(adj)
        merged_props["amod"] = existing_amod if len(existing_amod) > 1 else existing_amod[0]
        candidate = source.update_node_props(merged_props)
        return cls._status_node_with_state_type(candidate)

    @classmethod
    def _verbal_status_projection(cls, value, ctx):
        if not isinstance(value, Singleton) or value.kernel is None:
            return None
        edge = value.kernel.edgeLabel
        edge_name = str(getattr(edge, "named_entity", "") or "").strip().lower()
        status_type = cls._STATUS_VERB_TYPE_BY_LEMMA.get(edge_name)
        if status_type is None:
            return None

        target = value.kernel.target
        if cls._is_status_node(target, ctx):
            return cls._status_node_with_explicit_type(target, status_type)

        status_extra = cls._status_extra(target, ctx)
        if status_extra is not None:
            return cls._status_node_with_explicit_type(status_extra, status_type)
        return None

    @staticmethod
    def _status_node_with_explicit_type(value, status_type):
        if not isinstance(value, Singleton):
            return value
        props = dict(value.properties)
        props["type"] = status_type
        return value.update_node_props(props)

    @classmethod
    def _status_projection(cls, value, ctx):
        verbal_status = cls._verbal_status_projection(value, ctx)
        if verbal_status is not None:
            return verbal_status
        if cls._is_negated_lifecycle_value(value, ctx):
            return None
        if isinstance(value, SetOfSingletons):
            for entity in value.entities:
                projected = cls._status_projection(entity, ctx)
                if projected is not None:
                    return projected
            return None
        if not isinstance(value, Singleton):
            return None

        props = dict(value.properties)
        status_extra = cls._status_extra(value, ctx)
        if status_extra is not None and (not cls._is_status_node(value, ctx) or cls._has_colon_punct(value)):
            return cls._status_node_with_state_type(status_extra)

        if cls._is_status_node(value, ctx):
            return cls._status_node_with_state_type(value)

        if status_extra is not None:
            return cls._status_node_with_state_type(status_extra)
        return None

    @classmethod
    def _status_extra(cls, value, ctx):
        props = dict(value.properties)
        for extra in property_values(props, "extra"):
            if isinstance(extra, Singleton) and cls._is_status_node(extra, ctx):
                merged_props = dict(extra.properties)
                for inherited_key in ("amod", "advmod", "punct"):
                    if inherited_key in props and inherited_key not in merged_props:
                        merged_props[inherited_key] = props[inherited_key]
                return extra.update_node_props(merged_props)
        return None

    @staticmethod
    def _has_colon_punct(value):
        props = dict(value.properties)
        punct_values = property_values(props, "punct")
        return any(str(p) == ":" for p in punct_values)

    _NUMBERED_KEY_RE = re.compile(r"^\d+(?:\.\d+)?$")

    @classmethod
    def _status_node_with_state_type(cls, value):
        """Derive the status node's `type` from its structural context.

        For a status noun (e.g. `investigation`), the surface qualifier comes
        either from an `amod` (e.g. "Investigation complete") or from a
        positional preposition stored under a numeric key (e.g. `1:Under` for
        "Under investigation", `16:under` for "remains under investigation").

        The HOnK ontology may pre-set `type` to a composite phrase like
        `under_investigation`; this method overrides it with the structural
        qualifier so the rendered form is just `(type:under)` or
        `(type:complete)`, and consumes the property that supplied it.
        """
        if not isinstance(value, Singleton):
            return value
        props = dict(value.properties)

        qualifier, consumed_key, consumed_item = cls._pick_status_qualifier(props)
        if qualifier is None:
            return value

        props["type"] = qualifier
        if consumed_key == "amod":
            remaining = [
                item for item in property_values(props, "amod") if item is not consumed_item
            ]
            if not remaining:
                props.pop("amod", None)
            else:
                props["amod"] = remaining if len(remaining) > 1 else remaining[0]
        else:
            props.pop(consumed_key, None)

        return value.update_node_props(props)

    @classmethod
    def _pick_status_qualifier(cls, props):
        amod_items = property_values(props, "amod")
        for item in amod_items:
            name = item.named_entity if isinstance(item, Singleton) else item
            name = str(name).strip() if name is not None else ""
            if name and " " not in name:
                return name.lower(), "amod", item

        for key, raw in props.items():
            if not isinstance(key, str) or not cls._NUMBERED_KEY_RE.match(key):
                continue
            if not isinstance(raw, str):
                continue
            name = raw.strip()
            if name and " " not in name:
                return name.lower(), key, raw
        return None, None, None

    @staticmethod
    def _is_status_node(value, ctx):
        return isinstance(value, Singleton) and ctx.matchers.matches_honk_set(
            value, ctx.services.getHOnK().getStatusNouns()
        )

    @classmethod
    def _is_negated_lifecycle_value(cls, value, ctx):
        if not (isinstance(value, SetOfSingletons) and value.type == Grouping.NOT):
            return False
        return any(cls._is_lifecycle_value(entity, ctx) for entity in value.entities)

    @classmethod
    def _negated_lifecycle_projection(cls, value, ctx):
        if cls._is_negated_lifecycle_value(value, ctx):
            return value
        if not isinstance(value, Singleton) or value.kernel is None:
            return None
        for part in (value.kernel.source, value.kernel.target):
            if cls._is_negated_lifecycle_value(part, ctx):
                return part
        return None

    @classmethod
    def _is_lifecycle_value(cls, value, ctx):
        if isinstance(value, SetOfSingletons):
            return any(cls._is_lifecycle_value(entity, ctx) for entity in value.entities)
        if not isinstance(value, Singleton):
            return False
        lifecycle_phrases = cls._lifecycle_phrase_labels()
        candidates = {str(c).strip().lower() for c in ctx.matchers.name_candidates(value)}
        return bool(candidates & lifecycle_phrases)

    @staticmethod
    def _lifecycle_phrase_labels():
        try:
            from LaSSI.HOnK.TBox.LifecycleManager import _get_lifecycle_phrases
            return set(_get_lifecycle_phrases())
        except Exception:
            return set()


class LifecyclePropertyPromotionLateRule:
    """Second pass of `LifecyclePropertyPromotionRule` after SENTENCE properties
    are materialised.

    The base rule fires at `post_hoist_pre_dedupe`, which is before
    `promote_contextual_sentence_kernel` packages clauses into the `SENTENCE`
    property. Re-running the same logic after `rewrite_properties_logically`
    lets the SENTENCE-branch of `_rewrite` see those packaged clauses and
    surface copula-StatusNoun / NOT(lifecycle) shapes as TIME_STATUS /
    SPECIFICATION.
    """

    name = "lifecycle_property_promotion_late"
    phase = "post_logical_rewrite"

    def __init__(self):
        self._inner = LifecyclePropertyPromotionRule()

    def matches(self, kernel, ctx):
        return self._inner.matches(kernel, ctx)

    def apply(self, kernel, bindings, ctx):
        return self._inner.apply(kernel, bindings, ctx)

    def run(self, kernel, ctx):
        return self._inner.run(kernel, ctx)
