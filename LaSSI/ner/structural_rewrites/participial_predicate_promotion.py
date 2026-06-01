__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.string_functions import lemmatize_verb
from LaSSI.ner.structural_rewrites.declarative import structural_lexical_set
from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class ParticipialPredicatePromotionRule(StructuralRewriteRule):
    """Repair reduced-relative predicates that were misread as SPACE.

    In clauses like ``shoplifting ... involved goods stolen from a store``,
    the parser can make the participial action (``stolen``) the outer edge and
    leave the object introduced by ``involved`` under SPACE because of the
    surface preposition ``in``.  When the true event subject is the old target
    and the old source is existential, promote the ``amod`` verb to the main
    predicate and keep the old edge as an action on the new object.
    """

    name = "participial_predicate_promotion"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        source = kernel.kernel.source
        target = kernel.kernel.target
        edge_label = kernel.kernel.edgeLabel
        if not isinstance(source, Singleton) or source.type != "existential":
            return None
        if not isinstance(target, Singleton):
            return None
        if not isinstance(edge_label, Singleton) or not edge_label.named_entity:
            return None

        candidate = self._find_space_candidate(kernel, ctx)
        if candidate is None:
            return None

        action_lemma = candidate["action_lemma"]
        if action_lemma == lemmatize_verb(edge_label.named_entity):
            return None
        return {
            "candidate": candidate["node"],
            "action_lemma": action_lemma,
            "old_action": edge_label.named_entity,
        }

    def apply(self, kernel, bindings, ctx):
        candidate = bindings["candidate"]
        candidate_props = dict(candidate.properties)
        candidate_props.pop("amod", None)
        self._remove_space_residue(candidate_props, ctx)
        candidate_props.setdefault("actioned", bindings["old_action"])
        new_target = candidate.update_node_props(candidate_props)
        new_edge = kernel.kernel.edgeLabel.update_name(bindings["action_lemma"])
        new_source = kernel.kernel.target

        props = {}
        for key, value in dict(kernel.properties).items():
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            values = [item for item in values if getattr(item, "id", None) != candidate.id]
            if not values:
                continue
            props[key] = values if isinstance(value, (list, tuple)) else values[0]

        rewritten = kernel.update_kernel(new_source, "source")
        rewritten = rewritten.update_kernel(new_target, "target")
        rewritten = rewritten.update_kernel(new_edge, "edgeLabel")
        return rewritten.update_node_props(props)

    @staticmethod
    def _remove_space_residue(props, ctx):
        spatial_types = structural_lexical_set("spatial_relation_types") or {
            "stay in place",
            "near place",
            "motion to place",
            "motion from place",
            "motion through place",
        }
        if str(props.get("type", "")).strip().lower() in spatial_types:
            props.pop("type", None)

        prepositions = {
            str(term).strip().lower()
            for terms in (
                ctx.services.getHOnK().getPrepositions(),
                ctx.services.getHOnK().getPrototypicalPrepositions(),
            )
            for term in terms
            if term
        }
        for key, value in list(props.items()):
            try:
                float(str(key))
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value.strip().lower() in prepositions:
                props.pop(key, None)

    @classmethod
    def _find_space_candidate(cls, kernel, ctx):
        values = dict(kernel.properties).get("SPACE")
        if values is None:
            return None
        values = values if isinstance(values, (list, tuple)) else [values]
        action_verbs = cls._action_verbs(ctx)
        for value in values:
            if not isinstance(value, Singleton):
                continue
            if value.type in {"GPE", "LOC", "DATE", "TIME"}:
                continue
            props = dict(value.properties)
            amod_value = props.get("amod")
            amods = amod_value if isinstance(amod_value, (list, tuple)) else [amod_value]
            for amod in amods:
                if not isinstance(amod, str):
                    continue
                lemma = lemmatize_verb(amod)
                if lemma in action_verbs:
                    return {"node": value, "action_lemma": lemma}
        return None

    @staticmethod
    def _action_verbs(ctx):
        honk = ctx.services.getHOnK()
        return {
            str(term).strip().lower()
            for terms in (
                honk.getTransitiveVerbs(),
                honk.getStateVerbs(),
                honk.getCausativeVerbs(),
            )
            for term in terms
            if term
        }
