__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.node_functions import create_props_for_singleton
from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    Relationship,
    SetOfSingletons,
    Singleton,
)


class WeatherAmodConditionRule(StructuralRewriteRule):
    """Split weather-state adjectives out of mistaken metric modifiers.

    CoreNLP can attach terse forecast-list adjectives across punctuation, as in
    "Cloudy, 13.35°C, wind 3.64 mph", where `Cloudy` arrives as
    `wind[amod:Cloudy]`. In this domain that adjective is a sibling forecast
    condition, not a modifier of the metric noun.
    """

    name = "weather_amod_condition"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if (
            not isinstance(kernel, Singleton)
            or kernel.kernel is None
            or kernel.kernel.edgeLabel is None
            or str(kernel.kernel.edgeLabel.named_entity).lower() != "be"
        ):
            return None
        changed, _ = self._split_node(kernel.kernel.target, ctx)
        return {"_": True} if changed else None

    def apply(self, kernel, bindings, ctx):
        _, new_target = self._split_node(kernel.kernel.target, ctx)
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

    @classmethod
    def _split_node(cls, node, ctx):
        if isinstance(node, SetOfSingletons):
            changed = False
            entities = []
            for entity in node.entities:
                entity_changed, entity_new = cls._split_node(entity, ctx)
                changed = changed or entity_changed
                if (
                    entity_changed
                    and isinstance(entity_new, SetOfSingletons)
                    and entity_new.type == node.type == Grouping.AND
                ):
                    entities.extend(entity_new.entities)
                else:
                    entities.append(entity_new)
            if changed:
                return True, node.update_entities(entities)
            return False, node

        if not isinstance(node, Singleton):
            return False, node

        props = dict(node.properties) if node.properties else {}
        amods = props.get("amod", [])
        if isinstance(amods, str): amods = [amods]
        valid_amods = [a for a in amods if isinstance(a, str) and cls._is_weather_adjective(a, ctx)]
        if not valid_amods:
            return False, node
        if not cls._is_weather_noun(node, ctx):
            return False, node

        cleaned_props = dict(props)
        remaining_amods = [a for a in amods if a not in valid_amods]
        if remaining_amods:
            cleaned_props["amod"] = tuple(remaining_amods)
        else:
            cleaned_props.pop("amod", None)

        lemmas = cleaned_props.get("lemma", [])
        if isinstance(lemmas, str): lemmas = [lemmas]
        lemmas = [l for l in lemmas if l.lower() not in [va.lower() for va in valid_amods]]
        if lemmas:
            cleaned_props["lemma"] = lemmas[0] if len(lemmas) == 1 else lemmas
        else:
            cleaned_props.pop("lemma", None)

        cleaned = node.update_node_props(cleaned_props)

        entities = [cleaned]
        for va in valid_amods:
            adjective = Singleton(
                id=-(abs(node.id) + 100000 + len(entities)),
                named_entity=va,
                properties=create_props_for_singleton({}),
                min=node.min,
                max=node.max,
                type="JJ",
                confidence=node.confidence,
            )
            entities.append(adjective)

        return True, SetOfSingletons(
            id=node.id,
            type=Grouping.AND,
            entities=tuple(entities),
            min=node.min,
            max=node.max,
            confidence=node.confidence,
            root=False,
        )

    @staticmethod
    def _normalised_candidates(node):
        props = dict(node.properties) if getattr(node, "properties", None) else {}
        candidates = {getattr(node, "named_entity", None), props.get("lemma")}
        return {str(x).strip().lower() for x in candidates if x}

    @classmethod
    def _is_weather_noun(cls, node, ctx):
        terms = ctx.services.getHOnK().getWeatherConditionNouns()
        normalised_terms = {str(x).strip().lower() for x in terms if x}
        return bool(cls._normalised_candidates(node) & normalised_terms)

    @staticmethod
    def _is_weather_adjective(value, ctx):
        terms = ctx.services.getHOnK().getWeatherConditionAdjectives()
        normalised_terms = {str(x).strip().lower() for x in terms if x}
        return str(value).strip().lower() in normalised_terms
