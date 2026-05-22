__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.string_functions import lemmatize_verb
from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    append_unique_property_value,
)
from LaSSI.structures.internal_graph.EntityRelationship import (
    Relationship,
    SetOfSingletons,
    Singleton,
)


_LOCATION_TYPES = frozenset({"GPE", "LOC", "FAC"})
_DATE_TIME_TYPES = frozenset({"DATE", "TIME", "SUTime"})


def _is_location(node):
    return isinstance(node, Singleton) and str(getattr(node, "type", "")).upper() in _LOCATION_TYPES


def _is_date(node):
    return isinstance(node, Singleton) and str(getattr(node, "type", "")).upper() in _DATE_TIME_TYPES


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class AuxiliaryPeriphrasisPromotionRule(StructuralRewriteRule):
    """Strip a periphrastic auxiliary wrapper around a nominal/reporting verb
    that takes a location as its patient, lifting that verb into the subject
    slot of an implicit ``be``.

    The triggering shape is the passive periphrastic ``X is V to AUX Y``,
    which CoreNLP+GSM render as ``aux(?existential, V(?existential, X))``
    with the patient ``X`` (a location) buried as the inner verb's object
    and the actual content ``Y`` scattered across TOPIC/TOGETHERNESS/
    SPECIFICATION property buckets.  Semantically the inner verb is a
    reporting/prediction nominal (forecast, prediction, report, …) and the
    sentence is best read as ``be(V-as-noun, content)[SPACE: X]`` so it
    lines up with the verbless nominal pattern handled in
    ``SpecificationAndToTargetRule``.

    Pattern (kept structural so the rule isn't sentence-specific):
      * Outer kernel.source is an existential placeholder (passive lift).
      * Outer kernel.target or .source contains an inner Singleton with
        its own kernel.
      * The inner kernel's edge label matches a HOnK
        ``WeatherConditionNoun`` lemma (forecast, prediction, report-style
        nominals).  This keeps the rule scoped to verbs that *denote* a
        nominal content rather than firing on any arbitrary nested kernel.
      * The inner kernel's target is a location (GPE/LOC/FAC).

    Rewrite:
      * New kernel: ``be(<inner_verb_as_noun>, ?existential)``.
      * Inner location → SPACE on the outer kernel.
      * Merge outer + inner properties.  TOPIC entries that are dates land
        in TIME; the rest fall into SPECIFICATION so
        ``SpecificationAndToTargetRule`` can lift them into the target slot.
      * TOGETHERNESS that duplicates SPECIFICATION (same id) is dropped —
        the GSM construct rules emit both for one ``with``-clause."""

    name = "auxiliary_periphrasis_promotion"
    phase = "post_logical_rewrite"

    @staticmethod
    def _pick_richer_date(a, b):
        # Choose the TIME-slot date with the more informative property bag.
        # Priority: most case-marker entries (positional keys like "18:at"
        # and "type:defined"), then the longer character span — the entry
        # carrying the case markers is anchored to the actual surface
        # mention ("at 8pm"); the bare-year duplicate ("nummod:2026") is
        # only redundant SUTime metadata.
        def _score(node):
            if not hasattr(node, 'properties'):
                return (0, 0)
            props = dict(node.properties)
            case_keys = 0
            for k in props:
                try:
                    float(k)
                    case_keys += 1
                except (TypeError, ValueError):
                    if k in ('type',):
                        case_keys += 1
            span = (getattr(node, 'max', 0) or 0) - (getattr(node, 'min', 0) or 0)
            return (case_keys, span)
        return a if _score(a) >= _score(b) else b

    def _lemma_in_set(self, edge, terms):
        # Compare the edge's surface form *and* its lemmatised form to the
        # HOnK term list (stored as base verbs). CoreNLP lemmatises the
        # outer kernel edge in some shapes (``expect``) but leaves the
        # surface form in others (``forecast``), so accept either.
        if edge is None or not terms:
            return False
        surface = (edge.named_entity or "").strip().lower()
        if not surface:
            return False
        ontology = {str(w).lower() for w in terms if w}
        if surface in ontology:
            return True
        return lemmatize_verb(surface).lower() in ontology

    def _find_reporting_kernel(self, node, ctx):
        # Nested pattern: ``aux(?, V(?, GPE))`` where the inner kernel's
        # edge is a HOnK WeatherConditionNoun (the periphrastic "X is
        # forecast to have …" shape from weather_002 sentences 2–4).
        if not isinstance(node, Singleton) or node.kernel is None:
            return None
        weather_nouns = ctx.services.getHOnK().getWeatherConditionNouns()
        if not self._lemma_in_set(node.kernel.edgeLabel, weather_nouns):
            return None
        if not _is_location(node.kernel.target):
            return None
        return node

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None

        # Pattern A — nested: outer auxiliary wraps an inner kernel headed
        # by a weather-condition noun (forecast, prediction, …). Inner can
        # sit on either side of the outer depending on whether the outer
        # "have" took a direct object or a passive subject
        # ("is forecast to have rain" vs "is forecast to have conditions").
        for slot in ("target", "source"):
            cand = kernel.kernel.target if slot == "target" else kernel.kernel.source
            inner = self._find_reporting_kernel(cand, ctx)
            if inner is None:
                continue
            sibling = kernel.kernel.source if slot == "target" else kernel.kernel.target
            return {
                "shape": "nested",
                "inner_kernel_node": inner,
                "inner_subject": inner.kernel.source,
                "inner_object": inner.kernel.target,
                "inner_edge": inner.kernel.edgeLabel,
                "sibling": sibling,
            }

        # Pattern B — flat: outer kernel's edge label is itself a HOnK
        # PredictionVerb and its target is a location ("Newcastle is
        # expected to be cloudy …" — Stanza collapses the "is V to be JJ"
        # periphrasis to ``V(JJ_predicate, GPE)``, with V the actual
        # report/prediction verb). Same semantics as pattern A but the
        # report verb lives on the outer edge, not on an inner kernel.
        prediction_verbs = ctx.services.getHOnK().getPredictionVerbs()
        if (
            self._lemma_in_set(kernel.kernel.edgeLabel, prediction_verbs)
            and _is_location(kernel.kernel.target)
        ):
            return {
                "shape": "flat",
                "inner_kernel_node": None,
                "inner_subject": kernel.kernel.source,
                "inner_object": kernel.kernel.target,
                "inner_edge": kernel.kernel.edgeLabel,
                "sibling": None,
            }
        return None

    def apply(self, kernel, bindings, ctx):
        shape = bindings.get("shape", "nested")
        inner_kernel_node = bindings["inner_kernel_node"]
        inner_edge = bindings["inner_edge"]
        location = bindings["inner_object"]
        sibling = bindings.get("sibling")
        # Pattern B has no separate inner kernel; the outer kernel itself
        # carries the property bag and the report verb on its edge label.
        inner_props_source = inner_kernel_node if inner_kernel_node is not None else kernel
        # Flat-shape predicate (e.g. "cloudy" in `expect(cloudy, Newcastle)`)
        # is the outer source — preserve it so the SpecificationAndToTargetRule
        # downstream lifts it into the AND target alongside chance/etc.
        flat_predicate = bindings["inner_subject"] if shape == "flat" else None

        forecast_as_noun = Singleton(
            id=inner_edge.id,
            named_entity=inner_edge.named_entity,
            properties=inner_edge.properties,
            min=inner_edge.min,
            max=inner_edge.max,
            type="noun",
            confidence=inner_edge.confidence,
        )

        existential_target = Singleton(
            id=-(abs(kernel.id) + 1),
            named_entity="?",
            properties=frozenset(dict().items()),
            min=kernel.min,
            max=kernel.max,
            type="existential",
            confidence=1.0,
        )

        be_label = Singleton(
            id=-1,
            named_entity="be",
            properties=frozenset(dict().items()),
            min=inner_edge.min,
            max=inner_edge.max,
            type="verb",
            confidence=1.0,
        )

        new_relation = Relationship(
            source=forecast_as_noun,
            target=existential_target,
            edgeLabel=be_label,
            isNegated=kernel.kernel.isNegated,
        )

        # Merge outer + inner properties; inner property keys win on overlap
        # so TIME/SPACE/etc. computed locally to the forecast clause aren't
        # clobbered by stale outer values.
        outer_props = dict(kernel.properties)
        inner_props = dict(inner_props_source.properties) if inner_props_source is not kernel else {}
        merged = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in outer_props.items()}
        for k, v in inner_props.items():
            if isinstance(v, (list, tuple)):
                v = list(v)
            if k in merged and isinstance(merged[k], list) and isinstance(v, list):
                for item in v:
                    if item not in merged[k]:
                        merged[k].append(item)
            else:
                merged[k] = v

        # Split TOPIC entries: date-like nodes go to TIME, others stay
        # under SPECIFICATION.
        topic_values = _as_list(merged.pop('TOPIC', None))
        if topic_values:
            for item in topic_values:
                if _is_date(item):
                    append_unique_property_value(merged, 'TIME', item)
                else:
                    append_unique_property_value(merged, 'SPECIFICATION', item)

        # Catch-all `noun:[]` bucket holds nominal modifiers the construct
        # rules couldn't route (e.g. the bare object of "to have"
        # "light rain"). These belong in SPECIFICATION so the AND-to-target
        # rule can lift them alongside chance/conditions/etc. Drop nouns
        # that match the inner kernel's location target (already in SPACE)
        # or are simple date nodes (already in TIME).
        noun_values = _as_list(merged.pop('noun', None))
        location_id = getattr(location, 'id', None)
        for item in noun_values:
            if not isinstance(item, (Singleton, SetOfSingletons)):
                continue
            if location_id is not None and getattr(item, 'id', None) == location_id:
                continue
            if _is_date(item):
                append_unique_property_value(merged, 'TIME', item)
                continue
            append_unique_property_value(merged, 'SPECIFICATION', item)

        # SPECIFICATION may already contain DATE nodes (e.g. the `April 2026`
        # quantity that the of-rule attached to the kernel source). They
        # belong in TIME, not in the AND target. Move them, stripping any
        # spurious `extra` chain pointing back to the kernel source — the
        # source becomes a sibling of these dates under the AND target
        # below, so an `extra` link from a date back to it would just be
        # confusing.
        spec_values = _as_list(merged.get('SPECIFICATION'))
        if spec_values:
            kept_spec = []
            for item in spec_values:
                if _is_date(item):
                    if isinstance(item, Singleton):
                        cleaned = dict(item.properties)
                        cleaned.pop('extra', None)
                        item = item.update_node_props(cleaned) if cleaned != dict(item.properties) else item
                    append_unique_property_value(merged, 'TIME', item)
                else:
                    kept_spec.append(item)
            if kept_spec:
                merged['SPECIFICATION'] = kept_spec
            else:
                merged.pop('SPECIFICATION', None)

        # TOGETHERNESS that simply duplicates SPECIFICATION (same id) is a
        # GSM artefact — keep only the SPECIFICATION copy which carries the
        # `extra` chain.
        spec_ids = {
            getattr(item, 'id', None)
            for item in _as_list(merged.get('SPECIFICATION'))
            if hasattr(item, 'id')
        }
        if 'TOGETHERNESS' in merged:
            kept_togetherness = [
                item for item in _as_list(merged['TOGETHERNESS'])
                if getattr(item, 'id', None) not in spec_ids
            ]
            if kept_togetherness:
                merged['TOGETHERNESS'] = kept_togetherness
            else:
                merged.pop('TOGETHERNESS', None)

        append_unique_property_value(merged, 'SPACE', location)

        # If the auxiliary had a meaningful sibling (the object of "have"
        # when forecast sits as source — e.g. "have dry and clear
        # conditions"), fold it into SPECIFICATION so the
        # SpecificationAndToTargetRule downstream can lift it.
        if isinstance(sibling, (Singleton, SetOfSingletons)) and not (
            isinstance(sibling, Singleton) and sibling.type == 'existential'
        ):
            append_unique_property_value(merged, 'SPECIFICATION', sibling)

        # Pattern B (flat predict-verb): the original outer source carries
        # the actual content predicate (e.g. ``cloudy`` in
        # ``expect(cloudy, Newcastle)``). Fold it into SPECIFICATION so
        # SpecificationAndToTargetRule promotes it into the AND target.
        if isinstance(flat_predicate, (Singleton, SetOfSingletons)) and not (
            isinstance(flat_predicate, Singleton) and flat_predicate.type == 'existential'
        ):
            append_unique_property_value(merged, 'SPECIFICATION', flat_predicate)

        # Deduplicate TIME entries by their SUTime-normalised named_entity
        # and strip the redundant `nummod` SUTime artefact. CoreNLP/SUTime
        # collapses "8pm" and "2026" (from "8pm on 14 April 2026") to the
        # same ISO form ("2026-04-14T20-0"), so both surface mentions can
        # land in TIME each carrying a `nummod` (the bare number that
        # spawned them: "8" or "2026"). Those numbers are already encoded
        # in the ISO form and would confuse readers ("nummod:8" doesn't
        # obviously mean "8pm"), so drop them; keep the entry with the
        # most case-marker / position info, falling back to the longer
        # surface span.
        time_values = _as_list(merged.get('TIME'))
        if time_values:
            by_name = {}
            for item in time_values:
                if isinstance(item, Singleton) and _is_date(item):
                    props = dict(item.properties)
                    if 'nummod' in props:
                        new_props = {k: v for k, v in props.items() if k != 'nummod'}
                        item = item.update_node_props(new_props)
                key = getattr(item, 'named_entity', None) or id(item)
                existing = by_name.get(key)
                if existing is None:
                    by_name[key] = item
                else:
                    by_name[key] = self._pick_richer_date(existing, item)
            deduped = list(by_name.values())
            if deduped != time_values:
                merged['TIME'] = deduped

        new_kernel = Singleton(
            id=kernel.id,
            named_entity=kernel.named_entity,
            properties=kernel.properties,
            min=kernel.min,
            max=kernel.max,
            type=kernel.type,
            confidence=kernel.confidence,
            kernel=new_relation,
        )
        return new_kernel.update_node_props(merged)


# Backwards-compatible alias (was hard-named to "have/forecast"); registered
# under the new generic name.
HaveForecastPromotionRule = AuxiliaryPeriphrasisPromotionRule
