import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from LaSSI.ner.node_functions import create_props_for_singleton
from LaSSI.ner.structural_rewrites.auxiliary_periphrasis_promotion import (
    AuxiliaryPeriphrasisPromotionRule,
)
from LaSSI.ner.structural_rewrites.base import RuleRegistry, RewriteContext, as_list, dedupe_by_id_or_name
from LaSSI.ner.structural_rewrites.declarative import load_declarative_rules
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    Relationship,
    SetOfSingletons,
    Singleton,
)


class _FakeHOnK:
    def getWeatherConditionNouns(self):
        return {
            "weather",
            "condition",
            "conditions",
            "forecast",
            "rain",
            "precipitation",
            "cloud",
            "sunshine",
            "wind",
            "chance",
            "probability",
            "shower",
        }

    def getWeatherConditionAdjectives(self):
        return {"cloudy", "sunny", "clear", "dry", "rainy"}

    def getPredictionVerbs(self):
        return {"forecast", "expect", "predict"}

    def getCopulaSurfaceForms(self):
        return {"be", "is", "are", "'re", "being"}

    def getStatusNouns(self):
        return {"investigation", "suspect", "outcome"}

    def getReportingVerbs(self):
        return {"record"}

    def getLifecycleOutcomeVerbs(self):
        return {"conclude"}

    def getCausalSignalVerbs(self):
        return {"result"}


class _FakeServices:
    def getHOnK(self):
        return _FakeHOnK()


class _FakeMatchers:
    def __init__(self, G=None):
        self.G = G

    def is_location_like(self, node):
        return isinstance(node, Singleton) and node.type in {"GPE", "LOC", "FAC"}

    def dependency_children(self, node, labels):
        if self.G is None or not isinstance(node, Singleton) or node.id not in self.G:
            return []
        out = []
        for _, child_id, data in self.G.out_edges(node.id, data=True):
            if getattr(data.get("label"), "named_entity", None) in labels:
                out.append(self.G.nodes[child_id]["data"])
        return out

    def matches_class(self, value, class_name, *, kernel=None):
        name = getattr(value, "named_entity", value)
        name = str(name).strip().lower() if name is not None else ""
        if class_name == "ReportingVerb":
            return name in {"record", "recorded"}
        if class_name == "LifecycleOutcomeVerb":
            return name in {"conclude", "concluded"}
        if class_name == "CausalSignalVerb":
            return name in {"result", "resulted"}
        if class_name == "LocationLike":
            return self.is_location_like(value)
        if class_name == "Copula":
            return name in {"be", "is", "are", "'re", "being"}
        if class_name == "AccessPointLike":
            return name in {"public entrance", "access point"}
        if class_name == "CausalNode":
            return name in {"fire", "storm damage"}
        if class_name == "StateVerb":
            return name in {"open", "closed", "close"}
        if class_name == "ChangeOfStateVerb":
            return name in {"close", "closed"}
        if class_name == "OccurrenceVerb":
            return name in {"occur", "occurred", "happen", "happened", "take place"}
        if class_name == "RelativePronoun":
            return name in {"which", "that", "who", "whom", "whose"}
        if class_name == "WeatherConditionNoun":
            return name in {
                "weather",
                "condition",
                "conditions",
                "forecast",
                "rain",
                "rain shower",
                "precipitation",
                "cloud",
                "sunshine",
                "wind",
                "chance",
                "probability",
                "shower",
            }
        if class_name == "WeatherConditionAdjective":
            if name in {"cloudy", "rainy", "sunny", "clear", "dry"}:
                return True
            if isinstance(value, Singleton):
                props = dict(value.properties)
                prop_values = []
                for prop_value in props.values():
                    prop_values.extend(as_list(prop_value))
                return any(str(prop).strip().lower() in {"cloudy", "rainy", "sunny", "clear", "dry"}
                           for prop in prop_values)
            return False
        if class_name == "LifecycleHeadPhrase":
            return name in {"precipitation", "rain"}
        if class_name == "StatusNoun":
            return name in {"status", "outcome", "investigation", "charge", "arrest", "suspect"}
        if class_name == "EventClassifierHeadNoun":
            return name in {"offence", "offences", "offense", "offenses", "incident", "incidents"}
        if class_name == "ContentNode":
            return isinstance(value, (Singleton, SetOfSingletons)) and not self.matches_class(value, "ContextNode")
        if class_name == "ContextNode":
            return isinstance(value, Singleton) and value.type in {"DATE", "TIME", "SUTime", "GPE", "LOC", "FAC", "existential"}
        return False

    def matches_honk_set(self, value, honk_values):
        name = getattr(value, "named_entity", value)
        name = str(name).strip().lower() if name is not None else ""
        return name in {str(v).strip().lower() for v in (honk_values or set())}

    def prepositions_for_construct(self, construct_name):
        if construct_name == "space":
            return {"at", "in", "near", "on", "on or near"}
        return set()


class _ActionMatchers(_FakeMatchers):
    """Adds the action-verb classes used by ActionObjectPromotionRule."""

    def matches_class(self, value, class_name, *, kernel=None):
        name = getattr(value, "named_entity", value)
        name = str(name).strip().lower() if name is not None else ""
        if class_name == "CausativeVerb":
            return name == "abandon"
        if class_name == "MaterialisationVerb":
            return name == "replace"
        return super().matches_class(value, class_name, kernel=kernel)


def _props(values=None):
    return create_props_for_singleton(values or {})


def _node(node_id, name, node_type="noun", props=None):
    return Singleton(
        id=node_id,
        named_entity=name,
        properties=_props(props),
        min=node_id,
        max=node_id,
        type=node_type,
        confidence=1.0,
    )


_DEFAULT_TARGET = object()


def _kernel(edge_name="be", source=None, target=_DEFAULT_TARGET, props=None):
    return Singleton(
        id=100,
        named_entity="",
        type="SENTENCE",
        min=0,
        max=10,
        confidence=1.0,
        kernel=Relationship(
            source=source if source is not None else _node(1, "source"),
            target=_node(2, "target") if target is _DEFAULT_TARGET else target,
            edgeLabel=_node(3, edge_name, "verb"),
            isNegated=False,
        ),
        properties=_props(props),
    )


def _create_final_kernel_x_class():
    with patch("LaSSI.external_services.Services.Services.getInstance", return_value=_FakeServices()):
        from LaSSI.ner.CreateFinalKernelX import CreateFinalKernelX
    return CreateFinalKernelX


def _forecast_periphrasis(source=None, props=None):
    location = _node(90, "Newcastle", "GPE")
    inner = Singleton(
        id=91,
        named_entity="",
        type="SENTENCE",
        min=0,
        max=6,
        confidence=1.0,
        kernel=Relationship(
            source=_node(92, "?", "existential"),
            target=location,
            edgeLabel=_node(93, "forecast", "verb"),
            isNegated=False,
        ),
        properties=_props({}),
    )
    return _kernel(
        edge_name="have",
        source=source if source is not None else _node(94, "?", "existential"),
        target=inner,
        props=props,
    )


class TestStructuralRewrites(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = {rule.name: rule for rule in load_declarative_rules()}
        cls.ctx = RewriteContext(
            node_functions=None,
            services=_FakeServices(),
            matchers=_FakeMatchers(),
        )

    def test_as_list_and_dedupe_helpers(self):
        a = _node(1, "Newcastle", "GPE")
        b = _node(1, "Newcastle", "GPE")
        c = _node(2, "Sunderland", "GPE")

        self.assertEqual(as_list(None), [])
        self.assertEqual(as_list(a), [a])
        self.assertEqual(dedupe_by_id_or_name([a, b, c]), [a, c])

    def test_singleton_location_to_space(self):
        rule = self.rules["singleton_and_location_to_space"]
        loc = _node(8, "Newcastle", "GPE")
        kernel = _kernel(props={"AND": [loc]})

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)

        self.assertNotIn("AND", props)
        self.assertEqual(props["SPACE"], (loc,))

    def test_content_source_spatial_target_moves_target_to_space(self):
        rule = self.rules["content_source_spatial_target_to_space"]
        offence = _node(9, "possession offence")
        weapons = _node(10, "weapons", props={"extra": offence})
        station = _node(11, "Newcastle Bus Station", "LOC", props={"6.000000": "near"})
        kernel = _kernel(edge_name="record", source=weapons, target=station)

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)

        self.assertEqual(rewritten.kernel.source.type, "existential")
        self.assertEqual(rewritten.kernel.target, weapons)
        self.assertEqual(props["SPACE"], (station,))

    def test_content_source_location_target_without_spatial_preposition_is_left_alone(self):
        rule = self.rules["content_source_spatial_target_to_space"]
        weapons = _node(12, "weapons")
        station = _node(13, "Newcastle Bus Station", "LOC")
        kernel = _kernel(edge_name="record", source=weapons, target=station)

        rewritten = rule.run(kernel, self.ctx)

        self.assertEqual(rewritten.kernel.source, weapons)
        self.assertEqual(rewritten.kernel.target, station)
        self.assertNotIn("SPACE", dict(rewritten.properties))

    def test_location_source_spatial_target_is_left_alone(self):
        rule = self.rules["content_source_spatial_target_to_space"]
        origin = _node(14, "Newcastle", "GPE")
        station = _node(15, "Newcastle Bus Station", "LOC", props={"6.000000": "near"})
        kernel = _kernel(edge_name="record", source=origin, target=station)

        rewritten = rule.run(kernel, self.ctx)

        self.assertEqual(rewritten.kernel.source, origin)
        self.assertEqual(rewritten.kernel.target, station)
        self.assertNotIn("SPACE", dict(rewritten.properties))

    def test_quantity_drop_requires_and_target(self):
        rule = self.rules["quantity_redundant_drop"]
        target = SetOfSingletons(
            id=20,
            type=Grouping.AND,
            entities=(_node(4, "rain"), _node(5, "wind")),
            min=4,
            max=5,
            confidence=1.0,
        )
        kernel = _kernel(target=target, props={"QUANTITY": [_node(6, "% chance")]})

        rewritten = rule.run(kernel, self.ctx)

        self.assertNotIn("QUANTITY", dict(rewritten.properties))

    def test_lift_conjunct_context_to_kernel(self):
        rule = self.rules["lift_conjunct_context_to_kernel"]
        newcastle = _node(8, "Newcastle", "GPE")
        damage = _node(0, "damage", props={"SPACE": [newcastle]})
        arson = _node(1, "arson")
        target = SetOfSingletons(
            id=20, type=Grouping.AND, entities=(damage, arson),
            min=0, max=1, confidence=1.0,
        )
        church = _node(16, "St Nicholas' Church Yard", "LOC")
        kernel = _kernel(edge_name="record", target=target, props={"SPACE": [church]})

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)

        # Newcastle lifted off the conjunct and merged into the kernel SPACE list.
        space_names = [n.named_entity for n in as_list(props["SPACE"])]
        self.assertEqual(space_names, ["St Nicholas' Church Yard", "Newcastle"])
        lifted_damage = rewritten.kernel.target.entities[0]
        self.assertNotIn("SPACE", dict(lifted_damage.properties))

    def test_drop_temporal_from_space(self):
        rule = self.rules["drop_temporal_from_space"]
        church = _node(16, "St Nicholas' Church Yard", "LOC")
        newcastle = _node(6, "Newcastle", "GPE")
        date = _node(4, "2026-01", "DATE")
        kernel = _kernel(edge_name="record", props={"SPACE": [church, newcastle, date]})

        rewritten = rule.run(kernel, self.ctx)
        space_names = [n.named_entity for n in as_list(dict(rewritten.properties)["SPACE"])]

        self.assertEqual(space_names, ["St Nicholas' Church Yard", "Newcastle"])

    def test_promote_source_reduced_relative(self):
        rule = self.rules["promote_source_reduced_relative"]
        damage = _node(0, "damage", props={"amod": "criminal"})
        arson = _node(1, "arson")
        recorded = _node(2, "recorded", "verb")
        subject = SetOfSingletons(
            id=20, type=Grouping.AND, entities=(damage, arson, recorded),
            min=0, max=2, confidence=1.0,
        )
        rubbish_bin = _node(5, "rubbish bin", props={"det": "a", "actioned": "burning"})
        church = _node(9, "St Nicholas' Church Yard", "LOC")
        kernel = _kernel(edge_name="involve", source=subject, target=rubbish_bin,
                         props={"SPACE": [church]})

        rewritten = rule.run(kernel, self.ctx)

        # 'recorded' promoted to the predicate; subject nouns become the target.
        # (lemmatisation to "record" needs Stanza, unavailable in isolation.)
        self.assertTrue(rewritten.kernel.edgeLabel.named_entity.startswith("record"))
        self.assertEqual(rewritten.kernel.source.type, "existential")
        self.assertEqual(
            [n.named_entity for n in rewritten.kernel.target.entities],
            ["damage", "arson"],
        )
        # involve's object demoted to SPECIFICATION; SPACE preserved.
        spec_names = [n.named_entity for n in as_list(dict(rewritten.properties).get("SPECIFICATION"))]
        self.assertEqual(spec_names, ["rubbish bin"])
        self.assertIn("SPACE", dict(rewritten.properties))

    def test_promote_source_reduced_relative_skips_pure_noun_subject(self):
        rule = self.rules["promote_source_reduced_relative"]
        damage = _node(0, "damage")
        arson = _node(1, "arson")
        subject = SetOfSingletons(
            id=20, type=Grouping.AND, entities=(damage, arson),
            min=0, max=1, confidence=1.0,
        )
        kernel = _kernel(edge_name="involve", source=subject, target=_node(5, "bin"))

        self.assertIsNone(rule.matches(kernel, self.ctx))

    def test_flatten_nested_and(self):
        rule = self.rules["flatten_nested_and"]
        a = _node(10, "a")
        b = _node(11, "b")
        c = _node(12, "c")
        nested = SetOfSingletons(
            id=31,
            type=Grouping.AND,
            entities=(b, c),
            min=11,
            max=12,
            confidence=1.0,
        )
        target = SetOfSingletons(
            id=30,
            type=Grouping.AND,
            entities=(a, nested),
            min=10,
            max=12,
            confidence=1.0,
        )
        kernel = _kernel(target=target)

        rewritten = rule.run(kernel, self.ctx)

        self.assertEqual(rewritten.kernel.target.entities, (a, b, c))

    def test_rule_registry_trace_is_optional(self):
        rule = self.rules["quantity_redundant_drop"]
        target = SetOfSingletons(
            id=22,
            type=Grouping.AND,
            entities=(_node(13, "hail"), _node(14, "snow")),
            min=13,
            max=14,
            confidence=1.0,
        )
        kernel = _kernel(target=target, props={"QUANTITY": [_node(15, "% chance")]})
        trace = []
        ctx = RewriteContext(
            node_functions=None,
            services=_FakeServices(),
            matchers=_FakeMatchers(),
            trace=True,
            trace_log=trace,
        )

        rewritten = RuleRegistry([rule]).apply_phase(kernel, "post_logical_rewrite", ctx)

        self.assertNotIn("QUANTITY", dict(rewritten.properties))
        self.assertEqual(trace[0]["rule"], "quantity_redundant_drop")
        self.assertTrue(trace[0]["matched"])

    def test_specification_promotes_content_to_existential_target(self):
        rule = self.rules["specification_and_to_target"]
        existential = _node(16, "?", "existential")
        rain = _node(17, "rain")
        kernel = _kernel(target=existential, props={"SPECIFICATION": [rain]})

        rewritten = rule.run(kernel, self.ctx)

        self.assertEqual(rewritten.kernel.target, rain)
        self.assertNotIn("SPECIFICATION", dict(rewritten.properties))

    def test_auxiliary_periphrasis_moves_weather_causation_to_specification(self):
        rule = AuxiliaryPeriphrasisPromotionRule()
        chance = _node(51, "chance")
        rain_shower = _node(52, "rain shower", props={"amod": "light"})
        kernel = _forecast_periphrasis(
            props={"SPECIFICATION": [chance], "CAUSATION": [rain_shower]},
        )

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)
        spec_names = [node.named_entity for node in as_list(props.get("SPECIFICATION"))]

        self.assertEqual(rewritten.kernel.source.named_entity, "forecast")
        self.assertEqual(rewritten.kernel.target.type, "existential")
        self.assertEqual(spec_names, ["chance", "rain shower"])
        self.assertNotIn("CAUSATION", props)
        self.assertEqual([node.named_entity for node in as_list(props.get("SPACE"))], ["Newcastle"])

    def test_auxiliary_periphrasis_drops_empty_have_sibling(self):
        rule = AuxiliaryPeriphrasisPromotionRule()
        chance = _node(53, "chance")
        have = _node(54, "have", "verb", props={
            "mark": "to",
            "subjpass": "subjpass",
            "kernel": "root",
        })
        kernel = _forecast_periphrasis(
            source=have,
            props={"SPECIFICATION": [chance, have], "TOGETHERNESS": [have]},
        )

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)
        spec_names = [node.named_entity for node in as_list(props.get("SPECIFICATION"))]

        self.assertEqual(spec_names, ["chance"])
        self.assertNotIn("have", spec_names)
        self.assertNotIn("TOGETHERNESS", props)

    def test_auxiliary_periphrasis_drops_empty_have_inside_specification_and(self):
        rule = AuxiliaryPeriphrasisPromotionRule()
        chance = _node(61, "chance")
        have = _node(62, "have", "verb", props={
            "mark": "to",
            "subjpass": "subjpass",
            "kernel": "root",
        })
        grouped_spec = SetOfSingletons(
            id=63,
            type=Grouping.AND,
            entities=(chance, have),
            min=61,
            max=62,
            confidence=1.0,
        )
        kernel = _forecast_periphrasis(
            source=have,
            props={"SPECIFICATION": [grouped_spec]},
        )

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)
        spec_values = as_list(props.get("SPECIFICATION"))

        self.assertEqual(spec_values, [chance])

    def test_auxiliary_periphrasis_moves_weather_and_to_specification(self):
        rule = AuxiliaryPeriphrasisPromotionRule()
        rain = _node(58, "rain", props={"amod": "heavy"})
        conditions = _node(59, "conditions", props={"amod": "overcast"})
        have = _node(60, "have", "verb", props={
            "mark": "to",
            "subjpass": "subjpass",
            "kernel": "root",
        })
        kernel = _forecast_periphrasis(
            source=have,
            props={"AND": [rain, conditions]},
        )

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)
        spec_names = [node.named_entity for node in as_list(props.get("SPECIFICATION"))]

        self.assertEqual(spec_names, ["rain", "conditions"])
        self.assertNotIn("AND", props)
        self.assertNotIn("have", spec_names)

    def test_auxiliary_periphrasis_preserves_negated_weather_group(self):
        rule = AuxiliaryPeriphrasisPromotionRule()
        rain = _node(64, "rain")
        no_rain = SetOfSingletons(
            id=65,
            type=Grouping.NOT,
            entities=(rain,),
            min=64,
            max=64,
            confidence=1.0,
        )
        have = _node(66, "have", "verb", props={
            "mark": "to",
            "subjpass": "subjpass",
            "kernel": "root",
        })
        kernel = _forecast_periphrasis(
            source=have,
            props={"AND": [no_rain]},
        )

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)
        spec_values = as_list(props.get("SPECIFICATION"))

        self.assertEqual(spec_values, [no_rain])

    def test_auxiliary_periphrasis_recovers_negated_weather_from_have_object_graph(self):
        rule = AuxiliaryPeriphrasisPromotionRule()
        skies = _node(67, "skies", props={"amod": "clear"})
        group = _node(68, "skies rain")
        rain = _node(69, "rain")
        no = _node(70, "no", "NEG")
        have = _node(71, "have", "verb", props={
            "mark": "to",
            "subjpass": "subjpass",
            "kernel": "root",
        })
        graph = nx.DiGraph()
        for node in (have, group, skies, rain, no):
            graph.add_node(node.id, data=node)
        graph.add_edge(have.id, group.id, label=_node(72, "obj"))
        graph.add_edge(group.id, skies.id, label=_node(73, "orig"))
        graph.add_edge(group.id, rain.id, label=_node(74, "orig"))
        graph.add_edge(rain.id, no.id, label=_node(75, "neg"))
        ctx = RewriteContext(
            node_functions=None,
            services=_FakeServices(),
            matchers=_FakeMatchers(G=graph),
        )
        kernel = _forecast_periphrasis(
            source=have,
            props={"AND": [skies]},
        )

        rewritten = rule.run(kernel, ctx)
        props = dict(rewritten.properties)
        spec_values = as_list(props.get("SPECIFICATION"))
        negated_values = [
            value for value in spec_values
            if isinstance(value, SetOfSingletons) and value.type == Grouping.NOT
        ]

        self.assertEqual([value.entities[0].named_entity for value in negated_values], ["rain"])

    def test_auxiliary_periphrasis_recovers_negated_weather_from_have_edge_graph(self):
        rule = AuxiliaryPeriphrasisPromotionRule()
        skies = _node(76, "skies", props={"amod": "clear"})
        group = _node(77, "skies rain")
        rain = _node(78, "rain")
        no = _node(79, "no", "NEG")
        have_edge = _node(3, "have", "verb", props={"root": "root"})
        graph = nx.DiGraph()
        for node in (have_edge, group, skies, rain, no):
            graph.add_node(node.id, data=node)
        graph.add_edge(have_edge.id, group.id, label=_node(80, "obj"))
        graph.add_edge(group.id, skies.id, label=_node(81, "orig"))
        graph.add_edge(group.id, rain.id, label=_node(82, "orig"))
        graph.add_edge(rain.id, no.id, label=_node(83, "neg"))
        ctx = RewriteContext(
            node_functions=None,
            services=_FakeServices(),
            matchers=_FakeMatchers(G=graph),
        )
        kernel = _forecast_periphrasis(
            source=_node(84, "?", "existential"),
            props={"AND": [skies]},
        )

        rewritten = rule.run(kernel, ctx)
        props = dict(rewritten.properties)
        spec_values = as_list(props.get("SPECIFICATION"))
        negated_values = [
            value for value in spec_values
            if isinstance(value, SetOfSingletons) and value.type == Grouping.NOT
        ]

        self.assertEqual([value.entities[0].named_entity for value in negated_values], ["rain"])

    def test_auxiliary_periphrasis_preserves_non_weather_causation(self):
        rule = AuxiliaryPeriphrasisPromotionRule()
        chance = _node(55, "chance")
        damage = _node(56, "storm damage")
        have = _node(57, "have", "verb", props={
            "mark": "to",
            "subjpass": "subjpass",
            "kernel": "root",
        })
        kernel = _forecast_periphrasis(
            source=have,
            props={"SPECIFICATION": [chance], "CAUSATION": [damage]},
        )

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)
        spec_names = [node.named_entity for node in as_list(props.get("SPECIFICATION"))]
        cause_names = [node.named_entity for node in as_list(props.get("CAUSATION"))]

        self.assertEqual(spec_names, ["chance"])
        self.assertEqual(cause_names, ["storm damage"])

    def test_state_cause_access_point_is_data_backed(self):
        rule = self.rules["state_cause_access_point_swap"]
        fire = _node(18, "fire")
        station = _node(19, "station", "GPE")
        entrance = _node(20, "public entrance")
        kernel = _kernel(edge_name="open", source=fire, target=station, props={"SPACE": [entrance]})

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)

        self.assertEqual(rewritten.kernel.source.type, "existential")
        self.assertEqual(rewritten.kernel.target, entrance)
        self.assertEqual(props["CAUSATION"], (fire,))
        self.assertEqual(props["SPACE"], (station,))

    def test_occurrence_context_moves_to_causation(self):
        rule = self.rules["after_occurrence_context_to_causation"]
        event = _node(21, "storm damage", props={"type": "occur"})
        kernel = _kernel(edge_name="close", props={"TEMPORAL_CONTEXT": [event]})

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)

        self.assertNotIn("TEMPORAL_CONTEXT", props)
        self.assertEqual(props["CAUSATION"], (event,))

    def test_modifier_split_as_sibling(self):
        rule = self.rules["weather_amod_condition"]
        wind = _node(22, "wind", props={"amod": ["cloudy"], "lemma": ["wind", "cloudy"]})
        kernel = _kernel(edge_name="be", target=wind)

        rewritten = rule.run(kernel, self.ctx)
        target = rewritten.kernel.target

        self.assertIsInstance(target, SetOfSingletons)
        self.assertEqual([entity.named_entity for entity in target.entities], ["wind", "cloudy"])
        self.assertNotIn("amod", dict(target.entities[0].properties))

    def test_quantity_compound_merge_is_declarative(self):
        rule = self.rules["quantity_compound_merge"]
        chance = _node(24, "chance", props={"pos": "2"})
        target = SetOfSingletons(
            id=23,
            type=Grouping.AND,
            entities=(chance,),
            min=23,
            max=24,
            confidence=1.0,
        )
        percent = _node(25, "%", props={"pos": "1", "nummod": "57"})
        kernel = _kernel(target=target, props={"QUANTITY": [percent]})

        rewritten = rule.run(kernel, self.ctx)
        merged = rewritten.kernel.target.entities[0]

        self.assertEqual(merged.named_entity, "% chance")
        self.assertEqual(dict(merged.properties)["nummod"], "57")
        self.assertNotIn("QUANTITY", dict(rewritten.properties))

    def test_lifecycle_subject_promotion_swaps_extra_head(self):
        rule = self.rules["lifecycle_subject_promotion"]
        precipitation = _node(27, "precipitation")
        chance = _node(26, "chance", props={"extra": precipitation})
        kernel = _kernel(target=chance)

        rewritten = rule.run(kernel, self.ctx)
        target = rewritten.kernel.target

        self.assertEqual(target.named_entity, "precipitation")
        self.assertEqual(dict(target.properties)["extra"].named_entity, "chance")

    def test_action_object_promotion_abandon(self):
        from LaSSI.ner.structural_rewrites.action_object_promotion import (
            ActionObjectPromotionRule,
        )
        rule = ActionObjectPromotionRule()
        si = _node(32, "SI", "noun")
        m11_c = _node(33, "11m", "noun", props={"extra": _node(31, "approx", "")})
        six = _node(35, "6", "noun", props={"punct": '"'})
        m11_s = _node(34, "11m", "noun",
                      props={"extra": [_node(36, "approx", ""), six]})
        kernel = _kernel(edge_name="Abandon", source=_node(1, "?", "existential"),
                         target=None,
                         props={"CAUSATION": [si, m11_c], "SPECIFICATION": [m11_s]})
        ctx = RewriteContext(node_functions=None, services=_FakeServices(),
                             matchers=_ActionMatchers())

        rewritten = rule.run(kernel, ctx)
        target = rewritten.kernel.target
        props = dict(rewritten.properties)

        self.assertEqual(target.named_entity, "SI")
        measure_names = sorted(m.named_entity for m in dict(target.properties)["MEASURE"])
        self.assertEqual(measure_names, ["11m", "6"])
        self.assertNotIn("CAUSATION", props)
        self.assertNotIn("SPECIFICATION", props)

    def test_action_object_promotion_replace(self):
        from LaSSI.ner.structural_rewrites.action_object_promotion import (
            ActionObjectPromotionRule,
        )
        rule = ActionObjectPromotionRule()
        lppe = _node(42, "LP PE", "LOC", props={"5": "of"})
        m11_s = _node(43, "11m", "noun",
                      props={"extra": [_node(41, "approx", ""), lppe], "2": "with"})
        m11_t = _node(44, "11m", "noun",
                      props={"extra": _node(45, "approx", ""), "2": "with"})
        kernel = _kernel(edge_name="replace", source=_node(1, "?", "existential"),
                         target=None,
                         props={"SPECIFICATION": [m11_s], "TOGETHERNESS": [m11_t]})
        ctx = RewriteContext(node_functions=None, services=_FakeServices(),
                             matchers=_ActionMatchers())

        rewritten = rule.run(kernel, ctx)
        target = rewritten.kernel.target

        self.assertEqual(target.named_entity, "LP PE")
        self.assertEqual([m.named_entity for m in dict(target.properties)["MEASURE"]],
                         ["11m"])
        self.assertNotIn("TOGETHERNESS", dict(rewritten.properties))

    def test_action_object_promotion_skips_without_measure(self):
        # A causation clause with no measure must be left untouched.
        from LaSSI.ner.structural_rewrites.action_object_promotion import (
            ActionObjectPromotionRule,
        )
        rule = ActionObjectPromotionRule()
        damage = _node(50, "damage", "noun")
        kernel = _kernel(edge_name="Abandon", source=_node(1, "?", "existential"),
                         target=None, props={"CAUSATION": [damage]})
        ctx = RewriteContext(node_functions=None, services=_FakeServices(),
                             matchers=_ActionMatchers())

        rewritten = rule.run(kernel, ctx)
        self.assertIsNone(rewritten.kernel.target)
        self.assertIn("CAUSATION", dict(rewritten.properties))

    def test_reporting_reduced_relative_root_preferred_over_lifecycle_outcome(self):
        CreateFinalKernelX = _create_final_kernel_x_class()

        builder = object.__new__(CreateFinalKernelX)
        builder.G = nx.DiGraph()
        builder.post = SimpleNamespace(matchers=_FakeMatchers())
        subject = _node(70, "arson investigation")
        record_kernel = _kernel(
            edge_name="record",
            source=_node(71, "?1", "existential"),
            target=subject,
        )
        conclude_kernel = _kernel(
            edge_name="conclude",
            source=_node(72, "?2", "existential"),
            target=_node(73, "outcome"),
        )
        builder.G.add_node(10, data=record_kernel)
        builder.G.add_node(11, data=conclude_kernel)
        builder.G.add_node(subject.id, data=subject)
        builder.G.add_edge(10, subject.id, label=_node(74, "nsubjpass"))
        builder.G.add_edge(11, subject.id, label=_node(75, "nsubj"))

        self.assertEqual(
            builder._preferred_reduced_relative_reporting_root_id(11, [10, 11], [10, 11]),
            10,
        )

    def test_reporting_root_not_preferred_without_passive_subject(self):
        CreateFinalKernelX = _create_final_kernel_x_class()

        builder = object.__new__(CreateFinalKernelX)
        builder.G = nx.DiGraph()
        builder.post = SimpleNamespace(matchers=_FakeMatchers())
        subject = _node(76, "arson investigation")
        record_kernel = _kernel(edge_name="record", target=subject)
        conclude_kernel = _kernel(edge_name="conclude", target=_node(77, "outcome"))
        builder.G.add_node(10, data=record_kernel)
        builder.G.add_node(11, data=conclude_kernel)
        builder.G.add_node(subject.id, data=subject)
        builder.G.add_edge(10, subject.id, label=_node(78, "nsubj"))
        builder.G.add_edge(11, subject.id, label=_node(79, "nsubj"))

        self.assertIsNone(
            builder._preferred_reduced_relative_reporting_root_id(11, [10, 11], [10, 11])
        )

    def test_embedded_reporting_promoted_from_lifecycle_outcome(self):
        CreateFinalKernelX = _create_final_kernel_x_class()

        builder = object.__new__(CreateFinalKernelX)
        builder.post = SimpleNamespace(matchers=_FakeMatchers())
        recorded = _node(80, "recorded", "verb", props={"lemma": "record"})
        damage = _node(81, "damage")
        investigation = _node(82, "arson investigation")
        concluded = _node(85, "concluded", "verb")
        suspect = _node(86, "suspect")
        target = SetOfSingletons(
            id=83,
            type=Grouping.AND,
            entities=[recorded, damage],
            min=1,
            max=3,
            confidence=1.0,
            root=True,
        )
        kernel = _kernel(
            edge_name="conclude",
            source=_node(84, "?1", "existential"),
            target=target,
            props={"TIME_STATUS": [investigation], "SPECIFICATION": [concluded, suspect]},
        )

        rewritten = builder._promote_embedded_reduced_relative_reporting_root(kernel)

        self.assertEqual(rewritten.kernel.edgeLabel.named_entity, "record")
        self.assertEqual(
            [node.named_entity for node in rewritten.kernel.target.entities],
            ["damage", "arson"],
        )
        status = as_list(dict(rewritten.properties)["TIME_STATUS"])[0]
        self.assertEqual(status.named_entity, "investigation")
        self.assertEqual(dict(status.properties)["type"], "complete")
        specs = as_list(dict(rewritten.properties)["SPECIFICATION"])
        self.assertEqual(specs, [suspect])

    def test_lifecycle_outcome_sentence_projects_to_time_status(self):
        from LaSSI.ner.structural_rewrites.lifecycle_property_promotion import (
            LifecyclePropertyPromotionRule,
        )

        rule = LifecyclePropertyPromotionRule()
        investigation = _node(80, "investigation")
        outcome = _kernel(
            edge_name="conclude",
            source=_node(81, "?1", "existential"),
            target=None,
            props={"TIME_STATUS": [investigation]},
        )
        kernel = _kernel(edge_name="record", props={"SENTENCE": [outcome]})

        rewritten = rule.run(kernel, self.ctx)
        props = dict(rewritten.properties)
        status = as_list(props["TIME_STATUS"])[0]

        self.assertEqual(status.named_entity, "investigation")
        self.assertEqual(dict(status.properties)["type"], "complete")
        self.assertNotIn("SENTENCE", props)

    def test_causal_signal_result_builds_binary_cause_effect(self):
        from LaSSI.ner.structural_rewrites.causal_signal_result import (
            CausalSignalResultRule,
        )

        offence = _node(90, "offence")
        arson = _node(91, "arson")
        damage = _node(92, "damage", props={"amod": "criminal"})
        suspect = _node(93, "suspect", props={"det": "a", "19.000000": "in"})
        arrested = _node(94, "arrested", "verb", props={"actioned": "being", "pos": "20"})
        charged = _node(95, "charged", "verb", props={"pos": "22"})
        graph = nx.DiGraph()
        for node in (offence, arson, damage, suspect, arrested, charged):
            graph.add_node(node.id, data=node)
        graph.add_edge(offence.id, arson.id, label=_node(96, "compound"))
        ctx = RewriteContext(
            node_functions=None,
            services=_FakeServices(),
            matchers=_FakeMatchers(G=graph),
        )
        kernel = _kernel(
            edge_name="result",
            source=_node(97, "?1", "existential"),
            target=None,
            props={"CAUSATION": [arrested, charged, offence, arson, damage, suspect]},
        )

        rewritten = CausalSignalResultRule().run(kernel, ctx)
        props = dict(rewritten.properties)

        self.assertNotIn("CAUSATION", props)
        self.assertEqual(
            [node.named_entity for node in rewritten.kernel.source.entities],
            ["arson", "damage"],
        )
        self.assertEqual(as_list(dict(rewritten.kernel.source.entities[0].properties)["extra"]), [offence])
        self.assertEqual(rewritten.kernel.target.named_entity, "suspect")
        target_props = dict(rewritten.kernel.target.properties)
        self.assertEqual(target_props["actioned"], ("being", "arrested"))
        self.assertEqual([node.named_entity for node in target_props["extra"]], ["charged"])
        self.assertNotIn("19.000000", target_props)

    def test_causal_signal_result_preserves_unrelated_causation(self):
        from LaSSI.ner.structural_rewrites.causal_signal_result import (
            CausalSignalResultRule,
        )

        damage = _node(98, "damage")
        kernel = _kernel(
            edge_name="record",
            source=_node(99, "?1", "existential"),
            target=None,
            props={"CAUSATION": [damage]},
        )

        rewritten = CausalSignalResultRule().run(kernel, self.ctx)

        self.assertIn("CAUSATION", dict(rewritten.properties))
        self.assertIsNone(rewritten.kernel.target)

    # ------------------------------------------------------------------ #
    # transport_004: "out of use" footbridge notices                     #
    # ------------------------------------------------------------------ #

    def test_pseudo_verb_copula_recovery_rebuilds_copula(self):
        """S1 headline parse: footbridge⁽ᵛᵉʳᵇ⁾(?1[cop: out of use], None)
        [SPECIFICATION: station[extra: footbridge], SPACE: Whitley Bay]
        → be(footbridge, out of use)[SPACE: Whitley Bay[extra: station]]."""
        from LaSSI.ner.structural_rewrites.pseudo_verb_copula_recovery import (
            PseudoVerbCopulaRecoveryRule,
        )

        ctx = RewriteContext(
            node_functions=None,
            services=_FakeServices(),
            matchers=_AccessPointMatchers(),
        )
        out_of_use = _node(15, "out of use", "JJ")
        subject = _node(14, "footbridge", "noun", props={"amod": "smaller", "kernel": "kernel"})
        station = _node(9, "station", "noun", props={"extra": [subject]})
        whitley = _node(4, "Whitley Bay", "LOC")
        source = _node(1, "?1", "existential", props={"cop": out_of_use})
        kernel = _kernel(
            edge_name="footbridge",
            source=source,
            target=None,
            props={"SPECIFICATION": [station], "SPACE": [whitley]},
        )

        rewritten = PseudoVerbCopulaRecoveryRule().run(kernel, ctx)

        self.assertEqual(rewritten.kernel.edgeLabel.named_entity, "be")
        self.assertEqual(rewritten.kernel.source.named_entity, "footbridge")
        self.assertEqual(rewritten.kernel.target.named_entity, "out of use")
        props = dict(rewritten.properties)
        self.assertNotIn("SPECIFICATION", props)
        space = as_list(props.get("SPACE"))
        self.assertEqual(len(space), 1)
        self.assertEqual(space[0].named_entity, "Whitley Bay")
        space_extras = [e.named_entity for e in as_list(dict(space[0].properties).get("extra"))]
        self.assertIn("station", space_extras)

    def test_pseudo_verb_recovery_ignores_real_verb(self):
        """A genuine verb root must not be rewritten."""
        from LaSSI.ner.structural_rewrites.pseudo_verb_copula_recovery import (
            PseudoVerbCopulaRecoveryRule,
        )

        ctx = RewriteContext(
            node_functions=None,
            services=_FakeServices(),
            matchers=_AccessPointMatchers(),
        )
        kernel = _kernel(
            edge_name="record",
            source=_node(1, "?1", "existential", props={"cop": _node(2, "open", "JJ")}),
            target=None,
        )
        self.assertIsNone(PseudoVerbCopulaRecoveryRule().matches(kernel, ctx))

    def test_copula_complement_promotion_promotes_cop_to_target(self):
        """S4 causal parse: be(footbridge[cop: out of use], None)
        [SPECIFICATION: out of use, CAUSATION: concerns]
        → be(footbridge, out of use)[CAUSATION: concerns]."""
        from LaSSI.ner.structural_rewrites.copula_complement_promotion import (
            CopulaComplementPromotionRule,
        )

        out_of_use = _node(15, "out of use", "JJ")
        source = _node(1, "footbridge", "noun", props={"cop": out_of_use, "amod": "smaller"})
        spec_dup = _node(16, "out of use", "JJ")
        causation = _node(23, "concerns", "noun")
        kernel = _kernel(
            edge_name="be",
            source=source,
            target=None,
            props={"SPECIFICATION": [spec_dup], "CAUSATION": [causation]},
        )

        rewritten = CopulaComplementPromotionRule().run(kernel, self.ctx)

        self.assertEqual(rewritten.kernel.target.named_entity, "out of use")
        self.assertNotIn("cop", dict(rewritten.kernel.source.properties))
        props = dict(rewritten.properties)
        self.assertNotIn("SPECIFICATION", props)  # duplicate of the predicate dropped
        self.assertIn("CAUSATION", props)

    def test_copula_complement_promotion_ignores_well_formed_target(self):
        """A copula that already has a real target is left untouched."""
        from LaSSI.ner.structural_rewrites.copula_complement_promotion import (
            CopulaComplementPromotionRule,
        )

        kernel = _kernel(
            edge_name="be",
            source=_node(1, "footbridge", "noun", props={"cop": _node(2, "out of use", "JJ")}),
            target=_node(3, "operational", "JJ"),
        )
        self.assertIsNone(CopulaComplementPromotionRule().matches(kernel, self.ctx))

    def test_chained_preposition_markers_reassemble_out_of_use(self):
        """GraphBuilder pre-pass: use --case--> out(IN) --none--> of(IN) →
        rename the governed noun to the HOnK adjective "out of use" (JJ)."""
        from LaSSI.ner.GraphBuilder import GraphBuilder

        class _HOnKTypeOf:
            def typeOf(self, name):
                return {"https://ofox.co.uk/honk#JJ"} if str(name).lower() == "out of use" else set()

        gsm = [
            {"id": 0, "ell": ["noun"], "xi": ["use"], "properties": {"pos": "13.0"},
             "phi": [{"containment": "case", "content": 5, "score": {"parent": 0, "child": 5}}]},
            {"id": 5, "ell": ["IN"], "xi": ["out"], "properties": {"pos": "11.0"},
             "phi": [{"containment": "none", "content": 13, "score": {"parent": 5, "child": 13}}]},
            {"id": 13, "ell": ["IN"], "xi": ["of"], "properties": {"pos": "12.0"}, "phi": []},
        ]
        GraphBuilder(None, _HOnKTypeOf())._lift_chained_preposition_markers(gsm)

        self.assertEqual(gsm[0]["xi"], ["out of use"])
        self.assertEqual(gsm[0]["ell"], ["JJ"])

    def test_lifecycle_out_of_use_contradicts_operational(self):
        """The transport-availability dimension makes 'out of use' EXCLUDE
        'operational' (and not contradict another 'out of use')."""
        from LaSSI.structures.extended_fol.Formulae import FVariable, FUnaryPredicate
        from LaSSI.HOnK.TBox.LifecycleManager import _lifecycle_partition_verdict

        def pred(cop):
            return FUnaryPredicate(
                rel="be",
                arg=FVariable(name="footbridge", type="existential"),
                score=1.0,
                properties=frozenset({("cop", FVariable(name=cop, type="existential"))}),
            )

        self.assertEqual(
            _lifecycle_partition_verdict(pred("out of use"), pred("operational")),
            "contradiction",
        )
        self.assertIsNone(
            _lifecycle_partition_verdict(pred("out of use"), pred("out of use"))
        )


class _AccessPointMatchers(_FakeMatchers):
    """Adds "footbridge" to the access-point class for the recovery rule."""

    def matches_class(self, value, class_name, *, kernel=None):
        name = getattr(value, "named_entity", value)
        name = str(name).strip().lower() if name is not None else ""
        if class_name == "AccessPointLike":
            return name in {"footbridge", "footbridges", "public entrance", "access point"}
        return super().matches_class(value, class_name, kernel=kernel)


if __name__ == "__main__":
    unittest.main()
