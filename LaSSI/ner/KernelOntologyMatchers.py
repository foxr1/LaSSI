__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"
__status__ = "Production"

from LaSSI.external_services.Services import Services
from LaSSI.ner.HOnKLogicalRewriting import get_matching_logical_rules
from LaSSI.ner.string_functions import lemmatize_verb, surface_form_variants
from LaSSI.structures.internal_graph.EntityRelationship import Grouping, Singleton, SetOfSingletons
from LaSSI.structures.kernels.SentenceX import case_in_props


class KernelOntologyMatchers:
    """Pure HOnK-driven predicates over Singletons / SetOfSingletons.

    Held by `KernelPostProcessor` and `CreateFinalKernelX` so post-processing
    code can ask "is this node a location?" / "is this edge stative?" without
    reaching into the ontology directly. All lookups are mediated by HOnK
    getters; nothing here is hardcoded to a specific lemma."""

    def __init__(self, G=None):
        self.services = Services.getInstance()
        self.G = G

    # ---- name expansion / generic ontology-set membership ----

    def name_candidates(self, value):
        if value is None:
            return set()
        if isinstance(value, str):
            raw = {value, lemmatize_verb(value)}
            for variant in surface_form_variants(value):
                raw.add(variant)
                raw.add(lemmatize_verb(variant))
            for part in value.split():
                raw.add(part)
                raw.add(lemmatize_verb(part))
                for variant in surface_form_variants(part):
                    raw.add(variant)
                    raw.add(lemmatize_verb(variant))
            return {candidate for candidate in raw if candidate}
        if isinstance(value, Singleton):
            candidates = self.name_candidates(value.named_entity)
            for key, prop_value in dict(value.properties).items():
                candidates |= self.name_candidates(key)
                candidates |= self.name_candidates(prop_value)
            return candidates
        if isinstance(value, SetOfSingletons):
            candidates = set()
            for entity in value.entities:
                candidates |= self.name_candidates(entity)
            return candidates
        if isinstance(value, (list, tuple, set)):
            candidates = set()
            for item in value:
                candidates |= self.name_candidates(item)
            return candidates
        return self.name_candidates(str(value))

    def matches_honk_set(self, value, honk_values):
        if not honk_values:
            return False
        candidates = {str(candidate).lower() for candidate in self.name_candidates(value)}
        ontology_terms = {str(term).lower() for term in honk_values if term}
        return bool(candidates & ontology_terms)

    def matches_class(self, value, class_name, *, kernel=None):
        """Generic HOnK/raw-data class predicate used by declarative rewrites."""
        if value is None or not class_name:
            return False
        name = str(class_name)
        honk = self.services.getHOnK()
        class_getters = {
            "AccessPointNoun": honk.getAccessPointNouns,
            "CausativeVerb": honk.getCausativeVerbs,
            "MaterialisationVerb": honk.getMaterialisationVerbs,
            "ChangeOfStateVerb": honk.getChangeOfStateVerbs,
            "FacilityNoun": honk.getFacilityNouns,
            "LocationNoun": honk.getLocationNouns,
            "OccurrenceVerb": honk.getOccurrenceVerbs,
            "PredictionVerb": honk.getPredictionVerbs,
            "RelativePronoun": honk.getRelativePronouns,
            "RouteNoun": honk.getRouteNouns,
            "ServiceStateNoun": honk.getServiceStateNouns,
            "StateNoun": honk.getStateNouns,
            "StateVerb": honk.getStateVerbs,
            "StatusNoun": honk.getStatusNouns,
            "TransitiveVerb": honk.getTransitiveVerbs,
            "EventClassifierHeadNoun": honk.getEventClassifierHeadNouns,
            "DisruptionNoun": honk.getDisruptionNouns,
            "WeatherConditionAdjective": honk.getWeatherConditionAdjectives,
            "WeatherConditionNoun": honk.getWeatherConditionNouns,
        }
        if name == "LocationLike":
            return self.is_location_like(value)
        if name == "Copula":
            return self.is_copula_node(value)
        if name == "AccessPointLike":
            return self.has_access_point(value)
        if name == "CausalNode":
            return kernel is not None and self.is_causal_node(kernel, value)
        if name == "DateLike":
            return isinstance(value, Singleton) and str(getattr(value, "type", "")).upper() in {"DATE", "TIME", "SUTIME"}
        if name == "ContextNode":
            return self.is_context_node(value)
        if name == "ContentNode":
            return self.is_content_node(value)
        if name == "LifecycleHeadPhrase":
            return self.matches_lifecycle_head_phrase(value)
        getter = class_getters.get(name)
        return self.matches_honk_set(value, getter() if getter is not None else set())

    def is_context_node(self, node):
        if isinstance(node, SetOfSingletons):
            return all(self.is_context_node(entity) for entity in node.entities)
        if not isinstance(node, Singleton):
            return False
        try:
            from LaSSI.ner.structural_rewrites.declarative import structural_lexical_set
            context_types = structural_lexical_set("context_entity_types")
        except Exception:
            context_types = set()
        if not context_types:
            context_types = {"DATE", "TIME", "SUTime", "GPE", "LOC", "FAC", "existential"}
        return str(getattr(node, "type", "") or "") in context_types

    def is_content_node(self, node):
        if isinstance(node, SetOfSingletons):
            if node.type in {Grouping.AND, Grouping.OR}:
                return len(node.entities) > 0 and any(self.is_content_node(entity) for entity in node.entities)
            return True
        if not isinstance(node, Singleton):
            return False
        return not self.is_context_node(node)

    def matches_lifecycle_head_phrase(self, value):
        if not isinstance(value, Singleton) or not value.named_entity:
            return False
        try:
            from LaSSI.HOnK.TBox.LifecycleManager import _get_lifecycle_phrases
            phrases = _get_lifecycle_phrases()
        except Exception:
            phrases = {}
        label = value.named_entity.strip().lower()
        return any(
            part != "descriptive"
            for _, part in phrases.get(label, set())
        )

    def dependency_children(self, node, labels):
        graph = self.G
        if graph is None or not isinstance(node, Singleton) or node.id not in graph:
            return []
        label_names = set(labels or [])
        children = []
        for _, child_id, data in graph.out_edges(node.id, data=True):
            label = data.get("label")
            label_name = getattr(label, "named_entity", None)
            if label_name not in label_names:
                continue
            if child_id in graph:
                child = graph.nodes[child_id].get("data")
                if child is not None:
                    children.append(child)
        return children

    def reachable_ids(self, node):
        ids = set()
        if isinstance(node, Singleton):
            if node.id is not None:
                ids.add(node.id)
            for value in dict(node.properties or {}).values():
                ids.update(self.reachable_ids(value))
            if node.kernel is not None:
                ids.update(self.reachable_ids(node.kernel.source))
                ids.update(self.reachable_ids(node.kernel.target))
                ids.update(self.reachable_ids(node.kernel.edgeLabel))
        elif isinstance(node, SetOfSingletons):
            if node.id is not None:
                ids.add(node.id)
            for entity in node.entities:
                ids.update(self.reachable_ids(entity))
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                ids.update(self.reachable_ids(item))
        return ids

    # ---- copula / stative classification ----

    def is_stative_lemma(self, verb_lemma):
        """Stative verbs (remain/stay/pause/etc.) come from HOnK's StateVerb set;
        the bare copula 'be' surface forms function equivalently for the swap
        heuristic and are sourced from HOnK.getCopulaSurfaceForms()."""
        if not verb_lemma:
            return False
        lemma = verb_lemma.lower().strip()
        if not lemma:
            return False
        try:
            copula_forms = self.services.getHOnK().getCopulaSurfaceForms() or set()
        except Exception:
            copula_forms = set()
        if lemma in {str(f).lower() for f in copula_forms}:
            return True
        try:
            state_verbs = self.services.getHOnK().getStateVerbs() or set()
        except Exception:
            state_verbs = set()
        return lemma in {str(v).lower() for v in state_verbs}

    def is_preposition_name(self, name):
        """A node's named_entity counts as a bare preposition when HOnK's
        prototypical preposition set lists it. Used to spot orphan 'on'/'in'/etc.
        entries that leaked into SPACE without being absorbed into a location."""
        if not isinstance(name, str):
            return False
        token = name.strip().lower()
        if not token:
            return False
        try:
            preps = self.services.getHOnK().getPrototypicalPrepositions() or set()
        except Exception:
            preps = set()
        return token in {str(p).lower() for p in preps}

    def is_state_edge(self, edge_label):
        return isinstance(edge_label, Singleton) and self.matches_honk_set(
            edge_label.named_entity, self.services.getHOnK().getStateVerbs())

    def is_copula_node(self, value):
        """True when *value*'s surface form is a copula ('be' and its inflected
        / contracted forms). Backed by HOnK.getCopulaSurfaceForms() so the
        membership stays in the ontology, not a hardcoded list."""
        name = value.named_entity if isinstance(value, Singleton) else value
        if not isinstance(name, str) or not name.strip():
            return False
        try:
            copula_forms = self.services.getHOnK().getCopulaSurfaceForms() or set()
        except Exception:
            copula_forms = set()
        copula_lower = {str(form).lower() for form in copula_forms} or {"be"}
        parts = [part for part in name.split() if part]
        return all(
            part.lower() in copula_lower or lemmatize_verb(part).lower() in copula_lower
            for part in parts
        )

    # ---- entity-class predicates ----

    def is_location_like(self, node):
        if isinstance(node, SetOfSingletons):
            return any(self.is_location_like(entity) for entity in node.entities)
        if not isinstance(node, Singleton):
            return False
        honk = self.services.getHOnK()
        return (
                node.type in {"GPE", "LOC"} or
                self.matches_honk_set(node, honk.getLocationNouns()) or
                self.matches_honk_set(node, honk.getFacilityNouns()) or
                self.matches_honk_set(node, honk.getRouteNouns())
        )

    def has_access_point(self, node):
        if isinstance(node, SetOfSingletons):
            return any(self.has_access_point(entity) for entity in node.entities)
        if not isinstance(node, Singleton):
            return False
        return self.matches_honk_set(node, self.services.getHOnK().getAccessPointNouns())

    # ---- logical-rule introspection ----

    def logical_rule_context(self, node):
        target = node.kernel.target if (
            isinstance(node, Singleton) and node.kernel is not None and isinstance(node.kernel.target, Singleton)
        ) else node
        context = {
            "has_case": isinstance(target, Singleton) and case_in_props(dict(target.properties)),
            "incoming_edge_labels": tuple(),
            "dependency_labels": tuple(),
        }
        if not isinstance(target, Singleton) or self.G is None or target.id not in self.G:
            return context
        edge_labels = []
        for edge in self.G.in_edges(target.id, data=True):
            edge_label = edge[2].get('label')
            if isinstance(edge_label, Singleton):
                edge_labels.append(edge_label.named_entity)
        context["incoming_edge_labels"] = tuple(edge_labels)
        context["dependency_labels"] = tuple(edge_labels)
        return context

    def logical_construct_for_node(self, kernel, node):
        if not isinstance(node, Singleton):
            return None
        try:
            _, selected_rule = get_matching_logical_rules(
                kernel, node, False, structural_context=self.logical_rule_context(node))
        except Exception:
            return None
        return getattr(selected_rule, "logicalConstructName", None) if selected_rule is not None else None

    def is_causal_node(self, kernel, node):
        if not isinstance(node, Singleton):
            return False
        if self.logical_construct_for_node(kernel, node) == "causation":
            return True
        honk = self.services.getHOnK()
        return (
                self.matches_honk_set(node, honk.getCausativeVerbs()) or
                self.matches_honk_set(node, honk.getStateNouns())
        )

    # ---- preposition → logical construct lookup ----

    def prepositions_for_construct(self, construct_name):
        """All preposition surface forms (lowercase) that HOnK's logical
        rewriting rules classify under the given construct name (e.g.
        ``causation``, ``time``).  Derived from the ``Preposition: [...]``
        premises of rules in ``raw_data/logical_analysis.json``; nothing here
        is hardcoded to a specific lemma.

        Includes prepositions from a rule's primary classification and any
        ``additional_classifications`` (e.g. ``on or near`` contributes to
        both ``stay in place`` and ``near place``).

        Returns a frozenset, cached on the instance so repeated callers
        (e.g. a structural rewrite rule's ``matches``) don't re-scan the
        ontology each invocation."""
        cache = self.__dict__.setdefault('_prepositions_for_construct_cache', {})
        key = construct_name.lower()
        if key in cache:
            return cache[key]

        from LaSSI.ner.SemanticRoleRewriting import rule_classifications

        forms = set()
        try:
            rules = self.services.getHOnK().getLogicalRewritingRules() or {}
        except Exception:
            rules = {}
        for rule in rules.values():
            classifications = rule_classifications(rule)
            if not any(
                    c.construct_name and c.construct_name.lower() == key
                    for c in classifications
            ):
                continue
            for cond in getattr(rule, 'premises', None) or []:
                if cond.name != 'Preposition':
                    continue
                for val in cond.values:
                    if isinstance(val, str) and val:
                        forms.add(val.lower())

        result = frozenset(forms)
        cache[key] = result
        return result
