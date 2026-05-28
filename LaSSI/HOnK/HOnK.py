import dataclasses
import io
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass
import os.path
from enum import Enum
from functools import lru_cache
from typing import Optional, List

from FunctionalMatch.rdf.RDFGraph import Literal, XSD, RDFGraph

from LaSSI.HOnK import SentenceStructure, Prepositions


import logging


def _equivalence_closure(pairs):
    """
    Build a synonym dict from equivalence (eq) pairs using Union-Find.

    This replaces the naive O(n²·d) iterative transitive_closure for the
    special case of a symmetric+transitive (equivalence) relation.  Union-Find
    runs in O(n·α(n)) ≈ O(n) and produces the same result.

    Returns a defaultdict(set): term → set of all synonymous terms (excluding
    itself), matching the shape expected by getSynonymy().
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        parent.setdefault(x, x)
        parent.setdefault(y, y)
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for x, y in pairs:
        union(x, y)

    # Group all terms by their equivalence-class root
    groups: dict = defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)

    # Build the bidirectional lookup dict.
    # The original transitive_closure produces self-loops (A,A) from symmetric
    # pairs, so tmp[A] ends up containing A itself.  Replicate that here so
    # getSynonymy always returns a set that includes the queried term.
    # Using the shared group set object instead of copying it saves memory.
    result: defaultdict = defaultdict(set)
    for group in groups.values():
        for term in group:
            result[term] = group
    return result


def _dag_transitive_closure(pairs):
    """
    Compute the transitive closure of a directed relation (isA/partOf) using
    BFS from every source node.

    Replaces the naive O(n²·d) iterative approach.  BFS is O(V·(V+E)) which
    for sparse hierarchies is much faster in practice.

    Returns a set of (src, dst) pairs, matching the shape expected by
    getTransitiveClosureHier().
    """
    adj: defaultdict = defaultdict(set)
    nodes: set = set()
    for x, y in pairs:
        adj[x].add(y)
        nodes.add(x)
        nodes.add(y)

    closure: set = set()
    for start in nodes:
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in adj[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    closure.add((start, neighbour))
                    queue.append(neighbour)
    return closure

@dataclass()
class LogicalRewritingRule:
    label: str
    attachTo: str
    argument: str
    logicalConstructName: str
    logicalConstructProperty: Optional[str]

@dataclass()
class Condition:
    name: str
    values: list

@dataclass()
class Rule:
    id: int
    premises: List[Condition]
    not_premises: List[Condition]
    logicalConstructName: str
    logicalConstructProperty: Optional[str]
    # For rules that map to multiple logical types (e.g. "on or near" → both
    # "stay in place" and "near place"), extra (name, property) pairs beyond
    # the primary one are collected here so callers can apply all of them.
    additional_classifications: list = None

    def __post_init__(self):
        if self.additional_classifications is None:
            self.additional_classifications = []

class CasusHappening(Enum):
    EQUIVALENT = 0
    EXCLUSIVES = 1
    INDIFFERENT = 2
    NONE = 3
    GENERAL_IMPLICATION = 8
    LOSE_SPEC_IMPLICATION = 9
    INSTANTIATION_IMPLICATION = 10 #isa
    MISSING_1ST_IMPLICATION = 12 #miss


class HOnKSingleton(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @staticmethod
    def isReady():
        return (HOnKSingleton._instance is not None) and (HOnKSingleton._instance.honk is not None)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            cls._instance.honk = None
        return cls._instance

    @staticmethod
    def init(cache_path, user, password, hostame, port, onStorage, path, rules_path=None):
        if HOnKSingleton._instance.honk is None:
            HOnKSingleton._instance.honk = HOnK(cache_path, user, password, hostame, port, onStorage)
            HOnKSingleton._instance.honk.start(path, rules_path=rules_path)
        elif rules_path is not None:
            HOnKSingleton._instance.honk.configure_logical_rules_path(rules_path)

    @staticmethod
    def stop():
        if HOnKSingleton._instance is not None and HOnKSingleton._instance.honk is not None:
            HOnKSingleton._instance.honk.stop()
            HOnKSingleton._instance.honk = None
        HOnKSingleton._instance = None

    @staticmethod
    def get() -> 'HOnK':
        return HOnKSingleton._instance.honk


class HOnK(RDFGraph):

    def __init__(self, cache_path, user, password, hostame, port, onStorage =True):
        super().__init__("honk", "https://ofox.co.uk/honk#", user, password, hostame, port, "parmenides", databaseConn=onStorage)
        self.loaded = False
        self.cache_path = cache_path
        os.makedirs(self.cache_path, exist_ok=True)
        self.logger = logging.getLogger("HOnK")
        self.onStorage = onStorage
        self._store_path = None  # Set before super().start() to use RocksDB cache
        self._rules_json_path = None  # Set via start() to load rules from JSON
        self._loaded_rules_json_path = None
        self._loaded_rules_json_mtime = None
        self._init_lookup_sets()
        self._load_support_lookup_sets()

    def _type_lookup_map(self):
        return {
            "SemiModalVerb":           "semi_modal_verbs",
            "Pronoun":                 "pronouns",
            "PersonalPronoun":         "personal_pronouns",
            "PrototypicalPreposition": "prototypical_prepositions",
            "TransitiveVerb":          "transitive_verbs",
            "CausativeVerb":           "causative_verbs",
            "ConsumptionVerb":         "consumption_verbs",
            "MotionVerb":              "movement_verbs",
            "MeansVerb":               "means_verbs",
            "PredictionVerb":          "prediction_verbs",
            "StateVerb":               "state_verbs",
            "MaterialisationVerb":     "materialisation_verbs",
            "PhrasalVerb":             "phrasal_verbs",
            "UnitOfMeasure":           "units_of_measure",
            "AbstractEntity":          "abstract_entities",
            "Rejectable":              "rejected_edges",
            "Dependency":              "non_verbs",
            "TimeNoun":                "temporal_nouns",
            "LocationNoun":            "location_nouns",
            "StateNoun":               "state_nouns",
            "FacilityNoun":            "facility_nouns",
            "AccessPointNoun":         "access_point_nouns",
            "RouteNoun":               "route_nouns",
            "GeoSuffixNoun":           "geo_suffix_nouns",
            "ServiceStateNoun":        "service_state_nouns",
            "StatusNoun":              "status_nouns",
            "WeatherConditionNoun":    "weather_condition_nouns",
            "WeatherConditionAdjective": "weather_condition_adjectives",
            "FieldLabelNoun":          "field_label_nouns",
            "Conjunction":             "conjunctions",
            "DependantPreposition":    "prepositions",
            "IdiomaticPreposition":    "prepositions",
            "ComplexPreposition":      "prepositions",
        }

    def _init_lookup_sets(self):
        for attr in set(self._type_lookup_map().values()) | {
                "nouns_with_properties",
                "nouns_with_a",
                "logical_rewriting_rules",
                "prepositions",
                "copula_surface_forms",
                "change_of_state_verbs",
                "stative_verbs",
        }:
            if not hasattr(self, attr):
                setattr(self, attr, defaultdict() if attr == "logical_rewriting_rules" else set())

    def _add_lookup_terms_from_file(self, path, *attrs):
        if not os.path.exists(path):
            return
        with open(path, "r") as dep:
            for line in dep:
                term = line.strip()
                if not term or term.startswith("#"):
                    continue
                for attr in attrs:
                    getattr(self, attr).add(term)

    def _load_support_lookup_sets(self):
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "raw_data"))
        file_map = (
            ("non_verb_types.txt", ("non_verbs",)),
            ("units_of_measure.txt", ("units_of_measure",)),
            ("abstract_entity_concepts.txt", ("abstract_entities",)),
            (os.path.join("verbs", "causative_verbs.txt"), ("causative_verbs",)),
            (os.path.join("verbs", "causal_signal_verbs.txt"), ("causative_verbs",)),
            (os.path.join("verbs", "consumption_verbs.txt"), ("consumption_verbs",)),
            (os.path.join("verbs", "transitive_verbs.txt"), ("transitive_verbs",)),
            (os.path.join("verbs", "stative_verbs.txt"), ("stative_verbs",)),
            (os.path.join("verbs", "means_verbs.txt"), ("means_verbs",)),
            (os.path.join("verbs", "state_verbs.txt"), ("state_verbs",)),
            (os.path.join("verbs", "service_change_verbs.txt"), ("state_verbs", "change_of_state_verbs")),
            (os.path.join("verbs", "service_activity_verbs.txt"), ("state_verbs",)),
            (os.path.join("verbs", "copula_surface_forms.txt"), ("copula_surface_forms",)),
            (os.path.join("verbs", "movement_verbs.txt"), ("movement_verbs",)),
            (os.path.join("verbs", "materialisation_verbs.txt"), ("materialisation_verbs",)),
            (os.path.join("verbs", "phrasal_verbs.txt"), ("phrasal_verbs",)),
            (os.path.join("verbs", "prediction_verbs.txt"), ("prediction_verbs",)),
            (os.path.join("verbs", "semi_modal_verbs.txt"), ("semi_modal_verbs",)),
            (os.path.join("nouns", "facility_nouns.txt"), ("facility_nouns", "location_nouns")),
            (os.path.join("nouns", "access_point_nouns.txt"), ("access_point_nouns", "facility_nouns", "location_nouns")),
            (os.path.join("nouns", "route_nouns.txt"), ("route_nouns", "location_nouns")),
            (os.path.join("nouns", "geo_suffix_nouns.txt"), ("geo_suffix_nouns", "location_nouns")),
            (os.path.join("nouns", "service_state_nouns.txt"), ("service_state_nouns", "state_nouns")),
            (os.path.join("nouns", "status_nouns.txt"), ("status_nouns", "state_nouns")),
            (os.path.join("nouns", "weather_condition_nouns.txt"), ("weather_condition_nouns", "state_nouns")),
            (os.path.join("nouns", "field_label_nouns.txt"), ("field_label_nouns",)),
            (os.path.join("pronouns", "personal_pronouns.txt"), ("pronouns", "personal_pronouns")),
            (os.path.join("pronouns", "demonstrative_pronouns.txt"), ("pronouns",)),
            (os.path.join("pronouns", "relative_pronouns.txt"), ("pronouns",)),
            (os.path.join("pronouns", "indefinite_pronouns.txt"), ("pronouns",)),
            (os.path.join("pronouns", "interrogative_pronouns.txt"), ("pronouns",)),
        )
        for relative_path, attrs in file_map:
            self._add_lookup_terms_from_file(os.path.join(data_path, relative_path), *attrs)

    def start(self, filename=None, rules_path=None):
        if rules_path is not None:
            self._rules_json_path = rules_path
        use_cache = False

        if not self.onStorage and filename is not None:
            store_dir = os.path.join(self.cache_path, "honk_oxstore")
            mtime_file = store_dir + ".mtime"
            ttl_mtime = os.path.getmtime(filename)

            if os.path.exists(store_dir) and os.path.exists(mtime_file):
                try:
                    with open(mtime_file) as f:
                        use_cache = float(f.read().strip()) >= ttl_mtime
                except (ValueError, OSError):
                    use_cache = False

            if not use_cache and os.path.exists(store_dir):
                shutil.rmtree(store_dir)

            # Point _actual_start() at the persistent RocksDB path
            self._store_path = store_dir

        try:
            result = super().start()
        except RuntimeError:
            if not self.onStorage and filename is not None and use_cache:
                print(f"[HOnK] Cached RocksDB store is not readable; rebuilding: {store_dir}")
                if os.path.exists(store_dir):
                    shutil.rmtree(store_dir)
                if os.path.exists(mtime_file):
                    os.remove(mtime_file)
                use_cache = False
                self._store_path = store_dir
                result = super().start()
            else:
                raise
        if result:
            self.logger.info("HOnK started")
            if not self.onStorage and filename is not None:
                if not use_cache:
                    print(f"[HOnK] Parsing TTL (first-time, result will be cached): {filename}")
                    t0 = time.time()
                    self.parse(filename)
                    print(f"[HOnK] TTL parsed in {time.time() - t0:.1f}s — compacting store...")
                    self.graph.optimize()
                    print(f"[HOnK] Store compacted in {time.time() - t0:.1f}s total.")
                    with open(mtime_file, "w") as f:
                        f.write(str(ttl_mtime))
                else:
                    print(f"[HOnK] Loaded from RocksDB store cache ({store_dir}).")
            try:
                self._load()
            except RuntimeError:
                if not self.onStorage and filename is not None and use_cache:
                    print(f"[HOnK] Cached RocksDB store failed during load; rebuilding: {store_dir}")
                    super().stop()
                    if os.path.exists(store_dir):
                        shutil.rmtree(store_dir)
                    if os.path.exists(mtime_file):
                        os.remove(mtime_file)
                    use_cache = False
                    self._store_path = store_dir
                    result = super().start()
                    if result:
                        print(f"[HOnK] Parsing TTL (cache rebuild): {filename}")
                        t0 = time.time()
                        self.parse(filename)
                        print(f"[HOnK] TTL parsed in {time.time() - t0:.1f}s — compacting store...")
                        self.graph.optimize()
                        print(f"[HOnK] Store compacted in {time.time() - t0:.1f}s total.")
                        with open(mtime_file, "w") as f:
                            f.write(str(ttl_mtime))
                        self._load()
                else:
                    raise
        return result

    def configure_logical_rules_path(self, rules_path):
        self._rules_json_path = rules_path
        if getattr(self, "loaded", False):
            self._load_logical_rules_from_json_if_available(force=True)

    def _load_logical_rules_from_json_if_available(self, force=False):
        if self._rules_json_path is None or not os.path.exists(self._rules_json_path):
            return False

        rules_mtime = os.path.getmtime(self._rules_json_path)
        already_loaded = (
            self._loaded_rules_json_path == self._rules_json_path
            and self._loaded_rules_json_mtime == rules_mtime
            and getattr(self, "logical_rewriting_rules", None) is not None
        )
        if not force and already_loaded:
            return True

        self.logical_rewriting_rules = _load_logical_rules_from_json(self._rules_json_path)
        self._loaded_rules_json_path = self._rules_json_path
        self._loaded_rules_json_mtime = rules_mtime
        return True

    def stop(self):
        result = super().stop()
        if result:
            self._clear()
        return result

    def _load(self):
        if not self.hasDBStoredData():
            self._load_support_lookup_sets()
            self._load_logical_rules_from_json_if_available(force=True)
            return False
        if self.loaded:
            return True

        _t0 = time.time()
        print("[HOnK] Loading ontology data...")

        self.st = defaultdict(set)

        # Batch all 14 type-label lookups into a single SPARQL query instead of
        # making 14 individual round-trips.
        _type_map = self._type_lookup_map()
        _values_clause = " ".join(f"honk:{t}" for t in _type_map)
        _batch_query = f"""
            SELECT DISTINCT ?type_uri ?c
            WHERE {{
                VALUES ?type_uri {{ {_values_clause} }}
                ?a a ?type_uri.
                ?a rdfs:label ?c.
            }}"""
        _ns = str(self.namespace)
        _buckets = {attr: set(getattr(self, attr, set())) for attr in _type_map.values()}
        for row in self._iter_rows(_batch_query):
            type_local = str(row.type_uri)[len(_ns):]
            attr = _type_map.get(type_local)
            if attr is not None:
                _buckets[attr].add(str(row.c))
        for attr, s in _buckets.items():
            setattr(self, attr, s)
        # PrototypicalPreposition feeds prototypical_prepositions only via the map above;
        # include it in the full prepositions set too.
        self.prepositions = set(self.prepositions) | set(self.prototypical_prepositions)
        print(f"[HOnK]   type labels loaded ({time.time()-_t0:.1f}s)")

        ## get_logical_rewriting_rules
        print(f"[HOnK]   loading logical rewriting rules...")
        if self._load_logical_rules_from_json_if_available(force=True):
            pass
        else:
            knows_query = """
                     SELECT DISTINCT ?label ?rule_order ?Preposition ?logicalConstructName ?logicalConstructProperty ?MotionVerb ?SingletonHasBeenMatchedBy ?not ?AbstractEntity ?hasNMod ?hasNModPartOf ?hasNModIsA ?isSymmetricalIfComparedToNMod ?Number ?UnitOfMeasure ?StateVerb ?MeansVerb ?CausativeVerb ?MaterialisationVerb
                     WHERE {
                         ?a a honk:LogicalRewritingRule.
                         ?a rdfs:label ?label .
                         ?a honk:logicalConstructName ?logicalConstructName .
                         ?a honk:rule_order ?rule_order .
                         OPTIONAL { ?a honk:Preposition ?Preposition }
                         OPTIONAL { ?a honk:logicalConstructProperty ?logicalConstructProperty }
                         OPTIONAL { ?a honk:SingletonHasBeenMatchedBy ?SingletonHasBeenMatchedBy }
                         OPTIONAL { ?a honk:not ?not }
                         OPTIONAL { ?a honk:AbstractEntity ?AbstractEntity }
                         OPTIONAL { ?a honk:hasNMod ?hasNMod }
                         OPTIONAL { ?a honk:hasNModPartOf ?hasNModPartOf }
                         OPTIONAL { ?a honk:hasNModIsA ?hasNModIsA }
                         OPTIONAL { ?a honk:isSymmetricalIfComparedToNMod ?isSymmetricalIfComparedToNMod }
                         OPTIONAL { ?a honk:CausativeVerb ?CausativeVerb }
                         OPTIONAL { ?a honk:Number ?Number }
                         OPTIONAL { ?a honk:UnitOfMeasure ?UnitOfMeasure }
                         OPTIONAL { ?a honk:MotionVerb ?MotionVerb }
                         OPTIONAL { ?a honk:StateVerb ?StateVerb }
                         OPTIONAL { ?a honk:MeansVerb ?MeansVerb }
                         OPTIONAL { ?a honk:MaterialisationVerb ?MaterialisationVerb }
                     }"""

            not_query = """
                     SELECT DISTINCT ?Preposition ?MotionVerb ?SingletonHasBeenMatchedBy ?AbstractEntity ?hasNMod ?hasNModPartOf ?hasNModIsA ?isSymmetricalIfComparedToNMod ?Number ?UnitOfMeasure ?StateVerb ?MeansVerb ?CausativeVerb ?MaterialisationVerb
                     WHERE {
                         OPTIONAL { ?a honk:Preposition ?Preposition }
                         OPTIONAL { ?a honk:SingletonHasBeenMatchedBy ?SingletonHasBeenMatchedBy }
                         OPTIONAL { ?a honk:AbstractEntity ?AbstractEntity }
                         OPTIONAL { ?a honk:hasNMod ?hasNMod }
                         OPTIONAL { ?a honk:hasNModPartOf ?hasNModPartOf }
                         OPTIONAL { ?a honk:hasNModIsA ?hasNModIsA }
                         OPTIONAL { ?a honk:isSymmetricalIfComparedToNMod ?isSymmetricalIfComparedToNMod }
                         OPTIONAL { ?a honk:CausativeVerb ?CausativeVerb }
                         OPTIONAL { ?a honk:Number ?Number }
                         OPTIONAL { ?a honk:UnitOfMeasure ?UnitOfMeasure }
                         OPTIONAL { ?a honk:MotionVerb ?MotionVerb }
                         OPTIONAL { ?a honk:StateVerb ?StateVerb }
                         OPTIONAL { ?a honk:MeansVerb ?MeansVerb }
                         OPTIONAL { ?a honk:MaterialisationVerb ?MaterialisationVerb }
                     }"""

            str_premises = {"Preposition", "MotionVerb", "SingletonHasBeenMatchedBy", "AbstractEntity", "hasNMod",
                            "hasNModPartOf", "hasNModIsA", "isSymmetricalIfComparedToNMod", "Number", "UnitOfMeasure",
                            "StateVerb", "MeansVerb", "CausativeVerb", "MaterialisationVerb"}

            self.logical_rewriting_rules = defaultdict()

            query_rules = list(self._single_unary_query(knows_query, lambda x: x))
            query_rules.sort(key=lambda rule: rule.rule_order)  # So list is ordered numerically by rule_order
            grouped_rules = defaultdict(list)
            for rule in query_rules:
                grouped_rules[rule.rule_order].append(rule)

            for gr_key, grouped_rule in grouped_rules.items():
                premises = defaultdict(list)
                not_premises = defaultdict(list)
                logical_construct_name = None
                logical_construct_property = None
                for rule in grouped_rule:
                    logical_construct_name = rule.logicalConstructName.value if logical_construct_name is None else logical_construct_name
                    logical_construct_property = rule.logicalConstructProperty.value if logical_construct_property is None and rule.logicalConstructProperty is not None else logical_construct_property
                    for str_premise in str_premises:
                        if hasattr(rule, str_premise) and getattr(rule, str_premise) is not None:
                            premises[str_premise].append(getattr(rule, str_premise).value)
                    if hasattr(rule, "not") and getattr(rule, "not") is not None:
                        not_query_premises = list(self._iter_rows(not_query, {'a': getattr(rule, "not")}))
                        for not_query_premise in not_query_premises:
                            for str_premise in str_premises:
                                if hasattr(not_query_premise, str_premise) and getattr(not_query_premise,
                                                                                       str_premise) is not None:
                                    not_premises[str_premise].append(getattr(not_query_premise, str_premise).value)

                con_premises = list()
                for p_key, premise in premises.items():
                    con_premises.append(Condition(
                        name=p_key,
                        values=list(set(premise))
                    ))

                neg_con_premises = list()
                for np_key, not_premise in not_premises.items():
                    neg_con_premises.append(Condition(
                        name=np_key,
                        values=list(set(not_premise))
                    ))

                self.logical_rewriting_rules[gr_key.value] = Rule(
                    id=gr_key.value,
                    premises=con_premises,
                    not_premises=neg_con_premises,
                    logicalConstructName=logical_construct_name,
                    logicalConstructProperty=logical_construct_property
                )
        self.logger.info("Logical rewriting rules loaded")
        print(f"[HOnK]   logical rewriting rules loaded ({time.time()-_t0:.1f}s)")
        ## End: get_logical_rewriting_rules

        ## Nouuns with properties
        print(f"[HOnK]   loading nouns...")
        knows_query = """
                 SELECT DISTINCT ?hasProperty ?label
                 WHERE {
                     ?a a honk:Noun.
                     ?a rdfs:label ?label .
                     ?a honk:hasProperty ?hasProperty .
                 }"""

        self.logger.info("nouns_with_properties loaded")
        self.nouns_with_properties = set(self._single_unary_query(knows_query, lambda x: x))
        ## End: nouns with properties

        ## Start: nouns_with_a
        knows_query = """
                         SELECT DISTINCT ?isA ?label
                         WHERE {
                             ?a a honk:Noun.
                             ?a rdfs:label ?label .
                             ?a honk:isA ?isA .
                         }"""
        self.nouns_with_a = set(self._single_unary_query(knows_query, lambda x: x))
        self.logger.info("nouns_with_a loaded")
        print(f"[HOnK]   nouns loaded ({time.time()-_t0:.1f}s)")
        ## End: nouns_with_a

        # Materialise eq / isA / partOf as label-keyed Python adjacency dicts.
        # We bypass SPARQL on the hot path because pyoxigraph's planner picks an
        # O(N) scan plan for the BGP `?pred a honk:eq . ?src ?pred ?equiv` on
        # the current 22.4M-triple HOnK store — a single getSynonymy call hangs
        # for >5 minutes regardless of VALUES batching, which previously froze
        # _calculate_matrix at ~7/16 cells.  The dicts only carry URIs that
        # actually participate in eq/isA/partOf edges, so memory scales with
        # edge count (not the 10M total rdfs:label triples) — small for the
        # current cut-down TTL, ~200MB even at the WordNet scale described in
        # the original lazy-loading comment (~210K eq + ~788K hier pairs).
        self._build_relation_adjacency()
        self._syn_cache: dict = {}
        self._trcl_cache: dict = {}


        self.loaded = True
        # Invalidate any name_eq results computed during loading (e.g. from
        # typeOf calls used while populating noun lookups), since they may
        # have been answered against an incomplete ontology. The class-suffix
        # stripping branch in particular silently degrades to INDIFFERENT if
        # the facility/route/location noun lists weren't yet populated.
        self.name_eq.cache_clear()
        if hasattr(self, "_loc_variants_cache"):
            self._loc_variants_cache.clear()
        print(f"[HOnK] Ontology fully loaded in {time.time()-_t0:.1f}s.")
        return True



    def most_specific_type(self, types):
        types = list(map(lambda x: str(x).lower(), types))
        # Match plain "verb" OR any verb subclass (e.g. ChangeVerb → "changeverb" ends with "verb")
        if any(map(lambda x: x == "verb" or (x.endswith("verb") and not x.endswith("adverb")), types)):
            return "verb"
        elif "gpe" in types:
            return "GPE"
        elif "loc" in types:
            return "LOC"
        elif "org" in types:
            return "ORG"
        elif any(map(lambda x: x == "noun" or x.endswith("noun"), types)):
            return "noun"
        elif "entity" in types:
            return "ENTITY"
        elif any(map(lambda x: x == "adjective" or x.endswith("adjective"), types)):
            return "JJ"
        elif any(map(lambda x: x == "adverb" or x.endswith("adverb"), types)):
            return "RB"
        elif any(map(lambda x: x in {"preposition", "prepositionalphrase", "dependency"}
                                    or x.endswith("preposition") or x.endswith("prepositionalphrase"), types)):
            return "IN"
        else:
            return "None"

        # TODO: MAKE THIS BETTER EVENTUALLY
    def most_general_type(self, types):
        types = list(map(lambda x: str(x).upper(), types))
        tree = {
            "ENTITY": {
                "NOUN": {"ORG": {}, "PERSON": {}, "LOC": {"GPE": {}}},
                "JJ": {}
            }
        }

        if not types or not tree:
            return None
        if len(types) == 1:
            return types[0]

        def find_path(root, target_type, path):
            if root == target_type:
                return path
            if isinstance(root, dict):
                for key, value in root.items():
                    new_path = find_path(value, target_type, path + [key])
                    if new_path:
                        return new_path
            return None

        paths = []
        for type_ in types:
            path = find_path(tree, type_, [])
            if not path:
                return None  # Type not found in the tree
            paths.append(path)
        min_len = min(len(path) for path in paths)
        lca = None
        for i in range(min_len):
            if all(path[i] == paths[0][i] for path in paths):
                lca = paths[0][i]
            else:
                break

        return lca

    def get_label_is_a(self, label):
        self.logger.info(f"get_label_is_a  {label} started")
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a honk:%s.
             ?a rdfs:label ?c .
         }""" % label
        return self.string_query(knows_query, "c")

    def getAllEntitiesByImmediateType(self, t):
        ye = list(self.isA("^x", str(t)))
        if len(ye) == 0:
            return set()
        else:
            return {x["x"] for x in ye}

    def _getOutgoingNodesByClassInstance(self, srcLabel, classType):
        """Find outgoing nodes where the predicate is an instance of classType (numbered variant pattern)."""
        knows_query = """
         SELECT DISTINCT ?dst_label
         WHERE {
             ?pred_inst a ?class_uri .
             ?src ?pred_inst ?dst .
             ?src rdfs:label ?src_label .
             ?dst rdfs:label ?dst_label .
         }"""
        bindings = {
            "src_label": Literal(srcLabel, datatype=XSD.string),
            "class_uri": self.namespace[classType]
        }
        S = set()
        for d in self._run_custom_sparql_query(knows_query, bindings=bindings):
            if "dst_label" in d:
                S.add(str(d["dst_label"]))
        return S

    def _extractPureHierarchyByClass(self, classType):
        """Extract all (src, dst) pairs where predicate is an instance of classType."""
        knows_query = """
         SELECT DISTINCT ?src_label ?dst_label
         WHERE {
             ?pred_inst a ?class_uri .
             ?src ?pred_inst ?dst .
             ?src rdfs:label ?src_label .
             ?dst rdfs:label ?dst_label .
         }"""
        bindings = {"class_uri": self.namespace[classType]}
        S = set()
        for d in self._run_custom_sparql_query(knows_query, bindings=bindings):
            if "src_label" in d and "dst_label" in d:
                S.add((str(d["src_label"]), str(d["dst_label"])))
        return S

    def getOutgoingNodes(self, srcLabel, edgeType):
        """Override to support both direct predicates and numbered instance predicates."""
        S = super().getOutgoingNodes(srcLabel, edgeType)
        if S:
            return S
        return self._getOutgoingNodesByClassInstance(srcLabel, edgeType)

    def extractPureHierarchy(self, t, flip=False):
        """Override to support numbered instance predicates."""
        ye = list(self.single_edge("^src", t, "^dst"))
        if len(ye) > 0:
            if flip:
                return {(x["dst"], x["src"]) for x in ye}
            return {(x["src"], x["dst"]) for x in ye}
        pairs = self._extractPureHierarchyByClass(t)
        if flip:
            return {(y, x) for x, y in pairs}
        return pairs

    def _build_relation_adjacency(self):
        """Materialise the eq / isA / partOf / neqTo graph from the underlying
        store as four label-keyed Python adjacency dicts.  Result is cached
        next to the RocksDB store as `relation_adjacency.pkl` — first build
        is ~120s on the full 22.4M-triple HOnK; subsequent loads are <1s.

        Uses `quads_for_pattern` index lookups for edge enumeration and a
        single full-scan over the rdfs:label triples for URI→label resolution
        (much faster than ~1M individual round-trips).

        Edge directions follow the original SPARQL templates exactly:
            * eq: bidirectional — `(s, eq_pred, o)` adds `label(s) ↔ label(o)`.
            * isA: edges run supertype → subtype.  A triple `(s, isA_pred, o)`
              means subtype `o` *isA* supertype `s`, so we record
              `label(o) → {label(s), …}` (cur → supertypes).
            * partOf: edges run part → whole.  A triple `(s, partOf_pred, o)`
              means part `s` is partOf whole `o`, so we record
              `label(s) → {label(o), …}` (cur → wholes).
            * neqTo: bidirectional antonyms.
        """
        import pickle
        adj_cache = os.path.join(self.cache_path, "honk_oxstore.adj.v2.pkl")
        # The RocksDB store directory mtime updates on every open (LOG churn),
        # so anchor freshness on `honk_oxstore.mtime` (which records the TTL
        # mtime when the store was built — invariant unless TTL changes).
        ttl_mtime_file = os.path.join(self.cache_path, "honk_oxstore.mtime")
        try:
            if (os.path.exists(adj_cache)
                    and os.path.exists(ttl_mtime_file)
                    and os.path.getmtime(adj_cache) >= os.path.getmtime(ttl_mtime_file)):
                with open(adj_cache, "rb") as f:
                    blob = pickle.load(f)
                self._eq_adj = blob["eq"]
                self._isA_supers = blob["isA"]
                self._partOf_wholes = blob["partOf"]
                self._neqTo_adj = blob["neqTo"]
                n_eq = sum(len(v) for v in self._eq_adj.values())
                n_isA = sum(len(v) for v in self._isA_supers.values())
                n_part = sum(len(v) for v in self._partOf_wholes.values())
                n_neq = sum(len(v) for v in self._neqTo_adj.values())
                print(f"[HOnK]   relation adjacency loaded from cache: "
                      f"eq={n_eq} isA={n_isA} partOf={n_part} neqTo={n_neq} "
                      f"({len(self._eq_adj)}/{len(self._isA_supers)}/{len(self._partOf_wholes)}/{len(self._neqTo_adj)} terms)")
                return
        except (pickle.PickleError, EOFError, OSError) as e:
            print(f"[HOnK]   adjacency cache unreadable ({e!r}); rebuilding")

        import pyoxigraph
        ns = str(self.namespace)
        rdfs_label = pyoxigraph.NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
        rdf_type = pyoxigraph.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

        self._eq_adj: dict[str, set[str]] = {}
        self._isA_supers: dict[str, set[str]] = {}
        self._partOf_wholes: dict[str, set[str]] = {}
        self._neqTo_adj: dict[str, set[str]] = {}

        def _preds_of_class(cls_local_name):
            cls = pyoxigraph.NamedNode(ns + cls_local_name)
            return [q.subject for q in self.graph.quads_for_pattern(None, rdf_type, cls, None)]

        eq_preds = _preds_of_class("eq")
        isA_preds = _preds_of_class("isA")
        partOf_preds = _preds_of_class("partOf")
        neqTo_preds = _preds_of_class("neqTo")
        formOf_preds = _preds_of_class("formOf")

        # Pass 1: collect raw edge tuples (URIs) and the URIs we need labels for.
        eq_edges, isA_edges, partOf_edges, neqTo_edges, formOf_edges = [], [], [], [], []
        needed_uris: set = set()

        def _collect(pred_list, dest):
            for p in pred_list:
                for q in self.graph.quads_for_pattern(None, p, None, None):
                    s, o = q.subject, q.object
                    if isinstance(s, pyoxigraph.NamedNode) and isinstance(o, pyoxigraph.NamedNode):
                        dest.append((s.value, o.value))
                        needed_uris.add(s.value)
                        needed_uris.add(o.value)
        _t = time.time(); _collect(eq_preds, eq_edges)
        print(f"[HOnK]   adjacency: collected {len(eq_edges)} eq edges in {time.time()-_t:.1f}s", flush=True)
        _t = time.time(); _collect(isA_preds, isA_edges)
        print(f"[HOnK]   adjacency: collected {len(isA_edges)} isA edges in {time.time()-_t:.1f}s", flush=True)
        _t = time.time(); _collect(partOf_preds, partOf_edges)
        print(f"[HOnK]   adjacency: collected {len(partOf_edges)} partOf edges in {time.time()-_t:.1f}s", flush=True)
        _t = time.time(); _collect(neqTo_preds, neqTo_edges)
        print(f"[HOnK]   adjacency: collected {len(neqTo_edges)} neqTo edges in {time.time()-_t:.1f}s", flush=True)
        _t = time.time(); _collect(formOf_preds, formOf_edges)
        print(f"[HOnK]   adjacency: collected {len(formOf_edges)} formOf edges in {time.time()-_t:.1f}s", flush=True)

        # Pass 2: single full-scan over rdfs:label triples — ~50s for the
        # 10M-row label table, vs ~150s for ~1M individual URI round-trips.
        _t = time.time()
        uri_to_labels: dict[str, list[str]] = {}
        for q in self.graph.quads_for_pattern(None, rdfs_label, None, None):
            s = q.subject
            if not isinstance(s, pyoxigraph.NamedNode):
                continue
            uri = s.value
            if uri not in needed_uris:
                continue
            obj = q.object
            if isinstance(obj, pyoxigraph.Literal):
                uri_to_labels.setdefault(uri, []).append(obj.value)
        print(f"[HOnK]   adjacency: resolved {len(uri_to_labels)} URI labels in {time.time()-_t:.1f}s", flush=True)

        # Pass 3: build label-keyed adjacency dicts (cartesian over each URI's
        # labels — almost always 1×1, but correct for multi-labelled URIs).
        for s_uri, o_uri in eq_edges:
            s_ls = uri_to_labels.get(s_uri); o_ls = uri_to_labels.get(o_uri)
            if not s_ls or not o_ls:
                continue
            for ls in s_ls:
                for lo in o_ls:
                    self._eq_adj.setdefault(ls, set()).add(lo)
                    self._eq_adj.setdefault(lo, set()).add(ls)
        for s_uri, o_uri in isA_edges:
            s_ls = uri_to_labels.get(s_uri); o_ls = uri_to_labels.get(o_uri)
            if not s_ls or not o_ls:
                continue
            for ls in s_ls:
                for lo in o_ls:
                    self._isA_supers.setdefault(lo, set()).add(ls)
        for s_uri, o_uri in partOf_edges:
            s_ls = uri_to_labels.get(s_uri); o_ls = uri_to_labels.get(o_uri)
            if not s_ls or not o_ls:
                continue
            for ls in s_ls:
                for lo in o_ls:
                    self._partOf_wholes.setdefault(ls, set()).add(lo)
        for s_uri, o_uri in neqTo_edges:
            s_ls = uri_to_labels.get(s_uri); o_ls = uri_to_labels.get(o_uri)
            if not s_ls or not o_ls:
                continue
            for ls in s_ls:
                for lo in o_ls:
                    self._neqTo_adj.setdefault(ls, set()).add(lo)
                    self._neqTo_adj.setdefault(lo, set()).add(ls)
        # formOf (inflectional/morphological forms, e.g. offences → offence) are
        # treated as bidirectional equivalences so that plural/singular and
        # spelling-variant surface forms match the same ontology concept.
        for s_uri, o_uri in formOf_edges:
            s_ls = uri_to_labels.get(s_uri); o_ls = uri_to_labels.get(o_uri)
            if not s_ls or not o_ls:
                continue
            for ls in s_ls:
                for lo in o_ls:
                    self._eq_adj.setdefault(ls, set()).add(lo)
                    self._eq_adj.setdefault(lo, set()).add(ls)

        n_eq = sum(len(v) for v in self._eq_adj.values())
        n_isA = sum(len(v) for v in self._isA_supers.values())
        n_part = sum(len(v) for v in self._partOf_wholes.values())
        n_neq = sum(len(v) for v in self._neqTo_adj.values())
        print(f"[HOnK]   relation adjacency: eq={n_eq} isA={n_isA} partOf={n_part} neqTo={n_neq} "
              f"({len(self._eq_adj)}/{len(self._isA_supers)}/{len(self._partOf_wholes)}/{len(self._neqTo_adj)} terms)")

        try:
            with open(adj_cache, "wb") as f:
                pickle.dump({
                    "eq": self._eq_adj,
                    "isA": self._isA_supers,
                    "partOf": self._partOf_wholes,
                    "neqTo": self._neqTo_adj,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[HOnK]   adjacency: cached to {adj_cache}", flush=True)
        except OSError as e:
            print(f"[HOnK]   adjacency: failed to write cache ({e!r})", flush=True)

    def getSynonymy(self, k):
        """Return the set of all labels equivalent to *k* (including *k* itself).

        Pure-Python BFS over the preloaded `_eq_adj` adjacency dict — no
        SPARQL on the hot path.  Caches the result for every discovered
        synonym, so a single matrix-cell traversal warms every member of the
        equivalence class for O(1) future lookups.
        """
        if k in self._syn_cache:
            return self._syn_cache[k]
        synonyms = {k}
        frontier = {k}
        while frontier:
            next_frontier = set()
            for term in frontier:
                for n in self._eq_adj.get(term, ()):
                    if n not in synonyms:
                        synonyms.add(n)
                        next_frontier.add(n)
            frontier = next_frontier
        for t in synonyms:
            self._syn_cache[t] = synonyms
        return synonyms

    def typeOf2(self, src):
        knows_query = """
         SELECT DISTINCT ?dst 
         WHERE {
             ?src rdfs:subClassOf ?dst.
         }"""
        s = set()
        for x in self._iter_rows(knows_query, {"src": src}):
            s.add(str(x.dst))
        return s

    def hasTypedObject(self, src):
        return len(self.typeOf(src)) > 0

    def typeOf(self, src):
        from LaSSI.ner.string_functions import surface_form_variants
        src_str = str(src)
        src_terms = {src_str, src_str.lower()}
        for variant in surface_form_variants(src_str):
            src_terms.add(variant)
            src_terms.add(variant.lower())
        matched_types = {
            str(self.namespace) + type_name
            for type_name, attr in self._type_lookup_map().items()
            if src_terms & {str(value).lower() for value in getattr(self, attr, set())}
        }
        if matched_types or getattr(self, "graph", None) is None:
            return matched_types

        knows_query = """
         SELECT DISTINCT ?dst 
         WHERE {
             ?src a ?dst.
             ?src rdfs:label ?src_label.
        }"""
        s = set()
        try:
            for x in self._iter_rows(knows_query, {"src_label": Literal(src, datatype=XSD.string)}):
                s.add(str(x.dst))
        except RuntimeError:
            return matched_types
        return s

    def getSuperTypes(self, src):
        """Return the set of all label-supertypes of the entity *src*
        (including direct supertypes and their transitive ancestors).

        Adjacency-dict based: traverses `_isA_supers` only.  This replaces
        the original SPARQL `?s a ?type . ?type rdfs:subClassOf* ?super`
        which hung on the full 22.4M-triple HOnK store.

        Results are cached in self.st.  Returns an empty set if the closure
        exceeds `_REACHABLE_CAP` (treated as "too generic to be useful").
        """
        if src in self.st:
            return self.st[src]

        cap = self._REACHABLE_CAP
        supers: set[str] = set()
        frontier = {src}
        bailed = False
        while frontier:
            nxt = set()
            for t in frontier:
                for n in self._isA_supers.get(t, ()):
                    if n not in supers:
                        supers.add(n)
                        nxt.add(n)
                if len(supers) > cap:
                    bailed = True
                    break
            if bailed:
                break
            frontier = nxt
        if bailed:
            supers = set()
        self.st[src] = supers
        return supers

    def getTypedObjects(self):
            typing = """
             SELECT DISTINCT ?s ?src_label ?dst
    WHERE {
       ?s a ?dst .
       ?s rdfs:label ?src_label.
    } """
            d = defaultdict(set)
            for x in self._iter_rows(typing):
                d[str(x.src_label)].add(str(x.dst)[len(self.namespace):])
            return d

    def dumpTypedObjectsToTAB(self, filename: str | io.IOBase):
        l = self.getTypedObjects()
        n = len(l)
        f = filename if isinstance(filename, io.IOBase) else open(str(filename), "w")

        f.write(f"id\tidx\tt\ttype{os.linesep}")
        count = 1
        for k, v in l.items():
            k_escaped = k.replace("\\", "\\\\")

            # Get all unique generalised types for this word
            generalized_types = {self.most_specific_type([single_type]) for single_type in v}

            # Clean up "None" if other valid types exist
            if len(generalized_types) > 1:
                generalized_types.discard("None")
            elif not generalized_types:
                generalized_types.add("None")

            # Join them into a single string: e.g. "verb,noun"
            combined_types = ",".join(generalized_types)

            f.write(f"{count}\t{k_escaped}\t{k_escaped}\t{combined_types}{os.linesep}")
            count += 1

        return f

    def _cached_location_variants(self, name):
        """Helper for `name_eq`'s class-suffix-stripping branch.

        Returns `(variants, stripped)` where `variants` is the set of
        class-suffix-stripped forms of `name` (including `name` itself), and
        `stripped` is True iff at least one strip succeeded (i.e. `name` ends
        with a HOnK facility/access-point/route noun)."""
        from LaSSI.ner.string_functions import surface_form_variants, _ontology_class_suffix_terms
        cache = getattr(self, "_loc_variants_cache", None)
        if cache is None:
            cache = {}
            self._loc_variants_cache = cache
        if name in cache:
            return cache[name]
        variants = surface_form_variants(name) if name else set()
        result = (variants, len(variants) > 1)
        if _ontology_class_suffix_terms():
            cache[name] = result
        return result

    @lru_cache()
    def name_eq(self, src, dst):
        if (src == dst):
            return CasusHappening.EQUIVALENT
        elif ((src is None) or len(src) == 0) and ((dst is None) or len(dst) == 0):
            return CasusHappening.EQUIVALENT
        elif (src is None) or len(src) == 0:
            return CasusHappening.MISSING_1ST_IMPLICATION
        elif (dst is None) or len(dst) == 0:
            return CasusHappening.INDIFFERENT
        elif (src.startswith("?") and src[1:].isdigit()) or (dst.startswith("?") and dst[1].isdigit()):
            return CasusHappening.EQUIVALENT
        else:
            srcS = self.getSynonymy(src)
            dstS = self.getSynonymy(dst)
            neqTo_src = self._neqTo_adj.get(src, set())
            neqTo_dst = self._neqTo_adj.get(dst, set())
            # Direct (1-hop) neqTo edge between the heads is decisive: an
            # explicit ontology antonym overrides any sense-collapsed synonymy.
            if dst in neqTo_src or src in neqTo_dst:
                return CasusHappening.EXCLUSIVES
            # Direct synonymy (synsets share a member) outranks the broader
            # "synonym-of-mine matches antonym-of-theirs" path below — WordNet
            # routinely produces both verdicts when a term is polysemic, and
            # the same sense cannot self-contradict.
            if not srcS.isdisjoint(dstS):
                return CasusHappening.EQUIVALENT
            # Location-equivalence via class-suffix stripping. Fires when at
            # least one side sheds a HOnK-known facility / access-point /
            # route noun from its tail AND the resulting variant sets
            # intersect. Examples that match: "Haymarket Station" ≡
            # "Haymarket Metro station" (both reduce to "Haymarket");
            # "Haymarket" ≡ "Haymarket Station" (the latter reduces to
            # "Haymarket"). The "at least one stripped" condition keeps
            # non-location atoms ("close", "fire", "fire alarm") out — for
            # those, class_suffix_variants yields only the input string
            # (because no facility/access-point noun appears as a suffix),
            # so neither stripping nor a non-trivial intersection occurs.
            #
            # Variants are computed only for the raw input names, not the
            # full WordNet synsets. Location names are proper nouns with no
            # meaningful synonymy, so expanding `srcS`/`dstS` would multiply
            # the per-call cost 10-50× for no benefit. The fast path that
            # checks input-only variants converges on the same equivalence
            # ("Haymarket Station" vs "Haymarket Metro station" both strip
            # to "Haymarket") without iterating WordNet.
            src_variants, src_stripped = self._cached_location_variants(src)
            dst_variants, dst_stripped = self._cached_location_variants(dst)
            if (src_stripped or dst_stripped) and not src_variants.isdisjoint(dst_variants):
                return CasusHappening.EQUIVALENT
            if not dstS.isdisjoint(neqTo_src) or not srcS.isdisjoint(neqTo_dst):
                return CasusHappening.EXCLUSIVES
            # Supertype-intersection gate — short-circuit when the terms have
            # no common ancestor.  Adjacency-based, no SPARQL.
            resolveTypeFromOntologyLHS = self.getSuperTypes(src)
            resolveTypeFromOntologyRHS = self.getSuperTypes(dst)
            if len(resolveTypeFromOntologyLHS) == 0 or len(resolveTypeFromOntologyRHS) == 0:
                return CasusHappening.INDIFFERENT
            if resolveTypeFromOntologyLHS.isdisjoint(resolveTypeFromOntologyRHS):
                return CasusHappening.INDIFFERENT
            else:
                # Implication via isA/partOf reachability.  Old code did an
                # O(|srcS|*|dstS|) `_is_reachable` scan; replace with a single
                # union of reachable sets and an O(|dstS|) intersection check.
                src_reach: set = set()
                bailed = False
                for lhs in srcS:
                    r = self._get_reachable(lhs)
                    if r is None:
                        # closure too large to be useful — treat as INDIFFERENT
                        bailed = True
                        break
                    src_reach |= r
                if not bailed and not dstS.isdisjoint(src_reach):
                    return CasusHappening.GENERAL_IMPLICATION
                return CasusHappening.INDIFFERENT

    _REACHABLE_CAP = 5000

    def _get_reachable(self, term: str):
        """Return the set of all terms reachable from *term* via isA/partOf
        edges (lazy transitive closure), or None if the closure would exceed
        `_REACHABLE_CAP`.  Pure-Python BFS over the preloaded `_isA_supers`
        and `_partOf_wholes` dicts.

        Edge semantics, mirroring the original SPARQL:
            * isA: traverse upward — from *cur* to its supertypes.
            * partOf: traverse outward — from *cur* (the part) to its
              containing wholes.

        The cap prevents matrix cells from stalling on polysemic terms whose
        WordNet-style ancestor closure spans tens of thousands of nodes.
        """
        if term in self._trcl_cache:
            return self._trcl_cache[term]
        cap = self._REACHABLE_CAP
        reachable = {term}
        frontier = {term}
        bailed = False
        while frontier:
            next_frontier = set()
            for t in frontier:
                for n in self._isA_supers.get(t, ()):
                    if n not in reachable:
                        reachable.add(n)
                        next_frontier.add(n)
                for n in self._partOf_wholes.get(t, ()):
                    if n not in reachable:
                        reachable.add(n)
                        next_frontier.add(n)
                if len(reachable) > cap:
                    bailed = True
                    break
            if bailed:
                break
            frontier = next_frontier
        if bailed:
            self._trcl_cache[term] = None
            return None
        self._trcl_cache[term] = reachable
        return reachable

    def _is_reachable(self, lhs: str, rhs: str) -> bool:
        """Return True if *rhs* is reachable from *lhs* via isA/partOf edges."""
        r = self._get_reachable(lhs)
        return r is not None and rhs in r

    def getTransitiveClosureHier(self, t):
        # Kept for API compatibility; internal code uses _is_reachable directly.
        # Returns the cached reachable set for t (lazy, not the full closure).
        # Returns an empty set if the closure exceeds `_REACHABLE_CAP`.
        r = self._get_reachable(t)
        return r if r is not None else set()

    def getNounsWithProperties(self):
        return getattr(self, "nouns_with_properties", set())

    def getNounsWithA(self):
        return getattr(self, "nouns_with_a", set())

    def getSemiModalVerbs(self):
        return getattr(self, "semi_modal_verbs", set())

    def getPronouns(self):
        return getattr(self, "pronouns", set())

    def getPersonalPronouns(self):
        return getattr(self, "personal_pronouns", set())

    def getConsumptionVerbs(self):
        return getattr(self, "consumption_verbs", set())

    def getPrototypicalPrepositions(self):
        return getattr(self, "prototypical_prepositions", set())

    def getPhrasalVerbs(self):
        return getattr(self, "phrasal_verbs", set())

    def getTransitiveVerbs(self):
        return getattr(self, "transitive_verbs", set())

    def getRejectedVerbs(self):
        return getattr(self, "rejected_edges", set())

    def getNonVerbs(self):
        return getattr(self, "non_verbs", set())

    def getLogicalRewritingRules(self):
        self._load_logical_rules_from_json_if_available()
        return self.logical_rewriting_rules

    def getCausativeVerbs(self):
        return getattr(self, "causative_verbs", set())

    def getPredictionVerbs(self):
        return getattr(self, "prediction_verbs", set())

    def getChangeOfStateVerbs(self):
        """Verbs that license the causative alternation (cause-as-subject)
        but are *not* lexical causal signals. Used by the FOL rewriter to
        promote CAUSATION → subject; intentionally separate from
        `causative_verbs` because the TBox `CausativeVerb` class drives
        logical_analysis.json premises that would over-fire on these."""
        return getattr(self, "change_of_state_verbs", set())

    def isPartOf(self, part: str, whole: str, _cap: int = 256) -> bool:
        """Transitive meronymy check over the preloaded `_partOf_wholes` adjacency.

        Tries first an exact (case-insensitive) match on the full surface label,
        then falls back to head-noun matching: proper-noun-decorated entities
        like ``"Percy Street entrance"`` won't appear in the ontology, but
        ``entrance partOf station`` will — and the head noun of each side is
        what carries the meronymic type. We tokenise on whitespace and try the
        rightmost 1- and 2-token suffixes, which catches both single-word heads
        ("entrance", "station") and compound heads ("metro station").
        """
        if not part or not whole:
            return False
        adj = getattr(self, "_partOf_wholes", None)
        if not adj:
            return False
        lc_index = getattr(self, "_partOf_lc_index", None)
        if lc_index is None:
            lc_index = {k.lower(): k for k in adj.keys()}
            self._partOf_lc_index = lc_index

        def _reaches(start_label: str, target_lc_set: set) -> bool:
            seen = {start_label}
            frontier = {start_label}
            while frontier:
                nxt = set()
                for cur in frontier:
                    for w in adj.get(cur, ()):
                        if w.lower() in target_lc_set:
                            return True
                        if w not in seen:
                            seen.add(w)
                            nxt.add(w)
                            if len(seen) >= _cap:
                                return False
                frontier = nxt
            return False

        def _candidate_heads(label: str):
            toks = label.split()
            seen_c = []
            for cand in (label, " ".join(toks[-2:]) if len(toks) >= 2 else None,
                         toks[-1] if toks else None):
                if cand and cand not in seen_c:
                    seen_c.append(cand)
            return seen_c

        whole_targets = {c.lower() for c in _candidate_heads(whole)}
        for cand in _candidate_heads(part):
            start = lc_index.get(cand.lower())
            if start is None:
                continue
            if _reaches(start, whole_targets):
                return True
        return False

    def getMovementVerbs(self):
        return getattr(self, "movement_verbs", set())

    def getUnitsOfMeasure(self):
        return getattr(self, "units_of_measure", set())

    def getMeansVerbs(self):
        return getattr(self, "means_verbs", set())

    def getAbstractEntities(self):
        return getattr(self, "abstract_entities", set())

    def getStateVerbs(self):
        return getattr(self, "state_verbs", set())

    def getMaterialisationVerbs(self):
        return getattr(self, "materialisation_verbs", set())

    def getTemporalNouns(self):
        return getattr(self, "temporal_nouns", set())

    def getLocationNouns(self):
        return getattr(self, "location_nouns", set())

    def getStateNouns(self):
        return getattr(self, "state_nouns", set())

    def getFacilityNouns(self):
        return getattr(self, "facility_nouns", set())

    def getAccessPointNouns(self):
        return getattr(self, "access_point_nouns", set())

    def getRouteNouns(self):
        return getattr(self, "route_nouns", set())

    def getGeoSuffixNouns(self):
        return getattr(self, "geo_suffix_nouns", set())

    def getServiceStateNouns(self):
        return getattr(self, "service_state_nouns", set())

    def getStatusNouns(self):
        return getattr(self, "status_nouns", set())

    def getWeatherConditionNouns(self):
        return getattr(self, "weather_condition_nouns", set())

    def getWeatherConditionAdjectives(self):
        return getattr(self, "weather_condition_adjectives", set())

    def getFieldLabelNouns(self):
        return getattr(self, "field_label_nouns", set())

    def getConjunctions(self):
        return getattr(self, "conjunctions", set())

    def getCopulaSurfaceForms(self):
        """Inflected and contracted surface forms of the copula 'be'
        (am, is, are, was, were, be, been, being, 'm, 're, 's). Sourced
        from raw_data/verbs/copula_surface_forms.txt."""
        return getattr(self, "copula_surface_forms", set())

    def getPrepositions(self):
        """All honk:Preposition instances (union of all subclasses)."""
        return getattr(self, "prepositions", set())

    def collect_prepositions(self):
        return self.getPrepositions()

    def _clear(self):
        # Clear lru_cache entries that hold strong references to this instance,
        # otherwise the cache prevents GC and keeps the pyoxigraph Store open.
        self.name_eq.cache_clear()
        self.get_logical_functions.cache_clear()
        self.loaded = False
        self._syn_cache.clear()
        self._trcl_cache.clear()
        for attr in (
                "nouns_with_properties",
                "nouns_with_a",
                "semi_modal_verbs",
                "pronouns",
                "personal_pronouns",
                "consumption_verbs",
                "prototypical_prepositions",
                "transitive_verbs",
                "rejected_edges",
                "non_verbs",
                "logical_rewriting_rules",
                "causative_verbs",
                "change_of_state_verbs",
                "movement_verbs",
                "units_of_measure",
                "means_verbs",
                "abstract_entities",
                "state_verbs",
                "materialisation_verbs",
                "temporal_nouns",
                "location_nouns",
                "state_nouns",
                "facility_nouns",
                "access_point_nouns",
                "route_nouns",
                "geo_suffix_nouns",
                "service_state_nouns",
                "status_nouns",
                "weather_condition_nouns",
                "weather_condition_adjectives",
                "conjunctions",
                "prepositions",
                "copula_surface_forms",
        ):
            if hasattr(self, attr):
                getattr(self, attr).clear()
        self._loaded_rules_json_path = None
        self._loaded_rules_json_mtime = None

    @lru_cache(maxsize=128)
    def get_logical_functions(self, logical_construct_name, logical_construct_property):
        knows_query = """
         SELECT DISTINCT ?label ?attachTo ?argument ?logicalConstructProperty
         WHERE {
             ?a a honk:LogicalFunction.
             ?a rdfs:label ?label .
             ?a honk:logicalConstructName ?logicalConstructName .
             OPTIONAL { ?a honk:logicalConstructProperty ?logicalConstructProperty }
             ?a honk:attachTo ?attachTo .
             OPTIONAL { ?a honk:argument ?argument }
         }"""

        logical_functions = list()
        bindings = {
            'logicalConstructName': Literal(logical_construct_name, datatype=XSD.string),
        }
        for function in self._iter_rows(knows_query, bindings):
            # Manually filter by property to avoid OPTIONAL binding issues in SPARQL
            actual_prop = function.logicalConstructProperty.value if hasattr(function, 'logicalConstructProperty') and function.logicalConstructProperty is not None else None
            if actual_prop == logical_construct_property:
                logical_functions.append(LogicalRewritingRule(
                    label=function.label.value,
                    attachTo=function.attachTo.value,
                    argument=function.argument.value if hasattr(function, 'argument') and function.argument is not None else "property",
                    logicalConstructName=logical_construct_name,
                    logicalConstructProperty=actual_prop
                ))

        if not logical_functions and self._rules_json_path and os.path.exists(self._rules_json_path):
            # Fallback: Load directly from JSON if not in the RDF store
            log_defs, _ = SentenceStructure.load_logical_analysis(self._rules_json_path)
            if logical_construct_name in log_defs:
                for spec in log_defs[logical_construct_name].specs:
                    if spec.property == logical_construct_property:
                        logical_functions.append(LogicalRewritingRule(
                            label=f"log/{logical_construct_name}/{spec.property}" if spec.property else f"log/{logical_construct_name}",
                            attachTo=spec.attachTo,
                            argument=spec.argument,
                            logicalConstructName=logical_construct_name,
                            logicalConstructProperty=spec.property
                        ))

        return logical_functions

def _load_logical_rules_from_json(json_path: str):
    """Build logical_rewriting_rules directly from logical_analysis.json.

    Produces an identical defaultdict[int, Rule] to the SPARQL block in
    _load(), but reads the source JSON rather than querying the RDF store.
    """
    _, rules_list = SentenceStructure.load_logical_analysis(json_path)

    rules = defaultdict()
    rule_id = 1

    for matching in rules_list:
        not_premises = []
        raw_not = matching.premise.get("not")
        if raw_not:
            for nk, nv in raw_not.items():
                values = nv if isinstance(nv, list) else [nv]
                not_premises.append(Condition(name=nk, values=values))

        premises = []
        for key, val in matching.premise.items():
            if key == "not":
                continue
            values = val if isinstance(val, list) else [val]
            premises.append(Condition(name=key, values=values))

        # When a JSON rule has multiple classifications, create ONE Rule whose
        # additional_classifications list holds the extra (name, property) pairs.
        # This lets rewrite_node_logically apply all of them for a single match.
        if matching.classification:
            primary = matching.classification[0]
            extras = [
                (c.type, c.property)
                for c in matching.classification[1:]
            ]
            rules[rule_id] = Rule(
                id=rule_id,
                premises=premises,
                not_premises=not_premises,
                logicalConstructName=primary.type,
                logicalConstructProperty=primary.property,
                additional_classifications=extras,
            )
            rule_id += 1

    return rules


