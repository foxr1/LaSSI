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
    def init(cache_path, user, password, hostame, port, onStorage, path):
        if HOnKSingleton._instance.honk is None:
            HOnKSingleton._instance.honk = HOnK(cache_path, user, password, hostame, port, onStorage)
            HOnKSingleton._instance.honk.start(path)

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

    def start(self, filename=None):
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

        result = super().start()
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
            self._load()
        return result

    def stop(self):
        result = super().stop()
        if result:
            self._clear()
        return result

    def _load(self):
        if not self.hasDBStoredData():
            return False
        if self.loaded:
            return True

        _t0 = time.time()
        print("[HOnK] Loading ontology data...")

        self.st = defaultdict(set)

        # Batch all 14 type-label lookups into a single SPARQL query instead of
        # making 14 individual round-trips.
        _type_map = {
            "SemiModalVerb":           "semi_modal_verbs",
            "Pronoun":                 "pronouns",
            "PrototypicalPreposition": "prototypical_prepositions",
            "TransitiveVerb":          "transitive_verbs",
            "CausativeVerb":           "causative_verbs",
            "MotionVerb":              "movement_verbs",
            "MeansVerb":               "means_verbs",
            "StateVerb":               "state_verbs",
            "MaterialisationVerb":     "materialisation_verbs",
            "PhrasalVerb":             "phrasal_verbs",
            "UnitOfMeasure":           "units_of_measure",
            "AbstractEntity":          "abstract_entities",
            "Rejectable":              "rejected_edges",
            "Dependency":              "non_verbs",
        }
        _values_clause = " ".join(f"honk:{t}" for t in _type_map)
        _batch_query = f"""
            SELECT DISTINCT ?type_uri ?c
            WHERE {{
                VALUES ?type_uri {{ {_values_clause} }}
                ?a a ?type_uri.
                ?a rdfs:label ?c.
            }}"""
        _ns = str(self.namespace)
        _buckets = {attr: set() for attr in _type_map.values()}
        for row in self._iter_rows(_batch_query):
            type_local = str(row.type_uri)[len(_ns):]
            attr = _type_map.get(type_local)
            if attr is not None:
                _buckets[attr].add(str(row.c))
        for attr, s in _buckets.items():
            setattr(self, attr, s)
        print(f"[HOnK]   type labels loaded ({time.time()-_t0:.1f}s)")

        ## get_logical_rewriting_rules
        print(f"[HOnK]   loading logical rewriting rules...")
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
        # return self._single_unary_query(knows_query, lambda x: x)

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

        # syn is now computed lazily on demand in getSynonymy() to avoid
        # extracting and materialising all ~210K eq-pairs from the store at load
        # time, which causes OOM on large (WordNet-scale) TTL files.
        self._syn_cache: dict = {}

        # trcl is now computed lazily on demand in _get_reachable() to avoid
        # materialising the full transitive closure of ~788K isA/partOf pairs
        # at load time, which causes OOM on large (WordNet-scale) TTL files.
        self._trcl_cache: dict = {}


        # self.prepositions (full Preposition objects) is not used by the active
        # pipeline — collect_prepositions() has no call sites.  The only
        # preposition data the pipeline consumes is self.prototypical_prepositions,
        # which is already populated above by the batched type-label query.
        # The old loading block also contained a broken SPARQL UNION that left
        # ?src unbound in one branch, producing a cartesian product with every
        # labeled resource in the store and causing OOM on large TTL files.
        self.prepositions = {}
        self.loaded = True
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

    # SPARQL template used by getSynonymy for a single-term BFS step.
    # Finds all labels reachable from ?src_label in one hop via any predicate
    # that is an instance of honk:eq (both directions, since the TTL
    # stores symmetric pairs).
    _SYN_QUERY = """
        SELECT DISTINCT ?equiv_label
        WHERE {
            ?pred a honk:eq .
            {
                ?src rdfs:label ?src_label .
                ?src ?pred ?equiv .
                ?equiv rdfs:label ?equiv_label .
            }
            UNION
            {
                ?equiv rdfs:label ?src_label .
                ?equiv ?pred ?src .
                ?src rdfs:label ?equiv_label .
            }
        }"""

    def getSynonymy(self, k):
        """Return the set of all labels equivalent to *k* (including *k* itself).

        Uses SPARQL 1.1 property paths to follow all eq-instance edges in the
        store in one step. Results are cached across entire synonym groups.
        """
        if k in self._syn_cache:
            return self._syn_cache[k]

        # Use property paths (^?pred|?pred)* to follow symmetric eq edges.
        # This handles the full transitive/symmetric closure in one engine-level step.
        query = """
            SELECT DISTINCT ?equiv_label
            WHERE {
                ?src rdfs:label ?src_label .
                ?pred a honk:eq .
                ?src (^?pred|?pred)* ?equiv .
                ?equiv rdfs:label ?equiv_label .
            }"""
        bindings = {"src_label": Literal(k, datatype=XSD.string)}
        
        synonyms = {k}
        for row in self._iter_rows(query, bindings):
            if row.equiv_label:
                synonyms.add(str(row.equiv_label))

        # Cache the result for every discovered member.
        for t in synonyms:
            self._syn_cache[t] = synonyms
        return synonyms

        return visited

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
        knows_query = """
         SELECT DISTINCT ?dst 
         WHERE {
             ?src a ?dst.
             ?src rdfs:label ?src_label.
         }"""
        s = set()
        for x in self._iter_rows(knows_query, {"src_label": Literal(src, datatype=XSD.string)}):
            s.add(str(x.dst))
        return s

    def getSuperTypes(self, src):
        """Return the set of all superclasses of the entity *src* (including direct types and their ancestors).
        
        Uses SPARQL property paths (rdfs:subClassOf*) for fast, engine-level traversal.
        Results are cached in self.st.
        """
        if src in self.st:
            return self.st[src]
            
        query = """
            SELECT DISTINCT ?super
            WHERE {
                ?s rdfs:label ?label .
                ?s a ?type .
                ?type rdfs:subClassOf* ?super .
            }"""
        bindings = {"label": Literal(src, datatype=XSD.string)}
        
        supertypes = set()
        for row in self._iter_rows(query, bindings):
            if row.super:
                # Store full URIs as strings, matching previous behavior
                supertypes.add(str(row.super))
        
        self.st[src] = supertypes
        return supertypes

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
        f = None
        if isinstance(filename, io.IOBase):
            f = filename
        else:
            f = open(str(filename), "w")
        # Write a header row so FuzzyStringMatchDatabase.create()'s next(f) skip is harmless
        f.write(f"id\tidx\tt\ttype{os.linesep}")
        count = 1
        for k, v in l.items():
            t = self.most_specific_type(v)
            # Escape backslashes so PostgreSQL COPY doesn't treat them as escape sequences
            k_escaped = k.replace("\\", "\\\\")
            f.write(f"{count}\t{k_escaped}\t{k_escaped}\t{t}")
            count += 1
            if count <= n:
                f.write(os.linesep)
        return f

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
            resolveTypeFromOntologyLHS = set(self.getSuperTypes(src))
            resolveTypeFromOntologyRHS = set(self.getSuperTypes(dst))
            isect = resolveTypeFromOntologyLHS.intersection(resolveTypeFromOntologyRHS)
            if len(resolveTypeFromOntologyLHS) == 0:
                return CasusHappening.INDIFFERENT
            elif len(resolveTypeFromOntologyRHS) == 0:
                return CasusHappening.INDIFFERENT
            elif len(isect) == 0:
                return CasusHappening.INDIFFERENT
            else:
                srcS = self.getSynonymy(src)
                dstS = self.getSynonymy(dst)
                for k in isect:
                    # Check direct neqTo (both directions) via direct predicate and numbered instances
                    neqTo_src = (set(x["x"] for x in self.single_edge(src, "neqTo", "^x") if x.get("@^hasResult"))
                                 | self._getOutgoingNodesByClassInstance(src, "neqTo"))
                    neqTo_dst = (set(x["x"] for x in self.single_edge(dst, "neqTo", "^x") if x.get("@^hasResult"))
                                 | self._getOutgoingNodesByClassInstance(dst, "neqTo"))
                    # src neqTo dst, or any synonym of dst is neqTo of src
                    if dst in neqTo_src or len(set(dstS).intersection(neqTo_src)) > 0:
                        return CasusHappening.EXCLUSIVES
                    # dst neqTo src, or any synonym of src is neqTo of dst
                    if src in neqTo_dst or len(set(srcS).intersection(neqTo_dst)) > 0:
                        return CasusHappening.EXCLUSIVES
                    if len(set(srcS).intersection(set(dstS))) > 0:
                        return CasusHappening.EQUIVALENT
                    for lhs in self.getSynonymy(src):
                        for rhs in self.getSynonymy(dst):
                            if self._is_reachable(lhs, rhs):
                                return CasusHappening.GENERAL_IMPLICATION
                return CasusHappening.INDIFFERENT

    # SPARQL template used by _get_reachable for a single BFS step.
    # Reflects the two edge directions from the original trcl construction:
    #   isA  with flip=True  → edges run supertype → subtype
    #                          (lhs is destination; we follow incoming isA edges)
    #   partOf with flip=False → edges run part → whole
    #                          (lhs is source; we follow outgoing partOf edges)
    _HIER_QUERY = """
        SELECT DISTINCT ?next_label
        WHERE {
            {
                ?pred a honk:isA .
                ?next ?pred ?cur .
                ?cur  rdfs:label ?cur_label .
                ?next rdfs:label ?next_label .
            }
            UNION
            {
                ?pred a honk:partOf .
                ?cur  ?pred ?next .
                ?cur  rdfs:label ?cur_label .
                ?next rdfs:label ?next_label .
            }
        }"""

    def _get_reachable(self, term: str) -> set:
        """Return the set of all terms reachable from *term* via isA/partOf
        edges (lazy closure using SPARQL property paths)."""
        if term in self._trcl_cache:
            return self._trcl_cache[term]

        # Use SPARQL property paths to find the closure of isA/partOf in one step.
        # This handles the full reachability from *term* at the engine level.
        query = """
            SELECT DISTINCT ?dst_label
            WHERE {
                ?src rdfs:label ?src_label .
                ?pred_isa    a honk:isA .
                ?pred_partof a honk:partOf .
                ?src (^?pred_isa|?pred_partof)* ?dst .
                ?dst rdfs:label ?dst_label .
            }"""
        bindings = {"src_label": Literal(term, datatype=XSD.string)}
        
        reachable = {term}
        for row in self._iter_rows(query, bindings):
            if row.dst_label:
                reachable.add(str(row.dst_label))

        self._trcl_cache[term] = reachable
        return reachable

    def _is_reachable(self, lhs: str, rhs: str) -> bool:
        """Return True if *rhs* is reachable from *lhs* via isA/partOf edges."""
        return rhs in self._get_reachable(lhs)

    def getTransitiveClosureHier(self, t):
        # Kept for API compatibility; internal code uses _is_reachable directly.
        # Returns the cached reachable set for t (lazy, not the full closure).
        return self._get_reachable(t)

    def getNounsWithProperties(self):
        return self.nouns_with_properties

    def getNounsWithA(self):
        return self.nouns_with_a

    def getSemiModalVerbs(self):
        return self.semi_modal_verbs

    def getPronouns(self):
        return self.pronouns

    def getPrototypicalPrepositions(self):
        return self.prototypical_prepositions

    def getPhrasalVerbs(self):
        return self.phrasal_verbs

    def getTransitiveVerbs(self):
        return self.transitive_verbs

    def getRejectedVerbs(self):
        return self.rejected_edges

    def getNonVerbs(self):
        return self.non_verbs

    def getLogicalRewritingRules(self):
        return self.logical_rewriting_rules

    def getCausativeVerbs(self):
        return self.causative_verbs

    def getMovementVerbs(self):
        return self.movement_verbs

    def getUnitsOfMeasure(self):
        return self.units_of_measure

    def getMeansVerbs(self):
        return self.means_verbs

    def getAbstractEntities(self):
        return self.abstract_entities

    def getStateVerbs(self):
        return self.state_verbs

    def getMaterialisationVerbs(self):
        return self.materialisation_verbs

    def collect_prepositions(self):
        return self.prepositions

    def _clear(self):
        # Clear lru_cache entries that hold strong references to this instance,
        # otherwise the cache prevents GC and keeps the pyoxigraph Store open.
        self.name_eq.cache_clear()
        self.get_logical_functions.cache_clear()
        self.loaded = False
        self._syn_cache.clear()
        self._trcl_cache.clear()
        self.nouns_with_properties.clear()
        self.nouns_with_a.clear()
        self.semi_modal_verbs.clear()
        self.pronouns.clear()
        self.prototypical_prepositions.clear()
        self.transitive_verbs.clear()
        self.rejected_edges.clear()
        self.non_verbs.clear()
        self.logical_rewriting_rules.clear()
        self.causative_verbs.clear()
        self.movement_verbs.clear()
        self.units_of_measure.clear()
        self.means_verbs.clear()
        self.abstract_entities.clear()
        self.state_verbs.clear()
        self.materialisation_verbs.clear()
        self.prepositions.clear()

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
             ?a honk:argument ?argument .
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
                    logicalConstructName=logical_construct_name,
                    logicalConstructProperty=actual_prop
                ))

        return logical_functions

def load_from_txt_file(p:HOnK, path:str, classes:list, to_reject:set):
    with open(path, "r") as dep:
        for line in dep:
            line = line.strip()
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)

def generate_honk_graph(p:HOnK, data_path:str, result_path:str=None):
    p.create_property("hasAdjective")
    p.create_property("subject")
    p.create_property("d_object")
    p.create_property("composite_form_with")
    p.create_property("attachTo")
    p.create_property("argument")
    p.create_property("logicalConstructProperty")
    p.create_property("logicalConstructName")
    p.create_relationship("hasProperty")
    p.create_relationship("formOf")
    p.create_relationship("entryPoint")
    p.create_relationship("partOf")
    p.create_relationship("isA")
    p.create_relationship("relatedTo")
    p.create_relationship("capableOf")
    p.create_relationship("adjectivalForm")
    p.create_relationship("adverbialForm")
    p.create_relationship("eq")
    p.create_relationship("neqTo")
    _T = p.create_class("Dimensions")
    LOC_T = p.create_class("LOC", "Dimensions")
    GPE_T = p.create_class("GPE", ["Dimensions", "LOC"])
    gp_T = p.create_class("GraphParse")
    reject_T = p.create_class("Rejectable", "GraphParse",
                              comment="Whether the edge shall be rejected in the internal rewriting pipeline")
    meta_T = p.create_class("MetaGrammaticalFunction")
    dep_T = p.create_class("dependency", "MetaGrammaticalFunction")
    log_f_T = p.create_class("LogicalFunction", "MetaGrammaticalFunction",
                             comment="The sentence constructs at the logical level, similarly to English' Adverbial Phrases and Indirect Objects (https://it.wikipedia.org/wiki/Analisi_logica_della_proposizione vs. https://en.wikipedia.org/wiki/Adverbial_phrase)")
    log_f_T = p.create_class("LogicalRewritingRule", "MetaGrammaticalFunction",
                             comment="Defines how to capture the elements within the sentence structure and rewriting them in the most appropriate way as properties of the kernel/singleton they refer to")
    gr_obj_T = p.create_class("GrammaticalFunction", "MetaGrammaticalFunction")
    verb_T = p.create_class("Measure", "GrammaticalFunction")  # TODO: Is this a grammatical function
    verb_T = p.create_class("Concept", "GrammaticalFunction")  # TODO: Is this a grammatical function
    verb_T = p.create_class("Verb", "GrammaticalFunction")
    verb_T = p.create_class("Preposition", "GrammaticalFunction")
    noun_T = p.create_class("Noun", "GrammaticalFunction")
    adj_T = p.create_class("Adjective", "GrammaticalFunction")
    adj_T = p.create_class("Adverb", "GrammaticalFunction")
    adj_T = p.create_class("CompoundForm", "GrammaticalFunction")
    tverb_T = p.create_class("TransitiveVerb", "Verb")
    iverb_T = p.create_class("IntransitiveVerb", "Verb")
    causverb_T = p.create_class("CausativeVerb", "Verb")
    moveverb_T = p.create_class("MovementVerb", "Verb")
    meansverb_T = p.create_class("MeansVerb", "Verb")
    stateverb_T = p.create_class("StateVerb", "Verb")
    matverb_T = p.create_class("MaterialisationVerb", "Verb")
    phrasalverb_T = p.create_class("PhrasalVerb", "Verb")
    semimodalverb_T = p.create_class("SemiModalVerb", "Verb")
    proto_Prop = p.create_class("PrototypicalPreposition", "Preposition")
    dep_Prop = p.create_class("DependantPreposition", "Preposition")
    idio_Prop = p.create_class("IdiomaticPreposition", "Preposition")
    complex_Prop = p.create_class("ComplexPreposition", "Preposition")
    pronoun = p.create_class("Pronoun")
    pronoun_per = p.create_class("PersonalPronoun", "Pronoun")
    pronoun_dem = p.create_class("DemonstrativePronoun", "Pronoun")
    pronoun_rel = p.create_class("RelativePronoun", "Pronoun")
    pronoun_indef = p.create_class("IndefinitePronoun", "Pronoun")
    pronoun_interro = p.create_class("InterrogativePronoun", "Pronoun")
    unit_of_measure = p.create_class("UnitOfMeasure", "Measure")
    abstract_concept = p.create_class("AbstractEntity", "Concept")
    to_reject = None
    with open(os.path.join(data_path, "rejected_edge_types.txt"), "r") as dep:
        to_reject = {line.strip().lower() for line in dep}

    load_from_txt_file(p, os.path.join(data_path, "non_verb_types.txt"), ["dependency"], to_reject)
    load_from_txt_file(p, os.path.join(data_path, "verbs", "causative_verbs.txt"), ["CausativeVerb"], to_reject)
    load_from_txt_file(p, os.path.join(data_path, "verbs", "transitive_verbs.txt"), ["TransitiveVerb"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "units_of_measure.txt"), ["UnitOfMeasure"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "abstract_entity_concepts.txt"), ["AbstractEntity"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "verbs", "stative_verbs.txt"), ["Verb", "CausativeVerb"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "verbs", "means_verbs.txt"), ["Verb", "MeansVerb"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "verbs", "state_verbs.txt"), ["Verb", "StateVerb"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "verbs", "movement_verbs.txt"), ["Verb", "MovementVerb"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "verbs", "materialisation_verbs.txt"), ["Verb", "MaterialisationVerb"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "verbs", "phrasal_verbs.txt"), ["Verb", "PhrasalVerb"], to_reject)


    for preposition in Prepositions.load_prepositions(os.path.join(data_path, "prepositions.json")):
        classes = preposition.generate_classes(preposition.name.lower() in to_reject)
        p.create_entity(preposition.name, classes, **preposition.as_properties())

    log_defs, log_rewr_rules = SentenceStructure.load_logical_analysis(os.path.join(data_path, "logical_analysis.json"))
    for name, v in log_defs.items():
        for x in v.specs:
            d = dataclasses.asdict(x)
            if "property" in d and d["property"] is None:
                d.pop("property")
            else:
                d["logicalConstructProperty"] = d.pop("property")
            d["logicalConstructName"] = name
            entity_name = f"log/{name}/{d['logicalConstructProperty']}" if "logicalConstructProperty" in d else f"log/{name}"
            p.create_entity(entity_name, "LogicalFunction", entity_name, **d)
    ruleid = 1
    for rule in log_rewr_rules:
        for result in rule.classification:
            dres = dataclasses.asdict(result)
            if "property" in dres and dres["property"] is None:
                dres.pop("property")
            else:
                dres["logicalConstructProperty"] = dres.pop("property")
            dres["logicalConstructName"] = dres.pop("type")
            dres["rule_order"] = ruleid
            dres.update(rule.premise)
            p.create_entity(f"logrule/{ruleid}", "LogicalRewritingRule", **dres)
            ruleid += 1

    load_from_txt_file(p, os.path.join(data_path,  "pronouns","personal_pronouns.txt"), ["Pronoun", "PersonalPronoun"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "pronouns","demonstrative_pronouns.txt"), ["Pronoun", "DemonstrativePronoun"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "pronouns","relative_pronouns.txt"), ["Pronoun", "RelativePronoun"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "pronouns","indefinite_pronouns.txt"), ["Pronoun", "IndefinitePronoun"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "pronouns", "interrogative_pronouns.txt"), ["Pronoun", "InterrogativePronoun"], to_reject)
    load_from_txt_file(p, os.path.join(data_path,  "verbs", "semi_modal_verbs.txt"), ["Verb", "SemiModalVerb"], to_reject)

    p.create_concept("world", ["Noun"])
    p.create_concept("picture", ["Noun"])
    p.create_concept("hectic", "Adjective")
    p.create_concept("beautiful", "Adjective")
    p.create_concept("fabulous", "Adjective")
    p.create_concept("center", "Adjective")
    p.create_concept("centre", "Adjective")
    p.create_relationship_instance("center", "eq", "centre", True)
    p.create_concept("busy", "Adjective")
    p.create_concept("crowded", "Adjective")
    p.create_concept("fast", "Adjective")
    p.create_concept("busy", "Adjective")
    p.create_concept("slow#adj", "Adjective", entity_name="slow")
    p.create_concept("slow#v", "Verb", entity_name="slow")
    p.create_relationship_instance("slow#v", "adjectivalForm", "slow#adj")
    p.create_relationship_instance("slow#adj", "neqTo", "fast", True)
    p.create_concept("crowd#n", "Noun", entity_name="crowd")
    p.create_concept("crowd#v", "Verb", entity_name="crowd")
    p.create_relationship_instance("busy", "relatedTo", "crowd#n", True)
    p.create_relationship_instance("crowd#n", "relatedTo", "crowded", True)
    p.create_relationship_instance("crowd#v", "relatedTo", "crowded", True)
    p.create_concept("city", "LOC")
    p.create_concept("Newcastle", "GPE")
    p.create_relationship_instance("Newcastle", "isA", "city")
    p.create_concept("come back#v", "Verb", entity_name="come back")
    p.create_concept("traffic#v", "Verb", entity_name="traffic")
    p.create_concept("traffic#n", "Noun", entity_name="traffic")
    p.create_concept("flow in", "Verb")
    p.create_concept("flow#v", "Verb", entity_name="flow")
    p.create_relationship_instance("flow#v", "adjectivalForm", "fast")
    p.create_concept("flow#n", "Noun", entity_name="flow")
    p.create_concept("congestion", ["Noun"])
    p.create_concept("jam", ["Noun"])
    p.create_concept("busy city", ["Noun"], hasAdjective="busy", entryPoint="city")
    p.create_concept("traffic jam", ["Noun"], composite_with=["traffic#n", "jam"], entryPoint="traffic#n")
    p.create_concept("traffic congestion", ["Noun"], composite_with=["traffic#n", "congestion"], entryPoint="traffic#n")
    p.create_relationship_instance("traffic jam", "eq", "traffic congestion", True)
    p.create_relationship_instance("traffic jam", "eq", "traffic congestion", True)
    p.create_concept("flow fast", "CompoundForm", hasAdjective="fast", entryPoint="flow#v")
    p.create_relationship_instance("flow in", "eq", "flow#v", True)
    p.create_concept("traffic jam can slow traffic", "CompoundForm", entryPoint="slow", subject="traffic jam", d_object="traffic#n")
    p.create_concept("city centers", "LOC", hasAdjective="center", entryPoint="city")
    p.create_concept("city centres", "LOC", hasAdjective="centre", entryPoint="city")
    p.create_concept("city center", "LOC", hasAdjective="center", entryPoint="city")
    p.create_concept("city centre", "LOC", hasAdjective="centre", entryPoint="city")
    p.create_relationship_instance("city", "hasProperty", "busy", True)
    p.create_relationship_instance("city center", "partOf", "city")
    p.create_relationship_instance("city center", "eq", "city centre", True)
    p.create_relationship_instance("city centre", "partOf", "city")
    p.create_relationship_instance("city centers", "partOf", "city")
    p.create_relationship_instance("city centres", "partOf", "city")
    p.create_relationship_instance("city center", "eq", "city centers", True)
    p.create_relationship_instance("city center", "eq", "city centres", True)
    p.create_relationship_instance("city centre", "eq", "city centers", True)
    p.create_relationship_instance("city centre", "eq", "city centres", True)
    p.create_relationship_instance("city centers", "eq", "city centres", True)
    p.create_relationship_instance("busy", "relatedTo", "crowd#n", True)
    p.create_relationship_instance("congestion", "relatedTo", "traffic congestion", True)
    p.create_relationship_instance("crowd#n", "relatedTo", "congestion", True)
    p.create_relationship_instance("busy city", "relatedTo", "crowd#n", True)
    p.create_relationship_instance("traffic jam", "capableOf", "traffic jam can slow traffic")
    p.create_relationship_instance("hectic", "hasProperty", "traffic#n", True)
    p.create_relationship_instance("hectic", "eq", "busy", True)

    ## This is a tumor: single entity match in post-processing
    p.create_concept("embryoma_of_the_kidney#n", "Noun", entity_name="embryoma of the kidney")

    ## These are the examples leading to swapped specification properties via nmod, so, not specification(front,letter) as per nmod, but swapped specification(letter,front)
    p.create_concept("letter#n", "Noun", entity_name="letter")
    p.create_concept("front#n", "Noun", entity_name="front")
    p.create_relationship_instance("letter#n", "hasProperty", "front#n") ## Example of inverse occurrence of the relationship if compared to the nmod (isSymmetricalIfComparedToNMod,hasNModPartOf)
    p.create_concept("street#n", "Noun", entity_name="street")
    p.create_concept("corner#n", "Noun", entity_name="corner")
    p.create_relationship_instance("street#n", "hasProperty", "corner#n")
    p.create_concept("cell_division#n", "Noun", entity_name="cell division")
    p.create_concept("phase#n", "Noun", entity_name="phase")
    p.create_relationship_instance("cell_division#n", "hasProperty", "phase#n")
    p.create_concept("object#n", "Noun", entity_name="object")
    p.create_concept("surface#n", "Noun", entity_name="surface")
    p.create_relationship_instance("object#n", "hasProperty", "surface#n")
    p.create_concept("game#n", "Noun", entity_name="game")
    p.create_concept("chess#n", "Noun", entity_name="chess")
    p.create_relationship_instance("chess#n", "isA", "game#n") #(isSymmetricalIfComparedToNMod,hasNModIsA)

    if result_path is None:
        result_path = "HOnK.ttl"
    p.serialize(result_path)