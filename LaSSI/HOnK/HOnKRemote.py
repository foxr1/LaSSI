import os
import time
import logging
import requests
import re
import pyoxigraph
from collections import defaultdict
from typing import Optional, List, Set

# Import from HOnK to reuse its logic
from LaSSI.HOnK.HOnK import HOnK, HOnKSingleton, LogicalRewritingRule
# Import from FunctionalMatch.rdf.RDFGraph for types HOnK might check
from FunctionalMatch.rdf.RDFGraph import _TermWrapper, Literal, Namespace, XSD

class ProxySet:
    """
    A set-like object that lazy-loads its membership from a remote source.
    Useful for 'scoped' ingestion where we only check words present in the sentence.
    """
    def __init__(self, check_func):
        self._check = check_func
        self._cache = {}

    def __contains__(self, item):
        if item in self._cache:
            return self._cache[item]
        res = self._check(item)
        self._cache[item] = res
        return res

    def intersection(self, other):
        return {x for x in other if x in self}

    def __iter__(self):
        # We can't easily iterate over the remote set without fetching it all.
        # This will return an empty iterator for now to prevent accidental slow-downs.
        return iter([])

    def __len__(self):
        return 0

class RemoteTermWrapper:
    """
    Wraps a pyoxigraph term to provide the .value property and string conversion 
    expected by HOnK.
    """
    __slots__ = ("_term",)

    def __init__(self, term):
        self._term = term

    @property
    def value(self):
        if isinstance(self._term, pyoxigraph.Literal):
            dt = self._term.datatype.value if self._term.datatype else None
            v = self._term.value
            if dt == str(XSD.boolean):
                return v == "true"
            if dt == str(XSD.integer):
                try: return int(v)
                except: return v
            if dt in (str(XSD.double), str(XSD.float), str(XSD.decimal)):
                try: return float(v)
                except: return v
            return v
        return self._term.value

    def __str__(self) -> str:
        return self._term.value

    def __repr__(self) -> str:
        return f"RemoteTermWrapper({self._term!r})"

    def __eq__(self, other) -> bool:
        v = other.value if hasattr(other, 'value') else other
        return str(self.value) == str(v)

    def __lt__(self, other) -> bool:
        v = other.value if hasattr(other, 'value') else other
        try:
            return self.value < v
        except TypeError:
            return str(self.value) < str(v)

    def __bool__(self) -> bool:
        return self._term is not None

    def __hash__(self) -> int:
        return hash(str(self.value))

class RemoteQueryRow:
    __slots__ = ("_dict",)
    def __init__(self, row_dict):
        self._dict = row_dict

    def __getattr__(self, name):
        if name in self._dict:
            return self._dict[name]
        return None

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def asdict(self):
        return self._dict

class HOnKRemote(HOnK):
    """
    A remote version of HOnK that uses Fully Lazy Scoped Loading.
    Instead of bulk-loading any metadata at startup, it fetches everything
    from GraphDB on-demand for specific terms and caches them.
    """
    def __init__(self, endpoint_url, cache_path="cache"):
        super().__init__(cache_path, "remote", "remote", "remote", "0", onStorage=False)
        self.endpoint_url = endpoint_url
        self._ns_prefixes = {
            "honk": "https://ofox.co.uk/honk#",
            "honk": "https://ofox.co.uk/honk#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }
        self.namespace = Namespace(self._ns_prefixes["honk"])
        self.graph = None
        
        # Lazy caches
        self._lazy_syn = {}
        self._lazy_st = {}
        self._lazy_trcl = {}
        
        # Base class attributes required by LaSSI phases
        self.st = {}
        self.logical_rewriting_rules = defaultdict(list)
        
        # Initialize proxy sets for all categories
        self._type_map = {
            "SemiModalVerb":           "semi_modal_verbs",
            "Pronoun":                 "pronouns",
            "PrototypicalPreposition": "prototypical_prepositions",
            "TransitiveVerb":          "transitive_verbs",
            "CausativeVerb":           "causative_verbs",
            "MovementVerb":            "movement_verbs",
            "MeansVerb":               "means_verbs",
            "StateVerb":               "state_verbs",
            "MaterialisationVerb":     "materialisation_verbs",
            "PhrasalVerb":             "phrasal_verbs",
            "UnitOfMeasure":           "units_of_measure",
            "AbstractEntity":          "abstract_entities",
            "Rejectable":              "rejected_edges",
            "Dependency":              "non_verbs",
        }
        
        for type_local, attr_name in self._type_map.items():
            check_func = lambda term, t=type_local: self._check_type_remote(term, t)
            setattr(self, attr_name, ProxySet(check_func))

    def _check_type_remote(self, term: str, type_local: str) -> bool:
        """Check if a term has a specific type in the remote ontology."""
        query = f"""
            SELECT (COUNT(*) as ?count)
            WHERE {{
                ?s rdfs:label ?label .
                ?s a honk:{type_local} .
            }}"""
        bindings = {"label": Literal(term, datatype=XSD.string)}
        data = self._run_query(query, bindings)
        try:
            count = int(data['results']['bindings'][0]['count']['value'])
            return count > 0
        except (IndexError, KeyError, ValueError):
            return False

    def _actual_start(self):
        print(f"[HOnKRemote] SPARQL endpoint: {self.endpoint_url}")
        self.names = {}
        self.relationships = {}
        self.classes = {}
        return True

    def start(self, filename=None):
        """Skip parsing local TTL and fetch only required global structure."""
        self._actual_start()
        # Fetch logical rewriting rules (needed by rewriting phases)
        self._load()
        self.loaded = True
        return True

    def _load(self):
        """Fetch global ontology structure (logical rules) while keeping categories as lazy ProxySets."""
        # 1. Back up our ProxySets so the base class doesn't overwrite them
        proxies = {attr: getattr(self, attr) for attr in self._type_map.values()}
        
        # 2. Call the base class _load() which fetches logical rules and 
        # populates self.logical_rewriting_rules.
        print("[HOnKRemote] Fetching global ontology structure (logical rules)...")
        super()._load()
        
        # 3. Restore our ProxySets
        for attr, proxy in proxies.items():
            setattr(self, attr, proxy)
            
        # 4. Local Overrides for logical rules (since the remote ontology might be incomplete)
        # Add 'until' to time/continuous rule
        found_time_rule = None
        for rule in self.logical_rewriting_rules.values():
            if rule.logicalConstructName == 'time' and rule.logicalConstructProperty == 'continuous':
                found_time_rule = rule
                break
        if found_time_rule:
            for premise in found_time_rule.premises:
                if premise.name == 'Preposition' and 'until' not in premise.values:
                    premise.values.append('until')

        # Add 'at' to space/stay in place rule
        found_space_rule = None
        for rule in self.logical_rewriting_rules.values():
            if rule.logicalConstructName == 'space' and rule.logicalConstructProperty == 'stay in place':
                found_space_rule = rule
                # Preemptively choose Rule 3 or similar low-order rule if possible
                if int(rule.id) < 10:
                    break
        
        if found_space_rule:
            for premise in found_space_rule.premises:
                if premise.name == 'Preposition' and 'at' not in premise.values:
                    premise.values.append('at')

        # Add 'due' and 'due to' to causation rule
        found_causation_rule = None
        for rule in self.logical_rewriting_rules.values():
            if rule.logicalConstructName == 'causation' and int(rule.id) == 6:
                found_causation_rule = rule
                break
        if found_causation_rule:
            for premise in found_causation_rule.premises:
                if premise.name == 'Preposition':
                    if 'due' not in premise.values:
                        premise.values.append('due')
                    if 'due to' not in premise.values:
                        premise.values.append('due to')
            
        return True

    def _run_query(self, query: str, bindings: dict = None):
        """Execute SPARQL against the remote endpoint."""
        prefixes = "".join([f"PREFIX {p}: <{u}>\n" for p, u in self._ns_prefixes.items()])
        full_query = prefixes + query
        
        if bindings:
            values_clauses = []
            for var, val in bindings.items():
                if isinstance(val, pyoxigraph.NamedNode):
                    sparql_val = f"<{val.value}>"
                elif isinstance(val, pyoxigraph.Literal):
                    if val.datatype:
                        sparql_val = f'"{val.value}"^^<{val.datatype.value}>'
                    else:
                        sparql_val = f'"{val.value}"'
                elif hasattr(val, 'value'):
                    v = val.value
                    if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
                        sparql_val = f"<{v}>"
                    else:
                        sparql_val = f'"{v}"'
                else:
                    sparql_val = f'"{val}"'
                values_clauses.append(f"VALUES (?{var}) {{ ({sparql_val}) }}")
            full_query = re.sub(r"WHERE\s*\{", f"WHERE {{ {' '.join(values_clauses)} ", full_query, count=1, flags=re.IGNORECASE)

        headers = {'Accept': 'application/sparql-results+json', 'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(self.endpoint_url, data={'query': full_query}, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"SPARQL query failed: {response.text}")
        return response.json()

    def _iter_rows(self, query: str, bindings: dict = None):
        data = self._run_query(query, bindings)
        vars_ = data.get('head', {}).get('vars', [])
        for result in data.get('results', {}).get('bindings', {}):
            row_dict = {}
            for var in vars_:
                val_data = result.get(var)
                if val_data:
                    if val_data['type'] == 'uri':
                        term = pyoxigraph.NamedNode(val_data['value'])
                    elif val_data['type'] in ('literal', 'typed-literal'):
                        datatype = val_data.get('datatype')
                        term = pyoxigraph.Literal(val_data['value'], datatype=pyoxigraph.NamedNode(datatype) if datatype else None)
                    else:
                        term = pyoxigraph.Literal(val_data['value'])
                    row_dict[var] = RemoteTermWrapper(term)
                else:
                    row_dict[var] = None
            yield RemoteQueryRow(row_dict)

    # ------------------------------------------------------------------
    # Lazy methods: Fetch only what we need for the current term(s)
    # ------------------------------------------------------------------

    def getSynonymy(self, term: str) -> Set[str]:
        """Fetch synonyms for a single term lazily."""
        if term in self._lazy_syn:
            return self._lazy_syn[term]
            
        query = """
            SELECT DISTINCT ?equiv_label
            WHERE {
                ?s rdfs:label ?src_label .
                ?pred a honk:eq .
                ?s ?pred ?equiv .
                ?equiv rdfs:label ?equiv_label .
            }"""
        bindings = {"src_label": Literal(term, datatype=XSD.string)}
        
        synonyms = {term} 
        for row in self._iter_rows(query, bindings):
            if row.equiv_label:
                synonyms.add(str(row.equiv_label))
        
        self._lazy_syn[term] = synonyms
        return synonyms

    def getSuperTypes(self, term: str) -> Set[str]:
        """Fetch full supertype set (rdf:type + rdfs:subClassOf closure) for a term lazily."""
        if term in self._lazy_st:
            return self._lazy_st[term]
            
        query = """
            SELECT DISTINCT ?super
            WHERE {
                ?s rdfs:label ?label .
                ?s a ?type .
                ?type rdfs:subClassOf* ?super .
                FILTER(strstarts(str(?super), "https://ofox.co.uk/honk#"))
            }"""
        bindings = {"label": Literal(term, datatype=XSD.string)}
        
        supertypes = set()
        for row in self._iter_rows(query, bindings):
            if row.super:
                supertypes.add(str(row.super))
        
        self._lazy_st[term] = supertypes
        return supertypes

    def _get_reachable(self, term: str, predicate_uri: str) -> Set[str]:
        """Lazy reachability for specific predicate (isA or partOf)."""
        cache_key = (term, predicate_uri)
        if cache_key in self._lazy_trcl:
            return self._lazy_trcl[cache_key]
            
        query = f"""
            SELECT DISTINCT ?dst_label
            WHERE {{
                ?src rdfs:label ?src_label .
                ?src <{predicate_uri}>* ?dst .
                ?dst rdfs:label ?dst_label .
            }}"""
        bindings = {"src_label": Literal(term, datatype=XSD.string)}
        
        reachable = set()
        for row in self._iter_rows(query, bindings):
            if row.dst_label:
                reachable.add(str(row.dst_label))
        
        self._lazy_trcl[cache_key] = reachable
        return reachable

    def getTransitiveClosureHier(self, term: str, rel: str) -> Set[str]:
        """Fetch hierarchy reachability for isA or partOf lazily."""
        if rel == "isA":
            return self._get_reachable(term, str(self.namespace.isA))
        elif rel == "partOf":
            return self._get_reachable(term, str(self.namespace.partOf))
        return {term}

    # ------------------------------------------------------------------

    def fuzzyMatch(self, threshold: float, objectString: str):
        """
        Perform exact label matching remotely to fulfill the 'fuzzy' interface.
        Returns a mapping from score (1.0) to set of monads.
        """
        query = """
            SELECT DISTINCT ?label
            WHERE {
                ?s rdfs:label ?label .
            }"""
        bindings = {"label": Literal(objectString, datatype=XSD.string)}
        
        results = defaultdict(set)
        for row in self._iter_rows(query, bindings):
            results[1.0].add(str(row.label))
        return results

    def typedFuzzyMatch(self, threshold: float, objectString: str):
        """
        Perform exact label matching remotely with type information.
        Returns a mapping from score (1.0) to set of (monad, type) tuples.
        """
        query = """
            SELECT DISTINCT ?label ?type
            WHERE {
                ?s rdfs:label ?label .
                ?s a ?type .
                FILTER(strstarts(str(?type), "https://ofox.co.uk/honk#"))
            }"""
        bindings = {"label": Literal(objectString, datatype=XSD.string)}
        
        results = defaultdict(set)
        for row in self._iter_rows(query, bindings):
            type_name = str(row.type).split("#")[-1]
            results[1.0].add((str(row.label), type_name))
        return results

    def stop(self):
        self._lazy_syn.clear()
        self._lazy_st.clear()
        self._lazy_trcl.clear()
        return True

    def hasDBStoredData(self):
        return True

class HOnKRemoteSingleton:
    @staticmethod
    def init_remote(endpoint_url, cache_path="cache"):
        inst = HOnKSingleton.instance()
        if inst.honk is None:
            print(f"[HOnKRemote] Initializing Fully-Lazy Remote HOnK (Scoped Ingestion)")
            inst.honk = HOnKRemote(endpoint_url, cache_path)
            inst.honk.start()
        return inst
