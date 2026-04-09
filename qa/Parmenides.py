import dataclasses
import io
import pickle
from dataclasses import dataclass
import os.path
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from typing import Optional, List

import dacite
from rdflib import Literal, XSD
from FunctionalMatch.rdf.RDFGraph import RDFGraph
import logging




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



class ParmenidesSingleton(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @staticmethod
    def isReady():
        return (ParmenidesSingleton._instance is not None) and (ParmenidesSingleton._instance.parmenides is not None)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            cls._instance.parmenides = None
        return cls._instance

    @staticmethod
    def init(cache_path, user, password, hostame, port, onStorage, path):
        if ParmenidesSingleton._instance.parmenides is None:
            ParmenidesSingleton._instance.parmenides = Parmenides(cache_path, user, password, hostame, port, onStorage)
            ParmenidesSingleton._instance.parmenides.start(path)

    @staticmethod
    def stop():
        if ParmenidesSingleton._instance is not None and ParmenidesSingleton._instance.parmenides is not None:
            ParmenidesSingleton._instance.parmenides.stop()
            ParmenidesSingleton._instance.parmenides = None
        ParmenidesSingleton._instance = None

    @staticmethod
    def get() -> 'Parmenides':
        return ParmenidesSingleton._instance.parmenides


class Parmenides(RDFGraph):

    def __init__(self, cache_path, user, password, hostame, port, onStorage =True):
        super().__init__("parmenides", "https://logds.github.io/parmenides#", user, password, hostame, port, "parmenides", databaseConn=onStorage)
        self.loaded = False
        assert os.path.exists(cache_path) and os.path.isdir(cache_path)
        self.cache_path = cache_path
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.logger = logging.getLogger("Parmenides")
        self.onStorage = onStorage

    def start(self, filename=None):
        result = super().start()
        if result:
            self.logger.info("Parmenides started")
            if not self.onStorage and filename is not None:
                self.parse(filename)
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

        self.trcl = defaultdict(set)
        self.syn = defaultdict(set)
        self.st = defaultdict(set)
        self.semi_modal_verbs = set(self.get_label_is_a("SemiModalVerb"))
        self.pronouns = set(self.get_label_is_a("Pronoun"))
        self.prototypical_prepositions = set(self.get_label_is_a("PrototypicalPreposition"))
        self.transitive_verbs = set(self.get_label_is_a("TransitiveVerb"))
        self.causative_verbs = set(self.get_label_is_a("CausativeVerb"))
        self.movement_verbs = set(self.get_label_is_a("MovementVerb"))
        self.means_verbs = set(self.get_label_is_a("MeansVerb"))
        self.state_verbs = set(self.get_label_is_a("StateVerb"))
        self.materialisation_verbs = set(self.get_label_is_a("MaterialisationVerb"))
        self.phrasal_verbs = set(self.get_label_is_a("PhrasalVerb"))
        self.units_of_measure = set(self.get_label_is_a("UnitOfMeasure"))
        self.abstract_entities = set(self.get_label_is_a("AbstractEntity"))
        self.rejected_edges = set(self.get_label_is_a("Rejectable"))
        self.non_verbs = set(self.get_label_is_a("dependency"))

        ## get_logical_rewriting_rules
        knows_query = """
                 SELECT DISTINCT ?label ?rule_order ?preposition ?logicalConstructName ?logicalConstructProperty ?verb_of_motion ?SingletonHasBeenMatchedBy ?not ?abstract_entity ?hasNMod ?hasNModPartOf ?hasNModIsA ?isSymmetricalIfComparedToNMod ?hasNumber ?hasUnitOfMeasure ?verb_of_aims ?verb_of_state ?verb_of_means ?causative_verb ?isMaterializationVerb
                 WHERE {
                     ?a a parmenides:LogicalRewritingRule.
                     ?a rdfs:label ?label .
                     ?a parmenides:logicalConstructName ?logicalConstructName .
                     ?a parmenides:rule_order ?rule_order .
                     OPTIONAL { ?a parmenides:preposition ?preposition }
                     OPTIONAL { ?a parmenides:logicalConstructProperty ?logicalConstructProperty }
                     OPTIONAL { ?a parmenides:SingletonHasBeenMatchedBy ?SingletonHasBeenMatchedBy }
                     OPTIONAL { ?a parmenides:not ?not }
                     OPTIONAL { ?a parmenides:abstract_entity ?abstract_entity }
                     OPTIONAL { ?a parmenides:hasNMod ?hasNMod }
                     OPTIONAL { ?a parmenides:hasNModPartOf ?hasNModPartOf }
                     OPTIONAL { ?a parmenides:hasNModIsA ?hasNModIsA }
                     OPTIONAL { ?a parmenides:isSymmetricalIfComparedToNMod ?isSymmetricalIfComparedToNMod }
                     OPTIONAL { ?a parmenides:causative_verb ?causative_verb }
                     OPTIONAL { ?a parmenides:hasNumber ?hasNumber }
                     OPTIONAL { ?a parmenides:hasUnitOfMeasure ?hasUnitOfMeasure }
                     OPTIONAL { ?a parmenides:verb_of_motion ?verb_of_motion }
                     OPTIONAL { ?a parmenides:verb_of_aims ?verb_of_aims }
                     OPTIONAL { ?a parmenides:verb_of_state ?verb_of_state }
                     OPTIONAL { ?a parmenides:verb_of_means ?verb_of_means }
                     OPTIONAL { ?a parmenides:isMaterializationVerb ?isMaterializationVerb }
                 }"""
        # return self._single_unary_query(knows_query, lambda x: x)

        not_query = """
                 SELECT DISTINCT ?preposition ?verb_of_motion ?SingletonHasBeenMatchedBy ?abstract_entity ?hasNMod ?hasNModPartOf ?hasNModIsA ?isSymmetricalIfComparedToNMod ?hasNumber ?hasUnitOfMeasure ?verb_of_aims ?verb_of_state ?verb_of_means ?causative_verb ?isMaterializationVerb
                 WHERE {
                     OPTIONAL { ?a parmenides:preposition ?preposition }
                     OPTIONAL { ?a parmenides:SingletonHasBeenMatchedBy ?SingletonHasBeenMatchedBy }
                     OPTIONAL { ?a parmenides:abstract_entity ?abstract_entity }
                     OPTIONAL { ?a parmenides:hasNMod ?hasNMod }
                     OPTIONAL { ?a parmenides:hasNModPartOf ?hasNModPartOf }
                     OPTIONAL { ?a parmenides:hasNModIsA ?hasNModIsA }
                     OPTIONAL { ?a parmenides:isSymmetricalIfComparedToNMod ?isSymmetricalIfComparedToNMod }
                     OPTIONAL { ?a parmenides:causative_verb ?causative_verb }
                     OPTIONAL { ?a parmenides:hasNumber ?hasNumber }
                     OPTIONAL { ?a parmenides:hasUnitOfMeasure ?hasUnitOfMeasure }
                     OPTIONAL { ?a parmenides:verb_of_motion ?verb_of_motion }
                     OPTIONAL { ?a parmenides:verb_of_aims ?verb_of_aims }
                     OPTIONAL { ?a parmenides:verb_of_state ?verb_of_state }
                     OPTIONAL { ?a parmenides:verb_of_means ?verb_of_means }
                     OPTIONAL { ?a parmenides:isMaterializationVerb ?isMaterializationVerb }
                 }"""

        str_premises = {"preposition", "verb_of_motion", "SingletonHasBeenMatchedBy", "abstract_entity", "hasNMod",
                        "hasNModPartOf", "hasNModIsA", "isSymmetricalIfComparedToNMod", "hasNumber", "hasUnitOfMeasure",
                        "verb_of_aims", "verb_of_state", "verb_of_means", "causative_verb", "isMaterializationVerb"}

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
                    not_query_premises = self.graph.query(not_query, initBindings={'a': getattr(rule, "not")})
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
        ## End: get_logical_rewriting_rules

        ## Nouuns with properties
        knows_query = """
                 SELECT DISTINCT ?hasProperty ?label
                 WHERE {
                     ?a a parmenides:Noun.
                     ?a rdfs:label ?label .
                     ?a parmenides:hasProperty ?hasProperty .
                 }"""

        self.logger.info("nouns_with_properties loaded")
        self.nouns_with_properties = set(self._single_unary_query(knows_query, lambda x: x))
        ## End: nouns with properties

        ## Start: nouns_with_a
        knows_query = """
                         SELECT DISTINCT ?isA ?label
                         WHERE {
                             ?a a parmenides:Noun.
                             ?a rdfs:label ?label .
                             ?a parmenides:isA ?isA .
                         }"""
        self.nouns_with_a = set(self._single_unary_query(knows_query, lambda x: x))
        self.logger.info("nouns_with_a loaded")
        ## End: nouns_with_a

        self.syn = None
        syn_pickle = os.path.join(self.cache_path, "syn.pickle")
        if os.path.exists(syn_pickle):
            with open(syn_pickle, "rb") as f:
                self.syn = pickle.load(f)
        else:
            self.syn = self.extractPureHierarchy("eqTo", True) | self.extractPureHierarchy("eqTo", False)
            self.logger.info("syn eqTo extracted")
            from FunctionalMatch.utils import transitive_closure
            self.syn = transitive_closure(self.syn)
            self.logger.info("syn eqTo transitive closure")
            tmp = defaultdict(set)
            for (x, y) in self.syn:
                tmp[x].add(y)
                tmp[y].add(x)
            self.syn = tmp
            self.logger.info("syn eqTo transitive closure storing")
            with open(syn_pickle, "wb") as f:
                pickle.dump(self.syn, f, protocol=pickle.HIGHEST_PROTOCOL)

        hier_pickle = os.path.join(self.cache_path, "hier.pickle")
        if os.path.exists(hier_pickle):
            with open(hier_pickle, "rb") as f:
                self.trcl = pickle.load(f)
        if len(self.trcl) == 0:
            from FunctionalMatch.utils import transitive_closure
            s = self.extractPureHierarchy("isA", True) | self.extractPureHierarchy("partOf", False)
            self.logger.info("syn isA/partOf extracted")
            self.trcl = transitive_closure(s)
            self.logger.info("syn isA/partOf transitive closure")
            with open(hier_pickle, "wb") as f:
                pickle.dump(self.trcl, f, protocol=pickle.HIGHEST_PROTOCOL)


        ## Prepositions
        self.prepositions = None
        prep_pickle = os.path.join(self.cache_path, "prepositions.pickle")
        if os.path.exists(prep_pickle):
            with open(prep_pickle, "rb") as f:
                self.prepositions = pickle.load(f)
        if self.prepositions is None or len(self.prepositions) == 0:
            # from LaSSI.Parmenides.Prepositions import Preposition
            from Prepositions import Preposition
            query = """
            SELECT *
            WHERE {
                { ?src a <https://logds.github.io/parmenides#Preposition>. }
                UNION
            { ?s a ?t. ?t rdfs:subClassOf  <https://logds.github.io/parmenides#Preposition>. }
             ?src rdfs:label ?src_label.
            }"""
            result = defaultdict(dict)
            for d in self._run_custom_sparql_query(query):
                rel = str(d.get("t", ""))[len(self.namespace):]
                # rel = str(d.get("rel", ""))[len(Parmenides.parmenides_ns):]
                label = str(d["src_label"])
                local = result[label]
                # print(label)
                result[label] = Preposition.update_with_label(local, rel)
            query = """
                    SELECT *
                    WHERE {
                     ?src ?prop ?value.
                     ?prop a owl:ObjectProperty.
                     ?src rdfs:label ?label.
                    }"""
            for k in result.keys():
                binding = {"label": Literal(k, datatype=XSD.string)}
                for d in self._run_custom_sparql_query(query, bindings=binding):
                    rel = str(d.get("prop", ""))[len(self.namespace):]
                    if isinstance(d["value"], Literal):
                        # print(rel)
                        result[k][rel] = d["value"].value
            self.prepositions = dict()
            for k, v in result.items():
                v["name"] = k
                self.prepositions[k] = dacite.from_dict(Preposition, v)
            ## Prepositions: end
            with open(prep_pickle, "wb") as f:
                pickle.dump(self.prepositions, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.loaded = True
        return True



    def most_specific_type(self, types):
        types = list(map(lambda x: str(x).lower(), types))
        ### TODO: within .ttl and type inference
        if any(map(lambda x: "verb" in x, types)):
            return "VERB"
        elif "gpe" in types:
            return "GPE"
        elif "loc" in types:
            return "LOC"
        elif "org" in types:
            return "ORG"
        elif "noun" in types:
            return "noun"
        elif "entity" in types:
            return "ENTITY"
        elif "adjective" in types:
            return "JJ"
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
             ?a a parmenides:%s.
             ?a rdfs:label ?c .
         }""" % label
        return self.string_query(knows_query, "c")

    def getAllEntitiesByImmediateType(self, t):
        ye = list(self.isA("^x", str(t)))
        if len(ye) == 0:
            return set()
        else:
            return {x["x"] for x in ye}

    def getSynonymy(self, k):
        if k in self.syn:
            return self.syn[k]
        else:
            return {k}

    def typeOf2(self, src):
        knows_query = """
         SELECT DISTINCT ?dst 
         WHERE {
             ?src rdfs:subClassOf ?dst.
         }"""
        qres = self.graph.query(knows_query, initBindings={"src": src})
        s = set()
        for x in qres:
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
        qres = self.graph.query(knows_query, initBindings={"src_label": Literal(src, datatype=XSD.string)})
        s = set()
        for x in qres:
            s.add(str(x.dst))
        return s

    def getSuperTypes(self, src):
        if src in self.st:
            return self.st[src]
        s = list(self.typeOf(src))
        visited = set()
        while len(s) > 0:
            x = s.pop()
            if x not in visited:
                visited.add(x)
                for y in self.typeOf2(self.namespace[str(x)[len(self.namespace):]]):
                    s.append(y)
        self.st[src] = visited
        return visited

    def getTypedObjects(self):
            typing = """
             SELECT DISTINCT ?s ?src_label ?dst
    WHERE {
       ?s a ?dst .
       ?s rdfs:label ?src_label.
    } """
            qres = self.graph.query(typing)
            d = defaultdict(set)
            for x in qres:
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
        count = 1
        for k, v in l.items():
            k, t = k, self.most_specific_type(v)
            f.write(f"{count}\t{k}\t{k}\t{t}")
            count += 1
            if count <= n:
                f.write(os.linesep)
        return f

    @lru_cache()
    def name_eq(self, src, dst):
        if (src == dst):
            return CasusHappening.EQUIVALENT
        elif (src is None) or len(src) == 0:
            return CasusHappening.MISSING_1ST_IMPLICATION
        elif (dst is None) or len(dst) == 0:
            return CasusHappening.INDIFFERENT
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
                for k in isect:
                    if len(list(self.single_edge(src, "neqTo", dst))) > 0:
                        return CasusHappening.EXCLUSIVES
                    srcS = self.getSynonymy(src)
                    dstS = self.getSynonymy(dst)
                    if len(set(srcS).intersection(set(dstS))) > 0:
                        return CasusHappening.EQUIVALENT
                    for lhs in self.getSynonymy(src):
                        for rhs in self.getSynonymy(dst):
                            if (lhs, rhs) in self.getTransitiveClosureHier(k):
                                return CasusHappening.GENERAL_IMPLICATION
                return CasusHappening.INDIFFERENT

    def getTransitiveClosureHier(self, t):
        return self.trcl

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
        self.trcl.clear()
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
         SELECT DISTINCT ?label ?attachTo ?argument
         WHERE {
             ?a a parmenides:LogicalFunction.
             ?a rdfs:label ?label .
             ?a parmenides:logicalConstructName ?logicalConstructName .
             OPTIONAL { ?a parmenides:logicalConstructProperty ?logicalConstructProperty }
             ?a parmenides:attachTo ?attachTo .
             ?a parmenides:argument ?argument .
         }"""

        logical_functions = list()
        query_functions = self.graph.query(knows_query, initBindings={'logicalConstructName': Literal(logical_construct_name, datatype=XSD.string), 'logicalConstructProperty': Literal(logical_construct_property, datatype=XSD.string)})

        for function in query_functions:
            logical_functions.append(LogicalRewritingRule(
                label=function.label.value,
                attachTo=function.attachTo.value,
                logicalConstructName=logical_construct_name,
                logicalConstructProperty=logical_construct_property
            ))

        return logical_functions