import os
import time
import dataclasses as _dc
from collections import defaultdict

from LaSSI.structures.extended_fol.Formulae import FUnaryPredicate
from LaSSI.HOnK.HOnK import HOnKSingleton
from FunctionalMatch.utils import CountingDictionary


def knowledge_expansion_legacy(sentence, queries, filter=None):
    """
    :param sentence:    Single atom/proposition
    """
    from LaSSI.HOnK.HOnK import HOnKSingleton
    assert HOnKSingleton.isReady()
    from LaSSI.structures.extended_fol.Formulae import FAnd, FOr
    assert (not isinstance(sentence, FAnd)) and (not isinstance(sentence, FOr))
    S = dict()
    S[sentence] = list()
    # previous_size = 0
    toVisit = {sentence}
    while len(toVisit) > 0:
        # previous_size = len(S)
        tmp = set()
        for src in toVisit:
            for idx, query in enumerate(queries):
                hasResult, outcomes = query([src])
                if hasResult:
                    for x in outcomes:
                        if filter is not None and callable(filter) and filter(x):
                            S[src].append((idx,x))
                            if x not in S:
                                S[x] = list()
                                tmp.add(x)
        toVisit = tmp
    return S #Expanded knowledge



class KnowledgeExpansion:
    def __init__(self, filename):
        self.constituents = CountingDictionary()
        self.filename = filename
        self.Graph = dict()
        self._subGraph = defaultdict(set)

    def dump(self):
        with open(self.filename, "wb") as p:
            import pickle
            pickle.dump(self, p, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        filename = os.path.abspath(os.fspath(filename))
        obj = None
        if os.path.exists(filename):
            with open(filename, "rb") as p:
                    import pickle
                    obj = pickle.load(p)
                    assert isinstance(obj, KnowledgeExpansion)
                    obj.filename = filename
                    # obj.constituents = constituent
                    return obj
        if obj is None:
            obj = KnowledgeExpansion(filename)
        return obj

    def outgoingEdges(self, idx, predDstIdx=None, predRel=None):
        # idx = self.constituents.contains(x)
        if idx == -1:
            return []
        if (predDstIdx is None) and (predRel is None):
            return [(ruleId, self.constituents.fromId(dstId)) for (ruleId, dstId) in self.Graph[idx]]
        else:
            if predDstIdx is None:
                predDstIdx = lambda x: True
            if predRel is None:
                predRel = lambda x: True
            return [(ruleId, self.constituents.fromId(dstId)) for (ruleId, dstId) in self.Graph[idx] if
                    predDstIdx(dstId) and predRel(ruleId)]

    def outgoingEdgesIdx(self, idx, predDstIdx=None, predRel=None):
        if idx == -1:
            return []
        if (predDstIdx is None) and (predRel is None):
            return self.Graph[idx]
        else:
            if predDstIdx is None:
                predDstIdx = lambda x: True
            if predRel is None:
                predRel = lambda x: True
            return [(ruleId, dstId) for (ruleId, dstId) in self.Graph[idx] if
                    predDstIdx(dstId) and predRel(ruleId)]

    def idxGraph(self):
        return self.Graph

    def fullGraph(self):
        return {self.constituents.fromId(key): self.outgoingEdges(key) for key in self.Graph.keys()}

    def subGraph(self, x):
        idx = self.constituents.contains(x)
        if idx == -1:
            return dict()
        return {self.constituents.fromId(key): self.outgoingEdges(key, lambda x: x in self._subGraph[idx]) for key in self._subGraph[idx]}

    def subGraphPred(self, x, edgeLabelPred):
        idx = self.constituents.contains(x)
        if idx == -1:
            return dict()
        return {self.constituents.fromId(key): self.outgoingEdges(key, lambda x: x in self._subGraph[idx], edgeLabelPred) for key in self._subGraph[idx]}

    def subGraphIdx(self, x):
        idx = self.constituents.contains(x)
        if idx == -1:
            return dict()
        return {key: self.outgoingEdgesIdx(key, lambda x: x in self._subGraph[idx]) for key in self._subGraph[idx]}

    def subGraphIdxPred(self, x, edgeLabelPred):
        idx = self.constituents.contains(x)
        if idx == -1:
            return dict()
        return {key: self.outgoingEdgesIdx(key, lambda x: x in self._subGraph[idx], edgeLabelPred) for key in self._subGraph[idx]}

    def fromIdx(self, x):
        return self.constituents.fromId(x)

    def getIDx(self, obj):
        return self.constituents.contains(obj)

    def get_full_expansion_with_graph(self, sentence, ruleLabel, max_depth=1):
        entry_point, wasAlreadyPresent = self.constituents.add_with_wasPresent(sentence)
        assert wasAlreadyPresent
        finallyVisited = set()
        Q = [(entry_point, 0)]
        adj_list = dict()
        while len(Q)>0:
            curr, depth = Q.pop(0)
            if curr in finallyVisited:
                continue
            if curr not in adj_list:
                adj_list[curr] = defaultdict(list)
            finallyVisited.add(curr)
            
            if depth >= max_depth:
                continue
                
            if curr in self.Graph:
                for (lR, idxR), dst in self.Graph[curr]:
                    if lR == ruleLabel:
                        Q.append((dst, depth + 1))
                        adj_list[curr][idxR].append(dst)
        return entry_point, adj_list, {x: self.constituents.fromId(x) for x in finallyVisited}, {self.constituents.fromId(x) for x in finallyVisited}

    def get_full_expansion(self, sentence, ruleLabel, max_depth=1):
        idx, wasAlreadyPresent = self.constituents.add_with_wasPresent(sentence)
        assert wasAlreadyPresent
        finallyVisited = set()
        Q = [(idx, 0)]
        while len(Q)>0:
            curr, depth = Q.pop(0)
            if curr in finallyVisited:
                continue
            finallyVisited.add(curr)
            
            if depth >= max_depth:
                continue
                
            if curr in self.Graph:
                for (lR, idxR), dst in self.Graph[curr]:
                    if lR == ruleLabel:
                        Q.append((dst, depth + 1))
        return {self.constituents.fromId(x) for x in finallyVisited}

    def pruned_expansion(self, sentence, queries, ruleLabel=None, alreadyVisitedIdx=None, filter=None, max_depth=1):
        if alreadyVisitedIdx is None:
            alreadyVisitedIdx = dict()
        if isinstance(queries, list) or isinstance(queries, tuple):
            queries = {idx: q for idx, q in enumerate(queries)}
        assert isinstance(alreadyVisitedIdx, dict)
        from LaSSI.HOnK.HOnK import HOnKSingleton
        assert HOnKSingleton.isReady()

        idx, wasAlreadyPresent = self.constituents.add_with_wasPresent(sentence)

        if idx not in self.Graph:
            self.Graph[idx] = set()

        toVisit = {(idx, 0)}
        allVisited = set()
        if idx not in alreadyVisitedIdx:
            alreadyVisitedIdx[idx] = 0

        # Track which rules have been executed to avoid redundant calls.
        if not hasattr(self, '_executed_rules'):
            self._executed_rules = set()

        from tqdm import tqdm
        is_root_call = len(allVisited) == 0
        pbar = tqdm(desc=f"  Expanding {ruleLabel}", leave=False, disable=not is_root_call)

        while len(toVisit) > 0:
            tmp = set()
            current_level_indices = {i for i, d in toVisit}
            allVisited.update(current_level_indices)
            pbar.update(len(toVisit))

            t_it0 = time.time()
            for i_src, (srcIdx, depth) in enumerate(toVisit):
                src = self.constituents.fromId(srcIdx)

                # FIX 1: Semantic Caching
                # Use a string/canonical representation instead of the raw integer ID to
                # catch semantically identical objects that have different memory hashes.
                src_semantic_key = str(src)

                if time.time() - t_it0 > 5:
                    print(f"    - processing node {i_src}/{len(toVisit)} at depth {depth}: {src}")
                    t_it0 = time.time()

                if depth >= max_depth:
                    continue

                existing_rules = defaultdict(lambda: defaultdict(set))
                if srcIdx in self.Graph:
                    for (ruleLabel2, idx_rule2), dstIdx in self.Graph[srcIdx]:
                        existing_rules[idx_rule2][ruleLabel2].add(dstIdx)

                for idx_rule, query in queries.items():
                    # Check cache using the semantic key rather than just the ID
                    cache_key = (src_semantic_key, ruleLabel, idx_rule)
                    if cache_key in self._executed_rules:
                        pass

                    ruleWasPreviouslyExecuted = False
                    ruleWasStraightfowardlyExpanded = False

                    if idx_rule in existing_rules:
                        dstToIgnore = existing_rules[idx_rule].get(ruleLabel, set())

                        dstToConsider = set()
                        for label, dsts in existing_rules[idx_rule].items():
                            if label != ruleLabel:
                                dstToConsider.update(dsts)

                        final = dstToConsider.difference(dstToIgnore)
                        for x in final:
                            self.Graph[srcIdx].add(((ruleLabel, idx_rule), x))
                            # FIX 2: Strict Cycle Pruning (Standard BFS property)
                            if x not in alreadyVisitedIdx:
                                if x not in self.Graph:
                                    self.Graph[x] = set()
                                alreadyVisitedIdx[x] = depth + 1
                                tmp.add((x, depth + 1))

                        ruleWasPreviouslyExecuted = len(dstToIgnore) > 0
                        ruleWasStraightfowardlyExpanded = len(final) > 0

                        if ruleWasPreviouslyExecuted:
                            for x in dstToIgnore:
                                # FIX 2: Strict Cycle Pruning
                                if x not in alreadyVisitedIdx:
                                    if x not in self.Graph:
                                        self.Graph[x] = set()
                                    alreadyVisitedIdx[x] = depth + 1
                                    tmp.add((x, depth + 1))

                    if (not ruleWasPreviouslyExecuted) and (
                    not ruleWasStraightfowardlyExpanded) and cache_key not in self._executed_rules:
                        t_q0 = time.time()
                        hasResult, outcomes = query([src])
                        if time.time() - t_q0 > 1:
                            print(f"      - Rule {idx_rule} took {time.time() - t_q0:.1f}s on {src}")

                        self._executed_rules.add(cache_key)

                        if hasResult:
                            for x in outcomes:
                                if (filter is None) or (callable(filter) and filter(x)):
                                    try:
                                        hash(x)
                                    except TypeError:
                                        raise
                                    dstIdx, _ = self.constituents.add_with_wasPresent(x)
                                    self.Graph[srcIdx].add(((ruleLabel, idx_rule), dstIdx))

                                    # FIX 2: Strict Cycle Pruning
                                    if dstIdx not in alreadyVisitedIdx:
                                        if dstIdx not in self.Graph:
                                            self.Graph[dstIdx] = set()
                                        alreadyVisitedIdx[dstIdx] = depth + 1
                                        tmp.add((dstIdx, depth + 1))
                    else:
                        self._executed_rules.add(cache_key)

            toVisit = tmp
        pbar.close()
        self._subGraph[idx].update(allVisited)
        return alreadyVisitedIdx



def non_redundant_constituents(f, strictTyping = True):
    assert HOnKSingleton.isReady()
    p = HOnKSingleton.get()
    from LaSSI.structures.extended_fol.Formulae import is_selfstanding_variable
    return not (isinstance(f, FUnaryPredicate) and (f.rel == "be") and ((f.properties is None) or ((len(f.properties) == 0))) and ((is_selfstanding_variable(f.arg) and ((not strictTyping) or p.hasTypedObject(f.arg.name)))))

class TBoxReasoningSingleton(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @staticmethod
    def isReady():
        return (TBoxReasoningSingleton._instance is not None) and (TBoxReasoningSingleton._instance.rules is not None)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            cls._instance.rules = None
        return cls._instance

    @staticmethod
    def init(impl_file, eq_file, ke_file):
        impl_file = os.path.abspath(os.fspath(impl_file))
        eq_file = os.path.abspath(os.fspath(eq_file))
        ke_file = os.path.abspath(os.fspath(ke_file))
        if (TBoxReasoningSingleton._instance.rules is None or
                os.path.abspath(TBoxReasoningSingleton._instance.rules.ke_file) != ke_file):
            TBoxReasoningSingleton._instance.rules = TBoxReasoning(impl_file, eq_file, ke_file)

    def hasPersistedBefore(self):
        if TBoxReasoningSingleton._instance.rules is None:
            return None ## Do not know the state of it
        if not os.path.exists(TBoxReasoningSingleton._instance.rules.ke_file):
            return False
        with open(TBoxReasoningSingleton._instance.rules.ke_file) as f:
            import pickle
            obj = pickle.load(f)
            return isinstance(obj, KnowledgeExpansion)

    @staticmethod
    def subGraphIdx(x):
        return TBoxReasoningSingleton._instance.rules.ke.subGraphIdx(x)
    #
    # @staticmethod
    # def subGraphImpl(x):
    #     return TBoxReasoningSingleton._instance.rules.ke.subGraphIdxPred(x, lambda x: x.startswith("impl"))

    @staticmethod
    def getConstituentFromIdx(x):
        return TBoxReasoningSingleton._instance.rules.ke.fromIdx(x)

    @staticmethod
    def getConstituentIdx(x):
        return TBoxReasoningSingleton._instance.rules.ke.getIDx(x)

    @staticmethod
    def subGraphEq(x):
        return TBoxReasoningSingleton._instance.rules.ke.subGraphIdxPred(x, lambda x: x[0] == "eqR")

    @staticmethod
    def subGraphImpl(x):
        return TBoxReasoningSingleton._instance.rules.ke.subGraphIdxPred(x, lambda x: x[0] == "implR")

    @staticmethod
    def knowledge_expand(formula, isImpl):
        assert TBoxReasoningSingleton.isReady()
        rules = TBoxReasoningSingleton.get_eq_rules() if (not isImpl) else TBoxReasoningSingleton.get_impl_rules()
        label = "implR" if isImpl else "eqR"
        S = TBoxReasoningSingleton._instance.rules.impl_already_visited_set if isImpl else TBoxReasoningSingleton._instance.rules.eq_already_visited_set
        TBoxReasoningSingleton._instance.rules.ke.pruned_expansion(formula, rules, label, S, non_redundant_constituents)
        return TBoxReasoningSingleton._instance.rules.ke.get_full_expansion(formula, label)

    @staticmethod
    def explained_knowledge_expand(formula, isImpl):
        assert TBoxReasoningSingleton.isReady()
        rules = TBoxReasoningSingleton.get_eq_rules() if (not isImpl) else TBoxReasoningSingleton.get_impl_rules()
        label = "implR" if isImpl else "eqR"
        S = TBoxReasoningSingleton._instance.rules.impl_already_visited_set if isImpl else TBoxReasoningSingleton._instance.rules.eq_already_visited_set
        TBoxReasoningSingleton._instance.rules.ke.pruned_expansion(formula, rules, label, S, non_redundant_constituents)
        return TBoxReasoningSingleton._instance.rules.ke.get_full_expansion_with_graph(formula, label)

    @staticmethod
    def get_ke_file_name():
        assert TBoxReasoningSingleton.isReady()
        return TBoxReasoningSingleton._instance.rules.ke_file

    @staticmethod
    def get_eq_rules():
        assert TBoxReasoningSingleton.isReady()
        return TBoxReasoningSingleton._instance.rules.get_eq_rules()

    @staticmethod
    def get_impl_rules():
        assert TBoxReasoningSingleton.isReady()
        return TBoxReasoningSingleton._instance.rules.get_impl_rules()

    @staticmethod
    def dump():
        assert TBoxReasoningSingleton.isReady()
        TBoxReasoningSingleton._instance.rules.dump()

    @staticmethod
    def getIDXGraph():
        assert TBoxReasoningSingleton.isReady()
        return TBoxReasoningSingleton._instance.rules.ke.idxGraph()


class TBoxReasoning(object):
    def __init__(self, impl_file, eq_file, ke_file):
        self.eq_file = os.path.abspath(os.fspath(eq_file))
        self.impl_file = os.path.abspath(os.fspath(impl_file))
        self.ke_file = os.path.abspath(os.fspath(ke_file))
        self.rule_mapping = CountingDictionary.load(self.ke_file+"rules_")
        from FunctionalMatch.language.LanguageMainPoint import parse_query
        self.impl_rules = {self.rule_mapping.add(x): x for x in parse_query(self.impl_file)}
        self.eq_rules = {self.rule_mapping.add(x): x for x in parse_query(self.eq_file)}
        # self.constituents = CountingDictionary.load(ke_file+"const_")
        self.ke = KnowledgeExpansion.load(self.ke_file)
        # self.ke_impl = KnowledgeExpansion.load(self.constituents, ke_file)
        self.eq_already_visited_set = dict()
        self.impl_already_visited_set = dict()

    def dump(self):
        self.ke.dump()

    def get_eq_rules(self):
        return self.eq_rules

    def get_impl_rules(self):
        return self.impl_rules
