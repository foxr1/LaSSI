import os
from collections import defaultdict

# from LaSSI.structures.extended_fol.Formulae import FUnaryPredicate
# from LaSSI.HOnK.HOnK import HOnKSingleton
# from FunctionalMatch.utils import CountingDictionary
from LaSSI.HOnK.HOnK import HOnKSingleton
from Formulae import FUnaryPredicate, FAnd, FOr
from FunctionalMatch.utils import CountingDictionary


def knowledge_expansion_legacy(sentence, queries, filter=None):
    """
    :param sentence:    Single atom/proposition
    """
    #from LaSSI.HOnK.HOnK import HOnKSingleton
    assert HOnKSingleton.isReady()
    #from LaSSI.structures.extended_fol.Formulae import FAnd, FOr
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
        if not os.path.exists(self.filename):
            with open(self.filename, "wb") as p:
                import pickle
                pickle.dump(self, p, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        import os
        obj = None
        if os.path.exists(filename):
            with open(filename, "rb") as p:
                    import pickle
                    obj = pickle.load(p)
                    assert isinstance(obj, KnowledgeExpansion)
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

    def get_full_expansion(self, sentence, ruleLabel):
        idx, wasAlreadyPresent = self.constituents.add_with_wasPresent(sentence)
        assert wasAlreadyPresent
        allVisited = self._subGraph[idx]
        finallyVisited = set()
        Q = list(allVisited)
        while len(Q)>0:
            curr = Q.pop(0)
            if curr in finallyVisited:
                continue
            finallyVisited.add(curr)
            if curr in self.Graph:
                for (lR, idxR), dst in self.Graph[curr]:
                    if lR == ruleLabel:
                        Q.append(dst)
        return {self.constituents.fromId(x) for x in finallyVisited}

    def pruned_expansion(self, sentence, queries, ruleLabel=None, alreadyVisitedIdx=None, filter=None):
        if alreadyVisitedIdx is None:
            alreadyVisitedIdx = set()
        if isinstance(queries, list) or isinstance(queries, tuple):
            queries = {idx:q for idx, q in enumerate(queries)}
        assert isinstance(alreadyVisitedIdx, set)
        #from LaSSI.HOnK.HOnK import HOnKSingleton
        assert HOnKSingleton.isReady()
        #from LaSSI.structures.extended_fol.Formulae import FAnd, FOr
        assert (not isinstance(sentence, FAnd)) and (not isinstance(sentence, FOr))
        idx, wasAlreadyPresent = self.constituents.add_with_wasPresent(sentence)
        wasAlreadyPresent = wasAlreadyPresent and idx in alreadyVisitedIdx
        # if wasAlreadyPresent:
        #     print(f"Skipping: {idx}")
        #     return
        # else:
        #     print(f"{idx} for {sentence}")

        if idx not in self.Graph:
            self.Graph[idx] = set()
        toVisit = {idx}
        allVisited = set()
        alreadyVisitedIdx.add(idx)
        dstToIgnore = set()
        dstToConsider = set()
        while len(toVisit) > 0:
            tmp = set()
            allVisited.update(toVisit)
            for srcIdx in toVisit:
                src = self.constituents.fromId(srcIdx)
                for idx_rule, query in queries.items():

                    ## If srcIdx already appeared another time for another rule
                    ruleWasPreviouslyExecuted = False
                    ruleWasStraightfowardlyExpanded = False
                    if (srcIdx in self.Graph) and (len(self.Graph[srcIdx])>0):
                        ## If this lead to a computation, then very likely this was already re-computed, and
                        ## I don't have to take the effort of re-computing it, but just to add the new edge
                        ## label. This also allows not to re-compute the rules, and motivates why there
                        ## should be just one graph!
                        dstToIgnore.clear()
                        dstToConsider.clear()
                        for ((ruleLabel2, idx_rule2), dstIdx) in self.Graph[srcIdx]:
                            if (idx_rule == idx_rule2):
                                if (ruleLabel2 == ruleLabel):
                                    dstToIgnore.add(dstIdx)
                                else:
                                    dstToConsider.add(dstIdx)
                        final = dstToConsider.difference(dstToIgnore)
                        for x in final:
                            self.Graph[srcIdx].add(((ruleLabel, idx_rule), x))
                            if x not in alreadyVisitedIdx:
                                assert x in self.Graph
                                alreadyVisitedIdx.add(x)
                                tmp.add(x)
                        ruleWasPreviouslyExecuted = len(dstToIgnore)>0
                        ruleWasStraightfowardlyExpanded = len(final)>0

                    if (not ruleWasPreviouslyExecuted) and (not ruleWasStraightfowardlyExpanded):
                        ## Otherwise, recomputing it
                        hasResult, outcomes = query([src])
                        if hasResult:
                            for x in outcomes:
                                if (filter is None) or (callable(filter) and filter(x)):
                                    dstIdx, wasDstAlreadyPresent = self.constituents.add_with_wasPresent(x)
                                    wasDstAlreadyPresent = wasDstAlreadyPresent and (dstIdx in alreadyVisitedIdx)
                                    self.Graph[srcIdx].add(((ruleLabel,idx_rule), dstIdx))
                                    if not wasDstAlreadyPresent:
                                        if dstIdx not in self.Graph:
                                            self.Graph[dstIdx] = set()
                                        alreadyVisitedIdx.add(dstIdx)
                                        tmp.add(dstIdx)
            toVisit = tmp
        self._subGraph[idx].update(allVisited)
        return alreadyVisitedIdx



def non_redundant_constituents(f, strictTyping = True):
    assert HOnKSingleton.isReady()
    p = HOnKSingleton.get()
    from Formulae import is_selfstanding_variable
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
        if TBoxReasoningSingleton._instance.rules is None:
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
        result = TBoxReasoningSingleton._instance.rules.ke.pruned_expansion(formula, rules, label, S, non_redundant_constituents)
        if isImpl:
            TBoxReasoningSingleton._instance.rules.impl_already_visited_set.update(result)
        else:
            TBoxReasoningSingleton._instance.rules.eq_already_visited_set.update(result)
        return TBoxReasoningSingleton._instance.rules.ke.get_full_expansion(formula, label)

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
        self.eq_file = eq_file
        self.impl_file = impl_file
        self.ke_file = ke_file
        self.rule_mapping = CountingDictionary.load(ke_file+"rules_")
        from FunctionalMatch.language.LanguageMainPoint import parse_query
        self.impl_rules = {self.rule_mapping.add(x): x for x in parse_query(impl_file)}
        self.eq_rules = {self.rule_mapping.add(x): x for x in parse_query(eq_file)}
        # self.constituents = CountingDictionary.load(ke_file+"const_")
        self.ke = KnowledgeExpansion.load( ke_file)
        # self.ke_impl = KnowledgeExpansion.load(self.constituents, ke_file)
        self.eq_already_visited_set = set()
        self.impl_already_visited_set = set()

    def dump(self):
        self.ke.dump()

    def get_eq_rules(self):
        return self.eq_rules

    def get_impl_rules(self):
        return self.impl_rules