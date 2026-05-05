import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import pandas
from functools import reduce

from LaSSI.HOnK.TBox.ExpandConstituents import ExpandConstituents, isImplication, transformCaseWhenOneArgIsNegated
from LaSSI.structures.extended_fol.Formulae import Formula, FNot
from LaSSI.HOnK.formula_utils import latex_rendering, latex_rendering_to_raster_file, getAtoms
from FunctionalMatch.utils import CountingDictionary

from LaSSI.structures.extended_fol.TBoxReasoning import non_redundant_constituents

@dataclass
class ExplainSentence:
	formula: Formula
	atoms: dict[int, Formula]
	table: pandas.DataFrame
	holding: bool

@dataclass
class MutualTruthExplain:
    lhs: int
    rhs: int
    lhsF: Formula
    rhsF: Formula
    df: pandas.DataFrame
    logical_result: 'PairwiseCases'

    def __str__(self):
        from LaSSI.structures.extended_fol.Enums import print_case
        return print_case(self.lhs, self.logical_result, self.rhs)

@dataclass
class ExplainedUniversalTruth:
	constituent_implication: list[MutualTruthExplain]
	result: pandas.DataFrame

@dataclass
class FinalExplanation:
	confidence: float
	lhs: ExplainSentence
	rhs: ExplainSentence
	explained_joined_table: ExplainedUniversalTruth
	explained_joined_table_natural_joined_with_operands: pandas.DataFrame

def png_node(obj, key, dir, nodes_map,fillColor=None):
    import pydot
    local_file = os.path.join(dir, key + ".svg")
    local_file_exless = os.path.join(dir, key )
    if not os.path.exists(local_file_exless):
        latex_rendering_to_raster_file(obj, local_file_exless)
    d = {"image": local_file,
     "label": "",
     "width": "3cm",
     "height": "1cm",
     "shape": "box"}
    if fillColor is not None:
        d["fillcolor"] = fillColor
        d["style"] = "filled"
    nodes_map[key] = pydot.Node(key, **d)
    return nodes_map

def with_variables_from(f, l, minimal_constituents: CountingDictionary, fn, selection=False):
    from LaSSI.HOnK.formula_utils import semantic
    pdf = reduce(lambda x,y: x.merge(y, how="cross"),[pandas.DataFrame({str(x): [1,0]}) for x in l])
    L = []
    for x in pdf.to_dict(orient='records'):
        d = dict()
        for k,v in x.items():
            d[minimal_constituents.fromId(int(k))] = v
        x[fn] = semantic(f, d)
        if not selection or x[fn]>0.0:
            L.append(x)
    return pandas.DataFrame(L)

def with_variables_from(f, l, minimal_constituents: CountingDictionary, fn, selection=False):
    from LaSSI.HOnK.formula_utils import semantic
    pdf = reduce(lambda x,y: x.merge(y, how="cross"),[pandas.DataFrame({str(x): [1,0]}) for x in l])
    L = []
    for x in pdf.to_dict(orient='records'):
        d = dict()
        for k,v in x.items():
            d[minimal_constituents.fromId(int(k))] = v
        x[fn] = semantic(f, d)
        if not selection or x[fn]>0.0:
            L.append(x)
    return pandas.DataFrame(L)

def with_true_variables_from(l):
    return pandas.DataFrame({str(x): [1] for x in l})


def with_bdd_from(f, atom_to_bdd, manager):
    """BDD counterpart of `with_variables_from`: returns a BDD node representing
    the satisfying worlds of formula `f`, without materialising a truth table."""
    from LaSSI.HOnK.formula_utils import semantic_bdd
    return semantic_bdd(f, atom_to_bdd, manager)


class TabularCWASemantics:
    def __init__(self, sentence_list:List[Formula], cache_folder):
        self.sentence_list = []
        self.sentence_to_id = dict()
        for idx in range(len(sentence_list)):
            self.sentence_to_id[sentence_list[idx]] = idx
            self.sentence_list.append(sentence_list[idx])
        # self.kb = kb
        self.minimal_constituents = CountingDictionary()
        self.U = None
        self.minimal_constituent_dict = defaultdict(set)
        self.buildup = False
        self.ec = None
        self.cache_folder = cache_folder
        self.negation_resolution = dict()
        self.negations = set()

        #getSentenceAtomsFromId
        for sentence_id in range(len(self.sentence_list)):
            # collect_sentence_constituents
            from LaSSI.HOnK.formula_utils import getAtomsWithNegations
            # getSentenceAtomsFromId, for arg
            for x in getAtomsWithNegations(self.sentence_list[sentence_id]):
                current_x = self.minimal_constituents.add(x)
                self.minimal_constituent_dict[sentence_id].add(current_x)
                if isinstance(x, FNot):
                    # assert {x.arg} == getAtoms(x)
                    self.negations.add(current_x)
                    not_neg_x = self.minimal_constituents.add(x.arg)
                    self.negation_resolution[current_x] = not_neg_x

        ## --> Considering the constituent expansion without negation, so to avoid the rule deduplication within the constituent phase (i.e., nested match is not supported)
        N = len(self.minimal_constituents)
        # nonNegatedObjects = list(map(self.minimal_constituents.fromId, [x for x in range(N) if x not in self.negations]))
        nonNegatedObjects = sorted({x: self.minimal_constituents.fromId(x) for x in range(N) if x not in self.negations}.items())
        self.ec = ExpandConstituents(self.cache_folder, nonNegatedObjects)

        self._init_bdd()

    def _init_bdd(self):
        """Build the BDD manager, declare one boolean variable per non-negated
        minimal constituent, and pre-compute the per-atom BDD expression used
        by the symbolic similarity path.

        FNot atoms do *not* get their own BDD variable: instead they map to the
        symbolic NOT of the underlying atom's variable.  This satisfies the
        constraint that negation be handled with the BDD NOT operator rather
        than via the negation_resolution dictionary at evaluation time.
        """
        from dd.autoref import BDD
        self._bdd = BDD()
        self._var_name_for_id = {}
        N = len(self.minimal_constituents)
        for cid in range(N):
            if cid in self.negations:
                continue
            name = f"x{cid}"
            self._var_name_for_id[cid] = name
            self._bdd.declare(name)

        self._atom_to_bdd = {}
        self._bdd_for_id_cache = {}
        for cid in range(N):
            formula = self.minimal_constituents.fromId(cid)
            if cid in self.negations:
                underlying = self.negation_resolution[cid]
                node = self._bdd.apply('not', self._bdd.var(self._var_name_for_id[underlying]))
            else:
                node = self._bdd.var(self._var_name_for_id[cid])
            self._atom_to_bdd[formula] = node
            self._bdd_for_id_cache[cid] = node

        self._sentence_bdd_cache = {}

    def _bdd_for_id(self, cid):
        return self._bdd_for_id_cache[cid]

    def _bdd_for_sentence(self, sentence_id):
        if sentence_id in self._sentence_bdd_cache:
            return self._sentence_bdd_cache[sentence_id]
        node = with_bdd_from(self.sentence_list[sentence_id], self._atom_to_bdd, self._bdd)
        self._sentence_bdd_cache[sentence_id] = node
        return node

    def _universal_truth_bdd(self, S, T):
        """Symbolic counterpart of `_universal_truth`: returns a BDD encoding
        the conjunction of pairwise constraints between every i in S and j in
        T (i != j).

        The truth tables emitted by `_mutual_truth` map onto BDD operators as:
          Indifferent             -> TRUE  (no constraint)
          Implying(i, j)          -> ~i | j      (i => j)
          ConflictingImplication  -> i XOR j     (i and j must disagree)
          Equivalent(i, j)        -> ~(i XOR j)  (i <=> j)
        """
        from LaSSI.structures.extended_fol.Enums import PairwiseCases
        constraints = self._bdd.true
        union = S.union(T)
        if len(union) == 0:
            return constraints
        if len(union) == 1 and len(S.intersection(T)) == 1:
            return constraints
        for i in sorted(S):
            for j in sorted(T):
                if i == j:
                    continue
                test = self.determine(i, j)
                if test == PairwiseCases.Indifferent:
                    continue
                bi = self._bdd_for_id(i)
                bj = self._bdd_for_id(j)
                if test == PairwiseCases.Implying:
                    pair_c = self._bdd.apply('or', self._bdd.apply('not', bi), bj)
                elif test == PairwiseCases.ConflictingImplication:
                    pair_c = self._bdd.apply('xor', bi, bj)
                elif test == PairwiseCases.Equivalent:
                    pair_c = self._bdd.apply('not', self._bdd.apply('xor', bi, bj))
                else:
                    continue
                constraints = self._bdd.apply('and', constraints, pair_c)
        return constraints

    def getMinimalConstituentDict(self, sentence_id):
        S = set()
        for x in self.minimal_constituent_dict[sentence_id]:
            S.add(self.negation_resolution.get(x, x))
        return S

    def getIDXGraph(self):
        return self.ec.getIDXGraph()

    def getConstituentFromIDX(self, idx):
        return self.ec.getConstituentFromIDX(idx)

    def getConstituentIDX(self, obj):
        return self.ec.getConstituentIDX(obj)

    # def getImplExpansions(self, minimal_constituent_idx):
    #     return self.ec.getImplExpansions(minimal_constituent_idx)

    # def getEqExpansions(self, minimal_constituent_idx):
    #     return self.ec.getEqExpansions(minimal_constituent_idx)

    def getImplExpansionExplanation(self, constit):
        return self.ec.getImplExpansionExplanation(constit)

    def getEqExpansionExplanation(self, constituent):
        return self.ec.getEqExpansionExplanation(constituent)

    def __call__(self, i, j):
        return self.get_straightforward_id_similarity(self.sentence_to_id[i], self.sentence_to_id[j])

    def determine(self, i, j):
        from LaSSI.HOnK.HOnK import CasusHappening
        from LaSSI.structures.extended_fol.Enums import PairwiseCases
        x = self.minimal_constituents.fromId(i)
        y = self.minimal_constituents.fromId(j)
        if (isinstance(x, FNot) and isinstance(y, FNot)):
            val = self.ec.determine_raw(self.negation_resolution.get(i, i), self.negation_resolution.get(j, j))
            if isImplication(val):
                if isImplication(self.ec.determine_raw(self.negation_resolution.get(j, j), self.negation_resolution.get(i, i))):
                    val = CasusHappening.EQUIVALENT
                else:
                    val = CasusHappening.INDIFFERENT
            elif val == CasusHappening.EQUIVALENT:
                val = CasusHappening.EQUIVALENT
            return ExpandConstituents.rectify_implication(val)
        elif (x == FNot(y)) or (y == FNot(x)):
            return ExpandConstituents.rectify_implication(CasusHappening.EXCLUSIVES)
        elif isinstance(x, FNot):
            i_new = self.negation_resolution.get(i, i)
            val = self.ec.determine_raw(i_new, j, False, True)
            if isImplication(val):
                if isImplication(self.ec.determine_raw(j, i_new, isRightDrop=True)):
                    return PairwiseCases.ConflictingImplication
            elif val == CasusHappening.EQUIVALENT:
                return PairwiseCases.ConflictingImplication
            val = transformCaseWhenOneArgIsNegated(val)
            return ExpandConstituents.rectify_implication(val)
        elif isinstance(y, FNot):
            j_new = self.negation_resolution.get(j, j)
            val = self.ec.determine_raw(i, j_new, False, isRightDrop=True)
            if isImplication(val):
                if isImplication(self.ec.determine_raw(j_new, i, True)):
                    return PairwiseCases.ConflictingImplication
            elif val == CasusHappening.EQUIVALENT:
                return PairwiseCases.ConflictingImplication
            val = transformCaseWhenOneArgIsNegated(val)
            return ExpandConstituents.rectify_implication(val)
        else:
            return self.ec.determine(i,j)

    def explained_mutual_truth(self, i:int, j:int)->MutualTruthExplain:
        test = self.determine(i,j)#self.ec.determine(i, j) #self.get_mutual_truth(i, j)
        # relation = Relation()
        # relation.add_attributes([str(i), str(j)])
        from LaSSI.structures.extended_fol.Enums import PairwiseCases
        dataf = None
        # df = {"lhs": i,
        #       "rhs": j,
        #       "lhsF": self.minimal_constituents.fromId(i),
        #       "rhsF": self.minimal_constituents.fromId(j)}
        if (test == PairwiseCases.Indifferent):
            dataf= pandas.DataFrame({str(i): [0,0,1,1],
                     str(j): [0,1,0,1]})
        elif (test == PairwiseCases.Implying):
            dataf=  pandas.DataFrame({str(i): [0, 0, 1],
                            str(j): [0, 1, 1]  })
        elif (test == PairwiseCases.ConflictingImplication):
            dataf=  pandas.DataFrame({str(i): [0, 1],
                           str(j): [1,0]})
        elif (test == PairwiseCases.Equivalent):
            dataf=  pandas.DataFrame({str(i): [0, 1],
                           str(j): [0,1]})
        # df["logical_rsult"] = test
        # df["df"] = dataf
        return MutualTruthExplain(i, j,
                                  self.minimal_constituents.fromId(i),
                                  self.minimal_constituents.fromId(j),
                                  dataf,
                                  test)

    def _mutual_truth(self, i, j):
        test = self.determine(i,j)#self.ec.determine(i, j) #self.get_mutual_truth(i, j)
        # relation = Relation()
        # relation.add_attributes([str(i), str(j)])
        from LaSSI.structures.extended_fol.Enums import PairwiseCases
        if (test == PairwiseCases.Indifferent):
            return pandas.DataFrame({str(i): [0,0,1,1],
                     str(j): [0,1,0,1]})
        elif (test == PairwiseCases.Implying):
            return pandas.DataFrame({str(i): [0, 0, 1],
                            str(j): [0, 1, 1]  })
        elif (test == PairwiseCases.ConflictingImplication):
            return pandas.DataFrame({str(i): [0, 1],
                           str(j): [1,0]})
        elif (test == PairwiseCases.Equivalent):
            return pandas.DataFrame({str(i): [0, 1],
                           str(j): [0,1]})

    def explained_universal_truth(self, S:set[int], T:set[int])->ExplainedUniversalTruth:
        constituent_implication = []
        L = list()
        N = len(S.union(T))
        if N == 0:
            # relation = Relation(name="R")
            return ExplainedUniversalTruth(constituent_implication, pandas.DataFrame({}))
        if N == len(S.intersection(T)) and N == 1:
            return ExplainedUniversalTruth(constituent_implication, pandas.DataFrame({str(list(S)[0]):[0,1]}))
        else:
            for i in sorted(list(S)):
                for j in sorted(list(T)):
                    if i != j:
                        res = self.explained_mutual_truth(i,j)
                        constituent_implication.append(res)
                        L.append(res.df)
            return ExplainedUniversalTruth(constituent_implication, reduce(lambda x, y: x.merge(y), L))

    def _universal_truth(self, S, T):
        L = list()
        N = len(S.union(T))
        if N == 0:
            # relation = Relation(name="R")
            return pandas.DataFrame({})
        if N == len(S.intersection(T)) and N == 1:
            return pandas.DataFrame({str(list(S)[0]):[0,1]})
        else:
            for i in sorted(list(S)):
                for j in sorted(list(T)):
                    if i != j:
                        L.append(self._mutual_truth(i, j))
            return reduce(lambda x, y: x.merge(y), L)

    def explain_sentence(self, i:int, worldsWhereItAlwaysHolds:bool)->ExplainSentence:
        Ri = with_variables_from(self.sentence_list[i], self.minimal_constituent_dict[i], self.minimal_constituents,
                                 "R" + str(i), worldsWhereItAlwaysHolds)
        # return {"formula": self.sentence_list[i],
        #  "atoms": {x: self.minimal_constituents.fromId(x) for x in self.minimal_constituent_dict[i]},
        #  "table": Ri,
        #  "holding": worldsWhereItAlwaysHolds}
        return ExplainSentence(self.sentence_list[i],
                               {x: self.minimal_constituents.fromId(x) for x in self.minimal_constituent_dict[i]},
                               Ri,
                               worldsWhereItAlwaysHolds)

    def get_explained_id_similarity(self, i:int, j:int):
        Ri_explanation = self.explain_sentence(i, True)
        Rj_explanation = self.explain_sentence(j, False)
        ConstImplExpl = self.explained_universal_truth(set(self.minimal_constituent_dict[i]), set(self.minimal_constituent_dict[j]))
        relevantColumns = list(set(Ri_explanation.table.columns).union(set(Rj_explanation.table.columns)))
        tableSemantics = ConstImplExpl.result.merge(Ri_explanation.table).merge(Rj_explanation.table)[relevantColumns].drop_duplicates()
        semantics = tableSemantics[["R" + str(j)]].prod(axis=1)
        Rj_holding = len(semantics)
        total = semantics.sum(axis=0)/Rj_holding if Rj_holding>0.0 else 0.0
        # print(f"{i}~{j} := {total}")
        return FinalExplanation(total, Ri_explanation, Rj_explanation, ConstImplExpl, tableSemantics)
        # return {"confidence": total,
        #         "lhs": Ri_explanation,
        #         "rhs": Rj_explanation,
        #         "explained_joined_table": ConstImplExpl,
        #         "explained_joined_table_natural_joined_with_operands": tableSemantics}

    def get_straightforward_id_similarity(self, i:int, j:int):
        """Symbolic similarity between sentences i and j.

        Mirrors the original DataFrame-based ratio
            P(Sj=1 | Si=1, constraints)
        but evaluates it via BDD model counting, so the truth table is never
        materialised.  With n underlying boolean variables in the manager, the
        ratio is invariant to phantom variables (each contributes a factor 2 to
        both numerator and denominator), so we count over the full declared
        variable set.
        """
        Si_bdd = self._bdd_for_sentence(i)
        Sj_bdd = self._bdd_for_sentence(j)
        constraints = self._universal_truth_bdd(set(self.minimal_constituent_dict[i]),
                                                set(self.minimal_constituent_dict[j]))
        context = self._bdd.apply('and', Si_bdd, constraints)
        combined = self._bdd.apply('and', context, Sj_bdd)
        n_vars = len(self._var_name_for_id)
        if n_vars == 0:
            return 0.0
        ctx_count = self._bdd.count(context, nvars=n_vars)
        
        # DEBUG
        if ctx_count == 0:
            print(f"DEBUG: ctx_count is 0 for i={i}, j={j}")
            print(f"DEBUG: Si_bdd count = {self._bdd.count(Si_bdd, nvars=n_vars)}")
            print(f"DEBUG: Sj_bdd count = {self._bdd.count(Sj_bdd, nvars=n_vars)}")
            print(f"DEBUG: constraints count = {self._bdd.count(constraints, nvars=n_vars)}")
            
        if ctx_count == 0:
            return 0.0
        val = self._bdd.count(combined, nvars=n_vars) / ctx_count
        print(f"DEBUG: Sim({i}, {j}) = {val}")
        return val

    def get_implication(self, i, j):
        from LaSSI.structures.extended_fol.Enums import PairwiseCases
        val = self.get_straightforward_id_similarity(i, j)
        if val == 1.0:
            return PairwiseCases.Implying
        elif val == 0.0:
            return PairwiseCases.ConflictingImplication
        else:
            return PairwiseCases.Indifferent





    def buildReport(self, file, mathJax = True):
        from bs4 import Tag, BeautifulSoup
        import pydot
        from LaSSI.HOnK.formula_utils import latex_formula_rendering

        from pathlib import Path
        Path(file+"_dir").mkdir(parents=True, exist_ok=True)
        graph = pydot.Dot("my_graph", graph_type="digraph", rankdir="LR")
        nodes_map = dict()
        html = Tag(name="html")
        if mathJax:
            mathjax = """
            <script type="text/javascript" id="MathJax-script" async
      src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.0.0/es5/latest?tex-mml-chtml.js">
    </script>
    <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']]
      }
    };
    </script>
    <script id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
    </script><script src="//d3js.org/d3.v7.min.js"></script>
<script src="https://unpkg.com/@hpcc-js/wasm@2.20.0/dist/graphviz.umd.js"></script>
<script src="https://unpkg.com/d3-graphviz@5.6.0/build/d3-graphviz.js"></script>"""
            parser = BeautifulSoup(mathjax)
            for x in list(parser.children):
                html.append(x)
        body = Tag(name="body")

        p_ = Tag(name="h1")
        p_.append("Constutents DB")
        body.append(p_)
        ol = Tag(name="ol")
        # initial_constituents = self.minimal_constituents.getAllObjects()
        # constituent_id = CountingDictionary()

        for idx_orig, x in self.minimal_constituents.reverseConstituent.items():
            li = Tag(name="li")
            # idx = self.getConstituentIDX(x)
            li["id"] = f"constituent{idx_orig}"
            li.append(latex_formula_rendering(x, mathJax))

            pp = Tag(name="p")
            pp.append("Eq Rewriting:")
            li.append(pp)
            ool = Tag(name="ol")
            # eqex = self.getEqExpansions(idx)
            # meqex = {x:idx2+1 for idx2, x in enumerate(eqex)}
            fmeqex = self.getEqExpansionExplanation(x)
            for idx_ in fmeqex:
                lli = Tag(name="li")
                lli["value"] = idx_
                x = self.ec.getIthExpandedConstituent(idx_)
                lli.append(latex_formula_rendering(x, mathJax))
                uuul = Tag(name="ul")
                for (label, rule),dst in fmeqex[idx_]:
                    assert label == "eqR"
                    llli = Tag(name="li")
                    llli.append(str(idx_)+latex_rendering(f"\\xrightarrow{{eq {rule}}}")+str(dst))
                    uuul.append(llli)
                lli.append(uuul)
                ool.append(lli)
            li.append(ool)

            pp = Tag(name="p")
            pp.append("Impl Rewriting:")
            li.append(pp)
            # imex = self.getImplExpansions(idx)
            # mimex = {x:idx2+1 for idx2, x in enumerate(imex)}
            fmimex = self.getImplExpansionExplanation(idx_orig)
            ool = Tag(name="ol")
            for idx_ in fmimex:
                lli = Tag(name="li")
                lli["value"] = idx_
                x = self.ec.getIthExpandedConstituent(idx_)
                lli.append(latex_formula_rendering(x, mathJax))
                uuul = Tag(name="ul")
                for (label, rule), dst in fmimex[idx_]:
                    assert label == "implR"
                    llli = Tag(name="li")
                    llli.append(str(idx_)+latex_rendering(f"\\xrightarrow{{impl {rule}}}")+str(dst))
                    uuul.append(llli)
                lli.append(uuul)
                ool.append(lli)
            li.append(ool)

            ol.append(li)
        body.append(ol)
        from LaSSI.external_services.Services import Services
        Services.getInstance().log("Finished to write the rules")

        p_ = Tag(name="h1")
        p_.append("Sentences DB")
        body.append(p_)
        uul = Tag(name="ul")
        body.append(uul)

        for i, sentence in enumerate(self.sentence_list):
            minimal_constituents = self.minimal_constituent_dict[i]
            ref = f"Sentence{i}"
            Sentence = f"Sentence #{i}"
            Services.getInstance().log(ref)
            nodes_map[ref] = pydot.Node(ref, shape="circle",fillcolor="lightyellow",style="filled")
            graph.add_node(nodes_map[ref])


            lli = Tag(name="li")
            lla = Tag(name="a")
            lla["href"] = f"#{ref}"
            lla.append(Sentence)
            lli.append(lla)
            uul.append(lli)

            h1 = Tag(name="h2")
            a = Tag(name="a")
            h1["id"] = ref
            a.append(Sentence)
            h1.append(a)
            body.append(h1)

            p1 = Tag(name="p")
            p1.append("Logic form: ")
            p1.append(latex_formula_rendering(sentence, mathJax))
            body.append(p1)
            nodes_map = png_node(sentence, ref+"eq", file+"_dir", nodes_map)
            graph.add_node(nodes_map[ref+"eq"])
            graph.add_edge(pydot.Edge(ref, ref+"eq", label="hasFormula"))

            p2 = Tag(name="p")
            p2.append("Minimal Constituents:")
            body.append(p2)
            ol = Tag(name="ul")
            for x_idx in minimal_constituents:
                obj = self.minimal_constituents.reverseConstituent[x_idx]
                global_x_idx = self.getConstituentIDX(obj)
                li = Tag(name="li")
                li["value"] = global_x_idx
                ali = Tag(name="a")
                ali["href"] = f"#constituent{global_x_idx}"
                ali.append(str(global_x_idx))
                ali.append(latex_formula_rendering(obj, mathJax))
                nodes_map = png_node(self.minimal_constituents.reverseConstituent[x_idx], f"constituent{global_x_idx}", file + "_dir", nodes_map, fillColor="lightblue")
                graph.add_node(nodes_map[f"constituent{global_x_idx}"])
                graph.add_edge(pydot.Edge(ref, f"constituent{global_x_idx}", label="hasConstituent"))
                li.append(ali)
                ol.append(li)
            body.append(ol)
        html.append(body)
        html.decode()

        Services.getInstance().log("Printing html...")
        with open(file+".html", "w") as f:
            f.write(html.prettify())
        Services.getInstance().log("... done")

        graphIDX = self.getIDXGraph()
        for idx, ls in graphIDX.items():
            Services.getInstance().log(f"Node {idx}")
            label = f"constituent{idx}"
            if label not in nodes_map:
                src_obj = self.getConstituentFromIDX(idx)
                nodes_map = png_node(src_obj, label, file + "_dir", nodes_map)
                graph.add_node(nodes_map[label])
            # else:
            #     src_obj = nodes_map[label]
            for (ruleId, dstIdx) in ls:
                labelDst = f"constituent{dstIdx}"
                if labelDst not in nodes_map:
                    dst_obj = self.getConstituentFromIDX(dstIdx)
                    nodes_map = png_node(dst_obj, labelDst, file + "_dir", nodes_map)
                    graph.add_node(nodes_map[labelDst])
                # else:
                #     dst_obj = nodes_map[labelDst]
                graph.add_edge(pydot.Edge(label, labelDst, label=str(ruleId)))
        Services.getInstance().log("Graph finalised")

        with open(file+".dot", "w") as f:
            # As a string:
            output_raw_dot = graph.to_string()
            # tex_graph = convert_graph(output_raw_dot)
            f.write(output_raw_dot)
        Services.getInstance().log("Dot written")

        graph.write(file + ".pdf", format="pdf")
        Services.getInstance().log("PDF written")
        # with open(file+".dot", "w") as f:
        #     # As a string:
        #     output_raw_dot = graph.to_string()
        #     from FunctionalMatch.example.utils.doc2tex import convert_graph
        #     # tex_graph = convert_graph(output_raw_dot)
        #     f.write(output_raw_dot)
        #     # Or, save it as a DOT-file:
        #     # import pdflatex
        #     # bin_graph = str.encode(tex_graph)
        #     # pdfl = pdflatex.PDFLaTeX.from_binarystring(bin_graph, file)
        #     # pdfl.params["-output-directory"] = os.getcwd()
        #     # pdfl.create_pdf(keep_pdf_file=True)
        #     # fp = subprocess.run(["pdflatex", file+".tex"])


