import os.path
import pickle
from LaSSI.HOnK.HOnK import CasusHappening, HOnKSingleton
from LaSSI.structures.extended_fol.Enums import PairwiseCases
from LaSSI.structures.extended_fol.Formulae import FBinaryPredicate, FUnaryPredicate
from LaSSI.structures.extended_fol.ModelSearch import ModelSearch, ModelSearchBasis

# Re-export comparison utilities and logic for backward compatibility
from LaSSI.HOnK.TBox.ComparatorUtils import isImplication, transformCaseWhenOneArgIsNegated, isExistential
from LaSSI.HOnK.TBox.ConstituentComparator import (
    compare_variable,
    simplifyConstituentsAcross,
    simplifyConstituents,
    test_pairwise_sentence_similarity,
    is_direct_subset
)
from LaSSI.HOnK.TBox.Canonicaliser import canonicalize_atom_for_paraphrase_expansion

def instantiate_rules(constituents, expansion_dictionary, final_constituents, isImpl):
    ls = list(reversed(constituents))
    result_list = list()
    from tqdm import tqdm
    
    label = "implication" if isImpl else "equivalence"
    pbar = tqdm(ls, desc=f"Expanding {label} constituents", leave=False)
    
    for original, (idx, constituent) in enumerate(pbar):
        str1 = str(constituent)
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        entrypoint, adj_graph, id_to_constituent, s = TBoxReasoningSingleton.explained_knowledge_expand(constituent, isImpl)
        str2 = str(ls[original][1])
        if (str1 != str2):
            raise RuntimeError(str1+"!="+str2)
        expansion_dictionary[ls[original][1]] = s
        result = {"original": original,
                  "idx": idx,
                  "constituents": constituent,
                  "entrypoint": entrypoint,
                  "adj_graph": adj_graph,
                  "id_to_constituent": id_to_constituent}
        result_list.append(result)
    for y in expansion_dictionary.values():
        final_constituents.update(y)
    return result_list


class ExpandConstituents:
    def __init__(self, cache_folder, constituents):
        """
        This class provides the expansion for each of the sentences, as well as caching the direction of the implication for each of the formulae
        """
        print("Setting up the rule expander...")
        from LaSSI.external_services.Services import Services

        self.constituents = constituents
        _ied = os.path.join(cache_folder, "_ied.pickle")
        _ic = os.path.join(cache_folder, "_ic.pickle")
        _eed = os.path.join(cache_folder, "_eed.pickle")
        _ec = os.path.join(cache_folder, "_ec.pickle")
        explain_eq = os.path.join(cache_folder, "explain_eq.json")
        explain_impl = os.path.join(cache_folder, "explain_impl.json")

        if (os.path.exists(explain_impl) and os.path.exists(explain_eq) and os.path.exists(_ied) and os.path.exists(_ic) and os.path.exists(_eed) and os.path.exists(_ec)):
            with open(_ied, "rb") as f:
                self.impl_expansion_dictionary = pickle.load(f)
            with open(_ic, "rb") as f:
                self.impl_constituents = pickle.load(f)
            with open(_eed, "rb") as f:
                self.eq_expansion_dictionary = pickle.load(f)
            with open(_ec, "rb") as f:
                self.eq_constituents = pickle.load(f)

        else:
            self.impl_expansion_dictionary = dict()
            self.impl_constituents = set()
            self.eq_expansion_dictionary = dict()
            self.eq_constituents = set()

            Services.getInstance().log("Expanding the constituents...")
            self.eq_explained = instantiate_rules(self.constituents, self.eq_expansion_dictionary, self.eq_constituents,
                              False)
            from LaSSI.files.JSONDump import json_dumps

            with open(explain_eq, "w") as f:
                f.write(json_dumps(self.eq_explained))
            self.impl_explained = instantiate_rules(self.constituents, self.impl_expansion_dictionary, self.impl_constituents,
                              True)
            with open(explain_impl, "w") as f:
                f.write(json_dumps(self.impl_explained))

            with open(_ied, "wb") as f:
                pickle.dump(self.impl_expansion_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(_ic, "wb") as f:
                pickle.dump(self.impl_constituents, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(_eed, "wb") as f:
                pickle.dump(self.eq_expansion_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(_ec, "wb") as f:
                pickle.dump(self.eq_constituents, f, protocol=pickle.HIGHEST_PROTOCOL)

        from LaSSI.explainer.FullExplainer import load_expansion_graph_from_json_file
        self.impl_explained = load_expansion_graph_from_json_file(explain_impl)
        self.eq_explained = load_expansion_graph_from_json_file(explain_eq)
        self.result_cache = dict()
        self.result_cache_raw = dict()
        self.ms = ModelSearch()
        self.lhsOrigDict = dict()
        self.rhsOrigDict = dict()
        self.inv_idx = dict()
        Services.getInstance().log("Splitting across unary and binary constituents for each sentence...")
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        n = len(self.constituents)

        def _build_basis(item):
            row_i, row_sentence = item
            canonical = canonicalize_atom_for_paraphrase_expansion(row_sentence)
            t0 = time.time()
            impl_consts = list(self.impl_expansion_dictionary[row_sentence])
            if canonical != row_sentence:
                impl_consts.append(canonical)
            lhs = ModelSearchBasis(row_sentence, impl_consts)
            t_lhs = time.time() - t0
            t1 = time.time()
            eq_consts = list(self.eq_expansion_dictionary[row_sentence])
            if canonical != row_sentence:
                eq_consts.append(canonical)
            rhs = ModelSearchBasis(row_sentence, eq_consts)
            t_rhs = time.time() - t1
            return row_i, row_sentence, lhs, rhs, t_lhs, t_rhs

        max_workers = min(n, os.cpu_count() or 4, 8)
        slowest = (-1.0, None)  # (elapsed, row_i)
        total_elapsed = 0.0
        completed = 0
        with tqdm(total=n, desc="Splitting constituents", unit="sent") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_build_basis, item): item for item in self.constituents}
                for future in as_completed(futures):
                    row_i, row_sentence, lhs, rhs, t_lhs, t_rhs = future.result()
                    self.inv_idx[row_sentence] = row_i
                    self.lhsOrigDict[row_i] = lhs
                    self.rhsOrigDict[row_i] = rhs
                    elapsed = t_lhs + t_rhs
                    total_elapsed += elapsed
                    completed += 1
                    if elapsed > slowest[0]:
                        slowest = (elapsed, row_i)
                    pbar.set_postfix({
                        "last_row": row_i,
                        "last": f"{elapsed:.2f}s",
                        "lhs/rhs": f"{t_lhs:.2f}/{t_rhs:.2f}",
                        "avg": f"{total_elapsed / completed:.2f}s",
                        "slowest": f"row{slowest[1]}@{slowest[0]:.1f}s",
                    })
                    pbar.update(1)
        Services.getInstance().log(
            f"  Splitting constituents done: {completed} sentences in {total_elapsed:.1f}s "
            f"(avg {total_elapsed / max(completed, 1):.2f}s/sent, slowest row {slowest[1]} @ {slowest[0]:.1f}s)"
        )
        self.constituents = dict(self.constituents)

    def getImplExpansions(self, idx):
        return self.lhsOrigDict[idx].all() if idx in self.lhsOrigDict else []

    def getEqExpansions(self, idx):
        return self.rhsOrigDict[idx].all() if idx in self.lhsOrigDict else []

    def getConstituentIDX(self, ith):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getConstituentIdx(ith)

    def getIthExpandedConstituent(self, ith):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getConstituentFromIdx(ith)

    def getImplExpansionExplanation(self, constituent):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.subGraphImpl(constituent)

    def getEqExpansionExplanation(self, constituent):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.subGraphEq(constituent)

    def determine_raw(self, i: int, j: int, forceEquiv:bool=False, isLeftDrop = False, isRightDrop = False):
        assert i in self.constituents
        assert j in self.constituents
        lhsOrig = self.lhsOrigDict[i]
        rhsOrig = self.lhsOrigDict[j] if forceEquiv else self.rhsOrigDict[j]
        tmp = self.ms.compare(lhsOrig, rhsOrig, isLeftDrop, isRightDrop)
        self.result_cache_raw[(i, j)] = tmp
        return tmp

    @staticmethod
    def rectify_implication(tmp):
        if tmp == CasusHappening.EXCLUSIVES:
            return PairwiseCases.ConflictingImplication
        elif tmp == CasusHappening.EQUIVALENT:
            return PairwiseCases.Equivalent
        elif isImplication(tmp):
            return PairwiseCases.Implying
        else:
            return PairwiseCases.Indifferent

    def determine(self, i: int, j: int):
        if (i == j):
            self.result_cache[(i, j)] = PairwiseCases.Equivalent
        assert i in self.constituents
        assert j in self.constituents
        val = PairwiseCases.Indifferent

        lhsOrig = self.lhsOrigDict[i]
        rhsOrig = self.rhsOrigDict[j]
        tmp = self.ms.compare(lhsOrig, rhsOrig)
        if tmp == CasusHappening.EXCLUSIVES:
            val = PairwiseCases.ConflictingImplication
        elif tmp == CasusHappening.EQUIVALENT:
            val = PairwiseCases.Equivalent
        elif isImplication(tmp):
            val = PairwiseCases.Implying
        else:
            val = PairwiseCases.Indifferent

        self.result_cache[(i, j)] = val
        return val

    def getIDXGraph(self):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getIDXGraph()

    def getConstituentFromIDX(self, idx):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getConstituentFromIdx(idx)
