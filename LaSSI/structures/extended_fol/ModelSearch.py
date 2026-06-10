# from LaSSI.HOnK.TBox.ExpandConstituents import CasusHappening, test_pairwise_sentence_similarity, isImplication
# from logical_repr.Sentences import FUnaryPredicate, FBinaryPredicate, FNot
# from logical_repr.rewrite_kernels import make_not
from pydatagramdb import result

from LaSSI.structures.extended_fol.Formulae import *
from LaSSI.HOnK.HOnK import CasusHappening

_TRACE_COMPARE = False  # toggle for debug tracing
_TRACE_EXCLUSIVES_ONLY = True  # log only paths that return EXCLUSIVES


# Property keys whose `attachTo == "Kernel"` in raw_data/logical_analysis.json
# (the logical-context types). When one predicate carries one of these and the
# other doesn't — or they carry different ones — the predicates describe
# semantically distinct events and an expansion-derived implication must not
# silently override that distinction (e.g. REQUIREMENT vs CAUSATION).
# DERIVED from logical_analysis.json (`attachTo: Kernel` minus
# `similarity_semantics.kernel_context_excluded`, expanded with
# `key_spelling_aliases`) — a new Kernel-attached construct joins the guard
# automatically; SPACE/TIME exclusion is declared in the JSON, never here.
from LaSSI.utils.logical_analysis_reader import (
    kernel_context_keys,
    kernel_context_monotonicity,
    paraphrastic_slots,
)

_KERNEL_LOGICAL_CONTEXT_KEYS = kernel_context_keys()

# Partition of the kernel-context keys by declared monotonicity (see
# logical_analysis.json `_doc_monotonicity`): restrictive keys block an
# implication when RHS asserts them and LHS doesn't; intensional keys
# (MODALITY) block in the opposite direction — a modal LHS cannot imply a
# factual RHS ("expected to end X" does not entail "ends X"), while a factual
# LHS may imply a modal RHS (upward monotone under possibility).
_RESTRICTIVE_CONTEXT_KEYS, _INTENSIONAL_CONTEXT_KEYS = kernel_context_monotonicity()


# Paraphrastic kernel-key pairs, declared in logical_analysis.json
# (`similarity_semantics.paraphrastic_slots`). Each entry maps one surface key
# to a canonical form so that asymmetric appearances of the two keys across a
# pair of predicates are treated as the SAME logical slot when the guard below
# decides whether RHS asserts something LHS lacks. The value on each side is
# still compared point-by-point via Paraphrase.ttl, so a mapping here doesn't
# blanket-equate the values, only the SLOTS.
_KERNEL_KEY_EQUIVALENCES = paraphrastic_slots()


def _canonical_kernel_key(k: str) -> str:
    return _KERNEL_KEY_EQUIVALENCES.get(k, k)


def _kernel_logical_keys_block_implication(lhs_formula, rhs_formula) -> bool:
    """Return True iff the verdict ``LHS ⇒ RHS`` must be rejected because the
    two predicates' kernel-level logical-context properties (CAUSATION,
    REQUIREMENT, MODALITY, ...) make the implication unsound.

    Direction depends on each key's declared monotonicity
    (logical_analysis.json `_doc_monotonicity`):

    - *restrictive* keys (the default — intersective event modifiers): a
      more-specific LHS is allowed to imply a less-specific RHS (LHS may
      carry extra keys); the implication is invalid only when RHS introduces
      a restrictive key LHS lacks — RHS claiming something LHS did not
      commit to.
    - *intensional* keys (MODALITY — non-veridical operators): the modal
      does NOT entail the factual, so the implication is invalid when LHS
      carries the key and RHS does not ("expected to end X" ⇏ "ends X");
      the factual⇒modal direction stays valid (upward monotone under
      possibility).

    Keys listed in ``_KERNEL_KEY_EQUIVALENCES`` are canonicalised before
    the comparison, so e.g. an LHS carrying ``TEMPORAL_CONTEXT`` is
    treated as already covering an RHS-only ``AIM_OBJECTIVE``.
    """
    lhs_props = getattr(lhs_formula, 'properties', None)
    rhs_props = getattr(rhs_formula, 'properties', None)
    lhs_keys = {str(k).upper() for k, v in (lhs_props or ()) if v}
    rhs_keys = {str(k).upper() for k, v in (rhs_props or ()) if v}
    lhs_restrictive = {_canonical_kernel_key(k) for k in lhs_keys & _RESTRICTIVE_CONTEXT_KEYS}
    rhs_restrictive = {_canonical_kernel_key(k) for k in rhs_keys & _RESTRICTIVE_CONTEXT_KEYS}
    if rhs_restrictive - lhs_restrictive:
        return True
    lhs_intensional = {_canonical_kernel_key(k) for k in lhs_keys & _INTENSIONAL_CONTEXT_KEYS}
    rhs_intensional = {_canonical_kernel_key(k) for k in rhs_keys & _INTENSIONAL_CONTEXT_KEYS}
    return bool(lhs_intensional - rhs_intensional)


class ModelSearchBasis:
    def __init__(self, original, constituents):
        self.original = original
        self.unary = []
        self.binary = []
        if isinstance(original, FUnaryPredicate):
            self.unary.insert(0, original)
        elif isinstance(original, FBinaryPredicate):
            self.binary.insert(0, original)
        else:
            raise Exception("Unexpected expression: " + str(original))
        from LaSSI.structures.extended_fol.TBoxReasoning import non_redundant_constituents
        for constituent in constituents:
            if isinstance(constituent, FUnaryPredicate) or (isinstance(constituent, FNot) and isinstance(constituent.arg, FUnaryPredicate)):
                if not (constituent == original) and non_redundant_constituents(constituent, False):
                    self.unary.append(constituent)
            elif isinstance(constituent, FBinaryPredicate) or (isinstance(constituent, FNot) and isinstance(constituent.arg, FBinaryPredicate)):
                if not (constituent == original) and non_redundant_constituents(constituent, False):
                    self.binary.append(constituent)
            else:
                raise Exception("Unexpected expression: "+str(constituent))

    def all(self):
        return self.unary + self.binary



class ModelSearch:
    def __init__(self):
        self.pairwise_similarity_cache = dict()
        # self.kb = kb
        self.main_cache = dict()

    @staticmethod
    def _same_relation(lhs, rhs):
        if not (isinstance(lhs, (FUnaryPredicate, FBinaryPredicate)) and
                isinstance(rhs, (FUnaryPredicate, FBinaryPredicate))):
            return False
        if lhs.rel == rhs.rel:
            return True
        # Synonymous relations (e.g. 'carry out' / 'do') should also count
        # so the contextual guard fires and re-checks the originals' properties.
        from LaSSI.HOnK.HOnK import HOnKSingleton
        from LaSSI.HOnK.TBox.ExpandConstituents import isImplication
        if HOnKSingleton.isReady():
            rel_cmp = HOnKSingleton.get().name_eq(lhs.rel, rhs.rel)
            if rel_cmp == CasusHappening.EQUIVALENT or isImplication(rel_cmp):
                return True
        return False

    @staticmethod
    def _binary_implies_unary_drop(lhs, rhs):
        """True iff ``lhs`` is a binary predicate ``rel(a, b)`` and ``rhs`` is the
        unary ``rel(a)`` over the same relation and subject — i.e. ``rhs`` is the
        object-dropped (existential-generalisation) form of ``lhs``.

        Sound and directional: ``rel(a, b) ⇒ ∃y. rel(a, y)`` (modelled here as the
        unary ``rel(a)``). The reverse (unary ⇒ binary) does NOT hold, so this only
        fires for the binary-LHS / unary-RHS direction. Relation match goes through
        `_same_relation` (string-equal or ontology synonym)."""
        if not (isinstance(lhs, FBinaryPredicate) and isinstance(rhs, FUnaryPredicate)):
            return False
        if not ModelSearch._same_relation(lhs, rhs):
            return False
        a = getattr(lhs, 'src', None)
        u = getattr(rhs, 'arg', None)
        if a is None or u is None:
            return False
        if a == u:
            return True
        an = getattr(a, 'name', None)
        un = getattr(u, 'name', None)
        return an is not None and an == un

    @staticmethod
    def _has_properties(formula):
        return (
                isinstance(formula, (FUnaryPredicate, FBinaryPredicate)) and
                formula.properties is not None and len(formula.properties) > 0
        )

    def _guard_contextual_implication(self, objLHS, objRHS, verdict):
        from LaSSI.HOnK.TBox.ExpandConstituents import (
            test_pairwise_sentence_similarity, isImplication,
            compare_variable, simplifyConstituentsAcross, simplifyConstituents,
        )
        if not isImplication(verdict):
            return verdict
        # When either original carries properties (CAUSATION, SPACE, etc.),
        # re-check whether the originals' properties are compatible.
        # Expansion-derived implications strip optional properties and can
        # produce false positives when the originals disagree on those
        # properties (e.g. different causes).
        if not (self._has_properties(objLHS.original) or self._has_properties(objRHS.original)):
            if _TRACE_COMPARE:
                print(f"[GUARD] no properties on either original, returning verdict={verdict}")
            return verdict
        # First try a full direct comparison of the originals.  This covers
        # the common case where the relation names are string-equal or the
        # ontology recognises them as synonyms.
        if _TRACE_COMPARE:
            lhs_props = getattr(objLHS.original, 'properties', None)
            rhs_props = getattr(objRHS.original, 'properties', None)
            print(f"[GUARD] LHS original: rel={getattr(objLHS.original,'rel','?')}, dst={getattr(objLHS.original,'dst','?')}, props={lhs_props}")
            print(f"[GUARD] RHS original: rel={getattr(objRHS.original,'rel','?')}, dst={getattr(objRHS.original,'dst','?')}, props={rhs_props}")
            print(f"[GUARD] LHS str: {objLHS.original}")
            print(f"[GUARD] RHS str: {objRHS.original}")
        direct = test_pairwise_sentence_similarity(
            self.pairwise_similarity_cache,
            objLHS.original,
            objRHS.original,
            store=False,
            shift=False,
        )
        if _TRACE_COMPARE:
            print(f"[GUARD] direct comparison: {direct}")
            # Check if it was in cache
            cache_key = (objLHS.original, objRHS.original)
            if cache_key in self.pairwise_similarity_cache:
                print(f"[GUARD] NOTE: result was in pairwise_similarity_cache!")
        if direct == CasusHappening.INDIFFERENT:
            return CasusHappening.INDIFFERENT
        if direct == CasusHappening.EQUIVALENT:
            return CasusHappening.EQUIVALENT
        if direct == CasusHappening.GENERAL_IMPLICATION:
            # Mutual implication witnesses logical equivalence (A ⇒ B and
            # B ⇒ A ⊢ A ≡ B) — but only when both directions are *general*
            # implications. The weak implication kinds do not witness the
            # reverse entailment: LOSE_SPEC/INSTANTIATION mean one side is
            # strictly more specific (so the sides differ in content), and
            # MISSING_1ST means an argument is absent on one side, not that
            # the sides entail each other. Promoting on those collapsed
            # genuinely-asymmetric pairs (e.g. forward GENERAL + reverse
            # MISSING_1ST) into EQUIVALENT.
            reverse_direct = test_pairwise_sentence_similarity(
                self.pairwise_similarity_cache,
                objRHS.original,
                objLHS.original,
                store=False,
                shift=False,
            )
            if reverse_direct in (CasusHappening.EQUIVALENT, CasusHappening.GENERAL_IMPLICATION):
                return CasusHappening.EQUIVALENT
        # If the full comparison didn't return INDIFFERENT but also didn't
        # return the expansion-level implication (e.g. because the relation
        # names differ and aren't linked in the ontology), fall back to a
        # property-only comparison.  The expansion already established that
        # the predicates are semantically related, so we only need to check
        # whether the properties (CAUSATION, SPACE, etc.) are compatible.
        # Kernel-level logical-context properties (CAUSATION, REQUIREMENT,
        # TEMPORAL_CONTEXT, ...) carry meaningful semantic load. The verdict
        # here is an implication LHS ⇒ RHS derived from an expansion that
        # strips properties: it only stands if RHS doesn't assert any
        # logical-context key that LHS lacks. A more-specific LHS implying a
        # less-specific RHS is still fine (LHS can carry extra keys).
        if _kernel_logical_keys_block_implication(objLHS.original, objRHS.original):
            if _TRACE_COMPARE:
                print(f"[GUARD] RHS has logical-context key LHS lacks — downgrading to INDIFFERENT")
            return CasusHappening.INDIFFERENT
        if direct != CasusHappening.EQUIVALENT and not isImplication(direct):
            if _TRACE_COMPARE:
                print(f"[GUARD] direct was not eq/impl, checking properties directly")
            xorig = objLHS.original
            yorig = objRHS.original
            if (hasattr(xorig, 'properties') and hasattr(yorig, 'properties')
                    and xorig.properties and yorig.properties):
                from LaSSI.HOnK.TBox.ConstituentComparator import _filter_noise_props
                xprop = _filter_noise_props(xorig.properties)
                yprop = _filter_noise_props(yorig.properties)
                dLHS = dict(xprop)
                dRHS = dict(yprop)
                keys = set(dLHS.keys()) | set(dRHS.keys())
                d = self.pairwise_similarity_cache
                for key in keys:
                    if key in dLHS and key in dRHS:
                        lhs_bests = [
                            simplifyConstituents([compare_variable(d, xx, yy) for yy in dRHS[key]])
                            for xx in dLHS[key]
                        ]
                        prop_cmp = simplifyConstituentsAcross(lhs_bests) if lhs_bests else CasusHappening.EQUIVALENT
                        if _TRACE_COMPARE:
                            print(f"[GUARD] property key={key}: prop_cmp={prop_cmp}")
                        if prop_cmp == CasusHappening.INDIFFERENT:
                            return CasusHappening.INDIFFERENT
        if _TRACE_COMPARE:
            print(f"[GUARD] returning verdict={verdict}")
        return verdict

    def searchInSet(self, lhs, rhsSet, isRightDrop = False):
        from LaSSI.HOnK.TBox.ExpandConstituents import test_pairwise_sentence_similarity, isImplication
        for rrr in rhsSet:
            rhs = rrr.bogusCopula() if isRightDrop else rrr
            val = test_pairwise_sentence_similarity(self.pairwise_similarity_cache, lhs, rhs, shift=False)
            if val == CasusHappening.EXCLUSIVES:
                return val
            if val == CasusHappening.EQUIVALENT:
                return CasusHappening.EQUIVALENT
            if isImplication(val):
                return CasusHappening.GENERAL_IMPLICATION
        return CasusHappening.INDIFFERENT

    def compare(self, objLHS:ModelSearchBasis, objRHS:ModelSearchBasis, isLeftDrop = False, isRightDrop = False)->'CasusHappening':
        cp = (objLHS.original, objRHS.original)
        if (objLHS.original == objRHS.original):
            self.main_cache[cp] = CasusHappening.EQUIVALENT
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=EQUAL_ORIGINALS, result=EQUIVALENT")
            return self.main_cache[cp]
        from LaSSI.HOnK.TBox.LifecycleManager import _lifecycle_partition_verdict
        if _lifecycle_partition_verdict(objLHS.original, objRHS.original) == 'contradiction':
            self.main_cache[cp] = CasusHappening.EXCLUSIVES
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=LIFECYCLE_PARTITION_CONTRADICTION, result=EXCLUSIVES")
            return self.main_cache[cp]
        # Property-level SPACE contradictions are unambiguous: distinct named
        # geo entities in the SPACE slot mean the two predicates refer to
        # disjoint locations and cannot both hold of the same event.  The
        # expansion search strips properties, so this can't be derived
        # downstream — and the exhaustive loop's same-rel EXCLUSIVES
        # suppression (for WordNet synonym/antonym noise) would otherwise
        # mask the legitimate contradiction when both originals share rel.
        from LaSSI.HOnK.TBox.SpatialReasoner import _space_mismatch_contradiction, _space_city_mismatch_contradiction, _space_named_geo_entities
        _sp1 = _space_mismatch_contradiction(objLHS.original, objRHS.original)
        _sp2 = _space_city_mismatch_contradiction(objLHS.original, objRHS.original)
        if _sp1 or _sp2:
            from LaSSI.ner.string_functions import _ontology_class_suffix_terms
            _suffix = _ontology_class_suffix_terms()
            _lhs_names = _space_named_geo_entities(objLHS.original)
            _rhs_names = _space_named_geo_entities(objRHS.original)
            print(f"[SPACE-MISMATCH] LHS={_lhs_names} RHS={_rhs_names} near={_sp1} city={_sp2} suffix_terms_size={len(_suffix)} station_in_suffix={'station' in _suffix}")
            self.main_cache[cp] = CasusHappening.EXCLUSIVES
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=SPACE_MISMATCH, result=EXCLUSIVES")
            return self.main_cache[cp]
        # Property-level TIME contradictions: two predicates that each assert a
        # *defined* calendar date which differ describe mutually exclusive
        # timings of the same event (e.g. work "ends 2026-04-27" vs "ends on
        # 30 April"). Like SPACE, this is stripped by the expansion search, so
        # it must be detected on the originals here. Conservative: fires only
        # when BOTH sides carry a defined date and the date sets are disjoint —
        # a defined date vs no date stays indifferent.
        from LaSSI.HOnK.TBox.TemporalReasoner import _time_mismatch_contradiction
        if _time_mismatch_contradiction(objLHS.original, objRHS.original):
            self.main_cache[cp] = CasusHappening.EXCLUSIVES
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=TIME_MISMATCH, result=EXCLUSIVES")
            return self.main_cache[cp]
        if ((objLHS.original == make_not(objRHS.original)) or
              (objLHS.original == make_not(objLHS.original)) or
              (make_not(objLHS.original) in objRHS.unary) or
              (make_not(objLHS.original) in objRHS.binary) or
              (make_not(objRHS.original) in objLHS.unary) or
              (make_not(objRHS.original) in objLHS.binary)):
            self.main_cache[cp] = CasusHappening.EXCLUSIVES
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=NEGATION, result=EXCLUSIVES")
            return self.main_cache[cp]
        elif ((objLHS.original in objRHS.unary) or
              (objLHS.original in objRHS.binary)):
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=LHS_ORIG_IN_RHS_EXPANSION, lhs_rel={getattr(objLHS.original,'rel','?')}, rhs_rel={getattr(objRHS.original,'rel','?')}")
            self.main_cache[cp] = self._guard_contextual_implication(
                objLHS, objRHS, CasusHappening.GENERAL_IMPLICATION)
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=LHS_ORIG_IN_RHS_EXPANSION, result={self.main_cache[cp]}")
            return self.main_cache[cp]
        elif self._binary_implies_unary_drop(objLHS.original, objRHS.original):
            # Object-drop / existential generalisation: rel(a, b) ⇒ rel(a).
            # This is unconditionally sound: dropping the object of a predicate
            # cannot make it false. Hard contradictions on SPACE/TIME between the
            # originals are already returned as EXCLUSIVES earlier in `compare`,
            # and sentence-level distinguishing content is handled by the
            # similarity cap — so we do NOT route this through
            # `_guard_contextual_implication` (which would re-derive the
            # implication from a structural comparison that can't see the drop and
            # spuriously return INDIFFERENT).
            self.main_cache[cp] = CasusHappening.GENERAL_IMPLICATION
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=BINARY_IMPLIES_UNARY_DROP, result=GENERAL_IMPLICATION")
            return self.main_cache[cp]
        else:
            # Performing the constituents search:
            for lhs in objLHS.unary:
                negForm = make_not(lhs) if not isinstance(lhs, FNot) else lhs.arg
                if negForm in objRHS.unary:
                    self.main_cache[cp] = CasusHappening.EXCLUSIVES
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=NEG_UNARY_EXPANSION, result=EXCLUSIVES")
                    return self.main_cache[cp]
            for lhs in objLHS.binary:
                negForm = make_not(lhs) if not isinstance(lhs, FNot) else lhs.arg
                if negForm in objRHS.binary:
                    self.main_cache[cp] = CasusHappening.EXCLUSIVES
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=NEG_BINARY_EXPANSION, result=EXCLUSIVES")
                    return self.main_cache[cp]
            for lhs in objLHS.unary:
                if lhs in objRHS.unary:
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=UNARY_EXPANSION_MATCH, matched={lhs}")
                    self.main_cache[cp] = self._guard_contextual_implication(
                        objLHS, objRHS, CasusHappening.GENERAL_IMPLICATION)
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=UNARY_EXPANSION_MATCH, result={self.main_cache[cp]}")
                    return self.main_cache[cp]
            for lhs in objLHS.binary:
                if lhs in objRHS.binary:
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=BINARY_EXPANSION_MATCH, matched={lhs}")
                    self.main_cache[cp] = self._guard_contextual_implication(
                        objLHS, objRHS, CasusHappening.GENERAL_IMPLICATION)
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=BINARY_EXPANSION_MATCH, result={self.main_cache[cp]}")
                    return self.main_cache[cp]
            # Performing the exhaustive search:
            # Scan unary AND binary expansions for EXCLUSIVES first — a
            # genuine antonym anywhere must override any positive verdict
            # collected from the unary cross-product, otherwise a soft match
            # in unary would mask a real contradiction in binary.
            #
            # Guard against spurious EXCLUSIVES via synonym-of-mine ⇄
            # antonym-of-theirs WordNet paths: when the originals have the
            # same ontology-backed relation, expansion-derived antonymy is
            # untrustworthy ("record" vs the synonym "write off" both surface
            # as antonyms in WordNet despite both being live verbs of the same
            # event).  In that case, suppress EXCLUSIVES propagation.
            same_rel_originals = self._same_relation(objLHS.original, objRHS.original)
            elems = set()
            for lhs in objLHS.unary:
                if (isRightDrop) and isinstance(lhs, FNot):
                    continue
                tmp = lhs if not isLeftDrop else lhs.bogusCopula()
                val = self.searchInSet(tmp, objRHS.unary, isRightDrop)
                if val == CasusHappening.EXCLUSIVES:
                    if same_rel_originals:
                        continue
                    self.main_cache[cp] = val
                    return val
                elif val != CasusHappening.INDIFFERENT:
                    elems.add(val)
            for lhs in objLHS.binary:
                if (isRightDrop) and isinstance(lhs, FNot):
                    continue
                tmp = lhs if not isLeftDrop else lhs.bogusCopula()
                val = self.searchInSet(tmp, objRHS.binary, isRightDrop)
                if val == CasusHappening.EXCLUSIVES:
                    if same_rel_originals:
                        continue
                    self.main_cache[cp] = val
                    return val
                elif val != CasusHappening.INDIFFERENT:
                    elems.add(val)
            from LaSSI.HOnK.TBox.ExpandConstituents import simplifyConstituentsAcross
            result = simplifyConstituentsAcross(elems)
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=EXHAUSTIVE_SEARCH, elems={elems}, pre_guard_result={result}")
            result = self._guard_contextual_implication(objLHS, objRHS, result)
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=EXHAUSTIVE_SEARCH, post_guard_result={result}")
            self.main_cache[cp] = result
            return result
