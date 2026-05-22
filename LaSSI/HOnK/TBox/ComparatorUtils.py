from LaSSI.HOnK.HOnK import CasusHappening
from LaSSI.structures.extended_fol.Formulae import FVariable

def isImplication(x):
    return x == CasusHappening.GENERAL_IMPLICATION or x == CasusHappening.LOSE_SPEC_IMPLICATION or x == CasusHappening.INSTANTIATION_IMPLICATION or x == CasusHappening.MISSING_1ST_IMPLICATION

d_transformCaseWhenOneArgIsNegated = None

def transformCaseWhenOneArgIsNegated(orig: CasusHappening):
    """Negation of a more-valued logic (where we have more than just satisfiability or not)"""
    global d_transformCaseWhenOneArgIsNegated
    if d_transformCaseWhenOneArgIsNegated is None:
        d_transformCaseWhenOneArgIsNegated = {CasusHappening.NONE: CasusHappening.NONE,
                                              CasusHappening.INDIFFERENT: CasusHappening.INDIFFERENT,
                                              CasusHappening.EQUIVALENT: CasusHappening.EXCLUSIVES,
                                              CasusHappening.EXCLUSIVES: CasusHappening.EQUIVALENT,
                                              CasusHappening.GENERAL_IMPLICATION: CasusHappening.INDIFFERENT,
                                              CasusHappening.LOSE_SPEC_IMPLICATION: CasusHappening.INDIFFERENT,
                                              CasusHappening.INSTANTIATION_IMPLICATION: CasusHappening.INDIFFERENT,
                                              CasusHappening.MISSING_1ST_IMPLICATION: CasusHappening.INDIFFERENT}
    return d_transformCaseWhenOneArgIsNegated[orig]

def isExistential(x):
    if x is None:
        return False
    return isinstance(x, FVariable) and x.name[0] == "?" and x.name[1:].isdigit()
