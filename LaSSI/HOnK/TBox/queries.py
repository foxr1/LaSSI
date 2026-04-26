
from LaSSI.structures.extended_fol.Formulae import FVariable, FUnaryPredicate, FBinaryPredicate, FNot
from LaSSI.HOnK.HOnK import HOnKSingleton

## TODO: use single_edge_src_multipoint to retrive the target nodes given the name, the adjective, and the relationship

def getMultiwayAdjectivalPoint(d, **kwargs):
    main = kwargs.get("main", None)
    adj = kwargs.get("adj", None)
    rel = kwargs.get("rel", None)
    result = kwargs.get("result", None)
    if main is None or adj is None or rel is None:
        return []
    assert main is not None
    assert adj is not None
    assert rel is not None
    assert result is not None
    if (main not in d) or (adj not in d):
        return []
    # assert main in d
    assert adj in d
    main = d[main]
    adj = d[adj]
    if not isinstance(main, str):
        return []
    if not isinstance(adj, str):
        return []
    assert isinstance(result, str)
    S = list(HOnKSingleton.get().single_edge_src_multipoint(main, adj, rel, f"^{result}"))
    if len(S) == 0:
        return S
    else:
        L = []
        for x in S:
            dd = {k:v for k,v in d.items()}
            dd[result] = x[result]
            L.append(dd)
        return L

def getMultiwayTargetSimpleSentence(d, **kwargs):
    main = kwargs.get("main", None)
    if (main is None) or (main not in d):
        return []
    assert main is not None
    assert main in d
    main = d[main]
    if not isinstance(main, str):
        return []

    rel = kwargs.get("rel", None)
    if not isinstance(rel, str):
        return []

    resultVerb = kwargs.get("resultVerb", None)
    resultSubject = kwargs.get("resultSubject", None)
    resultObject = kwargs.get("resultObject", None)
    assert resultVerb is not None
    assert resultSubject is not None
    assert resultObject is not None
    S = list(HOnKSingleton.get().single_edge_dst_binary_capability(main, rel, f"^{resultVerb}", f"^{resultSubject}", f"^{resultObject}"))
    if len(S) == 0:
        return S
    else:
        L = []
        for x in S:
            dd = {k:v for k,v in d.items()}
            dd[resultVerb] = x[resultVerb]
            dd[resultSubject] = x[resultSubject]
            dd[resultObject] = x[resultObject]
            L.append(dd)
        return L

def extractJsonPath(d, **kwargs):
    jquery = kwargs.get("jquery", None)
    as_ = kwargs.get("result", None)
    if (jquery is None) or (as_ is None) or not isinstance(jquery, str):
        return []

    from FunctionalMatch.PropositionalLogic import var_interpret
    from FunctionalMatch.functions.structural_match import JSONPath
    result = var_interpret(JSONPath(jquery), d, keepList=True)
    L = []
    for x in result:
        if x is not None:
            dd = {k: v for k, v in d.items()}
            dd[as_] = x
            L.append(dd)
    return L

def extractProperties(d, **kwargs):
    from_obj = kwargs.get("from", None)
    assert from_obj is not None
    assert isinstance(from_obj, str)
    assert from_obj in d
    obj = d[from_obj]
    main = kwargs.get("field", None)
    assert main is not None
    assert isinstance(main, str)
    result = kwargs.get("result", None)
    assert result is not None
    assert isinstance(result, str)
    if not hasattr(obj, "properties") or (not isinstance(obj.properties, frozenset)):
        return [] ## if there are no properties field, then no match shall be considered
    L = []
    for k, v in obj.properties:
        if k == main:
            if isinstance(v, list) or isinstance(v, tuple):
                for x in v:
                    dd = {k: v for k, v in d.items()}
                    dd[result] = x
                    L.append(dd)
            else:
                dd = {k: v for k, v in d.items()}
                dd[result] = v
                L.append(dd)
    return L

def removeFromProperties(arg, **kwargs):
    if (arg is None):
        print("No variable to remove the adjective: returning NONE")
        return None
    if isinstance(arg, frozenset):
        var = kwargs.get("key", None)
        assert var is not None
        assert isinstance(var, str)
        return frozenset({k:v for k,v in arg if k != var}.items())
    else:
        raise RuntimeError("ERROR: we can remove properties only to something that has a properties field and which their type is a frozenset!")

def getOutgoingNodes(d, **kwargs):
    var = kwargs.get("variable", None)
    result = kwargs.get("result", None)
    rel = kwargs.get("rel", None)
    if var is None:
        return []
    assert result is not None
    assert rel is not None
    assert isinstance(var, str)
    assert isinstance(result, str)
    assert isinstance(rel, str)
    adjForm = d.get(var)
    if not isinstance(adjForm, str):
        return []
    S = HOnKSingleton.get().getOutgoingNodes(adjForm, rel)
    if len(S) == 0:
        return []
    else:
        L = []
        for x in S:
            dd = {k:v for k,v in d.items()}
            dd[result] = x
            L.append(dd)
        return L

def getIngoingNodes(d, **kwargs):
    var = kwargs.get("variable", None)
    result = kwargs.get("result", None)
    rel = kwargs.get("rel", None)
    if var is None:
        return []
    assert result is not None
    assert rel is not None
    assert isinstance(var, str)
    assert isinstance(result, str)
    assert isinstance(rel, str)
    adjForm = d.get(var)
    if not isinstance(adjForm, str):
        return []
    S = HOnKSingleton.get().getIngoingNodes(adjForm, rel)
    if len(S) == 0:
        return []
    else:
        L = []
        for x in S:
            dd = {k:v for k,v in d.items()}
            dd[result] = x
            L.append(dd)
        return L

def isOfType(kwargs):
    var = kwargs.get("variable", None)
    type_ = kwargs.get("type", None)
    assert var is not None
    assert type_ is not None
    assert isinstance(var, str)
    assert isinstance(type_, str)
    adjForm = kwargs.get(var)
    type_ = kwargs.get(type_, type_)
    if adjForm is None:
        return True
    val = 0
    for _ in HOnKSingleton.get().isA(adjForm, type_):
        val += 1
    return val > 0


def notNegated(kwargs):
    var = kwargs.get("variable", None)
    if var is None:
        return True
    assert isinstance(var, str)
    adjForm = kwargs.get(var)
    if adjForm is None:
        return True
    return not isinstance(adjForm, FNot)

def neitherNegatedNorNone(kwargs):
    var = kwargs.get("variable", None)
    if var is None:
        return False
    assert isinstance(var, str)
    adjForm = kwargs.get(var)
    if adjForm is None:
        return False
    return (not isinstance(adjForm, FNot)) and ((not  hasattr(adjForm, "name")) or ((adjForm.name is not None) and ((not adjForm.name.startswith("?")) or (not adjForm.name[1:].isdigit()))))

def NegatedOrNone(kwargs):
    return not neitherNegatedNorNone(kwargs)

def IsNone(kwargs):
    var = kwargs.get("variable", None)
    if var is None:
        return False
    assert isinstance(var, str)
    adjForm = kwargs.get(var)
    if adjForm is None:
        return False
    return (not isinstance(adjForm, FNot)) and ((not  hasattr(adjForm, "name")) or ((adjForm.name is None) or (( adjForm.name.startswith("?")) and ( adjForm.name[1:].isdigit()))))


def nonEmptyMatch(kwargs):
    var = kwargs.get("variable", None)
    if var is None:
        return False
    assert var is not None
    assert isinstance(var, str)
    adjForm = kwargs.get(var)
    if adjForm is None:
        return False
    return (not isinstance(adjForm, list)) or (len(adjForm) > 0)

def addAdjective(arg, **kwargs):
    if arg is None:
        print("No variable to add the adjective: returning NONE")
        return None
    if isinstance(arg, FVariable) or type(arg).__name__ == "FVariable":
        var = kwargs.get("adjective", None)
        assert var is not None
        assert isinstance(var, str)
        return arg.add_adjective(var)
    else:
        raise RuntimeError("ERROR: we can add an adjective only to something that is a variable!")

def dropCopula(arg, **kwargs):
    if arg is None:
        print("No variable to add the adjective: returning NONE")
        return None
    if isinstance(arg, FVariable) or type(arg).__name__ == "FVariable":
        return arg.dropCopula()
    else:
        raise RuntimeError("ERROR: we can add an adjective only to something that is a variable!")

def addSpecification(arg, **kwargs):
    if arg is None:
        print("No variable to add the property: returning NONE")
        return None
    if ((isinstance(arg, FVariable) or type(arg).__name__ == "FVariable")):
        value = kwargs.get("value", None)
        assert value is not None
        return arg.add_specification(value)
    else:
        raise RuntimeError("ERROR: we can add a specification only to something that is a variable!")

def addProperty(arg, **kwargs):
    if arg is None:
        print("No variable to add the property: returning NONE")
        return None
    if isinstance(arg, frozenset):
        var = kwargs.get("key", None)
        assert var is not None
        assert isinstance(var, str)
        value = kwargs.get("value", None)
        assert value is not None
        from LaSSI.structures.extended_fol.Formulae import update_property
        return update_property(arg, var, value)
    if ((isinstance(arg, FVariable) or type(arg).__name__ == "FVariable") or
            (isinstance(arg, FUnaryPredicate) or type(arg).__name__ == "FUnaryPredicate") or
            (isinstance(arg, FBinaryPredicate) or type(arg).__name__ == "FBinaryPredicate")):
        var = kwargs.get("key", None)
        assert var is not None
        assert isinstance(var, str)
        value = kwargs.get("value", None)
        assert value is not None
        return arg.add_property(var, value)
    else:
        raise RuntimeError("ERROR: we can add an adjective only to something that is a variable/(unary|binary)-predicate!")
