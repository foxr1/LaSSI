__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Dict, Union, Optional

from FunctionalMatch.functions.structural_match import Ignore


# class Formula:
#     def __str__(self):
#         return "Formula(?)"

def get_existential_variables(f):
    if f is None:
        yield from []
    elif isinstance(f, FNot) or type(f).__name__ == "FNot":
        yield from  get_existential_variables(f.arg)
    elif isinstance(f, FAnd) or type(f).__name__ == "FAnd":
        for x in f.args:
            yield from get_existential_variables(x)
    elif isinstance(f, FOr) or type(f).__name__ == "FOr":
        for x in f.args:
            yield from get_existential_variables(x)
    elif isinstance(f, FUnaryPredicate) or type(f).__name__ == "FUnaryPredicate":
        yield from get_existential_variables(f.arg)
        for k,v in f.properties:
            for x in v:
                yield from get_existential_variables(x)
    elif isinstance(f, FBinaryPredicate) or type(f).__name__ == "FBinaryPredicate":
        yield from get_existential_variables(f.src)
        yield from get_existential_variables(f.dst)
        for k,v in f.properties:
            for x in v:
                yield from get_existential_variables(x)
    elif isinstance(f, FVariable) or type(f).__name__ == "FVariable":
        if f.name[0] == "?" and f.name[1:].isdigit() and f.type == "existential":
            yield f.name
        yield from get_existential_variables(f.cop)
        for k,v in f.properties:
            for x in v:
                yield from get_existential_variables(x)
    else:
        yield from []

def print_proprieties(proprieties, cop=None):
    if isinstance(proprieties, dict) or isinstance(proprieties, defaultdict):
        proprieties = proprieties.items()
    L = []
    for k, v in proprieties:
        k = k.replace("_", "\\_").replace(" ", "\; ")
        if isinstance(v, list) or isinstance(v, tuple):
            for x in v:
                L.append("\\texttt{" + str(k) + "}: " + x.asLatexString() if not isinstance(x, str) else x.replace("_", "\\_"))
        else:
            L.append("\\texttt{" + str(k) + "}: " + v.asLatexString() if not isinstance(v, str) else v.replace("_", "\\_"))
    if cop is not None:
        L.append("\\texttt{JJ}: " + cop.asLatexString())
    if len(L) > 0:
        return "_{" + ",\; ".join(L) + "}"
    else:
        return ""

def update_property_function(prop, f):
    d = {k: v for k, v in prop} if prop is not None else {}
    for key in d:
        assert not isinstance(d[key], list)
        if isinstance(d[key], tuple):
            d[key] = tuple(map(f, d[key]))
        else:
            d[key] = f(d[key])
    return frozenset(d.items())

def update_bogus_copula(f):
    if isinstance(f, str):
        return f
    elif isinstance(f, Formula):
        return f.bogusCopula()
    raise RuntimeError("RROR")

def update_property(prop, key, value):
    d = {k: v for k, v in prop} if prop is not None else {}
    if key in d:
        assert not isinstance(d[key], list)
        if isinstance(d[key], tuple):
            if value not in d[key]:
                d[key] = d[key] + (value,)
        else:
            if d[key] != value:
                d[key] = (d[key], value)
    else:
        d[key] = (value,)
    return frozenset(d.items())

@dataclass(order=True, frozen=True, eq=True)
class ProvenanceInformation:
    name: str
    id: int
    specification: Optional[str]

@dataclass(order=True, frozen=True, eq=True)
class FVariable: ## TODO: rename to FTerm or FConstant
    name: str
    type: str
    specification: Optional[str|Ignore] = None  # extra
    cop: Optional['Formula'] = None
    id: Optional[int] = None
    properties: frozenset = field(default_factory=lambda: frozenset())
    spec_negation:bool = False
    meta: str = field(default_factory=lambda: "FVariable")
    asAll: bool = False ## By default, the interpretation is exitential. If not, this is interpreted as All
    # matched: bool = field(default_factory=lambda: False)


    def extract_provenance(self):
        yield ProvenanceInformation(self.name, self.id, self.specification)
        for k,v in self.properties:
            if isinstance(v, Formula):
                yield from v.extract_provenance()
            elif isinstance(v, tuple):
                for x in v:
                    if isinstance(x, Formula):
                        yield from x.extract_provenance()

    def instantiate_variable_with_entity(self, external_entity):
        if self.name[0] == "?" and self.name[1:].isdigit() and self.type == "existential" and isinstance(external_entity, FVariable):
            return FVariable(external_entity.name, external_entity.type, self.specification, self.cop, external_entity.id, external_entity.properties, self.spec_negation, self.meta, external_entity.asAll)
        else:
            return self

    def add_specification(self, spec):
        return FVariable(self.name, self.type, spec, self.cop, self.id, self.properties, self.spec_negation, self.meta, self.asAll)

    def add_adjective(self, adj, type="JJ"):
        return FVariable(self.name, self.type, self.specification, FVariable(adj, type, "", None, None), self.id, self.properties)

    def dropCopula(self):
        return FVariable(self.name, self.type, self.specification, None, self.id, self.properties)

    def bogusCopula(self):
        return FVariable(self.name, self.type, self.specification, FVariable("?0", "existential"), self.id, self.properties)

    def makeAsAll(self):
        return FVariable(self.name, self.type, self.specification, self.cop, self.id, self.properties, self.spec_negation, self.meta, True)

    def add_property(self, key, value):
        # d = {k:v for k,v in self.properties} if self.properties is not None else {}
        # if key in d:
        #     assert not isinstance(d[key], list)
        #     if isinstance(d[key], tuple):
        #         d[key] = d[key] + (value,)
        #     else:
        #         d[key] = (d[key], value)
        # else:
        #     d[key] = (value,)
        return FVariable(self.name, self.type, self.specification, self.cop, self.id, update_property(self.properties, key, value))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        premise = self.asLatexString()
        vars = list(set(get_existential_variables(self)))
        if len(vars) == 0:
            return premise
        else:
            return "\\exists " + (", ".join(vars)) + ".\," + premise
    def asLatexString(self):
        quant = ("\\square " if self.asAll else "\\lozenge ")
        name = self.name
        if name is None:
            name = "?"
        else:
            name = "\\textsf{" + name + "}"
        assert isinstance(self.specification, str) or (self.specification is None)
        if (self.specification is not None) and len(str(self.specification))>0:
            negation = "\\neg " if self.spec_negation else ""
            assert isinstance(self.specification, str)
            name += (" [\\textup{of}] " + negation + "\\textit{" + self.specification) + "}"
            name = "\\left[" + name + "\\right]^{\\texttt{" + str(self.id) + "}}"
        else:
            name = "{" + name + "}^{\\texttt{" + str(self.id) + "}}"
        return quant + name + print_proprieties(self.properties, self.cop)

    def add_specification(self, value):
        return FVariable(self.name, self.type, value, self.cop, self.id, self.properties)

def is_selfstanding_variable(f: 'Formula'):
    if not isinstance(f, FVariable):
        return False
    return ((f.cop is None) or (isinstance(f.cop, str) and len(f.cop) == 0)) and ((f.properties is None) or (len(f.properties) == 0))

@dataclass(order=True, frozen=True, eq=True)
class FUnaryPredicate:
    rel: str
    arg: 'Formula'
    score: float
    properties: frozenset = field(default_factory=lambda: frozenset())
    meta: str = field(default_factory=lambda: "FUnaryPredicate")
    # matched: bool = field(default_factory=lambda: False)

    def extract_provenance(self):
        if self.arg is not None:
            yield from self.arg.extract_provenance()
        for k,v in self.properties:
            for x in v:
                if isinstance(x, Formula):
                    yield from x.extract_provenance()

    def bogusCopula(self):
        return FUnaryPredicate(self.rel, self.arg.bogusCopula(), self.score, update_property_function(self.properties, update_bogus_copula))

    def instantiate_variable_with_entity(self, arg):
        return self

    def add_property(self, key, value):
        # d = {k:v for k,v in self.properties} if self.properties is not None else {}
        # if key in d:
        #     assert not isinstance(d[key], list)
        #     if isinstance(d[key], tuple):
        #         d[key] = d[key] + (value,)
        #     else:
        #         d[key] = (d[key], value)
        # else:
        #     d[key] = (value,)
        return FUnaryPredicate(self.rel, self.arg, self.score, update_property(self.properties, key, value))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        premise = self.asLatexString()
        vars = list(set(get_existential_variables(self)))
        if len(vars) == 0:
            return premise
        else:
            return "\\exists " + (", ".join(vars)) + ".\," + premise
    def asLatexString(self):
        name = self.rel
        if name is None:
            name = "?"
        else:
            name = "\\textit{" + name + "}"
        return name + print_proprieties(self.properties) + (("(" + self.arg.asLatexString() + ")") if self.arg is not None else "(?)")



@dataclass(order=True, frozen=True, eq=True)
class FBinaryPredicate:
    rel: str
    src: 'Formula'
    dst: 'Formula'
    score: float
    properties: frozenset
    meta: str = field(default_factory=lambda: "FBinaryPredicate")
    # matched: bool = field(default_factory=lambda: False)

    def extract_provenance(self):
        if self.src is not None:
            yield from self.src.extract_provenance()
        if self.dst is not None:
            yield from self.dst.extract_provenance()
        for k,v in self.properties:
            for x in v:
                if isinstance(x, Formula):
                    yield from x.extract_provenance()

    def bogusCopula(self):
        return FUnaryPredicate(self.rel, self.src.bogusCopula(), self.dst.bogusCopula(), self.score, update_property_function(self.properties, update_bogus_copula))

    def instantiate_variable_with_entity(self, arg):
        return self

    def add_property(self, key, value):
        # d = {k:v for k,v in self.properties} if self.properties is not None else {}
        # if key in d:
        #     assert not isinstance(d[key], list)
        #     if isinstance(d[key], tuple):
        #         d[key] = d[key] + (value,)
        #     else:
        #         d[key] = (d[key], value)
        # else:
        #     d[key] = (value,)
        return FBinaryPredicate(self.rel, self.src, self.dst, self.score, update_property(self.properties, key, value))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        premise = self.asLatexString()
        vars = list(set(get_existential_variables(self)))
        if len(vars) == 0:
            return premise
        else:
            return "\\exists " + (", ".join(vars)) + ".\," + premise
    def asLatexString(self):
        name = self.rel
        if name is None:
            name = "?"
        else:
            name = "\\textit{" + name + "}"
        s = "("
        if self.src is not None:
            s += (self.src.asLatexString() + ",")
        else:
            s += "?,"
        if self.dst is not None:
            if isinstance(self.dst, list):
                s += ("["+ (",\,".join([x.asLatexString() for x in self.dst]))  + "])")
            else:
                s += (self.dst.asLatexString() + ")")
        else:
            s += "?)"
        return name + print_proprieties(self.properties) +s




@dataclass(order=True, frozen=True, eq=True)
class FAnd:
    args: Tuple['Formula']
    meta: str = field(default_factory=lambda: "FAnd")
    # matched: bool = field(default_factory=lambda: False)

    def extract_provenance(self):
        for x in self.args:
            yield from x.extract_provenance()

    def bogusCopula(self):
        return FAnd([x.bogusCopula() for x in self.args])

    def instantiate_variable_with_entity(self, external_entity):
        return FAnd(args=tuple([x.instantiate_variable_with_entity(external_entity) for x in self.args]))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        premise = self.asLatexString()
        vars = list(set(get_existential_variables(self)))
        if len(vars) == 0:
            return premise
        else:
            return "\\exists " + (", ".join(vars)) + ".\," + premise
    def asLatexString(self):
        return "\\left(" + (" \\wedge ".join(map(lambda x: x.asLatexString(), self.args))) + "\\right)"



@dataclass(order=True, frozen=True, eq=True)
class FOr:
    args: Tuple['Formula']
    meta: str = field(default_factory=lambda: "FOr")
    # matched: bool = field(default_factory=lambda: False)

    def extract_provenance(self):
        for x in self.args:
            yield from x.extract_provenance()

    def bogusCopula(self):
        return FOr([x.bogusCopula() for x in self.args])

    def instantiate_variable_with_entity(self, external_entity):
        return FOr(args=tuple([x.instantiate_variable_with_entity(external_entity) for x in self.args]))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        premise = self.asLatexString()
        vars = list(set(get_existential_variables(self)))
        if len(vars) == 0:
            return premise
        else:
            return "\\exists " + (", ".join(vars)) + ".\," + premise

    def asLatexString(self):
        return "\\left(" + (" \\vee ".join(map(lambda x: x.asLatexString(), self.args))) + "\\right)"

@dataclass(order=True, frozen=True, eq=True)
class FNot:
    arg: 'Formula'
    meta: str = field(default_factory=lambda: "FNot")
    matched: bool = field(default_factory=lambda: False)

    def extract_provenance(self):
        if self.arg is not None:
            yield from self.arg.extract_provenance()

    def bogusCopula(self):
        return FNot(self.arg.bogusCopula())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        premise = self.asLatexString()
        vars = list(set(get_existential_variables(self)))
        if len(vars) == 0:
            return premise
        else:
            return "\\exists " + (", ".join(vars)) + ".\," + premise

    def asLatexString(self):
        return " \\neg \\left(" + self.arg.asLatexString() + "\\right)"


# @dataclass(order=True, frozen=True, eq=True)
# class FNot:
#     arg: 'Formula'
#     meta: str = field(default_factory=lambda: "FNot")
#     matched: bool = field(default_factory=lambda: False)
#
#     def __repr__(self):
#         return self.__str__()
#
#     def asLatexString(self):
#         return " \\neg \\left(" + self.arg.asLatexString() + "\\right)"

Formula = Union[FOr, FAnd, FUnaryPredicate, FBinaryPredicate, FVariable, FNot]

def make_not(param):
    return FNot(arg=param)

def prune_from_cop(var: FVariable):
    return FVariable(name=var.name, type=var.type, specification=var.specification, cop=None, id=var.id)

def id_formula(f:Formula):
    if isinstance(f, FNot) or type(f).__name__ == "FNot":
        yield from id_formula(f.arg)
    elif isinstance(f, FAnd) or type(f).__name__ == "FAnd":
        for x in f.args:
            yield from id_formula(x)
    elif isinstance(f, FOr) or type(f).__name__ == "FOr":
        for x in f.args:
            yield from id_formula(x)
    elif isinstance(f, FVariable) or type(f).__name__ == "FVariable":
        if f.cop is not None:
            yield from id_formula(f.cop)
        for k, v in f.properties:
            if isinstance(v, tuple):
                for x in v:
                    yield from id_formula(x)
            else:
                yield from id_formula(v)
        if f.id is not None and (f.id>=0):
            yield f.id
    elif isinstance(f, FUnaryPredicate) or type(f).__name__ == "FUnaryPredicate":
        if f.arg is not None:
            yield from id_formula(f.arg)
        for k, v in f.properties:
            if isinstance(v, tuple):
                for x in v:
                    yield from id_formula(x)
            else:
                yield from id_formula(v)
    elif isinstance(f, FBinaryPredicate) or type(f).__name__ == "FBinaryPredicate":
        if f.src is not None:
            yield from id_formula(f.src)
        if f.dst is not None:
            yield from id_formula(f.dst)
        for k, v in f.properties:
            if isinstance(v, tuple):
                for x in v:
                    yield from id_formula(x)
            else:
                yield from id_formula(v)
    else:
        yield from []

def type_atom(f:Formula):
    if isinstance(f, FNot) or type(f).__name__ == "FNot":
        yield from  type_atom(f.arg)
    elif isinstance(f, FAnd) or type(f).__name__ == "FAnd":
        for x in f.args:
            yield from id_formula(x)
    elif isinstance(f, FOr) or type(f).__name__ == "FOr":
        for x in f.args:
            yield from id_formula(x)
    elif isinstance(f, FVariable) or type(f).__name__ == "FVariable":
        yield f.type if f.type is not None else "ENTITY"
    else:
        raise RuntimeError("ERROR: wrongly expected type")



def formula_from_dict(f: Union[dict, str], useId=False):
    """
    Loading a json file in its object-dictionary rerpesentation into formulaes.
    This is mainly used to run the pipeline from one point at a time.
    """
    if f is None:
        return None
    if isinstance(f, str):
        return f
    from collections.abc import Iterable
    if isinstance(f, Iterable) and not (isinstance(f, dict)):
        return list(map(lambda x:formula_from_dict(x, useId), f))
    assert "meta" in f
    meta = f["meta"]
    if meta == "FNot":
        return FNot(arg=formula_from_dict(f["arg"], useId))
    if meta == "FOr":
        return FOr(args=tuple(map(lambda x:formula_from_dict(x, useId), f["args"])))
    if meta == "FAnd":
        return FAnd(args=tuple(map(lambda x:formula_from_dict(x, useId), f["args"])))
    if meta == "FVariable":
        spec_negation = bool(f["spec_negation"]) if "spec_negation" in f and f["spec_negation"] is not None else False
        name = str(f["name"]) if "name" in f and f["name"] is not None else None
        type = str(f["type"]) if "name" in f and f["type"] is not None else None
        id = int(f["id"]) if useId and ("id" in f and f["id"] is not None) else None
        specification = formula_from_dict(f["specification"], useId) if "specification" in f and f["specification"] is not None else None
        cop = formula_from_dict(f["cop"], useId) if "cop" in f else None
        properties = defaultdict(set)
        if "properties" in f:
            for k, v in f["properties"].items():
                for x in v:
                    properties[k].add(formula_from_dict(x, useId))
        properties = frozenset({k: tuple(v) for k, v in properties.items()}.items())
        return FVariable(name=name, type=type, specification=specification, cop=cop, id=id, spec_negation=spec_negation, properties=properties)
    if meta == "FUnaryPredicate":
        rel = str(f["rel"]) if "rel" in f and f["rel"] is not None else ""
        arg = formula_from_dict(f["arg"], useId) if "arg" in f and f["arg"] is not None else None
        score = float(f["score"]) if "score" in f else 1.0
        properties = defaultdict(set)
        if "properties" in f:
            for k, v in f["properties"].items():
                for x in v:
                    properties[k].add(formula_from_dict(x, useId))
        properties = {k: tuple(v) for k, v in properties.items()}
        return FUnaryPredicate(rel, arg, score, frozenset(properties.items()))
    if meta == "FBinaryPredicate":
        rel = str(f["rel"]) if "rel" in f else ""
        src = formula_from_dict(f["qa"], useId) if "qa" in f else None
        dst = formula_from_dict(f["dst"], useId) if "dst" in f else None
        score = float(f["score"]) if "score" in f else 1.0
        properties = defaultdict(set)
        if "properties" in f:
            for k, v in f["properties"].items():
                for x in v:
                    properties[k].add(formula_from_dict(x, useId))
        properties = {k: tuple(v) for k, v in properties.items()}
        return FBinaryPredicate(rel, src, dst, score, frozenset(properties.items()))
