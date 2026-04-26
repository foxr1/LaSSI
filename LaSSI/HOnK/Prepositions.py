__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2025, Functional Match"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPLv3"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import copy
import dataclasses
import json
from dataclasses import dataclass

import dacite


@dataclass(frozen=True, eq=True)
class Preposition:
    name: str
    canOmit:bool=False
    alwaysOmit:bool=False
    alwaysTakeInfinitive:bool=False
    alternativeOf:str=None
    abbreviationOf:str=None
    superlative:bool=False
    comparative:bool=False
    isPrototypical:bool=False
    isDependant:bool=False
    isIdiomatic:bool=False
    isComplex:bool=False

    def with_different_name(self, new_name, withAlternative=None, withAbbreviation=None):
        return Preposition(new_name, self.canOmit, self.alwaysOmit, self.alwaysTakeInfinitive, self.alternativeOf if withAlternative is None else withAlternative, self.abbreviationOf if withAbbreviation is None else withAbbreviation,
                           self.superlative, self.comparative, self.isPrototypical, self.isDependant, self.isIdiomatic, self.isComplex)

    def as_properties(self):
        d = dataclasses.asdict(self)
        for x in {"isPrototypical","isDependant","isIdiomatic", "isComplex","name"}:
            d.pop(x)
        return d

    def generate_classes(self, toReject=False):
        L = []
        if self.isPrototypical:
            L.append("PrototypicalPreposition")
        if self.isDependant:
            L.append("DependantPreposition")
        if self.isIdiomatic:
            L.append("IdiomaticPreposition")
        if self.isComplex:
            L.append("ComplexPreposition")
        if len(L) == 0:
            L.append("Preposition")
        if toReject:
            L.append("Rejectable")
        return L

    @staticmethod
    def update_with_label( d, case_):
        if case_ == "PrototypicalPreposition":
            d["isPrototypical"] = True
        if case_ == "DependantPreposition":
            d["isDependant"] = True
        if case_ == "IdiomaticPreposition":
            d["isIdiomatic"] = True
        if case_ == "ComplexPreposition":
            d["isComplex"] = True
        return d


def load_prepositions(file):
    data = None
    with open(file) as f:
        data = json.load(f)
    data.pop("__sources")
    d = dict()
    for k,v in data.items():
        v["name"] = k
        obj = dacite.from_dict(Preposition, v)
        d[k] = obj
    for v in list(d.values()):
        if v.abbreviationOf is not None:
            d[v.name] = d[v.abbreviationOf].with_different_name(v.name, withAbbreviation=v.abbreviationOf)
        elif v.alternativeOf is not None:
            d[v.name] = d[v.alternativeOf].with_different_name(v.name, withAlternative=v.alternativeOf)
    yield from d.values()


if __name__ == "__main__":
    data = list(load_prepositions("../../raw_data/prepositions.json"))
    print(data)
