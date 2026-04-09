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
from copy import deepcopy

from LaSSI.ner.node_functions import create_props_for_singleton

# from scipy._lib.array_api_compat.array_api_compat import numpy

type_conversion =             {"GPE": "SPACE",
             "LOC": "SPACE",
             "DATE": "TIME"}

from LaSSI.structures.internal_graph.EntityRelationship import NodeEntryPoint, Singleton, SetOfSingletons, Grouping
from LaSSI.structures.extended_fol.Formulae import FNot, FOr, FAnd, FUnaryPredicate, FVariable, FBinaryPredicate, \
    Formula, prune_from_cop, type_atom

def is_existential(obj):
    if (isinstance(obj, Singleton) or type(obj).__name__ == "Singleton") and (
                (len(obj.named_entity)>= 2) and obj.named_entity[0] == "?" and obj.named_entity[1:].isdigit() and obj.type == "existential"):
        return True
    if (not isinstance(obj, FVariable)) or (not type(obj).__name__ == "FVariable"):
        return False
    return obj.name[0] == "?" and obj.name[1:].isdigit() and obj.type == "existential"

bogus_dst = FVariable(name="there", type="non_verb", specification=None, cop=None, id=None)
bogus_src = {"it"}
discard_properties = {"end", "lemma", "begin", "kernel", "expl", "pos", "root", "common", "number", "adv", "conj", "mark", "specification", "nmod_poss", "amod"}
relative_pronouns = {"which","that", "who", "whom" }
interrogative_pronouns = {"what", "which", "who", "whom", "whose"}
demonstrative_pronouns = {"this", "these", "that", "those"}



def property_write(key, val: NodeEntryPoint) -> str:
    value = '""'
    # if isinstance(val, Singleton):
    return f'{key} : {value}'

def get_props(src):
    src_old_props = None
    if src is not None:
        if hasattr(src, "get_props"):
            src_old_props = src.get_props()
        elif hasattr(src, "properties"):
            src_old_props = src.properties
    return src_old_props

def make_and(entities):
    entities = tuple(entities)
    # assert all(map(lambda x: isinstance(x, Formula), entities))
    return FAnd(args=entities)


def make_or(entities):
    return FOr(args=tuple(entities))


def make_not(param):
    return FNot(arg=param)

def has_prop_just_one_negated_constituent(prop):
    """
    This function returns a pair of a boolean and of a rewritten set of properties

    If th proposition contains just one negated constituent and, therefore, the entire clause can be rewritten as
    one single logical negated constituent, then this function returns a rewritten non-null constituent. If this does not
    happen, it returns the same proposition. To disambiguate between the two, we use the boolean: if true, it means that
    the second argument is the cleaned version where the argument is negated
    :param prop:
    :return:
    """
    if (len(prop) != 1):
        return False, prop
    k, x = next(iter(prop))
    if k in Grouping.__members__.keys() or k == "SPECIFICATION":
        return False, prop
    assert isinstance(x, tuple) and len(x) == 1
    v = x[0]
    if isinstance(v,FNot):
        return True, frozenset({(k, (v.arg, ))})
    elif isinstance(v, FVariable):
        isCopNegated = ((v.cop is not None) and isinstance(v.cop, FNot))
        if isCopNegated:
            assert not isinstance(v.cop.arg, FNot) ## Not considering double negation at the moment, which should not be captured by the pipeline
        isSpecNegated = v.spec_negation
        if isCopNegated and isSpecNegated:
            return True, frozenset({(k, (FVariable(v.name, v.type, v.specification, v.cop.arg, v.id, v.properties), ))})
        elif isCopNegated:
            return True, frozenset(
                {(k, (FVariable(v.name, v.type, v.specification, v.cop.arg, v.id, v.properties), ))})
        elif isSpecNegated:
            return True, frozenset({(k, (FVariable(v.name, v.type, v.specification, v.cop, v.id, v.properties), ))})
        else:
            return False, prop
    # else:
    #     assert False
    return False, prop


def rewrite_predicate_with_new_first_argument(sentence, first_argument):
    if first_argument is None:
        return sentence
    if isinstance(sentence, FUnaryPredicate) or type(sentence).__name__ == "FUnaryPredicate":
        if (is_existential(sentence.arg)):
            return FUnaryPredicate(sentence.rel, first_argument, sentence.score, sentence.properties)
        else:
            return sentence
    elif isinstance(sentence, FBinaryPredicate) or type(sentence).__name__ == "FBinaryPredicate":
        if (is_existential(sentence.src)):
            return FBinaryPredicate(sentence.rel, first_argument, sentence.dst, sentence.score, sentence.properties)
        else:
            return sentence
    else:
        return sentence



class RewriteKernels:

    def __init__(self, obj, meu_db_row, useId=False):
        self.useId = useId
        self.meu_db_row = meu_db_row
        self.obj = obj
        from LaSSI.external_services.Services import Services
        self.p = Services.getInstance().getParmenides()
        self.e = Services.getInstance().getExistentials()
        self.dmin = None
        self.dmax = None
        self.dpos = None
        self.dmin, self.dmax, self.dpos = obj.update_map(defaultdict(lambda: 10000000), defaultdict(lambda:-1), defaultdict(lambda: 10000000))

    def derive_external_relationships(self, collection, arg):
        ls = []
        for node, mappa in collection:
            for k, v in mappa.items() if isinstance(mappa, dict) else mappa:
                if k == "SENTENCE":
                    for x in v:
                        if isinstance(x, Formula):
                            x = rewrite_predicate_with_new_first_argument(x, arg)
                            ls.append(x)
                        else:
                            result = self.rewrite_kernels(x)
                            result = rewrite_predicate_with_new_first_argument(result, arg)
                            ls.append(result)
        return ls

    def derive_kernel_properties(self, collection, properties):
        final_properties = defaultdict(list)
        for k,v in properties.items() if isinstance(properties, dict) else properties:
            for x in v:
                    final_properties[k].append(x)
        for node, mappa in collection:
            for k, v in mappa.items() if isinstance(mappa, dict) else mappa:
                if k == "nmod_poss":
                    if isinstance(v, str):
                        final_properties["SPECIFICATION"].append(node.add_specification(v))
                    elif isinstance(v, tuple):
                        for x in v:
                            if isinstance(x, str):
                                final_properties["SPECIFICATION"].append(node.add_specification(x))
                    elif hasattr(v, "name"):
                        final_properties["SPECIFICATION"].append(node.add_specification(v.name))
        return final_properties

    def make_properties(self, p):
        result = defaultdict(set)
        if "not" in set(map(lambda x: x.lower(), p.keys())):
            for k in filter(lambda x: x.lower() == "not", p.keys()):
                for single_val in p[k]:
                    if isinstance(single_val, SetOfSingletons):
                        assert len(single_val.entities) == 1
                        single_val = single_val.entities[0]
                    assert isinstance(single_val, Singleton)
                    type = single_val.type
                    single_val = make_not(self.make_arg(single_val))
                    result[type].add(single_val)
        for k, v in p.items():
            if str(k) == "\u2203" or str(k).lower() == "in" or str(k).lower() == "not":
                continue
            elif k not in discard_properties:
                for single_val in v:
                    if not isinstance(single_val, str):
                        tmp = self.make_arg(single_val)
                        neg_tmp = make_not(tmp)
                        if neg_tmp not in result[k]:
                            result[k].add(tmp)
        result2 = dict()
        for k, v in result.items():
            result2[k] = list(v)
        return result2  # dict(result2.items())

    def make_cop(self, entity) -> FVariable:
        if entity is None:
            return None
        elif isinstance(entity, str):
            return FVariable(name=entity, type="JJ", specification=None, cop=None, id=None)
        else:
            return self.make_arg(entity[0])  # TODO: Will we ever have more than one cop for a given entity?

    def make_arg(self, entity):
        if entity is None:
            return None
        elif isinstance(entity, FVariable) or isinstance(entity, FBinaryPredicate) or isinstance(entity,
                                                                                                 FUnaryPredicate) or isinstance(
                entity, FNot) or isinstance(entity, FAnd) or isinstance(entity, FOr):
            return entity
        elif hasattr(entity, "kernel") and entity.kernel is not None:
            return self.rewrite_kernels(entity)
        props = entity if isinstance(entity, dict) else entity.get_props()
        specifiaction = None
        if ("extra" in props) and (props["extra"] is not None) and (
                (not isinstance(props["extra"], tuple)) or len(props["extra"]) == 1):
            specifiaction = self.make_arg(props.pop("extra")[0]).name
        coplist = []
        cop = None
        if "cop" in props:
            cop = self.make_cop(props.pop("cop"))
        if cop is None:
            if "JJ" in props:
                cop = self.make_cop(props.pop("JJ"))
            for k in props:
                if k.endswith("mod"):
                    coplist.append(props[k])
        if len(coplist) == 1:
            cop = self.make_cop(coplist[0])
        elif len(coplist) > 1:
            cop = self.make_cop(" ".join(sorted(coplist, key=lambda x: self.meu_db_row.first_sentence.find(x))))
        id = None
        if self.useId:
            if isinstance(entity, dict) and "id" in entity:
                id = entity["id"]
            elif hasattr(entity, "id"):
                id = entity.id
        named_entity = props.pop("named_entity", None) if isinstance(entity,
                                                                     dict) else entity.get_name()  # TODO: Is this okay for getting the name of SetOfSingletons?
        type = props.pop("type", None) if isinstance(entity, dict) else entity.type
        if (type != "GPE") and (type != "SPACE"):
            named_entity = named_entity.lower()
        props2 = dict()
        asAll = False
        for k, v in props.items():
            if k == 'det' and isinstance(v, str):
                if v.lower() == "all":
                    asAll = True
            if k not in discard_properties and k not in {} and ((not isinstance(v, str)) or len(v) > 0):
                if isinstance(v, tuple):
                    props2[k] = tuple([self.make_arg(x) if isinstance(x, Singleton) else x for x in v])
                else:
                    props2[k] = self.make_arg(v) if isinstance(v, Singleton) else v
        if (cop == "usually") or (isinstance(cop, FVariable) and (cop.name == "usually")):  ## TODO:adverb
            cop = None
        props2 = self.props_as_unique_itemset(props2)
        test, props2 = has_prop_just_one_negated_constituent(props2)
        result = FVariable(name=named_entity, type=type, specification=specifiaction, cop=cop, id=id,
                         properties=props2, asAll=asAll)
        return FNot(result) if test else result

    def make_unary(self, rel, src, score, prop):
        if rel == "be":  # TODO: generalise
            if src is not None and src.cop is not None and (prune_from_cop(src).type != "JJ"):  # TODO: generalise
                return self.make_binary("have", prune_from_cop(src), src.cop, score, prop)
            if src is not None and (
                    src.type == "DATE" or src.type == "GPE" or src.type == "LOC" or src.type == "SPACE") and src.cop is not None:  # TODO: generalise
                dstType = type_conversion.get(src.type, src.type)
                if dstType not in prop:
                    prop[dstType] = []
                prop[dstType].append(src)
                return self.make_unary(rel, src.cop, score, prop)
        if "non_verb" in prop:
            prop.pop("non_verb")
        s = set(prop.keys())
        for x in s:
            if isinstance(prop[x], list):
                if len(prop[x]) == 0:
                    prop[x] = prop[x][0]
                else:
                    prop[x] = tuple(prop[x])
        prop = self.props_as_unique_itemset(prop)
        test, prop = has_prop_just_one_negated_constituent(prop)
        rel = "be" if rel == "None" else rel
        result = FUnaryPredicate(rel=rel, arg=src, score=score, properties=prop)
        return FNot(result) if test else result

    def make_binary(self, rel, src, dst, score, prop):
        if isinstance(dst, FAnd):
            return make_and([self.make_binary(rel, src, x, score, prop) for x in dst.args])
        elif isinstance(dst, FOr):
            return make_or([self.make_binary(rel, src, x, score, prop) for x in dst.args])
        elif isinstance(dst, FNot):
            return make_not(self.make_binary(rel, src, dst.arg, score, prop))
        elif (rel == "be" and (dst is None or (not isinstance(dst, FBinaryPredicate) and not isinstance(dst,
                                                                                                      FUnaryPredicate) and dst.type == "existential"))) or dst is None:
            from LaSSI.ner.MergeSetOfSingletons import merge_multiway_static_properties
            return self.make_unary(rel, src, score, merge_multiway_static_properties(prop, dst.properties))
        if rel == "have":  # TODO: generalise
            if src is not None and (
                    src.type == "DATE" or src.type == "GPE" or src.type == "LOC" or src.type == "SPACE") and src.cop is None:  # TODO: generalise
                dstType = type_conversion.get(src.type, src.type)
                if dstType not in prop:
                    prop[dstType] = []
                prop[dstType].append(src)
                return self.make_unary("be", dst, score, prop)
        if "non_verb" in prop:
            prop.pop("non_verb")
        s = set(prop.keys())
        for x in s:
            if isinstance(prop[x], list):
                if len(prop[x]) == 0:
                    prop[x] = prop[x][0]
                else:
                    prop[x] = tuple(prop[x])
        prop = self.props_as_unique_itemset(prop)
        test, prop = has_prop_just_one_negated_constituent(prop)
        result =  FBinaryPredicate(rel=rel, src=src, dst=dst, score=score, properties=prop)
        return FNot(result) if test else result

    def make_prop(self, src, rel, negated, score, properties, dst):
        if (dst is not None):
            if isinstance(dst, SetOfSingletons):
                if dst.type == Grouping.AND:
                    return make_and(map(lambda x: self.make_prop(src, rel, negated, score, properties, x), dst.entities))
                elif dst.type == Grouping.OR:
                    return make_or(map(lambda x: self.make_prop(src, rel, negated, score, properties, x), dst.entities))
                elif dst.type == Grouping.NOT:
                    return make_not(self.make_prop(src, rel, negated, score, properties, list(dst.entities)[0]))
                else:
                    n = dst.type.name
                    raise RuntimeError(f"Unknown source type: {n}")
            else:
                if negated:
                    return make_not(self.make_prop(src, rel, False, score, properties, dst))
                else:
                    result = None
                    p = dict()
                    foundSingleton = Grouping.NONE
                    argument = None
                    forKey = None
                    for k, v in properties.items():
                        if k not in discard_properties:
                            if isinstance(v, list) or isinstance(v, tuple):
                                j = []
                                for x in v:
                                    if (foundSingleton == Grouping.NONE) and isinstance(x, SetOfSingletons) and (
                                            x.type != Grouping.NOT):
                                        assert len(x.entities) == 1
                                        foundSingleton = x.type
                                        argument = x.entities[0]
                                        forKey = k
                                    elif isinstance(x, str) or isinstance(x, Formula):
                                        j.append(x)
                                    elif isinstance(x, Singleton):
                                        j.append(self.make_arg(x))
                                    elif isinstance(x, SetOfSingletons):
                                        if x.type == Grouping.NOT:
                                            if isinstance(x.entities[0], Singleton):
                                                j.append(make_not(self.make_arg(x.entities[0])))
                                            else:
                                                raise RuntimeError(f"Unknown argument type: {x}")
                                        else:
                                            raise RuntimeError(f"Unknown argument type: {x}")
                                    else:
                                        raise RuntimeError(f"Unknown argument type: {x}")
                                if len(j) > 0:
                                    p[k] = tuple(j)
                            else:
                                p[k] = v
                    props_to_merge = []
                    src_old_props = get_props(src)
                    src = self.make_arg(src)
                    p["qa"] = []
                    p["dst"] = []
                    if src is not None:
                        p["qa"].append(src)
                        if src_old_props is not None:
                            props_to_merge.append((src, src_old_props))
                        if hasattr(src, "properties"):
                            props_to_merge.append((src, src.properties))
                    dst_old_props = get_props(dst)
                    # dst_old_props = dst.get_props() if dst is not None else None
                    dst = self.make_arg(dst)
                    if "ENTITY" in p and len(p["ENTITY"])==1:
                        orig_dst = dst
                        dst = dst.instantiate_variable_with_entity(p["ENTITY"][0])
                        if dst != orig_dst:
                            del p["ENTITY"]
                    if dst is not None:
                        p["dst"].append(dst)
                        if dst_old_props is not None:
                            props_to_merge.append((dst, dst_old_props))
                        if hasattr(dst, "properties"):
                            props_to_merge.append((dst, dst.properties))
                    prop = self.make_properties(p)
                    del prop["qa"]
                    del prop["dst"]
                    prop = self.derive_kernel_properties(props_to_merge, prop)
                    props_to_merge.append((None, prop))
                    prop = deepcopy(prop)
                    if "SENTENCE" in prop:
                        del prop["SENTENCE"]
                    if src.name.lower() in bogus_src and rel.lower() == "be":
                        if src.cop is None:
                            result = self.make_unary(rel, dst, score, prop)
                        else:
                            src = src.cop
                    if result is None:
                        if dst == bogus_dst:
                            result = self.make_unary(rel, src, score, prop)
                        else:
                            result = self.make_binary(rel, src, dst, score, prop)

                    other_props = []
                    if isinstance(result, FUnaryPredicate) or isinstance(result, FBinaryPredicate):
                        arg = None
                        if isinstance(result, FBinaryPredicate):
                            if result.rel == "have":
                                arg = result.src
                            else:
                                arg = result.dst
                        else:
                            if result.rel != "be":
                                arg = result.arg
                        other_props = self.derive_external_relationships(props_to_merge, arg)
                    if len(other_props) > 0:
                        other_props.append(result)
                        result = make_and(other_props)
                    return self.onFoundSingleton(foundSingleton, result, prop, forKey, argument, src, rel, negated, score, dst)
        else:
            if negated:
                return make_not(self.make_prop(src, rel, False, score, properties, None))
            else:
                argument = None
                forKey = None
                foundSingleton = Grouping.NONE
                p = dict()
                for k, v in properties.items():
                    if k not in discard_properties:
                        if isinstance(v, list) or isinstance(v, tuple):
                            j = []
                            for x in v:
                                if (foundSingleton == Grouping.NONE) and isinstance(x, SetOfSingletons) and (
                                        x.type != Grouping.NOT):
                                    assert len(x.entities) == 1
                                    foundSingleton = x.type
                                    argument = x.entities[0]
                                    forKey = k
                                elif isinstance(x, str) or isinstance(x, Formula):
                                    j.append(x)
                                elif isinstance(x, Singleton):
                                    j.append(self.make_arg(x))
                                elif isinstance(x, SetOfSingletons):
                                    if x.type == Grouping.NOT:
                                        if isinstance(x.entities[0], Singleton):
                                            j.append(make_not(self.make_arg(x.entities[0])))
                                        else:
                                            raise RuntimeError(f"Unknown argument type: {x}")
                                    else:
                                        raise RuntimeError(f"Unknown argument type: {x}")
                                else:
                                    raise RuntimeError(f"Unknown argument type: {x}")
                            if len(j) > 0:
                                p[k] = tuple(j)
                        else:
                            p[k] = v
                p["qa"] = []
                p["dst"] = []
                props_to_merge = []
                src_old_props = get_props(src)
                src = self.make_arg(src)
                if src is not None:
                    p["qa"].append(src)
                    if src_old_props is not None:
                        props_to_merge.append((src, src_old_props))
                    if hasattr(src, "properties"):
                        props_to_merge.append((src, src.properties))
                prop = self.make_properties(p)
                del prop["qa"]
                if "dst" in prop:
                    del prop["dst"]
                prop = self.derive_kernel_properties(props_to_merge, prop)
                props_to_merge.append((None, prop))
                prop = deepcopy(prop)
                if "SENTENCE" in prop:
                    del prop["SENTENCE"]
                result = self.make_unary(rel, src, score, prop)
                other_props = []
                if isinstance(result, FUnaryPredicate) or isinstance(result, FBinaryPredicate):
                    arg = None
                    if isinstance(result, FUnaryPredicate):
                        arg = result.arg
                    elif result.rel == "have":
                        arg = result.src
                    else:
                        arg = result.dst
                    other_props = self.derive_external_relationships(props_to_merge, arg)
                if len(other_props) > 0:
                    other_props.append(result)
                    result = make_and(other_props)
                return self.onFoundSingleton(foundSingleton, result, prop, forKey, argument, src, rel, negated, score, dst)

    def src_make_prop(self, src, rel, negated, score, properties, dst):
        if src is not None:
            if isinstance(src, SetOfSingletons):
                if src.type == Grouping.AND:
                    return make_and(map(lambda x: self.src_make_prop(x, rel, negated, score, properties, dst), src.entities))
                elif src.type == Grouping.OR:
                    return make_or(map(lambda x: self.src_make_prop(x, rel, negated, score, properties, dst), src.entities))
                elif src.type == Grouping.NOT:
                    return make_not(self.src_make_prop(list(src.entities)[0], rel, negated, score, properties, dst))
                elif src.type == Grouping.NEITHER:
                    return make_and(map(lambda x: make_not(self.src_make_prop(x, rel, negated, score, properties, dst)), src.entities))
                else:
                    n = src.type.name
                    raise RuntimeError(f"Unknown source type: {n}")
            else:
                return self.make_prop(src, rel, negated, score, properties, dst)
        else:
            return self.make_prop(None, rel, negated, score, properties, dst)

    def props_as_unique_itemset(self, prop):
        d = dict()
        for k, v in prop.items():
            if k in discard_properties or len(k) == 0 or isinstance(v, str):
                continue
            if len(v)==1 or k in Grouping.__members__.keys() or k == "SPECIFICATION":
                d[k] = tuple(set(v))
            else:
                assert len(v)==2
                type_per_item = []
                negated_args = []
                for idx, x in enumerate(v):
                    if isinstance(x, FNot):
                        negated_args.append(idx)
                    else:
                        assert isinstance(x, FVariable)
                    type_per_item.append(self.p.most_specific_type(list(set(type_atom(x)))))
                if "VERB" in type_per_item:
                    d[k] = tuple(set(v))
                else:
                    mst = self.p.most_specific_type(type_per_item)
                    idx = 0 if type_per_item[0] == mst else 1
                    opp = 1-idx
                    if type_per_item[opp] == mst:
                        ## TODO: Both belong to the same type. It might be an error with the interpretation if I have a negation
                        assert False
                    else:
                        if idx in negated_args:
                            neg_arg = v[idx].arg
                            assert (neg_arg.specification is None) or len(neg_arg.specification) == 0
                            d[k] = (FNot(FVariable(neg_arg.name, neg_arg.type, v[opp].name if not opp in negated_args else v[opp].arg.name, neg_arg.cop, neg_arg.id,
                                              neg_arg.properties, spec_negation=opp in negated_args)),)
                        else:
                            assert (v[idx].specification is None) or len(v[idx].specification) == 0
                            d[k] = (FVariable(v[idx].name, v[idx].type, v[opp].name if not opp in negated_args else v[opp].arg.name, v[idx].cop,  v[idx].id, v[idx].properties, spec_negation=opp in negated_args), )
        return frozenset(d.items())

    def rewrite_kernels(self, obj=None) -> Formula:
        if obj is None:
            obj = self.obj
        main_pop = obj.kernel
        properties = obj.properties
        from LaSSI.ner.MergeSetOfSingletons import merge_multiway_static_properties
        if main_pop.edgeLabel is not None:
            properties = merge_multiway_static_properties(properties, main_pop.edgeLabel.properties)
        rel = main_pop.edgeLabel.named_entity if main_pop.edgeLabel is not None else "None"  # TODO: Do we want string "None" or None? (e.g. when we have no verb?)
        negated = main_pop.isNegated

        score = main_pop.edgeLabel.confidence if main_pop.edgeLabel is not None else 1
        if main_pop.source is not None:
            score *= main_pop.source.confidence
        if main_pop.target is not None:
            score *= main_pop.target.confidence

        source = main_pop.source
        target = main_pop.target
        if rel == "be" and is_existential(target):
            if len(target.properties) > 0:
                source = source.update_node_props(merge_multiway_static_properties(main_pop.source.properties, target.properties))
            target = None
        return self.src_make_prop(source, rel, negated, score, dict(properties), target)

    def onFoundSingleton(self, foundSingleton, result, prop, forKey, argument, src, rel, negated, score, dst):
        if foundSingleton == Grouping.NONE:
            return result
        else:
            dcp = copy.deepcopy(prop)
            dcp[forKey] = (argument,)
            if foundSingleton == Grouping.AND:
                return make_and([result, self.make_prop(src, rel, negated, score, dcp, dst)])
            elif foundSingleton == Grouping.OR:
                return make_or([result, self.make_prop(src, rel, negated, score, dcp, dst)])
            else:
                n = dst.type.name
                raise RuntimeError(f"Unknown source type: {n}")


def rewrite_kernels(obj, meudb, useId=False):
    r = RewriteKernels(obj, meudb, useId)
    tmp = r.rewrite_kernels()
    return tmp
