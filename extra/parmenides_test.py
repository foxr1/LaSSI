import logging

from LaSSI.structures.extended_fol.TBoxReasoning import KnowledgeExpansion, non_redundant_constituents
from LaSSI.structures.extended_fol.TabularCWASemantics import TabularCWASemantics
from LaSSI.structures.extended_fol.Formulae import FUnaryPredicate, FVariable, FBinaryPredicate, formula_from_dict
from FunctionalMatch.language.LanguageMainPoint import parse_query

from LaSSI.structures.extended_fol.rewrite_kernels import make_and


def build_mock_test_sentences():
    traffic = FVariable("traffic", "Noun", None, None, 1)
    city_center = FVariable("city center", "Noun", None, None, 1)
    adj1 = FVariable("busy", "JJ", None, None, 1)
    adj2 = FVariable("fast", "JJ", None, None, 1)
    ncl = FVariable("Newcastle", "LOC", None, None, 1)
    fast_traffic = FVariable("traffic", "Noun", None, adj2, 1)
    datum = FUnaryPredicate("flow", traffic, 1.0, frozenset({"ciao": "giacomo"}.items()))
    datum2 = FBinaryPredicate("have", city_center, adj1, -1, frozenset())
    datum3 = FBinaryPredicate("have", city_center, city_center, -1, frozenset())
    datum4 = FBinaryPredicate("have", city_center, adj2, -1, frozenset())
    datum5 = FUnaryPredicate("be", fast_traffic, 1.0, frozenset({"ciao": "giacomo"}.items()))
    datum6 = FBinaryPredicate("have", ncl, traffic, -1, frozenset())
    datum7 = FUnaryPredicate("be", adj2, -1, frozenset({"SPACE": ncl}.items()))


    sentences = [datum, datum2, datum3, datum4, datum5, datum6, datum7]

    return sentences

def load_paper_sentences():
    with open("/home/giacomo/projects/LaSSI/catabolites/all_newcastle/logical_rewriting.json", "r") as f:
        import json
        data = json.load(f)
    assert isinstance(data, list)
    return [formula_from_dict(dct) for dct in data]
    # for dct in data:
    #     this = formula_from_dict(dct)
    #     print(this)

def tst_query(sentences = None):
    ## INIT PARMENIDES
    import os
    parmenides = "parmenides.ttl" if os.path.isfile("parmenides.ttl") else None
    from LaSSI.HOnK.HOnK import HOnKSingleton
    logging.basicConfig(level=logging.INFO)
    HOnKSingleton.instance()
    HOnKSingleton.init("/home/giacomo/PyCharmProjects/FunctionalMatch/data/cache", "giacomo", "omocaig",
                             "localhost", 5432, False, parmenides)

    # queries = parse_query("/query_impl.txt")

    if sentences is None:
        sentences = load_paper_sentences() #[load_paper_sentences()[-1]]

    from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
    print("Init TBox Reasoning Rules")
    TBoxReasoningSingleton.instance()
    # TBoxReasoningSingleton.init("/home/giacomo/PyCharmProjects/FunctionalMatch/query_test.txt",
    #                             "/home/giacomo/PyCharmProjects/FunctionalMatch/query_test.txt")
    TBoxReasoningSingleton.init("/home/giacomo/PyCharmProjects/FunctionalMatch/query_impl.txt",
                                "/home/giacomo/PyCharmProjects/FunctionalMatch/query_eq.txt",
                                "/home/giacomo/PyCharmProjects/FunctionalMatch/data/cache/_kexp.pickle")

    print("Initializing all the data and performing the expansion")
    cwa = TabularCWASemantics(sentences, "/home/giacomo/PyCharmProjects/FunctionalMatch/data/cache")

    print("Building Report!")
    TBoxReasoningSingleton.dump()

    # l1 = Node.leaf(1)
    # l2 = Node.leaf(2)
    # l3 = Node.leaf(3)
    # s1 = Node(0, l2, l3)
    # datum = Node(5, s1, l1)
    # result = ObjDepthDeterminer.get_depth_dictionary(datum)
    # print(knowledge_expansion(datum, queries))
    # print(knowledge_expansion(datum2, queries))
    # print(knowledge_expansion(datum3, queries))
    # print(knowledge_expansion(datum4, queries))
    # print(knowledge_expansion(datum6, queries))
    # print(knowledge_expansion(datum7, queries))
    # for query in queries:
    #     result = query([datum, datum2, datum3])
    #     print(result)

    ## STOP PARMENIDES
    HOnKSingleton.stop()

def parmenides_db_write():
    from LaSSI.HOnK.HOnK import HOnK
    logging.basicConfig(level=logging.INFO)
    p = HOnK("/home/giacomo/PyCharmProjects/FunctionalMatch/data/cache","giacomo", "omocaig", "localhost", 5432, onStorage=False)
    import os

    p.start("parmenides.ttl" if os.path.isfile("parmenides.ttl") else None)
    #generate_parmenides_graph(p, "/home/giacomo/PyCharmProjects/FunctionalMatch/data")
    p.stop()

def simpler_test():
    # adj2 = FVariable("fast", "JJ", None, None, 1)
    # adj3 = FVariable("busy", "JJ", None, None, 20)
    # adj4 = FVariable("crowd", None, None, None, 20)
    # ncl = FVariable("Newcastle", "LOC", None, None, 31)
    # becl = FUnaryPredicate("be", adj3, -1, frozenset({"SPACE": ncl}.items()))
    # have2 = FBinaryPredicate("have", ncl, adj4, 1.0, frozenset())
    Sentence = load_paper_sentences()[6]
    print(Sentence)

    ## INIT PARMENIDES
    import os
    parmenides = "parmenides.ttl" if os.path.isfile("parmenides.ttl") else None
    from LaSSI.HOnK.HOnK import HOnKSingleton
    logging.basicConfig(level=logging.INFO)
    HOnKSingleton.instance()
    HOnKSingleton.init("/home/giacomo/PyCharmProjects/FunctionalMatch/data/cache", "giacomo", "omocaig",
                             "localhost", 5432, False, parmenides)
    queries = parse_query("/home/giacomo/PyCharmProjects/FunctionalMatch/query_test.txt")
    ke = KnowledgeExpansion("/home/giacomo/PyCharmProjects/FunctionalMatch/data/cache/_kexp.pickle")
    ke.pruned_expansion(Sentence, queries, "implDebug", filter=non_redundant_constituents)

    # from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import knowledge_expansion
    for k,v in ke.fullGraph().items():
        for idx, element in v:
            print(element)

    ## STOP PARMENIDES
    HOnKSingleton.stop()


# def formulae_rewriting_test():
#     # import os
#     # parmenides = "parmenides.ttl" if os.path.isfile("parmenides.ttl") else None
#     # from FunctionalMatch.example.parmenides.Parmenides import HOnKSingleton
#     # logging.basicConfig(level=logging.INFO)
#     # HOnKSingleton.instance()
#     # HOnKSingleton.init("/home/giacomo/PyCharmProjects/FunctionalMatch/data/cache","giacomo", "omocaig", "localhost", 5432, False, parmenides)
#
#     # l = HOnKSingleton.get().getOutgoingNodes("flow", "adjectivalForm")
#     # HOnKSingleton.stop()


if __name__ == '__main__':
    var = FVariable("?1", "existential", None, None, 1)
    cable = FVariable("cable", "ENTITY", None, None, 1)
    he = FVariable("equipment", "ENTITY", "home entertainment", None, 1)
    q1 = FBinaryPredicate("isA", var, he, -1, frozenset())
    q2 = FBinaryPredicate("require", var, cable, -1, frozenset())
    what_home_entertainment = print(make_and([q1, q2]))
    # formulae_rewriting_test()
    # tst_query()
    # simpler_test()