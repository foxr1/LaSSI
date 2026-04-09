import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

with open("cn_edge_mappings.json", "r") as f:
    mappings = json.load(f)

class HuggingFace:
    def __init__(self, model=None):
        if model is None:
            model = 'sentence-transformers/all-MiniLM-L6-v2'
        self.model = SentenceTransformer(model)

    def getEmbedding(self, x):
        return self.model.encode(x)

    def string_similarity(self, x: str, y: str) -> float:  # between 0 and 1
        x_vec = self.model.encode(x)
        y_vec = self.model.encode(y)
        strictSim = float(util.pairwise_dot_score(x_vec, y_vec))
        if (strictSim > np.finfo(float).eps):  # ReLU
            return strictSim
        return 0.0


s = HuggingFace()
h = s.string_similarity
# print(h("hey there", "hi there"))
# print(h("form_of", "FormOf"))
# print(h("form_of", "DistinctFrom"))


extract_fields_edges = ["form_of", "part_of", "eq", "not_eq", "shares_isa_with", "isa", "related_to"]


class ConceptNet5RelationType():
    Antonym = 0
    AtLocation = 1
    CapableOf = 2
    Causes = 3
    CausesDesire = 4
    CreatedBy = 5
    DefinedAs = 6
    DerivedFrom = 7
    Desires = 8
    DistinctFrom = 9
    Entails = 10
    EtymologicallyDerivedFrom = 11
    EtymologicallyRelatedTo = 12
    ExternalURL = 13
    FormOf = 14
    HasA = 15
    HasContext = 16
    HasFirstSubevent = 17
    HasLastSubevent = 18
    HasPrerequisite = 19
    HasProperty = 20
    HasSubevent = 21
    InstanceOf = 22
    IsA = 23
    LocatedNear = 24
    MadeOf = 25
    MannerOf = 26
    MotivatedByGoal = 27
    NotCapableOf = 28
    NotDesires = 29
    NotHasProperty = 30
    NotUsedFor = 31
    ObstructedBy = 32
    PartOf = 33
    ReceivesAction = 34
    RelatedTo = 35
    SimilarTo = 36
    SymbolOf = 37
    Synonym = 38
    UsedFor = 39
    capital = 40
    field = 41
    genre = 42
    genus = 43
    influencedBy = 44
    knownFor = 45
    language = 46
    leader = 47
    occupation = 48
    product = 49


conceptnet_edges = list(vars(ConceptNet5RelationType).keys())[1:-3]
# print(conceptnet_edges)

# edges_to_remove = {}
# edge_changes = {}
# for extract_field_edge in extract_fields_edges:
#     print(extract_field_edge)
#     for conceptnet_edge in conceptnet_edges:
#         similarity = h(extract_field_edge, conceptnet_edge)
#         if similarity < 0.5: continue
#
#         if conceptnet_edge in edges_to_remove:
#             edges_to_remove[conceptnet_edge] += 1
#         else:
#             edges_to_remove[conceptnet_edge] = 1
#             edge_changes[conceptnet_edge] = extract_field_edge
#
# print(edges_to_remove)  ### it prints {'FormOf': 1, 'PartOf': 1, 'IsA': 2, 'RelatedTo': 1}
#
#

labels_without_mapping = set()

def get_edge(edge_label):
    # if current_edge in extract_fields_edges:
    #     return current_edge
    label = mappings.get(edge_label)
    if not label: labels_without_mapping.add(edge_label)
    return label

def show_non_mapped_labels():
    print("Labels that havent been mapped:")
    print(len(labels_without_mapping))
    for l in labels_without_mapping:
        print(l)
    # so on the first run it showed SubwordOf, with_pos, with_sense, inflection


