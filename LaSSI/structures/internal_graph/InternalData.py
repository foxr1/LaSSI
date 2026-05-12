import dataclasses
from dataclasses import dataclass
from typing import List

from LaSSI.structures.internal_graph.EntityRelationship import Singleton, deserialize_NodeEntryPoint
from LaSSI.structures.internal_graph.Graph import Graph

@dataclass
class InternalRepresentation:
    internal_graph: Graph
    sentences: List[Singleton] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, data):
        return deserialize_NodeEntryPoint(data)
        # return cls(
        #     internal_graph=Graph.from_dict(data.get('internal_graph')),
        #     sentences=[Singleton.from_dict(x) for x in data.get('sentences')]
        # )
