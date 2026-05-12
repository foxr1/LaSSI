__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox, Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, DefaultDict

from LaSSI.structures import DependencyRoles


class Grouping(Enum):
    AND = 0
    OR = 1
    NEITHER = 2
    NOT = 3
    NONE = 4
    GROUPING = 5
    MULTIINDIRECT = 6


# class Properties(TypedDict):  # Key-Value association
#     property: str  # Key
#     value: Union[int, str, float, bool]  # Value


class NodeEntryPoint:
    pass


@dataclass(order=True, frozen=True, eq=True)
class SingletonProperties():
    begin: str
    end: str
    pos: str
    specification: str = None
    number: str = None
    extra: str = None

def replaceNamed(entity: 'Singleton', s: str) -> 'Singleton':
    return Singleton(id=entity.id,
                     named_entity=s,
                     properties=entity.properties,
                     min=entity.min,
                     max=entity.max,
                     type=entity.type,
                     confidence=entity.confidence)


@dataclass(order=True, frozen=True, eq=True)
class SetOfSingletons(NodeEntryPoint):  # Graph node representing conjunction/disjunction/exclusion between entities
    id: int
    type: Grouping  # Type of node grouping
    entities: List[NodeEntryPoint]  # A list of entity nodes
    min: int
    max: int
    confidence: float
    root: bool = False

    @property
    def properties(self):
        from LaSSI.ner.node_functions import create_props_for_singleton
        return create_props_for_singleton(self.get_props())

    def min_f(self):
        return min(int(self.min), min([x.min_f() for x in self.entities if x is not None]))

    def max_f(self):
        return max(int(self.max), max([x.max_f() for x in self.entities if x is not None]))

    def pos_f(self):
        return min([x.pos_f() for x in self.entities if x is not None])

    def update_map(self, dmin, dmax, dpos):
        if self.entities is not None:
            for x in self.entities:
                dmin, dmax, dpos = x.update_map(dmin, dmax, dpos)
        m = self.min_f()
        M = self.max_f()
        p = self.pos_f()
        dmin[self.id] = min(dmin[self.id], m)
        dpos[self.id] = min(dpos[self.id], p)
        dmax[self.id] = max(dmax[self.id], M)
        return dmin, dmax, dpos

    def extract_properties(self, p):
        if (self.entities is None) or len(self.entities) == 0:
            yield from []
        else:
            for x in self.entities:
                yield from x.extract_properties(p)

    def update_entities(self, new_entities):
        return SetOfSingletons(
            id=self.id,
            type=self.type,
            entities=tuple(new_entities),
            min=self.min,
            max=self.max,
            confidence=self.confidence,
            root=self.root
        )

    def strip_root_properties(self):
        return SetOfSingletons(
            id=self.id,
            type=self.type,
            entities=tuple(
                entity.strip_root_properties() if entity is not None and hasattr(entity, "strip_root_properties") else entity
                for entity in self.entities
            ),
            min=self.min,
            max=self.max,
            confidence=self.confidence,
            root=False
        )

    def get_props(self, properties=None):
        from LaSSI.ner.MergeSetOfSingletons import merge_properties
        if properties is None:
            properties = dict()
        for entity in self.entities:
            if entity is not None and hasattr(entity, "get_props"):
                properties = merge_properties(properties, entity.get_props())
        return properties

    def update_node_props(self, node_props):
        new_entities = []
        for entity in self.entities:
            if entity is not None and hasattr(entity, "update_node_props"):
                new_entities.append(entity.update_node_props(node_props))
            else:
                new_entities.append(entity)
        return self.update_entities(new_entities)

    def add_property(self, prop_key, prop_value):
        node_props = self.get_props()
        node_props[prop_key] = prop_value
        return self.update_node_props(node_props)

    def get_name(self):
        # sorted_entities = sorted(self.entities, key=lambda x: float(dict(x.properties)['pos']))
        sorted_entity_names = list(
            map(lambda x: x.named_entity if hasattr(x, 'named_entity') else x.get_name() if hasattr(x, "get_name") else str(x), self.entities))
        return f" {self.type}".join(sorted_entity_names)

    def get_node_properties_string(self, node_to_use):
        if node_to_use is None or not isinstance(node_to_use, Singleton):
            return ''

        props_to_ignore = ['begin', 'pos', 'end', 'kernel', 'lemma', 'specification', 'number', 'root', 'expl', 'cc',
                           'conj', 'neg', 'adv', 'subjpass']
        properties_list = defaultdict(list)
        for key in dict(node_to_use.properties):
            if key not in props_to_ignore:
                properties_key_ = dict(node_to_use.properties)[key]
                if isinstance(properties_key_, str) and properties_key_ != '':
                    try:
                        float(key)
                        continue  # Skip float-keyed preposition properties (handled by assign_kernel)
                    except ValueError:
                        pass

                    properties_list[key].append(properties_key_)
                else:
                    if isinstance(properties_key_, Singleton):
                        properties_list[key].append(self.get_node_string(properties_key_))
                    elif isinstance(properties_key_, SetOfSingletons):
                        properties_list[key].append(self.get_node_string(properties_key_))
                    else:
                        for node in properties_key_:
                            if key == 'SENTENCE':  # It is a node with kernel (most likely)
                                properties_list[key].append(self.to_string(node))
                            else:
                                if key in {'nmod', 'nmod_poss', 'acl_relcl'}:
                                    properties_list[key].append(self.get_node_string(node))
                                else:
                                    properties_list[key].append(self.get_node_string(node))

        return re.sub(r"(nmod|nmod_poss|acl_relcl):\1", r"\1",
                      f"""[{", ".join(f'({k}:{v[0] if len(v) == 1 else "[" + ", ".join(v) + "]"})' for k, v in properties_list.items())}]""" if len(
                          properties_list) > 0 else '')

    def get_node_string(self, node=None):
        if node is None:
            node = self

        node_string = node.named_entity if node is not None and isinstance(node, Singleton) else 'None'

        # If node_string is empty, it is a kernel so return that
        if node_string == '':
            node_string = node.to_string(node) if node_string == "" else node_string
        else:
            # Join SetOfSingletons
            node_string = f"{node.type.name}({', '.join(self.get_node_string(entity) for entity in node.entities)})" if isinstance(
                node, SetOfSingletons) and node_string == 'None' else node_string

            # Add properties
            node_string = f"{node_string}{self.get_node_properties_string(node)}"
        return node_string


def _deserialize_property_value(value):
    if isinstance(value, list):
        return tuple(_deserialize_property_value(x) for x in value)
    if isinstance(value, dict):
        if ('entities' in value and 'id' in value) or 'named_entity' in value:
            return deserialize_NodeEntryPoint(value)
        return {k: _deserialize_property_value(v) for k, v in value.items()}
    return value


def _deserialize_properties(properties):
    if properties is None:
        return frozenset()
    return frozenset((k, _deserialize_property_value(v)) for k, v in properties.items())


def _deserialize_relationship(data):
    if data is None:
        return None
    return Relationship(
        source=deserialize_NodeEntryPoint(data.get('source')),
        target=deserialize_NodeEntryPoint(data.get('target')),
        edgeLabel=deserialize_NodeEntryPoint(data.get('edgeLabel')),
        isNegated=data.get('isNegated', False),
    )


def deserialize_NodeEntryPoint(data: dict) -> NodeEntryPoint:
    if data is None:
        return data
    if 'entities' in data:
        return SetOfSingletons(id=int(data.get('id')),
                               type=Grouping(int(data.get('type'))),
                               entities =tuple(map(deserialize_NodeEntryPoint, data.get('entities'))),
            min=int(data.get('min')),
            max=int(data.get('max')),
            confidence=float(data.get('confidence')
                                             ))
    elif 'named_entity' in data:
        return Singleton(
            id=int(data.get('id')),
            named_entity=data.get('named_entity'),
            properties=_deserialize_properties(data.get('properties')),
            min=int(data.get('min')),
            max=int(data.get('max')),
            confidence=float(data.get('confidence')),
                             type=data.get('type'),
            kernel=_deserialize_relationship(data.get('kernel')),
                             )



@dataclass(order=True, frozen=True, eq=True)
class Relationship:  # Representation of an edge
    source: NodeEntryPoint  # Source node
    target: NodeEntryPoint  # Target node
    edgeLabel: 'Singleton'  # Edge label, also represented as an entity with properties
    isNegated: bool = False  # Whether the edge expresses a negated action


    def min_f(self):
        return min([x.min_f() for x in [self.source, self.target, self.edgeLabel] if x is not None])

    def pos_f(self):
        return min([x.pos_f() for x in [self.source, self.target, self.edgeLabel] if x is not None])

    def max_f(self):
        return max([x.max_f() for x in [self.source, self.target, self.edgeLabel] if x is not None])

    def update_map(self, dmin, dmax, dpos):
        if self.source is not None:
            dmin, dmax, dpos = self.source.update_map(dmin, dmax, dpos)
        if self.target is not None:
            dmin, dmax, dpos = self.target.update_map(dmin, dmax, dpos)
        if self.edgeLabel is not None:
            dmin, dmax, dpos = self.edgeLabel.update_map(dmin, dmax, dpos)
        return dmin, dmax, dpos

    def extract_properties(self, p):
        if (self.source is None) and (self.target is None):
            yield from []
        else:
            if self.source is not None:
                yield from self.source.extract_properties(p)
            if self.target is not None:
                yield from self.target.extract_properties(p)

    @staticmethod
    def from_nodes(r, nodes):
        return Relationship(source=nodes[r.source.id] if r.source.id >= 0 else r.source, target=nodes[r.target.id] if r.target.id >= 0 else r.target, edgeLabel=r.edgeLabel, isNegated=r.isNegated)

    @classmethod
    def from_dict(cls, c):
        return cls(source=deserialize_NodeEntryPoint(c.get('source')),
                   target=deserialize_NodeEntryPoint(c.get('target')),
                   edgeLabel=deserialize_NodeEntryPoint(c.get('edgeLabel')),
                   isNegated=bool(c.get('Singleton', False))
                   )

    def update_vertex(self, node, vertex_type):
        if vertex_type == 'source':
            return Relationship(
                source=node,
                target=self.target,
                edgeLabel=self.edgeLabel,
                isNegated=self.isNegated
            )
        elif vertex_type == 'target':
            return Relationship(
                source=self.source,
                target=node,
                edgeLabel=self.edgeLabel,
                isNegated=self.isNegated
            )
        elif vertex_type == 'edgeLabel':
            return Relationship(
                source=self.source,
                target=self.target,
                edgeLabel=node,
                isNegated=self.isNegated
            )




@dataclass(order=True, frozen=True, eq=True)
class Singleton(NodeEntryPoint):  # Graph node representing just one entity
    id: int
    named_entity: str  # String representation of the entity
    properties: frozenset  # Key-Value association for such entity
    min: int
    max: int
    type: str
    confidence: float
    kernel: Relationship = None

    def min_f(self):
        return self.min

    def pos_f(self):
        if self.properties is None:
            return 0
        else:
            for k, v in self.properties:
                if k == "pos":
                    return int(float(v))
            return 0

    def max_f(self):
        return self.max

    def update_map(self, dmin, dmax, dpos):
        if self.kernel is not None:
            dmin, dmax, dpos = self.kernel.update_map(dmin, dmax, dpos)
        mins = [int(self.min),dmin[self.id]]
        maxs = [int(self.max),dmax[self.id]]
        dposes = [self.pos_f(),dpos[self.id]]
        if self.properties is not None:
            for k, v in self.properties:
                if k =="begin":
                    mins.append(int(float(v)))
                elif k ==  "end":
                    maxs.append(int(float(v)))
                elif k ==  "pos":
                    dposes.append(int(float(v)))
                elif isinstance(v, tuple):
                   for x in v:
                       if isinstance(x, Singleton) or isinstance(x, Relationship) or isinstance(x, SetOfSingletons):
                            dmin, dmax, dpos = x.update_map(dmin, dmax, dpos)
                elif isinstance(v, Singleton) or isinstance(v, Relationship) or isinstance(v, SetOfSingletons):
                    dmin, dmax, dpos = v.update_map(dmin, dmax, dpos)
        dmin[self.id] = min(mins)
        dpos[self.id] = min(dposes)
        dmax[self.id] = max(maxs)
        return dmin, dmax, dpos

    def get_name(self):
        return self.named_entity

    def extract_properties(self, p):
        if ((self.properties is None) or len(self.properties) == 0) and self.kernel is None:
            yield from []
        else:
            for k, v in self.properties:
                if isinstance(v, list) or isinstance(v, tuple):
                   for x in v:
                       if p(k, x):
                           yield k, x
                       if type(x).__name__ == 'Relationship' or type(x).__name__ == 'Singleton' or type(x).__name__ == 'SetOfSingletons':
                            yield from x.extract_properties(p)
                elif p(k, v):
                    yield k, v
                if type(v).__name__ == 'Relationship' or type(v).__name__ == 'Singleton' or type(
                            v).__name__ == 'SetOfSingletons':
                    yield from v.extract_properties(p)
            if self.kernel is not None:
                yield from self.kernel.extract_properties(p)

    def get_props(self, properties=None):
        return dict(self.properties)

    def update_node_props(self, node_props):
        from LaSSI.ner.node_functions import create_props_for_singleton
        return Singleton(
            id=self.id,
            named_entity=self.named_entity,
            min=self.min,
            max=self.max,
            type=self.type,
            confidence=self.confidence,
            kernel=self.kernel,
            properties=create_props_for_singleton(node_props),
        )

    def update_name(self, new_name):
        return Singleton(
            id=self.id,
            named_entity=new_name,
            min=self.min,
            max=self.max,
            type=self.type,
            confidence=self.confidence,
            kernel=self.kernel,
            properties=self.properties,
        )

    def update_type(self, new_type):
        return Singleton(
            id=self.id,
            named_entity=self.named_entity,
            min=self.min,
            max=self.max,
            type=new_type,
            confidence=self.confidence,
            kernel=self.kernel,
            properties=self.properties,
        )

    def update_kernel(self, new_node, kernel_part):
        return Singleton(
            id=self.id,
            named_entity='',
            type='SENTENCE',
            min=self.min,
            max=self.max,
            confidence=1,
            kernel=Relationship(
                source=new_node if kernel_part == "source" else self.kernel.source,
                target=new_node if kernel_part == "target" else self.kernel.target,
                edgeLabel=new_node if kernel_part == "edgeLabel" else self.kernel.edgeLabel,
                isNegated=self.kernel.isNegated,
            ),
            properties=self.properties,
        )

    def remove_prop(self, prop_name):
        node_props = dict(self.properties)
        if prop_name in node_props:
            node_props.pop(prop_name)
        return self.update_node_props(node_props)

    def add_property(self, prop_key, prop_value):
        if isinstance(self, SetOfSingletons):
            return SetOfSingletons(
                id=self.id,
                type=self.type,
                entities=self.entities,
                min=self.min,
                max=self.max,
                confidence=self.confidence,
                root=True
            )

        node_props = dict(self.properties)
        node_props[prop_key] = prop_value
        return self.update_node_props(node_props)

    def strip_root_properties(self):
        sing_props = dict(self.properties)
        if 'kernel' in sing_props:
            sing_props.pop('kernel')
        if 'root' in sing_props:
            sing_props.pop('root')

        return Singleton(
            id=self.id,
            named_entity=self.named_entity,
            properties=frozenset(sing_props.items()),
            min=self.min,
            max=self.max,
            type=self.type,
            confidence=self.confidence,
            kernel=self.kernel,
        )

    @classmethod
    def from_dict(cls, c):
        import dacite
        c["properties"] = frozenset(c["properties"].items())
        dacite.from_dict(Singleton, c)

    def get_node_properties_string(self, node_to_use):
        if node_to_use is None or not isinstance(node_to_use, Singleton):
            return ''

        # 'adv' is consumed by phrasal-verb construction (check_for_adv). Any remaining
        # 'adv' string property was not a phrasal verb and is spurious display noise.
        props_to_ignore = ['begin', 'pos', 'end', 'kernel', 'lemma', 'specification', 'number', 'root', 'expl', 'adv',
                           'subjpass']
        properties_list = defaultdict(list)
        for key in dict(node_to_use.properties):
            if key not in props_to_ignore:
                properties_key_ = dict(node_to_use.properties)[key]
                if isinstance(properties_key_, str) and properties_key_ != '':
                    try:
                        float(key)
                        key = int(float(key))
                    except ValueError:
                        pass

                    properties_list[key].append(properties_key_)
                else:
                    if isinstance(properties_key_, Singleton):
                        properties_list[key].append(self.get_node_string(properties_key_))
                    elif isinstance(properties_key_, SetOfSingletons):
                        properties_list[key].append(self.get_node_string(properties_key_))
                    elif properties_key_ is not None:
                        for node in properties_key_:
                            if key == 'SENTENCE':  # It is a node with kernel (most likely)
                                properties_list[key].append(self.to_string(node))
                            else:
                                if key in {'nmod', 'nmod_poss', 'acl_relcl'}:
                                    properties_list[key].append(self.get_node_string(node))
                                else:
                                    properties_list[key].append(self.get_node_string(node))

        return re.sub(r"(nmod|nmod_poss|acl_relcl):\1", r"\1",
                      f"""[{", ".join(f'({k}:{v[0] if len(v) == 1 else "[" + ", ".join(v) + "]"})' for k, v in properties_list.items())}]""" if len(
                          properties_list) > 0 else '')

    def get_node_string(self, node=None):
        if node is None:
            node = self

        node_string = node.named_entity if node is not None and isinstance(node, Singleton) else 'None'

        # If node_string is empty, it is a kernel so return that
        if node_string == '':
            node_string = node.to_string(node) if node_string == "" else node_string
        else:
            # Join SetOfSingletons
            node_string = f"{node.type.name}({', '.join(self.get_node_string(entity) for entity in node.entities)})" if isinstance(
                node, SetOfSingletons) and node_string == 'None' else node_string

            # Add properties
            node_string = f"{node_string}{self.get_node_properties_string(node)}"
        return node_string

    # Rewrite Singleton(kernel) in form edgeLabel[props](source[props], target[props])[props]
    def to_string(self, node=None):
        if node is None:
            node = self

        if node.kernel:
            edge_label = self.get_node_string(node.kernel.edgeLabel) if node.kernel.edgeLabel is not None else 'None'
            source = self.get_node_string(node.kernel.source) if node.kernel.source is not None else 'None'
            target = self.get_node_string(node.kernel.target) if node.kernel.target is not None else 'None'

            properties = self.get_node_properties_string(node)

            if edge_label == 'be' and node.kernel.target is not None:
                from LaSSI.external_services.Services import Services
                p = Services.getInstance().getHOnK()
                target_name = node.kernel.target.named_entity
                is_verb_in_ontology = False
                if target_name:
                    is_verb_in_ontology = (target_name in p.state_verbs or target_name in p.transitive_verbs or 
                                           target_name in p.movement_verbs or target_name in p.phrasal_verbs or 
                                           target_name in p.causative_verbs or target_name in p.semi_modal_verbs or 
                                           target_name in p.means_verbs or target_name in p.materialisation_verbs)
                
                if node.kernel.target.type in DependencyRoles.copula_complement_pos_tags() or is_verb_in_ontology:
                    return f"{target}(?, {source}){properties}" if not node.kernel.isNegated else f"NOT({target}(?, {source}){properties})"

            return f"{edge_label}({source}, {target}){properties}" if not node.kernel.isNegated else f"NOT({edge_label}({source}, {target}){properties})"
        else:
            return node.named_entity
