import networkx as nx
import json
import os
from LaSSI.structures.internal_graph.EntityRelationship import Singleton
from LaSSI.ner.MergeSetOfSingletons import merge_properties

class GraphPreprocessor:
    def __init__(self, node_functions, existentials, honk, shouldDrawGraphs=False):
        self.node_functions = node_functions
        self.existentials = existentials
        self.honk = honk
        self.shouldDrawGraphs = shouldDrawGraphs
        self._load_preprocessing_rules()

    def _load_preprocessing_rules(self):
        self.rules = {}
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'raw_data', 'preprocessing_rules.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.rules = json.load(f)
        else:
            self.rules = {
                "edge_labels_to_merge_properties": ["inherit_edge"],
                "edge_labels_to_merge_and_add_target_name": ["mark", "punct", "amod", "advmod", "case", "adv"],
                "edge_labels_for_special_case_propagation": ["case"],
                "edge_labels_to_flip_and_make_existential": ["cop"]
            }

    def preprocess(self, G):
        nodes_to_remove = []
        edges_to_remove = []

        personal_pronouns = self.honk.getPersonalPronouns()
        if personal_pronouns:
            for node_id, node_data in list(G.nodes(data=True)):
                singleton = node_data.get('data')
                if not isinstance(singleton, Singleton):
                    continue
                raw_type = getattr(singleton, 'type', '')
                if not isinstance(raw_type, str):
                    continue
                node_type = (raw_type or '').upper()
                if node_type not in {'PRP', 'PRP$', 'PRONOUN'}:
                    continue
                props = dict(singleton.properties) if singleton.properties is not None else {}
                lemma = (props.get('lemma') or singleton.named_entity or '').lower()
                if lemma not in personal_pronouns:
                    continue
                preserved_props = {}
                for key in ('pos', 'begin', 'end', 'kernel', 'root', 'subjpass'):
                    if key in props:
                        preserved_props[key] = props[key]
                new_singleton = Singleton(
                    id=singleton.id,
                    named_entity="?" + str(self.existentials.increaseAndGetExistential()),
                    properties=frozenset(preserved_props.items()),
                    min=singleton.min,
                    max=singleton.max,
                    type='existential',
                    confidence=singleton.confidence,
                )
                nx.set_node_attributes(G, {node_id: new_singleton}, 'data')

        for hyph_node in [node[1]['data'] for node in G.nodes(data=True) if node[1]['data'].type == 'HYPH']:
            first_word = next((node for node in G.nodes(data=True) if node[1]['data'].max == hyph_node.min), None)
            second_word = next((node for node in G.nodes(data=True) if node[1]['data'].min == hyph_node.max), None)

            if first_word and second_word:
                nx.set_node_attributes(G, {first_word[0]: first_word[1]['data'].update_name(f"{first_word[1]['data'].named_entity}{hyph_node.named_entity}{second_word[1]['data'].named_entity}")}, 'data')
                nodes_to_remove.extend([hyph_node.id, second_word[0]])

                for node in [n[1]['data'] for n in G.nodes(data=True) if dict(second_word[1]['data'].properties)['pos'] in dict(n[1]['data'].properties)]:
                    nx.set_node_attributes(G, {node.id: node.remove_prop(dict(second_word[1]['data'].properties)['pos'])}, 'data')

        for edge in G.edges(data=True):
            if any([x for x in nodes_to_remove if x in edge]):
                continue

            source, target, edge_label = edge
            source_data = G.nodes[source]['data']
            target_data = G.nodes[target]['data']
            edge_label_name = edge_label['label'].named_entity

            if edge_label_name in self.rules.get("edge_labels_to_merge_properties", []):
                source_props = source_data.get_props()
                target_props = target_data.get_props()
                if not target_data.type in source_props:
                    new_properties = merge_properties(source_props, target_props, {'begin', 'end', 'pos'})
                    nx.set_node_attributes(G, {edge[0]: source_data.update_node_props(new_properties)}, 'data')
                if edge[1] not in nodes_to_remove:
                    nodes_to_remove.append(edge[1])
            elif edge_label_name in self.rules.get("edge_labels_to_merge_and_add_target_name", []):
                if isinstance(target_data, Singleton):
                    type_key = self.node_functions.get_node_type(target_data) if edge_label_name != 'case' else 'case'
                else:
                    type_key = self.honk.most_general_type(map(lambda x: x.type, target_data.entities))

                if type_key != 'existential':
                    target_props = target_data.get_props()
                    source_props = source_data.get_props()
                    new_properties = merge_properties(source_props, target_props, {'begin', 'end', 'pos'})
                    source_data = source_data.update_node_props(new_properties)
                    source_data = source_data.add_property(edge_label_name, target_data.get_name())
                    nx.set_node_attributes(G, {edge[0]: source_data}, 'data')
                    
                    if edge[1] not in nodes_to_remove:
                        edges_to_remove.append(edge)
                    
                    if edge_label_name in self.rules.get("edge_labels_for_special_case_propagation", []):
                        case_target_pos = dict(G.nodes[edge[1]]['data'].properties).get('pos')
                        if case_target_pos is not None:
                            source_data = source_data.add_property(case_target_pos, G.nodes[edge[1]]['data'].get_name())
                            nx.set_node_attributes(G, {edge[0]: source_data}, 'data')
                            
                        for conj_edge in list(G.out_edges(edge[1], data=True)):
                            if conj_edge[2]['label'].named_entity == 'conj' and conj_edge[1] in G.nodes:
                                conj_node = G.nodes[conj_edge[1]]['data']
                                conj_pos = dict(conj_node.properties).get('pos')
                                if conj_pos is not None:
                                    source_data = source_data.add_property(conj_pos, conj_node.get_name())
                                    nx.set_node_attributes(G, {edge[0]: source_data}, 'data')
                                for cc_edge in list(G.out_edges(conj_edge[1], data=True)):
                                    if cc_edge[2]['label'].named_entity == 'cc' and cc_edge[1] in G.nodes:
                                        cc_node = G.nodes[cc_edge[1]]['data']
                                        cc_pos = dict(cc_node.properties).get('pos')
                                        if cc_pos is not None:
                                            source_data = source_data.add_property(cc_pos, cc_node.get_name())
                                            nx.set_node_attributes(G, {edge[0]: source_data}, 'data')
                                        if cc_edge[1] not in nodes_to_remove:
                                            nodes_to_remove.append(cc_edge[1])
                                if conj_edge[1] not in nodes_to_remove:
                                    nodes_to_remove.append(conj_edge[1])
            elif edge_label_name in self.rules.get("edge_labels_to_flip_and_make_existential", []) and str(target_data.type).lower() == 'verb':
                G.add_edge(edge[1], edge[0], label=target_data, isNegated=edge[2]['isNegated'])
                
                has_existing_root = any(
                    G.nodes[n].get('data') is not None and
                    hasattr(G.nodes[n]['data'], 'properties') and
                    n != edge[1] and
                    ('kernel' in dict(G.nodes[n]['data'].properties) or 'root' in dict(G.nodes[n]['data'].properties))
                    for n in G.nodes
                )
                new_existential = self.node_functions.create_existential_node(G, edge[1])
                if not has_existing_root:
                    new_existential = new_existential.add_property('kernel', 'root')
                nx.set_node_attributes(G, {edge[1]: new_existential}, 'data')
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            G.remove_edge(*edge[:2])

        for node in nodes_to_remove:
            G = self._remove_node(G, node)

        if self.shouldDrawGraphs:
            self._draw_graph(G)

        return G

    def _remove_node(self, G, node):
        for parent in [edge for edge in G.in_edges(node, data=True)]:
            for child in [edge for edge in G.out_edges(node, data=True)]:
                G.add_edge(parent[0], child[1], label=child[2]['label'], isNegated=child[2]['isNegated'])

        G.remove_node(node)
        return G

    def _draw_graph(self, G):
        from LaSSI.ner.CreateInternalGraph import CreateInternalGraph
        dummy = CreateInternalGraph(False, None)
        dummy.shouldDrawGraphs = True
        dummy.drawNetworkXGraph(G)
