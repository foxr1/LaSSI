import networkx as nx
import math
import itertools
from LaSSI.ner.MergeSetOfSingletons import merge_properties, GraphNER_withProperties
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons, Grouping
from LaSSI.structures.kernels.Sentence import is_kernel_in_props, case_in_props

class NodeMerger:
    def __init__(self, dependency_role_rewriter, honk, existentials, is_simplistic_rewriting, meu_db_row, node_functions, shouldDrawGraphs=False):
        self.dependency_role_rewriter = dependency_role_rewriter
        self.honk = honk
        self.existentials = existentials
        self.is_simplistic_rewriting = is_simplistic_rewriting
        self.meu_db_row = meu_db_row
        self.node_functions = node_functions
        self.shouldDrawGraphs = shouldDrawGraphs

    def merge(self, G):
        nodes_to_remove = []

        compound_prt_edges = [edge for edge in G.edges(data=True) if edge[2]['label'].named_entity in ['compound_prt']]
        for edge in compound_prt_edges:
            source = G.nodes[edge[0]]['data']
            target = G.nodes[edge[1]]['data']
            parts = [source, target]

            sorted_entities = sorted(parts, key=lambda x: (x.min_f(), x.pos_f()))
            sorted_entity_names = [x.get_name() for x in sorted_entities]

            all_types = list(map(getattr, sorted_entities, itertools.repeat('type')))
            specific_type = self.honk.most_specific_type(all_types)
            name = " ".join(sorted_entity_names)

            new_node = Singleton(
                id=source.id,
                named_entity=name,
                properties=source.properties,
                min=min(parts, key=lambda x: x.min).min,
                max=max(parts, key=lambda x: x.max).max,
                type=specific_type,
                confidence=1
            )
            nx.set_node_attributes(G, {source.id: new_node}, 'data')
            nodes_to_remove.append(target.id)

        merge_edges = [edge for edge in G.edges(data=True) if edge[2]['label'].named_entity in ['orig', 'compound', 'conj', 'appos']]
        for idx, edge in enumerate(merge_edges):
            if edge[0] not in G or edge[1] not in G:
                continue

            nx.set_node_attributes(G, {
                edge[0]: G.nodes[edge[0]]['data'].type if edge[2]['label'].named_entity in ['orig'] else edge[2]['label'].named_entity
            }, 'type')

            if G.nodes[edge[0]]['type'] == Grouping.NONE:
                nx.set_node_attributes(G, {
                    edge[0]: Grouping.AND if 'conj' in G.nodes[edge[0]]['data'].get_props() else Grouping.NONE
                }, 'type')

            if edge[0] in G and edge[1] in G:
                if self.dependency_role_rewriter.preserve_dependency_role(G, edge):
                    if edge[1] not in nodes_to_remove:
                        nodes_to_remove.append(edge[1])
                    continue

                G = nx.contracted_nodes(G, edge[0], edge[1], self_loops=False)

                if ((
                    idx + 1 < len(merge_edges) and merge_edges[idx + 1][0] != edge[0] and merge_edges[idx + 1][1] != edge[0] or
                        (
                            idx + 1 < len(merge_edges) and merge_edges[idx + 1][0] == edge[0] and
                            merge_edges[idx + 1][2]['label'].named_entity != edge[2]['label'].named_entity
                        )
                ) or idx == len(merge_edges) - 1):
                    node = G.nodes[edge[0]]
                    node_type = node['type']

                    grouped_nodes = (lambda f: f(f, node))(lambda f, node: ([node['data']] if 'data' in node and hasattr(node['data'], 'type') and not (isinstance(node['data'], Singleton) and 'conj' in dict(node['data'].properties)) else []) + [item for sub_node in node.get('contraction', {}).values() for item in f(f, sub_node)])

                    for n in grouped_nodes[1:]:
                        if grouped_nodes[0].id in G.nodes:
                            root_node = G.nodes[grouped_nodes[0].id]['data']

                            if isinstance(root_node, SetOfSingletons) and root_node.type == Grouping.NOT and len(
                                    root_node.entities) == 1:
                                root_node = root_node.entities[0]

                            if isinstance(root_node, Singleton):
                                new_properties = merge_properties(root_node.get_props(), n.get_props(), {'begin', 'end', 'pos'})
                                nx.set_node_attributes(G, {root_node.id: root_node.update_node_props(new_properties)}, 'data')

                    if node['type'] == 'conj':
                        cc_nodes = [edge[1] for edge in G.out_edges(edge[0], data=True) if edge[2]['label'].named_entity == 'cc']
                        if len(cc_nodes) > 0:
                            cc_node = cc_nodes[0]
                            node_type = self._get_group_enum(G.nodes[cc_node]['data'].named_entity)
                            nodes_to_remove.append(cc_node)
                        else:
                            node_type = Grouping.AND
                    elif node['type'] == 'existential':
                        gid = grouped_nodes[0].id
                        grouped_nodes = [G.nodes[gid]['data'] if gid in G.nodes else grouped_nodes[0]]
                    else:
                        node_type = self._get_group_enum(node_type) if not isinstance(node_type, Grouping) and self._get_group_enum(node_type) is not Grouping.NONE else node_type if isinstance(node_type, Grouping) else Grouping.GROUPING

                    norm_confidence = 1.0

                    if len(grouped_nodes) > 1:
                        new_node = SetOfSingletons(
                            id=node['data'].id,
                            type=node_type,
                            entities=tuple(grouped_nodes),
                            min=min(grouped_nodes, key=lambda x: x.min).min,
                            max=max(grouped_nodes, key=lambda x: x.max).max,
                            confidence=norm_confidence * math.prod([node.confidence for node in grouped_nodes]),
                            root=any(map(is_kernel_in_props, grouped_nodes))
                        )

                        # For a *compound* merge, pass the dependency governor
                        # (grouped_nodes[0] = node['data'], the edge[0] the merged
                        # compound edges point from) so GraphNER can correct a
                        # named-entity geo modifier that wrongly usurped the head
                        # (only when the governor is not itself a place).
                        compound_head_hint = (
                            grouped_nodes[0]
                            if edge[2]['label'].named_entity == 'compound' else None
                        )
                        new_node = GraphNER_withProperties(
                            new_node,
                            self.is_simplistic_rewriting,
                            self.meu_db_row,
                            self.honk,
                            self.existentials,
                            head_hint=compound_head_hint
                        ) if node_type == Grouping.GROUPING else new_node
                    else:
                        new_node = grouped_nodes[0]

                    if new_node.id not in list(G.nodes):
                        G.add_node(new_node.id, data=new_node)
                    else:
                        nx.set_node_attributes(G, {node['data'].id: new_node}, 'data')
                    self._keepDataKey(G, node['data'].id)

                    if isinstance(new_node, Singleton):
                        from LaSSI.ner.TypeResolver import TypeResolver
                        resolver = TypeResolver(self.meu_db_row, self.honk)
                        nx.set_node_attributes(G, {node['data'].id: resolver.nodeTypeResolution(new_node, resolver.associateNodeToBestMeuMatch(new_node), G)}, 'data')

                    for parent_id in [n for n in [edge[0] for edge in G.in_edges(node['data'].id)] if ((isinstance(G.nodes[n]['data'], Singleton) and G.nodes[n]['data'].named_entity == 'but') or (isinstance(G.nodes[n]['data'], SetOfSingletons) and getattr(G.nodes[n]['data'].entities[0], 'named_entity', None) == 'but')) and n not in nodes_to_remove]:
                        nodes_to_remove.append(node['data'].id)

                        negation_nodes = [
                            e for e in G.out_edges(
                                [n for n in [edge[0] for edge in G.in_edges(node['data'].id)]
                                 if (isinstance(G.nodes[n]['data'], Singleton) and G.nodes[n]['data'].named_entity == 'but') or (isinstance(G.nodes[n]['data'], SetOfSingletons) and getattr(G.nodes[n]['data'].entities[0], 'named_entity', None) == 'but')][0], data=True
                            ) if e[2]['label'].named_entity == 'neg'
                        ]

                        if len(negation_nodes) > 0:
                            new_node = SetOfSingletons(
                                id=new_node.id,
                                type=Grouping.NOT,
                                entities=tuple([new_node]),
                                min=new_node.min,
                                max=new_node.max,
                                confidence=new_node.confidence,
                                root=any(map(is_kernel_in_props, [new_node]))
                            )
                            nodes_to_remove.extend([m[1] for m in negation_nodes])

                        nx.set_node_attributes(G, {parent_id: SetOfSingletons(
                            id=parent_id,
                            type=Grouping.AND,
                            entities=tuple([new_node]),
                            min=new_node.min,
                            max=new_node.max,
                            confidence=new_node.confidence,
                            root=new_node.root
                        )}, 'data')

                    self.check_for_negations(G, nodes_to_remove)

        self.check_for_negations(G, nodes_to_remove)

        G.remove_nodes_from(nodes_to_remove)

        G.remove_nodes_from([node for node in nx.isolates(G) if (
            isinstance(G.nodes[node]['data'], Singleton) and
            'kernel' not in dict(G.nodes[node]['data'].properties) and
            not case_in_props(dict(G.nodes[node]['data'].properties))
        )]) if len(G.nodes()) > 1 else None

        if len(G.nodes()) > 1:
            root_node_ids = [
                nid for nid in G.nodes()
                if isinstance(G.nodes[nid]['data'], Singleton) and is_kernel_in_props(G.nodes[nid]['data'])
            ]
            if root_node_ids:
                root_id = root_node_ids[0]
                for node_id in list(nx.isolates(G)):
                    if node_id == root_id:
                        continue
                    node_data = G.nodes[node_id]['data']
                    inner_data = node_data
                    if isinstance(node_data, SetOfSingletons) and node_data.entities:
                        inner_data = node_data.entities[0]
                        if isinstance(inner_data, SetOfSingletons) and inner_data.entities:
                            inner_data = inner_data.entities[0]
                    if isinstance(inner_data, Singleton) and case_in_props(dict(inner_data.properties)):
                        obl_label = Singleton(
                            id=self.node_functions.fresh_id(),
                            named_entity='obl',
                            properties=frozenset(),
                            min=-1,
                            max=-1,
                            type='non_verb',
                            confidence=1.0
                        )
                        G.add_edge(root_id, node_id, label=obl_label, isNegated=False)

        if self.shouldDrawGraphs:
            self._draw_graph(G)

        return G

    def check_for_negations(self, G, nodes_to_remove):
        for e in [n for n in [edge for edge in G.edges(data=True)] if
                  G.nodes[n[1]]['data'].type == 'NEG' and n[1] not in nodes_to_remove]:
            node = G.nodes[e[0]]['data']
            neg_node = G.nodes[e[1]]['data']
            nodes_to_remove.append(e[1])

            if (isinstance(node, SetOfSingletons) and node.type == Grouping.AND
                    and neg_node.min >= 0):
                neg_pos = neg_node.min
                entities_before = [ent for ent in node.entities if ent.min <= neg_pos]
                entities_after = [ent for ent in node.entities if ent.min > neg_pos]
                if entities_after:
                    negated = [
                        SetOfSingletons(
                            id=ent.id,
                            type=Grouping.NOT,
                            entities=tuple([ent]),
                            min=ent.min,
                            max=ent.max,
                            confidence=ent.confidence,
                            root=is_kernel_in_props(ent)
                        )
                        for ent in entities_after
                    ]
                    new_node = SetOfSingletons(
                        id=node.id,
                        type=Grouping.AND,
                        entities=tuple(entities_before + negated),
                        min=node.min,
                        max=node.max,
                        confidence=node.confidence,
                        root=node.root
                    )
                else:
                    new_node = SetOfSingletons(
                        id=node.id,
                        type=Grouping.NOT,
                        entities=tuple([node]),
                        min=node.min,
                        max=node.max,
                        confidence=node.confidence,
                        root=any(map(is_kernel_in_props, [node]))
                    )
            else:
                new_node = SetOfSingletons(
                    id=node.id,
                    type=Grouping.NOT,
                    entities=tuple([node]),
                    min=node.min,
                    max=node.max,
                    confidence=node.confidence,
                    root=any(map(is_kernel_in_props, [node]))
                )

            nx.set_node_attributes(G, {new_node.id: new_node}, 'data')

    def _keepDataKey(self, G, node_id):
        node_attributes = G.nodes[node_id]
        keys_to_remove = [key for key in node_attributes if key != 'data']
        for key in keys_to_remove:
            del node_attributes[key]

    def _get_group_enum(self, name):
        if 'and' in name or 'but' in name or 'appos' in name:
            return Grouping.AND
        elif ('nor' in name) or ('neither' in name):
            return Grouping.NEITHER
        elif 'or' in name:
            return Grouping.OR
        elif 'not' in name:
            return Grouping.NOT
        elif 'multipleindobj' in name:
            return Grouping.MULTIINDIRECT
        else:
            return Grouping.NONE

    def _draw_graph(self, G):
        from LaSSI.ner.CreateInternalGraph import CreateInternalGraph
        dummy = CreateInternalGraph(False, None)
        dummy.shouldDrawGraphs = True
        dummy.drawNetworkXGraph(G)
