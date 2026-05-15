import math
import networkx as nx
import json
import os
from LaSSI.structures.internal_graph.EntityRelationship import Singleton
from LaSSI.structures.kernels.Sentence import case_in_props

class GraphSanitiser:
    def __init__(self, shouldDrawGraphs=False):
        self.shouldDrawGraphs = shouldDrawGraphs


    def sanitise(self, G):
        G = self.sanitiseCompoundChainsByPosition(G)
        G = self.dropSpuriousApposEdges(G)
        G = self.reconnectOrphanComponents(G)
        G = self.resolveMultipleInDobj(G)
        return G

    def sanitiseCompoundChainsByPosition(self, G):
        if G.number_of_nodes() == 0:
            return G

        def _pos_of(node):
            if node is None or not hasattr(node, 'properties'):
                return None
            props = dict(node.properties)
            raw = props.get('pos')
            if raw is None:
                return None
            try:
                return int(float(raw))
            except (TypeError, ValueError):
                return None

        pos_to_node_id = {}
        for nid, data in G.nodes(data=True):
            pos = _pos_of(data.get('data'))
            if pos is not None:
                pos_to_node_id[pos] = nid

        edges_to_remove = []
        edges_to_add = []

        for src, dst, key, edge_data in G.edges(data=True, keys=True):
            label = edge_data.get('label')
            if label is None:
                continue
            edge_name = label.named_entity if hasattr(label, 'named_entity') else str(label)
            if edge_name != 'compound':
                continue

            src_node = G.nodes[src].get('data')
            dst_node = G.nodes[dst].get('data')
            src_pos = _pos_of(src_node)
            dst_pos = _pos_of(dst_node)
            if src_pos is None or dst_pos is None:
                continue

            if abs(src_pos - dst_pos) <= 1:
                continue

            if abs(src_pos - dst_pos) == 2:
                mid_pos = min(src_pos, dst_pos) + 1
                mid_id = pos_to_node_id.get(mid_pos)
                if mid_id is not None:
                    mid_node = G.nodes[mid_id].get('data')
                    if mid_node is not None:
                        props = dict(mid_node.properties) if getattr(mid_node, 'properties', None) else {}
                        if str(getattr(mid_node, 'type', '')).upper() == 'PUNCT' or \
                           str(props.get('pos', '')).upper() in ['PUNCT', ','] or \
                           getattr(mid_node, 'named_entity', '') in ',;:()[]{}':
                            continue

            edges_to_remove.append((src, dst, key))

            right_id = pos_to_node_id.get(dst_pos + 1)
            if right_id is None or right_id == src or right_id == dst:
                continue
            right_node = G.nodes[right_id].get('data')
            if right_node is None or getattr(right_node, 'type', None) != 'noun':
                continue

            new_label = Singleton(
                id=right_id,
                named_entity='compound',
                properties=frozenset(),
                min=right_node.min,
                max=right_node.max,
                type='non_verb',
                confidence=right_node.confidence,
            )
            edges_to_add.append((right_id, dst, new_label))

        for src, dst, key in edges_to_remove:
            if G.has_edge(src, dst, key):
                G.remove_edge(src, dst, key)

        for src, dst, label in edges_to_add:
            G.add_edge(src, dst, label=label, isNegated=False)

        if self.shouldDrawGraphs:
            self._draw_graph(G)

        return G

    def dropSpuriousApposEdges(self, G):
        def _has_determiner(node_id):
            node = G.nodes[node_id].get('data')
            if isinstance(node, Singleton):
                props = dict(node.properties) if node.properties else {}
                if props.get('det'):
                    return True
            for _, child_id, edge_data in G.out_edges(node_id, data=True):
                label = edge_data.get('label')
                if label is not None and label.named_entity == 'det':
                    return True
                if child_id in G:
                    child = G.nodes[child_id].get('data')
                    if isinstance(child, Singleton) and child.type == 'det':
                        return True
            return False

        edges_to_remove = []
        for src, dst, key, data in G.edges(data=True, keys=True):
            label = data.get('label')
            if label is None or label.named_entity != 'appos':
                continue
            src_data = G.nodes[src].get('data')
            if not isinstance(src_data, Singleton):
                continue
            src_props = dict(src_data.properties) if src_data.properties else {}
            source_is_pp_head = case_in_props(src_props) or src_props.get('punct') == ','
            if source_is_pp_head and _has_determiner(dst):
                edges_to_remove.append((src, dst, key))
        for src, dst, key in edges_to_remove:
            if G.has_edge(src, dst, key):
                G.remove_edge(src, dst, key)
        return G

    def reconnectOrphanComponents(self, G):
        if G.number_of_nodes() == 0:
            return G

        kernel_id = None
        for nid, data in G.nodes(data=True):
            node = data.get('data')
            if node is None or not hasattr(node, 'properties'):
                continue
            if 'kernel' in dict(node.properties):
                kernel_id = nid
                break
        if kernel_id is None:
            return G

        undirected = G.to_undirected()
        primary_component = nx.node_connected_component(undirected, kernel_id)

        for component in nx.connected_components(undirected):
            if component == primary_component:
                continue
            head_id = None
            for nid in component:
                node = G.nodes[nid].get('data')
                if node is None or getattr(node, 'type', None) != 'noun':
                    continue
                in_edges = [e for e in G.in_edges(nid) if e[0] in component]
                out_edges = list(G.out_edges(nid))
                if len(in_edges) == 0 and len(out_edges) > 0:
                    head_id = nid
                    break
            if head_id is None:
                continue

            kernel_node = G.nodes[kernel_id]['data']
            edge_singleton = Singleton(
                id=kernel_id,
                named_entity='nmod',
                properties=frozenset(dict().items()),
                min=kernel_node.min,
                max=kernel_node.max,
                type='non_verb',
                confidence=kernel_node.confidence,
            )
            G.add_edge(kernel_id, head_id, label=edge_singleton, isNegated=False)

        return G

    def resolveMultipleInDobj(self, G):
        nodes_to_remove = []
        multiindobj_nodes = [node for node in G.nodes(data=True) if node[1]['data'].type == 'multipleindobj']
        for node in multiindobj_nodes:
            h_id = node[0]
            h_data = node[1]['data']
            h_props = dict(h_data.properties)

            in_edges = list(G.in_edges(h_id, data=True))
            out_edges = list(G.out_edges(h_id, data=True))

            for child_edge in out_edges:
                child_id = child_edge[1]
                child_data = G.nodes[child_id]['data']

                for parent_edge in in_edges:
                    G.add_edge(parent_edge[0], child_id, label=parent_edge[2]['label'], isNegated=parent_edge[2]['isNegated'])

                if h_props:
                    child_props = dict(child_data.properties)
                    for k, v in h_props.items():
                        if k == 'orig': continue
                        if k not in child_props:
                            child_props[k] = v
                    new_child_data = child_data.update_node_props(child_props)
                    nx.set_node_attributes(G, {child_id: new_child_data}, 'data')

            nodes_to_remove.append(h_id)

        G.remove_nodes_from(nodes_to_remove)

        if self.shouldDrawGraphs:
            self._draw_graph(G)

        return G

    def _draw_graph(self, G):
        from LaSSI.ner.CreateInternalGraph import CreateInternalGraph
        dummy = CreateInternalGraph(False, None)
        dummy.shouldDrawGraphs = True
        dummy.drawNetworkXGraph(G)
