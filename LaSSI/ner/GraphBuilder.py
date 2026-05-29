import networkx as nx
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping

from LaSSI.ner.string_functions import does_string_have_negations

class GraphBuilder:
    def __init__(self, existentials, honk, shouldDrawGraphs=False):
        self.existentials = existentials
        self.honk = honk
        self.negations = {'not', 'no'}
        self.shouldDrawGraphs = shouldDrawGraphs

    def build(self, gsm_json, node_functions):
        self._promote_detached_passive_content_root(gsm_json)

        for gsm_item in gsm_json:
            amods = []
            puncts = []
            keys_to_remove = []
            if 'properties' not in gsm_item:
                continue
            for k, v in gsm_item['properties'].items():
                if str(k).startswith('amod_'):
                    val = v[0] if isinstance(v, list) and v else v
                    amods.append(val)
                    keys_to_remove.append(k)
                elif str(k).startswith('punct_'):
                    val = v[0] if isinstance(v, list) and v else v
                    puncts.append(val)
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del gsm_item['properties'][k]
            if amods:
                if 'amod' in gsm_item['properties']:
                    existing = gsm_item['properties']['amod']
                    existing_list = list(existing) if isinstance(existing, (list, tuple)) else [existing]
                    gsm_item['properties']['amod'] = tuple(existing_list + amods)
                else:
                    gsm_item['properties']['amod'] = tuple(amods)
            if puncts:
                if 'punct' in gsm_item['properties']:
                    existing = gsm_item['properties']['punct']
                    existing_list = list(existing) if isinstance(existing, (list, tuple)) else [existing]
                    gsm_item['properties']['punct'] = tuple(existing_list + puncts)
                else:
                    gsm_item['properties']['punct'] = tuple(puncts)

        G = nx.MultiDiGraph()

        # Create Singletons and add them as nodes
        for item in gsm_json:
            node_id = item.get('id')
            if node_id is not None:
                G.add_node(node_id, data=self.create_singleton(item))

        # Add all edges to the graph
        for item in gsm_json:
            for edge in item['phi']:
                score = edge.get('score', {})
                parent_id = score.get('parent')
                child_id = score.get('child')

                # Ensure these IDs exist in newly created list of nodes
                if parent_id in G and child_id in G:
                    edge_label_text = edge.get('containment', '').strip().replace(":", "_")

                    non_verbs = self.honk.getNonVerbs()
                    non_verb_set = {nv.strip() for nv in non_verbs}
                    if edge_label_text in non_verb_set:
                        edge_type = "non_verb"
                    else:
                        edge_type = "verb"

                    has_negations = does_string_have_negations(edge_label_text)
                    query_words = edge_label_text.split()
                    result_words = [word for word in query_words if word.lower() not in self.negations]
                    edge_label_text = ' '.join(result_words)
                    
                    # Ensure edge_label_text isn't empty if it consisted entirely of negations
                    if not edge_label_text and query_words:
                        edge_label_text = query_words[0]

                    G.add_edge(parent_id, child_id, label=Singleton(
                        id=parent_id,
                        named_entity=edge_label_text,
                        properties=frozenset(dict().items()),
                        min=G.nodes[parent_id]['data'].min,
                        max=G.nodes[parent_id]['data'].max,
                        type=edge_type,
                        confidence=G.nodes[parent_id]['data'].confidence
                    ), isNegated=has_negations)

        # Get the nodes in (reverse) lexicographical topological order and create new graph in that order
        G = node_functions.sort_G(G)

        if self.shouldDrawGraphs:
            self._draw_graph(G)

        return G

    def _promote_detached_passive_content_root(self, gsm_json):
        """Repair a grammar split where a passive side clause steals subjpass.

        DatagramDB can promote an oblique reduced-clause host as the passive
        root while leaving the real infinitival content as a detached root.
        The shape is structural: a `subjpass` node with an `acl` child plus a
        later root whose child is a nominal `multipleindobj` content group.
        Move the passive subject edge onto the detached content root so the
        kernel builder sees the same shape as ordinary passive-content cases.
        """
        by_id = {item.get('id'): item for item in gsm_json if item.get('id') is not None}

        def has_xi(item, value):
            return value in item.get('xi', [])

        def has_ell(item, value):
            return value in item.get('ell', [])

        def edge_child(edge):
            return edge.get('score', {}).get('child')

        def edge_parent(edge):
            return edge.get('score', {}).get('parent')

        def edge_label(edge):
            return str(edge.get('containment', '')).strip()

        def is_nominal_content_group(item):
            if item is None or not has_ell(item, 'multipleindobj'):
                return False
            orig_children = [
                by_id.get(edge_child(edge))
                for edge in item.get('phi', [])
                if edge_label(edge) == 'orig'
            ]
            return any(
                child is not None and child.get('ell', [None])[0] not in {'verb', 'TO', 'RB', '<dot>'}
                for child in orig_children
            )

        def content_group_max_pos(item):
            if item is None:
                return float('-inf')
            positions = []
            for edge in item.get('phi', []):
                if edge_label(edge) != 'orig':
                    continue
                child = by_id.get(edge_child(edge))
                if child is None:
                    continue
                try:
                    positions.append(float(child.get('properties', {}).get('pos', 'nan')))
                except ValueError:
                    pass
            return max(positions) if positions else float('-inf')

        passive_holders = []
        for item in gsm_json:
            if not has_xi(item, 'subjpass'):
                continue
            edges = item.get('phi', [])
            if not any(edge_label(edge) == 'acl' for edge in edges):
                continue
            transferable = [
                edge for edge in edges
                if edge_label(edge) not in {'acl', 'inherit_edge'}
            ]
            if transferable:
                passive_holders.append((item, transferable))

        if not passive_holders:
            return

        for root in gsm_json:
            if root.get('properties', {}).get('root') != 'root' or has_xi(root, 'subjpass'):
                continue
            content_edges = [
                edge for edge in root.get('phi', [])
                if is_nominal_content_group(by_id.get(edge_child(edge)))
            ]
            if not content_edges:
                continue

            content_max_pos = max(
                content_group_max_pos(by_id.get(edge_child(edge)))
                for edge in content_edges
            )
            matching_holders = [
                (holder, edges)
                for holder, edges in passive_holders
                if float(holder.get('properties', {}).get('pos', 'inf')) > content_max_pos
            ]
            if not matching_holders:
                continue

            _, passive_edges = matching_holders[0]
            if 'subjpass' not in root.setdefault('xi', []):
                root['xi'].append('subjpass')

            existing = {
                (edge_label(edge), edge_child(edge))
                for edge in root.get('phi', [])
            }
            for edge in passive_edges:
                key = (edge_label(edge), edge_child(edge))
                if key in existing:
                    continue
                copied = {
                    'containment': edge.get('containment', ''),
                    'content': edge.get('content'),
                    'properties': edge.get('properties', {}),
                    'score': dict(edge.get('score', {})),
                }
                copied['score']['parent'] = root.get('id')
                root.setdefault('phi', []).append(copied)
                existing.add(key)

    def create_singleton(self, gsm_item):
        min_value = -1
        max_value = -1
        if len(gsm_item['xi']) > 0 and gsm_item['xi'][0] != '' and ((len(gsm_item['ell']) > 0 and gsm_item['ell'][0] != '∃') or (len(gsm_item['ell']) == 0)):  
            name = gsm_item['xi'][0]

            if 'begin' in dict(gsm_item['properties'].items()):
                min_value = int(gsm_item['properties']['begin'])
                max_value = int(gsm_item['properties']['end'])
            node_type = gsm_item['ell'][0] if len(gsm_item['ell']) > 0 else "None"
            node_type = self._get_group_enum(gsm_item['properties']['conj']) if 'conj' in gsm_item['properties'] else node_type
        else:
            # xi might be empty if the node is invented, therefore existential
            name = "?" + str(self.existentials.increaseAndGetExistential())
            node_type = 'existential'

        # If we have "root" in "ell", add it to properties
        if len(gsm_item['ell']) > 1:
            gsm_item['properties']['kernel'] = gsm_item['ell'][1]
        elif gsm_item.get('properties', {}).get('root') == 'root':
            gsm_item['properties']['kernel'] = 'root'
            
        if len(gsm_item['xi']) > 1 and 'subjpass' in gsm_item['xi'][1]:
            gsm_item['properties']['subjpass'] = gsm_item['xi'][1]

        # Add 'det' to properties
        if len(gsm_item['ell']) > 0 and 'det' in gsm_item['ell']:
            gsm_item['properties']['det'] = gsm_item['ell'][0]

        # TODO: We are dropping this as it is currently just used for GSM parsing, no implementation here (yet)
        if 'xpos' in gsm_item['properties'].keys():
            gsm_item['properties'].pop('xpos')

        return Singleton(
            id=gsm_item['id'],
            named_entity=name,
            properties=frozenset(gsm_item['properties'].items()),
            min=min_value,
            max=max_value,
            type=node_type,
            confidence=1.0
        )

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
