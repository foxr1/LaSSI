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
