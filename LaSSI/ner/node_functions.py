from collections import deque, defaultdict
from LaSSI.external_services.Services import Services
from LaSSI.ner.string_functions import has_auxiliary
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping, Relationship


def create_existential_node(node_id=-1):
    return Singleton(
        id=node_id,
        named_entity="?" + str(Services.getInstance().getExistentials().increaseAndGetExistential()),
        properties=frozenset(dict().items()),
        min=-1,
        max=-1,
        type="existential",
        confidence=-1
    )


def create_props_for_singleton(properties):
    return frozenset({tuple(k) if isinstance(k, list) else k: tuple(v) if isinstance(v, list) else v for k, v in properties.items()}.items())


def get_min_position(node):
    if node is None:
        return None

    if isinstance(node, Singleton):
        node_props = dict(node.properties)
        if not 'pos' in node_props:
            return -1
        else:
            return int(float(node_props['pos']))
    else:
        positions = [
            pos for pos in map(lambda x: get_min_position(x), getattr(node, "entities", ()))
            if pos is not None and pos > -1
        ]
        return min(positions) if positions else -1


class NodeFunctions:
    def __init__(self, max_id):
        self.services = Services.getInstance()
        self.existentials = self.services.getExistentials()
        self.node_id_map = dict()
        self.max_id = max_id

    def get_gsm_item_from_id(self, gsm_id, gsm_json):
        gsm_id_json = {row['id']: row for row in gsm_json}  # Associate ID to key value
        if gsm_id in gsm_id_json:
            return gsm_id_json[gsm_id]
        else:
            return None

    def remove_gsm_item_by_id(self, gsm_id, gsm_json, ids_to_remove):
        gsm_id_json = {row['id']: idx for idx, row in enumerate(gsm_json)}  # Associate ID to key value
        if gsm_id in gsm_id_json:
            ids_to_remove.append(gsm_id_json[gsm_id])

    def get_node_parents(self, node, gsm_json, return_edge=False):
        parents = []
        number_of_nodes = range(len(gsm_json))
        for row in number_of_nodes:
            gsm_item = gsm_json[row]
            if len(gsm_item['phi']) > 0:
                for edge in gsm_item['phi']:
                    if (isinstance(node, Singleton) and self.get_node_id(edge['score']['child']) == node.id) or (
                            not isinstance(node, Singleton) and edge['score']['child'] == node['id']):
                        if return_edge:
                            parents.append([edge['score']['parent'], edge['containment']])
                        else:
                            parents.append(edge['score']['parent'])

        return parents

    def get_node_id(self, node_id):
        value = self.node_id_map.get(node_id, node_id)
        if value is not None and value in self.node_id_map.keys():  # Check if value exists and is a key
            return self.node_id_map.get(value)
        else:
            return value

    def get_original_node_id(self, node_id, nodes):
        for old_id, new_id in self.node_id_map.items():
            # If we have MULTIINDIRECT, we just want the original ID as the passed on
            if new_id == node_id and old_id in nodes and nodes[old_id].type == Grouping.MULTIINDIRECT and len(
                    nodes[old_id].entities) == 1:
                return node_id

            if new_id == node_id:
                return old_id
        return node_id

    def resolve_node_id(self, node_id, nodes):
        if node_id is None:
            return create_existential_node()
        else:
            return nodes[self.get_node_id(node_id)]

    def fresh_id(self):
        self.max_id += 1
        return self.max_id

    def fresh_id_and_add_to_node_map(self, id_to_map):
        fresh_id = self.fresh_id()
        self.node_id_map[id_to_map] = fresh_id
        return fresh_id

    # Insert at position to retain topological order
    def add_to_nodes_at_pos(self, nodes, new_node, pos):
        items = list(nodes.items())
        items.insert(pos, (new_node.id, new_node))
        return dict(items)

    def is_node_but(self, gsm_item):
        return gsm_item is not None and len(gsm_item['xi']) > 0 and 'but' in gsm_item['xi'][0].lower() and 'cc' in gsm_item['ell'][0].lower()

    def node_bfs(self, edges, root_node_id):
        nodes = defaultdict(set)
        for x in edges:
            nodes[x.source.id].add(x.target.id)

        visited = set()
        q = deque()  # Create a queue for BFS

        # Mark the source node as visited and enqueue it
        visited.add(root_node_id)
        q.append(root_node_id)

        while q:
            id = q.popleft()
            for dst in nodes[id]:
                if dst not in visited:
                    visited.add(dst)
                    q.append(dst)

        return visited

    def get_node_type(self, node):
        return node.type if isinstance(node.type, str) else node.type.name

    def get_valid_nodes(self, nodes):
        valid_nodes = []
        for node in nodes:
            if node is not None:
                valid_nodes.append(node)

        if len(valid_nodes) > 0:
            return valid_nodes
        else:
            return -1

    def get_max_from_nodes(self, nodes):
        valid_nodes = self.get_valid_nodes(nodes)
        return max(valid_nodes, key=lambda x: x.max).max if valid_nodes != -1 else -1

    def get_min_from_nodes(self, nodes):
        valid_nodes = self.get_valid_nodes(nodes)
        return min(valid_nodes, key=lambda x: x.min).min if valid_nodes != -1 else -1

    def check_node_coordinations_for_auxiliary(self, current_edge, all_edges):
        if 'cc' in dict(current_edge.source.properties):
            return any(map(has_auxiliary, [edge.edgeLabel.named_entity for edge in all_edges if edge.source.id == current_edge.source.id]))
        else:
            return has_auxiliary(current_edge.edgeLabel.named_entity)

    def convert_relationship_to_sentence(self, sentence_id, kernel, properties=None):
        valid_nodes = self.get_valid_nodes([kernel.source, kernel.target])
        return Singleton(
            id=sentence_id,
            named_entity="",
            type="SENTENCE",
            min=self.get_min_from_nodes(valid_nodes),
            max=self.get_max_from_nodes(valid_nodes),
            confidence=1,
            kernel=Relationship(
                source=kernel.source,
                target=kernel.target,
                edgeLabel=kernel.edgeLabel,
                isNegated=kernel.isNegated
            ),
            properties=frozenset() if properties is None else create_props_for_singleton(properties),
        )
