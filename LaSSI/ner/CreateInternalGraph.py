__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

from collections import defaultdict
import networkx as nx

from LaSSI.external_services.Services import Services
from LaSSI.ner.SemanticRoleRewriting import DependencyRoleRewriter
from LaSSI.ner.node_functions_X import NodeFunctions
from LaSSI.structures.internal_graph.EntityRelationship import Singleton

from LaSSI.ner.GraphBuilder import GraphBuilder
from LaSSI.ner.GraphSanitiser import GraphSanitiser
from LaSSI.ner.TypeResolver import TypeResolver
from LaSSI.ner.GraphPreprocessor import GraphPreprocessor
from LaSSI.ner.NodeMerger import NodeMerger


class CreateInternalGraph:
    def __init__(self, is_simplistic_rewriting, meu_db_row):
        self.negations = {'not', 'no'}
        self.nodes = dict()
        self.edges = None
        self.services = Services.getInstance()
        self.existentials = self.services.getExistentials()
        self.is_simplistic_rewriting = is_simplistic_rewriting
        self.meu_db_row = meu_db_row
        self.shouldDrawGraphs = False

    def runGraphCreation(self, gsm_json, honk):
        self.max_id = max(map(lambda x: int(x["id"]), gsm_json)) + 1
        self.node_functions = NodeFunctions(self.max_id)
        self.honk = honk
        self.dependency_role_rewriter = DependencyRoleRewriter(honk)

        from LaSSI.ner.TypeResolver import _filter_spurious_meus
        self.meu_db_row = _filter_spurious_meus(self.meu_db_row, honk)

        builder = GraphBuilder(self.existentials, self.honk, self.shouldDrawGraphs)
        sanitiser = GraphSanitiser(self.shouldDrawGraphs, self.honk)
        type_resolver = TypeResolver(self.meu_db_row, self.honk)
        preprocessor = GraphPreprocessor(self.node_functions, self.existentials, self.honk, self.shouldDrawGraphs)
        merger = NodeMerger(
            self.dependency_role_rewriter,
            self.honk,
            self.existentials,
            self.is_simplistic_rewriting,
            self.meu_db_row,
            self.node_functions,
            self.shouldDrawGraphs
        )

        # Phase 1
        G = builder.build(gsm_json, self.node_functions)

        # Phases 1.05, 1.1, 1.2, 1.25, 1.5
        G = sanitiser.sanitise(G)

        # Phases 2, 2.1, 2.5
        G = type_resolver.resolve(G)

        # Phase 3
        G = preprocessor.preprocess(G)

        # Phase 4
        G = merger.merge(G)

        return G

    def drawNetworkXGraph(self, G):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, seed=42, k=0.8)
        node_size = 2500

        # Draw nodes and node labels
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="skyblue")
        node_labels = {node: f"[{data['data'].type}] {node}, {Singleton.get_node_string(data['data'])}" for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

        # Draw edges with different curvatures for parallel edges
        ax = plt.gca()
        edge_groups = defaultdict(list)
        for u, v, key in G.edges(keys=True):
            edge_groups[(u, v)].append(key)

        for (u, v), keys in edge_groups.items():
            n = len(keys)
            for i, key in enumerate(keys):
                if n > 1:
                    curvature = (i // 2 + 1) * 0.15
                    if i % 2 != 0:
                        curvature *= -1  # Curve other way
                else:
                    curvature = 0  # Straight line for single edges

                connection_style = f'arc3,rad={curvature}'
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v, key)],
                    connectionstyle=connection_style,
                    arrows=True,
                    arrowsize=25,
                    node_size=node_size,
                    min_target_margin=15,  # Gap for arrow
                    ax=ax
                )

        # Combine labels for parallel edges and draw them
        combined_edge_labels = defaultdict(list)
        for u, v, data in G.edges(data=True):
            combined_edge_labels[(u, v)].append(data.get('label', '').named_entity)

        # Join labels with newlines for display
        final_edge_labels = {k: '\n'.join(v) for k, v in combined_edge_labels.items()}

        nx.draw_networkx_edge_labels(G, pos, edge_labels=final_edge_labels, font_color='red')

        if hasattr(self, 'meu_db_row'):
            plt.title(f"\"{self.meu_db_row.first_sentence}\"")
        plt.show()
