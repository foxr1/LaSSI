from typing import List

from LaSSI.external_services.Services import Services
from LaSSI.ner.AssignTypeToSingleton import AssignTypeToSingleton
from LaSSI.ner.CreateFinalKernel import CreateFinalKernel
from LaSSI.ner.CreateFinalKernelX import CreateFinalKernelX
from LaSSI.ner.CreateInternalGraph import CreateInternalGraph
from LaSSI.structures.internal_graph.EntityRelationship import Singleton
from LaSSI.structures.internal_graph.Graph import Graph
from LaSSI.tests.test_assertions import TestLaSSI


class GraphProvenance:
    def __init__(self, gsm_json_graph, meu_db, is_simplistic_rewriting):
        self._internal_graph = None
        self._sentence = None
        self._nodes = None
        self._edges = None
        self.gsm_json_graph = gsm_json_graph
        self.is_simplistic_rewriting = is_simplistic_rewriting
        self.meu_db_row = meu_db
        self.graph_creation = CreateInternalGraph(is_simplistic_rewriting, meu_db)
        self.atts_global = AssignTypeToSingleton(is_simplistic_rewriting, meu_db)
        self.services = Services.getInstance()
        self.parmenides = self.services.getParmenides()
        self.existentials = self.services.getExistentials()

    def internal_graph(self) -> Graph:
        if self.is_simplistic_rewriting:
            # Phase 0-4
            self.gsm_json_graph = self.atts_global.groupGraphNodes(self.gsm_json_graph)
            # Phase 5
            self.atts_global.checkForNegation(self.gsm_json_graph)

            # Parmenides Information
            rejected_edges = self.parmenides.getRejectedVerbs()
            non_verbs = self.parmenides.getNonVerbs()

            # Now, create the internal graph from now created edges
            self._internal_graph = self.atts_global.constructIntermediateGraph(self.gsm_json_graph, rejected_edges,
                                                                               non_verbs)

            return self._internal_graph
        else:
            self.G = self.graph_creation.runGraphCreation(self.gsm_json_graph, self.parmenides)
            return self.G

    def sentence(self) -> Singleton:
        create_final_kernel_X = CreateFinalKernelX(self.G, self.graph_creation.negations, self.graph_creation.node_functions)
        self.x_sentence = create_final_kernel_X.constructSentence()

        # create_final_kernel = CreateFinalKernel(self.atts_global.nodes, self.gsm_json_graph, self.atts_global.edges, self.atts_global.negations, self.atts_global.node_functions)
        # self._sentence = create_final_kernel.constructSentence()

        # if (  # Sentences that have changed representation but might be better now?
        #         not TestLaSSI.compare_internal_representations(TestLaSSI(), self.x_sentence.to_string(), self._sentence.to_string()) and
        #         self.meu_db_row.first_sentence not in {
        #             'you score well on tests', 'a coin falls and rolls', 'grab it before it stops ringing',
        #             'be frightful and/or learning', 'privately administered region', 'come closer', 'the effect of dissolving the sugar',
        #             'the essence of seeing is everywhere', 'Saturdays have usually busy city centers', 'Newcastle has traffic but not in the city centre'
        #         }
        # ):
        #     assert False
        #     print("FAiL")

        return self.x_sentence
