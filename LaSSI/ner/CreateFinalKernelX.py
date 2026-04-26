import itertools
import re
import string
from collections import defaultdict
from types import SimpleNamespace

import networkx as nx
import numpy

from LaSSI.external_services.Services import Services
from LaSSI.ner.MergeSetOfSingletons import merge_properties
from LaSSI.ner.HOnKLogicalRewriting import get_matching_logical_rules
from LaSSI.ner.node_functions_X import create_props_for_singleton, get_min_position, NodeFunctions
from LaSSI.ner.string_functions import is_label_verb, check_semi_modal, lemmatize_verb
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons, Relationship, Grouping
from LaSSI.structures.kernels.SentenceX import is_kernel_in_props, create_edge_kernel, create_existential, \
    is_node_in_kernel_nodes, create_sentence, case_in_props, find_action_ed_node_in_kernel, \
    rewrite_action_ed_node, get_prepositions


class CreateFinalKernelX:
    def __init__(self, G, negations, node_functions):
        self.services = Services.getInstance()
        self.existentials = self.services.getExistentials()
        self.negations = negations
        self.G = G
        self.node_functions = node_functions

    def constructSentence(self) -> Singleton:
        from LaSSI.structures.kernels.SentenceX import create_sentence

        # Phase 1
        true_targets, found_preposition_labels  = self.find_prepositions_and_true_targets()  # Get list of edges to be used for kernel

        # Phase 2
        filtered_top_node_ids = self.get_topological_root_node_ids(true_targets)

        # Identify connected components so that genuinely disconnected kernel clauses
        # (e.g. main passive clause + advcl clause produced by p3pass + p3) can be
        # combined correctly rather than discarding all but the last.
        G_undirected = self.G.to_undirected()
        component_map = {
            nid: cid
            for cid, comp in enumerate(nx.connected_components(G_undirected))
            for nid in comp
        }
        roots_by_component = defaultdict(list)
        for node_id in filtered_top_node_ids:
            roots_by_component[component_map.get(node_id, -1)].append(node_id)
        is_multi_component = len(roots_by_component) > 1

        # For multi-component graphs identify the primary root before the loop modifies
        # node data.  Primary = the root whose Singleton has a non-empty 'adv' property
        # (set by p3pass to record the advcl verb that links the two components).
        primary_root_id = None
        if is_multi_component:
            for node_id in filtered_top_node_ids:
                node_data = self.G.nodes[node_id]['data']
                if (
                    isinstance(node_data, Singleton) and
                    'adv' in dict(node_data.properties) and
                    dict(node_data.properties)['adv']  # non-empty
                ):
                    primary_root_id = node_id
                    break

        # Phase 3
        used_edges = set()
        acl_relcl_map = dict()
        position_pairs = self.get_position_pairs()
        for node_id in filtered_top_node_ids:
            loop_settings = SimpleNamespace(shouldLoop=True, edgeForKernel=None, previousKernel=None)
            while loop_settings.shouldLoop:
                descendant_node_ids = list(nx.bfs_tree(self.G, node_id))

                # Ensure the "descendant nodes" are in the edge, and not in used_edges (previous loop), unless we have preposition labels
                filtered_edges = [
                    x for x in self.G.edges(data=True, keys=True) if
                    # Edge source and target are in descendent nodes OR target has preposition label
                    (
                        (x[0] in descendant_node_ids and x[1] in descendant_node_ids) or
                        (x[1] in found_preposition_labels)
                    )
                    # Edge source and target are NOT in used edges from previous loop OR they both are and we have preposition labels
                    and
                    (
                        (x[0], x[1]) not in used_edges or
                        ((x[0], x[1]) in used_edges and len(found_preposition_labels) > 0)
                    )
                    # Edge target is not equal to current root node in loop and target is not a verb
                    and not (x[1] == node_id and self.G.nodes[x[1]]['data'].type.lower() == 'verb')
                ]

                # If we have an edge from the previous iteration use this as our edges
                if loop_settings.edgeForKernel is not None:
                    filtered_edges = [loop_settings.edgeForKernel]

                used_edges = set(map(lambda y: (y[0], y[1]), filtered_edges))
                descendant_nodes = {key: x for key, x in self.G.nodes(data=True) if key in descendant_node_ids}

                # Phase 3.1
                self.G, kernel, loop_settings, acl_relcl_map = create_sentence(
                    self.G, filtered_edges, descendant_nodes, self.negations, node_id, found_preposition_labels,
                    self.node_functions, loop_settings, acl_relcl_map
                )
                kernel = self.kernel_post_processing(kernel, position_pairs)

                # Only check for empty kernel if more than one root node, as if there is only 1 root node we need something, even if it is empty...
                if len(filtered_top_node_ids) > 1:
                    kernel = self.check_if_empty_kernel(kernel)  # Check we do not have be(?, ?) as a kernel
                    if kernel is not None: # (not empty)
                        # if not loop_settings.edgeForKernel:
                        nx.set_node_attributes(self.G, {node_id: kernel}, 'data')
                else:
                    nx.set_node_attributes(self.G, {node_id: kernel}, 'data')

                # Remove 'root' property from nodes, so we do not reuse same node again as it has been accounted for
                attributes_to_update = {
                    node_id: node['data'].strip_root_properties()
                    for node_id, node in self.G.nodes(data=True)
                    if node_id in descendant_node_ids
                }
                nx.set_node_attributes(self.G, attributes_to_update, 'data')

                # If this current "kernel" is none, then remove so the kernel used is the last occurring correct one
                #  (only if we have at least one other kernel available)
                if (
                        ((isinstance(kernel, Singleton) and kernel.kernel is None) or (kernel is None)) and
                        len(filtered_top_node_ids) > 1 and
                        node_id == filtered_top_node_ids[-1]
                ):
                    filtered_top_node_ids.pop()

        # Return the last node ('highest' topological kernel)
        sorted_G = list(NodeFunctions.sort_G(self.G))
        final_kernel = self.G.nodes[[n_id for n_id in sorted_G if n_id in filtered_top_node_ids][-1] if len(filtered_top_node_ids) > 0 else sorted_G[-1]]['data']

        # Multi-component: if the kernels came from genuinely disconnected graph
        # components, combine them.  The primary (identified before the loop by its
        # non-empty 'adv' property) becomes the top-level kernel; every other
        # component's kernel is appended as a SENTENCE property of the primary.
        if (
            is_multi_component and
            primary_root_id is not None and
            primary_root_id in filtered_top_node_ids
        ):
            primary_kernel = self.G.nodes[primary_root_id]['data']
            subordinate_kernels = [
                self.G.nodes[nid]['data']
                for nid in filtered_top_node_ids
                if nid != primary_root_id
                and self.G.nodes[nid]['data'] is not None
                and isinstance(self.G.nodes[nid]['data'], Singleton)
            ]
            if subordinate_kernels and isinstance(primary_kernel, Singleton):
                new_props = defaultdict(list)
                for k, v in dict(primary_kernel.properties).items():
                    if isinstance(v, (list, tuple)):
                        new_props[k] = list(v)
                    else:
                        new_props[k] = v
                for sk in subordinate_kernels:
                    new_props['SENTENCE'].append(sk)
                primary_kernel = primary_kernel.update_node_props(new_props)
            final_kernel = primary_kernel

        # Replace acl_relcl occurrences
        final_kernel = self.acl_replacement(final_kernel, acl_relcl_map)

        # If "final kernel" does not have a kernel
        final_kernel = self.check_for_action_ed_node(acl_relcl_map, final_kernel, SimpleNamespace(shouldLoop=True, edgeForKernel=None, previousKernel=None), position_pairs)

        final_kernel = self.remove_duplicate_properties(final_kernel)
        final_kernel = self.rewrite_properties_logically(final_kernel)
        final_kernel = self.remove_duplicate_properties(final_kernel)  # Remove duplicates added during logical rewriting (e.g. SPECIFICATION)
        final_kernel = self.check_for_adv(final_kernel)

        print(f"{final_kernel.to_string()}\n")
        return final_kernel

    def find_prepositions_and_true_targets(self):
        found_preposition_labels = {}
        true_targets = set()

        # Remove 'dep' edges, given there are other edges. TODO: Could this be done at the preprocessing stage?
        edge_labels = defaultdict(set)  # (source ID, target ID) : edge label name
        for edge in self.G.edges(data=True, keys=True):
            edge_labels[(edge[0], edge[1])].add(edge[3]['label'].named_entity)
        edge_labels = {key: "dep" in value and len(value) > 1 for key, value in edge_labels.items()}  # (source ID, target ID): True/False

        edges_to_remove = []
        for edge in self.G.edges(data=True, keys=True):
            if edge[3]['label'].named_entity == 'dep' and edge_labels[(edge[0], edge[1])]:
                edges_to_remove.append((edge[0], edge[1], edge[2]))

        self.G.remove_edges_from(edges_to_remove)

        if len(self.G.edges(data=True)) > 0:
            # A single word that precedes a noun phrase complement and expresses spatial relations (*in* the house)
            prototypical_prepositions = Services.getInstance().getHOnK().getPrototypicalPrepositions()

            if prototypical_prepositions:
                prepositions_pattern = r"\b(" + "|".join(
                    map(re.escape, sorted(prototypical_prepositions, key=len, reverse=True))) + r")\b"
                prepositions_regex = re.compile(prepositions_pattern)
            else:
                prepositions_regex = None

            for edge in self.G.edges(data=True):
                source, target, edge_label = edge
                source = self.G.nodes[source]['data']
                target = self.G.nodes[target]['data']
                edge_label = edge_label['label'].named_entity

                is_prepositional_phrase = False
                if prepositions_regex:
                    match = prepositions_regex.search(edge_label)
                    if (match and
                            match.group(1) != edge_label and  # Check the preposition is part of a larger phrase
                            not is_label_verb(edge_label.split()[0]) and
                            not case_in_props(dict(target.properties))):
                        is_prepositional_phrase = True

                is_gerund_phrase = (
                        edge_label.endswith('ing') and
                        source.type != "existential" and
                        not self.node_functions.check_node_coordinations_for_auxiliary(edge, self.G)
                )

                if is_prepositional_phrase or is_gerund_phrase:
                    # Add root property to target
                    nx.set_node_attributes(self.G, {target.id: target.add_property('kernel', 'root')}, 'data')
                    found_preposition_labels[target.id] = lemmatize_verb(edge_label)
                elif (is_label_verb(edge_label) and
                      is_kernel_in_props(source) and
                      target.type != 'existential'):
                    nx.set_node_attributes(self.G, {target.id: target.strip_root_properties()}, 'data')
                elif not is_kernel_in_props(target):
                    # A "true target" is the target of an edge where the target itself is not a 'root'
                    true_targets.add(target.id)

        return true_targets, found_preposition_labels

    def get_topological_root_node_ids(self, true_targets):
        filtered_nodes = set()
        filtered_top_node_ids = set()

        # Loop over every source and target for every edge
        for node_id in itertools.chain.from_iterable(map(lambda x: [x[0], x[1]], self.G.edges(data=True))):
            edge_node = self.G.nodes[node_id]['data']

            # Check if edge node is NOT None, NOT in true targets, and IS a root
            if (
                    edge_node is not None and
                    edge_node.id not in true_targets and
                    is_kernel_in_props(edge_node)
                    # or
                    # (isinstance(edge_node, Singleton) and edge_node.type == 'existential')
            ):
                filtered_top_node_ids.add(node_id)

                # Remove SetOfSingleton children from filtered nodes TODO: SetOfSingleton ID share
                # if isinstance(edge_node, SetOfSingletons):
                #     for entity in edge_node.entities:
                #         filtered_nodes = filtered_nodes - {entity.id}

            # if edge_node is None or edge_node.id in filtered_top_node_ids:
            #     continue
            # filtered_top_node_ids.add(edge_node.id)

        # Keep only genuine roots: nodes with no incoming edges
        candidates = {nid for nid in filtered_top_node_ids if len(list(self.G.in_edges(nid))) == 0}
        if candidates:
            filtered_top_node_ids = candidates

        if len(filtered_top_node_ids) == 0:
            # Get either the only node that remains, or all nodes that contain a root
            filtered_top_node_ids = \
                [list(self.G.nodes)[-1]] if len(self.G.nodes) == 1 else \
                    [
                        self.node_functions.get_node_id(x[0]) for x in self.G.nodes(data=True) if
                        is_kernel_in_props(x[1]['data'], False)
                    ]
            # Remove duplicate IDs
            filtered_top_node_ids = list(map(int, numpy.unique(filtered_top_node_ids)))

        return [id for id in list(NodeFunctions.sort_G(self.G)) if id in filtered_top_node_ids]

    def check_for_action_ed_node(self, acl_relcl_map, final_kernel, loop_settings, position_pairs):
        action_ed_node = find_action_ed_node_in_kernel(final_kernel)
        # If no kernel at all OR we have a kernel with edge label "be" or equal to a found action(ed) node
        if (
                not hasattr(final_kernel, 'kernel') or
                final_kernel.kernel is None or
                (
                    action_ed_node and
                    (
                        final_kernel.kernel.edgeLabel.named_entity == 'be' or
                        final_kernel.kernel.edgeLabel == action_ed_node
                    )
                )
        ):
            # If a Singleton of type verb, make this an edge with no source or target
            if isinstance(final_kernel, Singleton) and final_kernel.type.lower() == 'verb':
                final_kernel = create_edge_kernel(final_kernel)
                final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
            else:
                nodes = {final_kernel.id: self.G.nodes[final_kernel.id]}
                #
                if not hasattr(final_kernel, 'properties') or not action_ed_node:
                    self.G, _ = create_existential(self.G, nodes, self.node_functions)
                    self.G, final_kernel, loop_settings, acl_relcl_map = (
                        create_sentence(
                            self.G, self.G.edges(data=True, keys=True), nodes, self.negations, final_kernel.id, {},
                            self.node_functions, loop_settings, acl_relcl_map
                        ))
                    final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
                else:
                    final_kernel = rewrite_action_ed_node(final_kernel, action_ed_node, self.negations)
                    final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
        return final_kernel

    def check_for_adv(self, kernel):
        if kernel.kernel is None:
            return kernel

        source_props = dict(kernel.kernel.source.properties) if isinstance(kernel.kernel.source, Singleton) else None
        if source_props is not None and 'adv' in source_props and source_props['adv']:
            word_permutations = itertools.permutations(kernel.kernel.edgeLabel.named_entity.split(' ') + source_props['adv'].split(' '))
            combined_permutations = {' '.join(p) for p in word_permutations}

            # If 'adv' name is in edge label, we don't need it in properties
            if source_props['adv'] in kernel.kernel.edgeLabel.named_entity:
                kernel = kernel.update_kernel(kernel.kernel.source.remove_prop('adv'), 'source')
                nx.set_node_attributes(self.G, {kernel.id: kernel}, 'data')

            # If the concatenation is not present in the list of phrasal verbs, strip the spurious adv property
            phrasal_verbs = Services.getInstance().getHOnK().getPhrasalVerbs()
            found_phrasal_verbs = phrasal_verbs.intersection(combined_permutations)
            if len(found_phrasal_verbs) == 0:
                kernel = kernel.update_kernel(kernel.kernel.source.remove_prop('adv'), 'source')
                nx.set_node_attributes(self.G, {kernel.id: kernel}, 'data')
                return kernel

            new_edge_label_name = list(found_phrasal_verbs)[0] # TODO: What if more than one element?

            edge_label = kernel.kernel.edgeLabel.update_name(new_edge_label_name)
            edge_source = kernel.kernel.source.remove_prop('adv')

            kernel = kernel.update_kernel(edge_source, "source")
            kernel = kernel.update_kernel(edge_label, "edgeLabel")
        else:
            if kernel.kernel.source.type == "SENTENCE":
                kernel = kernel.update_kernel(self.check_for_adv(kernel.kernel.source), "source")
            elif kernel.kernel.target is not None and kernel.kernel.target.type == "SENTENCE":
                kernel = kernel.update_kernel(self.check_for_adv(kernel.kernel.target), "target")

            properties_to_keep = defaultdict(list)
            for key in dict(kernel.properties):
                properties_key_ = dict(kernel.properties)[key]
                if isinstance(properties_key_, str):
                    properties_to_keep[key] = properties_key_
                elif properties_key_ is not None:
                    for node in properties_key_:
                        if key == 'SENTENCE':
                            properties_to_keep[key].append(self.check_for_adv(node))
                        else:
                            properties_to_keep[key].append(node)

            kernel = kernel.update_node_props(properties_to_keep)

        return kernel

    def rewrite_properties_logically(self, kernel):
        # TODO: Need to check that properties of properties are encompassed
        properties_to_add = defaultdict(list)
        force_update = False
        if isinstance(kernel, Singleton):
            for key in dict(kernel.properties):
                properties_key_ = dict(kernel.properties)[key]
                if key == 'extra':
                    # As to not continually rewrite previously rewritten nodes
                    properties_to_add[key] = properties_key_
                    continue

                if not isinstance(properties_key_, str):
                    if isinstance(properties_key_, Singleton):
                        properties_to_add = self.rewrite_node_logically(
                            kernel, properties_key_, dict(kernel.properties), type_key=key)
                    else:
                        for prop_node in properties_key_:
                            if isinstance(prop_node, Singleton):
                                if prop_node.kernel is not None:
                                    if prop_node.kernel.edgeLabel.named_entity in {"nmod", "nmod_poss", "obl", "acl", "acl_relcl"}:
                                        if isinstance(prop_node.kernel.target, SetOfSingletons):
                                            target_sos = prop_node.kernel.target
                                            inner = target_sos
                                            while isinstance(inner, SetOfSingletons) and inner.entities:
                                                inner = inner.entities[0]
                                            _, key_to_use = self.rewrite_node_logically(
                                                kernel, inner, defaultdict(list), False, True)
                                            if key_to_use is not None:
                                                properties_to_add[key_to_use].append(target_sos)
                                        else:
                                            properties_to_add = self.rewrite_node_logically(
                                                kernel, prop_node, properties_to_add, True)
                                            kernel, properties_to_add, force_update = self.check_property_replacement(kernel, properties_to_add) # TODO: Do we want to do this?
                                    elif prop_node.type == "SENTENCE":
                                        prop_node = self.rewrite_properties_logically(prop_node)
                                        properties_to_add[key].append(prop_node)
                                    # else:
                                        # continue  # TODO
                                else:
                                    properties_to_add = self.rewrite_node_logically(kernel, prop_node, properties_to_add, type_key=key)
                            elif isinstance(prop_node, SetOfSingletons):
                                # TODO: Is it - SPACE:AND:NOT(x[type:y])
                                #  Or - AND:NOT(x[type:y])[SPACE:]
                                rewritten_entities = []
                                key_to_use = ""
                                for entity in prop_node.entities:
                                    # TODO: Might need to be recursive?
                                    entity, key_to_use = self.rewrite_node_logically(kernel, entity, defaultdict(list), False, True)
                                    rewritten_entities.append(entity)
                                prop_node = prop_node.update_entities(rewritten_entities)

                                if key_to_use is not None:
                                    # Create new SetOfSingletons encompassing previous key
                                    grouping_type = Grouping[key] if key in Grouping.__members__ else Grouping.NONE
                                    properties_to_add[key_to_use].append(SetOfSingletons(
                                        id=prop_node.id,
                                        type=grouping_type,
                                        entities=tuple([prop_node]),
                                        min=min(rewritten_entities, key=lambda x: x.min).min,
                                        max=max(rewritten_entities, key=lambda x: x.max).max,
                                        confidence=1
                                    ))
                                    kernel.update_node_props(properties_to_add)
                                else:
                                    # properties_to_add[key] = properties_key_
                                    properties_to_add[key].append(prop_node)
                else:
                    properties_to_add[key] = properties_key_

        # Rewrite source logically if SetOfSingletons
        # if kernel.kernel is not None and isinstance(kernel.kernel, SetOfSingletons):
        #     kernel = Singleton.update_kernel(kernel, SetOfSingletons.update_entities(kernel.kernel.source, [self.rewrite_properties_logically(x) for x in kernel.kernel.source.entities]), "source")

        # Rewrite source and target properties
        if hasattr(kernel, "kernel") and kernel.kernel is not None:
            kernel = kernel.update_kernel(self.rewrite_properties_logically(kernel.kernel.source), 'source') if kernel.kernel.source is not None else kernel
            kernel = kernel.update_kernel(self.rewrite_properties_logically(kernel.kernel.target), 'target') if kernel.kernel.target is not None else kernel

        if len(properties_to_add) > 0 or force_update:
            return kernel.update_node_props(properties_to_add)
        else:
            return kernel

    def rewrite_node_logically(self, kernel, initial_node, properties, has_nmod=False, return_key=False, type_key=None):
        honk = Services.getInstance().getHOnK()
        prop_node, selected_rule = get_matching_logical_rules(kernel, initial_node, has_nmod)
        prepositions = get_prepositions(prop_node)
        number_value = dict(prop_node.properties)["nummod"] if "nummod" in dict(prop_node.properties) else None

        if selected_rule is not None:
            # Remove prepositions (so long as construct property is not None) as no longer needed
            node_props = {k: v for k, v in dict(prop_node.properties).items() if (
                    isinstance(v, str) and
                    v.lower() not in prepositions
            ) or (
                    not isinstance(v, str)
            ) or (
                    k not in {'nummod'} and
                    selected_rule.logicalConstructName in {'quantity', 'measure'}
            )}
            # prop_node = Singleton.update_node_props(prop_node, node_props)  # TODO: If we do not remove the properties, then rewriting of source/target might re-rewrite the same preposition? We might lose information however

            selected_function = honk.get_logical_functions(selected_rule.logicalConstructName, selected_rule.logicalConstructProperty)

            if len(selected_function) > 0:
                selected_function = selected_function[0]  # Use first function in list
                type_key = selected_function.logicalConstructName.upper()
                logical_type = selected_function.logicalConstructProperty

                if selected_function.attachTo == "Singleton":
                    if logical_type is not None:
                        if type_key == "SPECIFICATION":
                            if logical_type == "inverse":
                                if 'extra' in node_props:
                                    if initial_node.kernel.source.id not in [x.id for x in node_props["extra"] if isinstance(x, Singleton)]:
                                        node_props["extra"] = list(node_props["extra"])
                                        node_props["extra"].append(initial_node.kernel.source)
                                else:
                                    node_props["extra"] = [initial_node.kernel.source]
                            else:
                                node_to_add = prop_node
                                prop_node = initial_node.kernel.source
                                node_props = dict(prop_node.properties)
                                if 'extra' in node_props:
                                    if node_to_add.id not in [x.id for x in node_props["extra"] if isinstance(x, Singleton)]:
                                        node_props["extra"] = list(node_props["extra"])
                                        node_props["extra"].append(node_to_add)
                                else:
                                    node_props["extra"] = [node_to_add]
                        elif logical_type is not None:
                            node_props["type"] = logical_type  # TODO: node_props[key] and properties[key] would be duplicated, so use "type" instead?
                        prop_node = prop_node.update_node_props(node_props)
                    
                    val_to_add = prop_node if number_value is None or (number_value is not None and not hasattr(selected_function, "hasNumber")) else number_value
                    if isinstance(val_to_add, Singleton):
                        val_to_add = self.rewrite_properties_logically(val_to_add)
                        if val_to_add.id not in [x.id for x in properties[type_key] if isinstance(x, Singleton)]:
                            properties[type_key].append(val_to_add)
                    else:
                        if val_to_add not in properties[type_key]:
                            properties[type_key].append(val_to_add)

                elif selected_function.attachTo == "Kernel":
                    if logical_type is not None:
                        node_props["type"] = logical_type
                        prop_node = prop_node.update_node_props(node_props)
                    prop_node = self.rewrite_properties_logically(prop_node)
                    if prop_node.id not in [x.id for x in properties[type_key] if isinstance(x, Singleton)]:
                        properties[type_key].append(prop_node)

        if return_key:
            return prop_node, type_key
        else:
            # If we don't find a rule, or function, add the initial node to its original key
            if selected_rule is None or (isinstance(selected_function, list) and len(selected_function) == 0):
                type_key = type_key if type_key is not None else initial_node.type
                if not isinstance(properties[type_key], (list, tuple)):
                    properties[type_key] = [initial_node]
                else:
                    if initial_node.id not in [x.id for x in properties[type_key] if isinstance(x, Singleton)]:
                        properties[type_key].append(initial_node)
            return properties

    def check_property_replacement(self, kernel, properties):
        _be_forms = frozenset({'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', "'m", "'re", "'s"})
        _copula_types = {'JJ', 'JJS', 'RB'}
        edge_label_name = kernel.kernel.edgeLabel.named_entity.lower().strip() if (kernel.kernel and kernel.kernel.edgeLabel) else ""
        is_copula_kernel = (
            edge_label_name in _be_forms and
            kernel.kernel.target is not None and
            kernel.kernel.target.type in _copula_types
        )
        nodes_to_remove = []
        force_update = False
        for key in properties:
            for prop_node in properties[key]:
                if isinstance(prop_node, Singleton):
                    # TODO: Do we consider edgeLabel too?
                    if kernel.kernel.source.id == prop_node.id:
                        kernel = kernel.update_kernel(prop_node, "source")
                        nodes_to_remove.append(prop_node)
                    elif not is_copula_kernel and kernel.kernel.target is not None and (kernel.kernel.target.id == prop_node.id or ('extra' in dict(prop_node.properties) and len([x for x in list(dict(prop_node.properties)['extra']) if x.id == kernel.kernel.target.id]) > 0)):  # TODO: Copy logic for checking extra in source
                        if key == 'INSTRUMENT':
                            kernel = kernel.update_kernel(None, "target")
                        else:
                            kernel = kernel.update_kernel(prop_node, "target")
                            nodes_to_remove.append(prop_node)
        for remove_node in nodes_to_remove:
            for key, value in dict(properties).items():
                force_update = True
                properties[key] = [x for x in properties[key] if x.id != remove_node.id]
        return kernel, properties, force_update

    def acl_replacement(self, kernel, acl_relcl_map):
        # TODO: This isn't entirely recursive...
        if len(acl_relcl_map.values()) > 0 and kernel.kernel is not None:
            kernel_source = self.get_acl_replacement(acl_relcl_map, kernel.kernel.source)
            kernel_target = self.get_acl_replacement(acl_relcl_map, kernel.kernel.target)

            properties_to_keep = defaultdict(list)

            # Check if properties has any Singleton's and replace those
            for key in dict(kernel.properties):
                properties_key_ = dict(kernel.properties)[key]
                if not isinstance(properties_key_, str):
                    if isinstance(properties_key_, Singleton):
                        new_prop = self.get_acl_replacement(acl_relcl_map, properties_key_)
                        properties_to_keep[key].append(new_prop)
                    else:
                        for prop_node in properties_key_:
                            if prop_node.kernel is not None and prop_node.kernel.target is not None and prop_node.kernel.target.id in acl_relcl_map.keys():
                                continue

                            if prop_node.kernel is None:
                                properties_to_keep[key].append(prop_node)
                            elif prop_node.type == 'SENTENCE':
                                prop_sing_source = self.get_acl_replacement(acl_relcl_map, prop_node.kernel.source)
                                prop_sing_target = self.get_acl_replacement(acl_relcl_map, prop_node.kernel.target)

                                properties_to_keep[key].append(Singleton(
                                    id=prop_node.id,
                                    named_entity='',
                                    type='SENTENCE',
                                    min=prop_node.min,
                                    max=prop_node.max,
                                    confidence=1,
                                    kernel=Relationship(
                                        source=prop_sing_source,
                                        target=prop_sing_target,
                                        edgeLabel=prop_node.kernel.edgeLabel,
                                        isNegated=prop_node.kernel.isNegated,
                                    ),
                                    properties=prop_node.properties,
                                ))
                else:
                    properties_to_keep[key].append(properties_key_)

            return Singleton(
                id=kernel.id,
                named_entity='',
                type='SENTENCE',
                min=kernel.min,
                max=kernel.max,
                confidence=1,
                kernel=Relationship(
                    source=kernel_source,
                    target=kernel_target,
                    edgeLabel=kernel.kernel.edgeLabel,
                    isNegated=kernel.kernel.isNegated,
                ),
                properties=create_props_for_singleton(properties_to_keep),
            )
        else:
            return kernel

    def get_acl_replacement(self, acl_relcl_map, node):
        return acl_relcl_map[node.id] if node is not None and node.id in acl_relcl_map.keys() else node

    def get_position_pairs(self):
        position_pairs = {}
        for edge in self.G.edges(data=True):
            source, target, _ = edge
            source = self.G.nodes[source]['data']
            target = self.G.nodes[target]['data']

            source_pos = get_min_position(source)
            target_pos = get_min_position(target)
            if target_pos > source_pos:
                if source.id in position_pairs:
                    if target_pos < position_pairs[source.id]:
                        position_pairs[source.id] = target_pos
                else:
                    position_pairs[source.id] = target_pos
        return position_pairs

    # Check if final kernel is "empty" be(?, ?) and use the properties of 'SENTENCE'
    def check_if_empty_kernel(self, kernel, force=False):
        properties_to_keep = dict()
        new_kernel = None
        if (
                isinstance(kernel,
                           Singleton) and kernel.kernel is not None and kernel.kernel.edgeLabel is not None and (
                kernel.kernel.edgeLabel.named_entity == "be" if not force else True)
                and
                (
                        ((kernel.kernel.source is not None and kernel.kernel.source.type == 'existential') and (
                                kernel.kernel.target is not None and kernel.kernel.target.type == 'existential'))
                        or
                        ((kernel.kernel.source is None) and (kernel.kernel.target is None))
                )
        ):
            node_props = dict(kernel.properties)
            if len(node_props) > 0 and 'SENTENCE' in node_props:
                for key in node_props:
                    if key == 'SENTENCE':
                        new_kernel = node_props['SENTENCE'][0]  # TODO: Safe to use 0th element?
                        new_kernel = self.check_if_empty_kernel(new_kernel)
                    else:
                        properties_to_keep[key] = node_props[key]
            # elif len(node_props) == 0:
                #     return None

        if new_kernel is not None:
            if len(properties_to_keep) > 0:
                return new_kernel.update_node_props(properties_to_keep)
            else:
                return new_kernel
        else:
            return kernel

    # Rewrite kernel if positions are not correct
    def kernel_post_processing(self, kernel, position_pairs):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return kernel

        # If target is an adjective, and there is a pronoun or entity in the properties, swap these round
        if kernel.kernel.target is not None and isinstance(kernel.kernel.target, Singleton) and kernel.kernel.target.type.startswith("JJ"):
            for key in dict(kernel.properties):
                properties_key = dict(kernel.properties)[key]
                for node in properties_key:
                    if node.type in {'PRONOUN', 'ENTITY'}:
                        new_kernel_properties = defaultdict(list)
                        new_kernel_properties[kernel.kernel.target.type].append(kernel.kernel.target)
                        new_kernel_properties = merge_properties(dict(kernel.properties), new_kernel_properties)
                        kernel = kernel.update_node_props(new_kernel_properties)
                        kernel = kernel.update_kernel(properties_key[0], "target")
                        break

        # TODO: Temporary patch, investigate how applicable this is to other scenarios
        # be(ENTITY1, ?)[SENTENCE:verb(?, ENTITY2)[]]
        if (
            kernel.kernel.target is not None and
            kernel.kernel.target.type == 'existential' and
            kernel.kernel.edgeLabel.named_entity == 'be' and
            'SENTENCE' in dict(kernel.properties)
        ):
            for key in dict(kernel.properties):
                if key == 'SENTENCE':
                    sentences = dict(kernel.properties)[key]
                    for sentence in sentences:
                        if sentence.kernel.source.type == 'existential':
                            for sentence_key in dict(sentence.properties):
                                sentence_properties = dict(sentence.properties)[sentence_key]
                                for prop_node in sentence_properties:
                                    if prop_node.id == kernel.kernel.source.id:
                                        kernel = sentence.update_kernel(prop_node, "source")

        # if kernel.kernel.source in [node[1][0] for node in dict(list(dict(kernel.properties)['SENTENCE'])[0].properties).items()]:
        #     for key in dict(kernel.properties):
        #         properties_key = dict(kernel.properties)[key]
        #         for node in properties_key:
        #     matched_key = [node[1][0] for node in dict(list(dict(kernel.properties)['SENTENCE'])[0].properties).items()]


        properties_to_keep = dict(kernel.properties)
        new_target = kernel.kernel.target

        # Check kernel positions, if kernel is in position pairs AND the edge is SEMI MODAL
        #  (e.g. scissors need sharpening: need(scissors, None)[SENTENCE:sharpening(?, None)] => need(scissors, sharpening(?, None)))
        if (
                kernel.id in position_pairs and
                kernel.kernel.edgeLabel is not None and
                check_semi_modal(kernel.kernel.edgeLabel.named_entity)
        ):
            position_value = position_pairs[kernel.id]
            if 'SENTENCE' in dict(kernel.properties):
                properties_to_keep['SENTENCE'] = []
                for sentence_elm in list(dict(kernel.properties)['SENTENCE']):
                    # If a property should be considered as the target based on position of property
                    if int(float(dict(sentence_elm.kernel.edgeLabel.properties)['pos'])) == position_value:
                        new_target = sentence_elm
                    else:
                        properties_to_keep['SENTENCE'].append(sentence_elm)

            return Singleton(
                id=kernel.id,
                named_entity="",
                type="SENTENCE",
                min=kernel.min,
                max=kernel.max,
                confidence=kernel.confidence,
                kernel=Relationship(
                    source=kernel.kernel.source,
                    target=new_target,
                    edgeLabel=kernel.kernel.edgeLabel,
                    isNegated=kernel.kernel.isNegated
                ),
                properties=create_props_for_singleton(properties_to_keep),
            )
        # If we do NOT have an edge label, and source OR target are adjectives (e.g. None(clear, vision) => be(vision[clear]), ?))
        elif (
                kernel.kernel.edgeLabel is None and
                kernel.kernel.source is not None and
                kernel.kernel.target is not None and
                (
                    (kernel.kernel.source.type.startswith('JJ') and kernel.kernel.target.type == 'ENTITY') or
                    (kernel.kernel.target.type.startswith('JJ') and kernel.kernel.source.type == 'ENTITY')
                )
        ):
            new_edges = []
            root_id = None
            # Update source/target with adjective as properties
            if kernel.kernel.source.type.startswith('JJ') and kernel.kernel.target.type == 'ENTITY':
                node_props = merge_properties(
                    dict(kernel.kernel.target.properties),
                    {kernel.kernel.source.type: kernel.kernel.source}
                )
                root_id = kernel.kernel.target.id
                nx.set_node_attributes(self.G, {root_id: kernel.kernel.target.update_node_props(node_props)}, 'data')
                create_existential(self.G, {kernel.kernel.target.id: self.G.nodes[kernel.kernel.target.id]}, self.node_functions)
            elif kernel.kernel.target.type.startswith('JJ') and kernel.kernel.source.type == 'ENTITY':
                node_props = merge_properties(
                    dict(kernel.kernel.source.properties),
                    {kernel.kernel.target.type: kernel.kernel.target}
                )
                root_id = kernel.kernel.source.id
                nx.set_node_attributes(self.G, {root_id: kernel.kernel.source.update_node_props(node_props)}, 'data')
                create_existential(self.G, {kernel.kernel.target.id: self.G.nodes[kernel.kernel.target.id]}, self.node_functions)

            self.G, kernel, edge_to_loop, acl_relcl_map = create_sentence(
                self.G, new_edges, {key: x for key, x in self.G.nodes(data=True)}, self.negations, root_id, {}, self.node_functions,
                SimpleNamespace(shouldLoop=False, edgeForKernel=None, previousKernel=None), {}
            )
            kernel = self.kernel_post_processing(kernel, position_pairs)
        return kernel

    def remove_duplicate_properties(self, kernel, kernel_nodes=None):
        if kernel.kernel is None:
            return kernel

        properties_to_keep = defaultdict(list)

        if kernel_nodes is None:
            kernel_nodes = set()
        kernel_nodes = self.add_to_kernel_nodes(kernel, kernel_nodes)

        # Add 'nmod' source and target to kernel nodes, so duplicate nodes are not added to properties
        for key in dict(kernel.properties):
            properties_key_ = dict(kernel.properties)[key]
            if isinstance(properties_key_, str):
                continue
            else:
                for node in properties_key_:
                    if key in {'nmod', 'nmod_poss', 'obl', 'acl', 'acl_relcl'}:
                        properties_to_keep[key].append(self.remove_duplicate_properties(node, kernel_nodes))
                        kernel_nodes = self.add_to_kernel_nodes(node if key not in {'nmod', 'obl', 'acl', 'acl_relcl'} else node.kernel.target, kernel_nodes)

        # Check if empty kernel is in properties and remove, recursively iterate through kernels to remove duplicate properties
        for key in dict(kernel.properties):
            properties_key_ = dict(kernel.properties)[key]
            if isinstance(properties_key_, str):
                continue
            else:
                for node in properties_key_:
                    if key == 'SENTENCE':
                        emptied_node = self.check_if_empty_kernel(node, True)
                        if emptied_node is not None:
                            # If given property is `be(? OR in kernel_nodes, ? OR in kernel_nodes)`, then do not add as property as it is redundant
                            if (hasattr(node, 'kernel') and not (node.kernel.edgeLabel.named_entity == "be" and (
                                    (
                                            (
                                                    kernel_nodes is not None and node.kernel.source.type != 'existential' and node.kernel.source in kernel_nodes)
                                            and (
                                                    node.kernel.target is not None and node.kernel.target.type == 'existential')
                                    )
                                    or
                                    (
                                            (
                                                    kernel_nodes is not None and node.kernel.target is not None and node.kernel.target.type != 'existential' and node.kernel.target in kernel_nodes)
                                            and (
                                                    node.kernel.source is not None and node.kernel.source.type == 'existential')
                                    )
                            ))) or kernel_nodes is None or not hasattr(node, 'kernel'):
                                if not is_node_in_kernel_nodes(emptied_node, kernel_nodes):
                                    properties_to_keep[key].append(self.remove_duplicate_properties(node, kernel_nodes))
                                else:
                                    self.remove_duplicate_properties(node, kernel_nodes)
                    elif not is_node_in_kernel_nodes(node, kernel_nodes):
                        if hasattr(node, 'properties'):
                            inner_properties_to_keep = dict()
                            for inner_key in dict(node.properties):
                                value = dict(node.properties)[inner_key]
                                if ((isinstance(value, str) and value in string.punctuation) or (
                                        kernel.kernel.edgeLabel is not None and
                                        isinstance(value, str) and
                                        not re.search(r"\b" + value + r"\b", kernel.kernel.edgeLabel.named_entity)
                                ) or not isinstance(value, str) or kernel.kernel.edgeLabel is None):
                                    inner_properties_to_keep[inner_key] = value
                            properties_to_keep[key].append(node.update_node_props(inner_properties_to_keep))
                        else:
                            properties_to_keep[key].append(node)

        # Check if we have duplicate kernels now they are all added, and keep most relevant one (i.e. two equal kernels but only one has properties)
        if 'SENTENCE' in properties_to_keep:
            sentence_properties = properties_to_keep['SENTENCE']
            if len(sentence_properties) > 1:  # Check for more than one SENTENCE in props
                equal_kernels = [sentence for sentence in sentence_properties if
                                 sentence.kernel == sentence_properties[0].kernel]
                if len(equal_kernels) > 1:  # Check we have at leasts two "equal" sentences
                    sentences_with_props = [sentence for sentence in equal_kernels if len(sentence.properties) > 0]

                    found_properties = defaultdict(list)
                    for given_sentence in sentences_with_props:
                        found_properties = merge_properties(found_properties, dict(given_sentence.properties))
                    properties_to_keep["SENTENCE"] = [Singleton(
                        id=equal_kernels[0].id,
                        named_entity=equal_kernels[0].named_entity,
                        type=equal_kernels[0].type,
                        min=equal_kernels[0].min,
                        max=equal_kernels[0].max,
                        confidence=equal_kernels[0].confidence,
                        kernel=equal_kernels[0].kernel,
                        properties=create_props_for_singleton(found_properties),
                    )]

        return Singleton(
            id=kernel.id,
            named_entity=kernel.named_entity,
            type=kernel.type,
            min=kernel.min,
            max=kernel.max,
            confidence=kernel.confidence,
            kernel=kernel.kernel,
            properties=create_props_for_singleton(properties_to_keep),
        )

    def add_to_kernel_nodes(self, node, kernel_nodes):
        if isinstance(node, SetOfSingletons):
            kernel_nodes.add(node)
            for entity in node.entities:
                self.add_to_kernel_nodes(entity, kernel_nodes)
        else:
            if node.kernel is not None:
                kernel_nodes.add(node)

                # Check if properties has any Singleton's and add to kernel nodes also
                if node.kernel.edgeLabel is not None:
                    self.add_singletons_from_node_properties(node.kernel.edgeLabel, kernel_nodes)
                    self.add_to_kernel_nodes(node.kernel.edgeLabel, kernel_nodes)
                if node.kernel.source is not None:
                    if node.kernel.edgeLabel is not None and node.kernel.edgeLabel.named_entity not in {'nmod', 'obl', 'acl', 'acl_relcl'}:  # TODO: We might lose information from the source when nmod relationship is logically rewritten, so do not add source to kernel_nodes?
                        self.add_singletons_from_node_properties(node.kernel.source, kernel_nodes)
                        self.add_to_kernel_nodes(node.kernel.source, kernel_nodes)
                if node.kernel.target is not None:
                    # If it is nmod or obl, we are rewriting it, but the target (e.g. "station") should be considered "consumed"
                    # so it doesn't appear as a redundant property elsewhere.
                    self.add_singletons_from_node_properties(node.kernel.target, kernel_nodes)
                    self.add_to_kernel_nodes(node.kernel.target, kernel_nodes)
            else:
                kernel_nodes.add(node)

        return kernel_nodes

    def add_singletons_from_node_properties(self, node, kernel_nodes):
        if isinstance(node, Singleton):
            for key in dict(node.properties):
                properties_key_ = dict(node.properties)[key]
                if not isinstance(properties_key_, str):
                    if isinstance(properties_key_, Singleton):
                        self.add_to_kernel_nodes(properties_key_, kernel_nodes)
                    else:
                        for prop_node in properties_key_:
                            if isinstance(prop_node, Singleton):
                                self.add_to_kernel_nodes(prop_node, kernel_nodes)
                            elif isinstance(prop_node, SetOfSingletons):
                                kernel_nodes.add(node)
                                for prop_entity in prop_node.entities:
                                    self.add_to_kernel_nodes(prop_entity, kernel_nodes)
