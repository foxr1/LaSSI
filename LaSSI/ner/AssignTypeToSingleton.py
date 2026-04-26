__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

from collections import defaultdict
from copy import copy
from itertools import repeat

from LaSSI.external_services.Services import Services
from LaSSI.ner.MergeSetOfSingletons import merge_properties
from LaSSI.ner.node_functions import NodeFunctions
from LaSSI.ner.string_functions import is_label_verb, does_string_have_negations
from LaSSI.ner.topological_sort import topologicalSort
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping, SetOfSingletons, Relationship
from LaSSI.structures.internal_graph.Graph import Graph
from LaSSI.structures.kernels.Sentence import is_kernel_in_props, case_in_props


class AssignTypeToSingleton:
    def __init__(self, is_simplistic_rewriting, meu_db_row, negations=None):
        if negations is None:
            negations = {'not', 'no'}
        self.negations = negations
        self.associations = set()
        self.meu_entities = defaultdict(set)
        self.nodes = dict()
        self.edges = None
        self.services = Services.getInstance()
        self.existentials = self.services.getExistentials()
        self.is_simplistic_rewriting = is_simplistic_rewriting
        self.meu_db_row = meu_db_row

    def clear_meu_node_associations(self):
        self.associations.clear()
        self.meu_entities.clear()
        self.nodes.clear()

    def get_current_state_nodes(self):
        return self.nodes

    def groupGraphNodes(self, gsm_json):
        self.max_id = max(map(lambda x: int(x["id"]), gsm_json)) + 1
        self.node_functions = NodeFunctions(self.max_id)

        # Phase -1
        gsm_json = topologicalSort(gsm_json)

        # Phase 0
        gsm_json = self.preProcessing(gsm_json)

        # Phase 1
        self.extractLogicalConnectorsAsGroups(gsm_json)

        # Phase 1.5
        self.check_for_but(gsm_json)  # TODO: Move to kernel creation

        # TODO: These phases may be removed, as it already happens in Phase 1.1, as it might only need to happen for SetOfSingletons
        # Phase 2
        for key in self.nodes:
            self.associateNodeToMeuMatches(self.nodes[key])
        # Phase 3
        for item in self.associations:
            self.singletonTypeResolution(item, gsm_json)

        # Phase 4
        self.resolveGraphNERs()

        return gsm_json

    # Phase 0
    def preProcessing(self, gsm_json):
        # Pre-processing not semantically driven
        # Scan for 'inherit' edges and contain them in the node that has that edge
        nodes_by_id = {item['id']: item for item in gsm_json}
        nodes_by_begin = {item['properties']['begin']: item for item in gsm_json if
                          'begin' in item.get('properties', {})}
        nodes_by_end = {item['properties']['end']: item for item in gsm_json if 'end' in item.get('properties', {})}

        # List of parent ID's for a given child ID
        parent_map = defaultdict(list)
        for node in gsm_json:
            for edge in node['phi']:
                child_id = edge['score']['child']
                if child_id is not None:
                    parent_map[child_id].append(node['id'])

        ids_to_remove = set()
        hyph_ids_to_remove = set()

        # Concatenate hyphenated words from node properties
        for gsm_item in gsm_json:
            # Look for hyphens, to properly concatenate
            if 'HYPH' in gsm_item['ell'] and gsm_item['xi'][0] == '-': # TODO: Handle '/' being typed as HYPH?
                first_word = nodes_by_end.get(gsm_item['properties'].get('begin'))
                second_word = nodes_by_begin.get(gsm_item['properties'].get('end'))

                if first_word and second_word:
                    first_word['xi'][0] = f"{first_word['xi'][0]}-{second_word['xi'][0]}"
                    hyph_ids_to_remove.add(second_word['id']) # Mark the hyphen node for removal

        for pos, gsm_item in enumerate(gsm_json):
            if gsm_item['id'] in ids_to_remove:
                continue

            # Skip empty nodes that provide no additional information
            if len(gsm_item["phi"]) == 0 and not parent_map.get(gsm_item['id']):
                ids_to_remove.add(gsm_item['id'])
                continue

            # Remove unwanted 'subjpass' value
            if 'subjpass' in gsm_item['xi']:
                gsm_item['xi'].remove('subjpass')

            # Remove possible '-' dangling second word
            for remove_id in hyph_ids_to_remove:
                remove_node = nodes_by_id.get(remove_id)
                for key, value in dict(gsm_item['properties']).items():
                    try:
                        x = float(key)
                        if x == float(remove_node['properties']['pos']):
                            del gsm_item['properties'][key]
                    except:
                        continue

            edges_to_keep = []
            for edge in gsm_item['phi']:
                containment = edge['containment']
                child_id = edge['score']['child']
                node_to_inherit = nodes_by_id.get(child_id)

                if not node_to_inherit:
                    edges_to_keep.append(edge)
                    continue

                if 'inherit_' in containment:
                    if containment.endswith('_edge'):
                        if len(gsm_item['ell']) == 0:
                            gsm_item['ell'] = node_to_inherit['ell']
                        else:
                            gsm_item['ell'][len(gsm_item['ell']):] = node_to_inherit['ell'][1:]

                        if self.shouldInheritNode(node_to_inherit, gsm_item, gsm_json):
                            new_properties = merge_properties(dict(gsm_item['properties']), dict(node_to_inherit['properties']), {'begin', 'end', 'pos'})
                            gsm_item['properties'] = new_properties

                        # If the current node is the only parent, mark the inherited node for removal
                        if parent_map.get(child_id) == [gsm_item['id']]:
                            ids_to_remove.add(child_id)

                    # Inherit edges from the target node
                    for edge_to_inherit in node_to_inherit['phi']:
                        edge_to_inherit['score']['parent'] = gsm_item['id']
                        gsm_item['phi'].append(dict(edge_to_inherit))

                    # Remove edges from node that has been inherited
                    node_to_inherit['phi'] = []
                elif 'mark' in containment and ('IN' in node_to_inherit['ell'] or 'TO' in node_to_inherit['ell']):
                    gsm_item['properties']['mark'] = node_to_inherit.get('xi', [""])[0]
                elif 'punct' in containment:
                    gsm_item['properties']['punct'] = node_to_inherit.get('xi', [""])[0]
                else:
                    edges_to_keep.append(edge)

            # Ignore 'inherit_edge' as they are accounted for, keep all other edges
            gsm_item['phi'] = edges_to_keep
        return [item for item in gsm_json if item['id'] not in ids_to_remove]

    def shouldInheritNode(self, node_to_inherit, gsm_item, gsm_json):
        # Check if 'orig' descendant positions are > 'inherit_edge' positions
        if not node_to_inherit['ell'][0] in dict(gsm_item['properties']):
            min_orig_pos = self.getMinMaxNodePos(gsm_json, gsm_item, 'orig', min)
            max_inherit_pos = self.getMinMaxNodePos(gsm_json, gsm_item, '_edge', max)

            if min_orig_pos < 0 or max_inherit_pos < 0 or min_orig_pos < max_inherit_pos:
                return True

        return False

    def getMinMaxNodePos(self, gsm_json, gsm_item, edge_label, min_or_max, pos=-1):
        # Recursively get children from edge_label
        # TODO: Sometimes children are already removed through inherit/mark edges etc. (does this affect the logic?)
        for edge in gsm_item['phi']:
            if edge_label == True or edge_label in edge['containment']:
                child_node = self.node_functions.get_gsm_item_from_id(edge['score']['child'], gsm_json)
                if 'pos' in child_node['properties']:
                    child_node_pos = int(float(child_node['properties']['pos']))
                    if pos < 0:
                        pos = child_node_pos
                    else:
                        pos = min_or_max(pos, child_node_pos)

                    if len(child_node['phi']) > 0:
                        self.getMinMaxNodePos(gsm_json, child_node, True, min_or_max, pos)
        return pos


    # Phase 1
    def extractLogicalConnectorsAsGroups(self, gsm_json):
        # Get all nodes from resulting graph and create list of Singletons
        number_of_nodes = range(len(gsm_json))

        # Phase 1.1
        self.create_all_singleton_nodes(gsm_json, number_of_nodes)

        # Check if conjugation ('conj') or ('multipleindobj') exists and if true exists, merge into SetOfSingletons
        # Also if 'compound' relationship is present, merge parent and child nodes
        for row in number_of_nodes:
            gsm_item = gsm_json[row]
            grouped_nodes = []
            has_conj = 'conj' in gsm_item['properties']
            has_multipleindobj = 'multipleindobj' in gsm_item['ell']
            is_compound = False
            has_compound_prt = False
            group_type = None
            norm_confidence = 1.0

            # TODO: Change this property to broader name?
            if len(gsm_item['xi']) > 1 and 'subjpass' in gsm_item['xi'][1]:
                gsm_item['properties']['subjpass'] = gsm_item['xi'][1]

            # Get all nodes from edges of root node
            if has_conj or has_multipleindobj:
                for edge in gsm_item['phi']:
                    if 'orig' in edge['containment'] and (
                            (not has_multipleindobj) or is_label_verb(edge['containment'])):
                        node_id = self.node_functions.get_node_id(edge['content'])
                        node = self.nodes[node_id]

                        # Merge properties from parent into children
                        # TODO: Should we really only be merging desired properties like 'kernel' (e.g. we now get conj property in children)?
                        if isinstance(node, Singleton):
                            node_props = merge_properties(dict(node.properties), gsm_item['properties'], {'pos'})
                            # gsm_item['properties'] = node_props
                            self.nodes[node_id] = node.update_node_props(node_props)

                        grouped_nodes.append(self.nodes[node_id])
                        norm_confidence *= node.confidence
            else:
                # If not 'conjugation' or 'multipleindobj', then check for compound edges
                is_compound, norm_confidence = self.get_relationship_entities(
                    grouped_nodes,
                    gsm_json,
                    gsm_item,
                    is_compound,
                    norm_confidence,
                    {'compound', 'none'}, # Treat 'none' as compound also
                    False)
                if not is_compound:
                    has_conj, norm_confidence = self.get_relationship_entities(
                        grouped_nodes,
                        gsm_json,
                        gsm_item,
                        has_conj,
                        norm_confidence,
                        {'conj'},
                        False)
                    if has_conj:
                        grouped_nodes.insert(0, self.nodes[self.node_functions.get_node_id(gsm_item['id'])])
                    else:
                        has_compound_prt, norm_confidence = self.get_relationship_entities(
                            grouped_nodes,
                            gsm_json,
                            gsm_item,
                            has_compound_prt,
                            norm_confidence,
                            {'compound_prt'})

            # Determine conjugation type
            if has_conj:
                group_type = self.get_conj_group_type(group_type, gsm_item, gsm_json)

            if self.is_simplistic_rewriting and len(grouped_nodes) > 0:
                sorted_entities = sorted(grouped_nodes, key=lambda x: float(dict(x.properties)['pos']))
                sorted_entity_names = list(map(getattr, sorted_entities, repeat('named_entity')))

                all_types = list(map(getattr, sorted_entities, repeat('type')))
                specific_type = self.services.getHOnK().most_specific_type(all_types)

                if group_type == Grouping.OR:
                    name = " or ".join(sorted_entity_names)
                elif group_type == Grouping.AND:
                    name = " and ".join(sorted_entity_names)
                elif group_type == Grouping.NEITHER:
                    name = " nor ".join(sorted_entity_names)
                    name = f"neither {name}"
                else:
                    name = " ".join(sorted_entity_names)

                self.nodes[gsm_item['id']] = Singleton(
                    id=gsm_item['id'],
                    named_entity=name,
                    properties=frozenset(gsm_item['properties'].items()),
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    type=specific_type,
                    confidence=norm_confidence
                )
            elif not self.is_simplistic_rewriting:
                # if group_type is not None:
                #     for g_type in group_type:
                #         self.create_set_of_singletons(g_type, grouped_nodes, gsm_item, has_conj, has_multipleindobj, is_compound, has_compound_prt, norm_confidence, gsm_json, row)
                # else:
                #     self.create_set_of_singletons(group_type, grouped_nodes, gsm_item, has_conj, has_multipleindobj, is_compound, has_compound_prt, norm_confidence, gsm_json, row)
                self.create_set_of_singletons(group_type, grouped_nodes, gsm_item, has_conj, has_multipleindobj,
                                              is_compound, has_compound_prt, norm_confidence, gsm_json, row)

        self.remove_duplicate_nodes()

    def get_conj_group_type(self, group_type, gsm_item, gsm_json):
        conj = gsm_item['properties']['conj'].strip() if 'conj' in gsm_item['properties'] else ""
        if len(conj) == 0:
            from LaSSI.structures.internal_graph.from_raw_json_graph import bfs
            def f(nodes, id):
                cc_names = list(map(lambda y: nodes[y['score']['child']]['xi'][0],
                                    filter(lambda x: x['containment'] == 'cc', nodes[id]['phi'])))

                cc_properties = list(
                    map(lambda z: z['properties']['cc'] if 'cc' in z['properties'] else None, nodes.values()))
                cc_properties = [val for val in cc_properties if val is not None]
                cc_names.extend(cc_properties)

                if len(cc_names) > 0:
                    # TODO: Check if current node has another 'cc' edge, then add to cc_names
                    #  Then duplicate and add both AND and OR to properties
                    #  Might need improving...
                    # for edge in nodes[id]['phi']:
                    #     for inner_edge in nodes[edge['score']['child']]['phi']:
                    #         if 'cc' in inner_edge['containment']:
                    #             cc_names.append(nodes[inner_edge['score']['child']]['xi'][0])
                    return ' '.join(cc_names) #cc_names
                else:
                    return None

            conj = bfs(gsm_json, gsm_item['id'], f)
        # group_type = []
        # # TODO: If both AND/OR, only use OR (for time being)
        # if len(conj) > 0:
        #     for g_type in conj:
        #         group_type.append(self.get_group_enum(g_type))
        # else:
        #     group_type = [Grouping.NONE]
        # return group_type
        if 'and' in conj or 'but' in conj:
            group_type = Grouping.AND
        elif ('nor' in conj) or ('neither' in conj):
            group_type = Grouping.NEITHER
        elif 'or' in conj:
            group_type = Grouping.OR
        else:
            group_type = Grouping.NONE
        return group_type

    def get_group_enum(self, name):
        if 'and' in name or 'but' in name:
            group_type = Grouping.AND
        elif ('nor' in name) or ('neither' in name):
            group_type = Grouping.NEITHER
        elif 'or' in name:
            group_type = Grouping.OR
        elif 'not' in name:
            group_type = Grouping.NOT
        elif 'multipleindobj' in name:
            group_type = Grouping.MULTIINDIRECT
        else:
            group_type = Grouping.NONE
        return group_type

    # Phase 1.1
    def create_all_singleton_nodes(self, gsm_json, number_of_nodes):
        for row in number_of_nodes:
            gsm_item = gsm_json[row]
            if 'conj' in gsm_item['properties'] or 'multipleindobj' in gsm_item['ell']:
                continue  # Add SetOfSingletons later as we need ALL Singletons first
            else:
                self.generate_singleton(gsm_item, gsm_json)

    def generate_singleton(self, gsm_item, gsm_json):
        min_value = -1
        max_value = -1
        if len(gsm_item['xi']) > 0 and gsm_item['xi'][0] != '' and gsm_item['ell'][
            0] != '∃':  # TODO: Checking for ∃ might not be valid...
            name = gsm_item['xi'][0]
            min_value = int(gsm_item['properties']['begin'])
            max_value = int(gsm_item['properties']['end'])
            node_type = gsm_item['ell'][0] if len(gsm_item['ell']) > 0 else "None"
        else:
            # xi might be empty if the node is invented, therefore existential
            name = "?" + str(self.existentials.increaseAndGetExistential())
            node_type = 'existential'
        # If we have "root" in "ell", add it to properties
        if len(gsm_item['ell']) > 1:
            gsm_item['properties']['kernel'] = gsm_item['ell'][1]
        if len(gsm_item['xi']) > 1 and 'subjpass' in gsm_item['xi'][1]:
            gsm_item['properties']['subjpass'] = gsm_item['xi'][1]
        # Add 'det' to properties
        if len(gsm_item['ell']) > 0 and 'det' in gsm_item['ell']:
            gsm_item['properties']['det'] = gsm_item['ell'][0]
        self.nodes[gsm_item['id']] = Singleton(
            id=gsm_item['id'],
            named_entity=name,
            properties=frozenset(gsm_item['properties'].items()),
            min=min_value,
            max=max_value,
            type=node_type,
            confidence=1.0
        )
        # TODO: Need to resolve grouped entities before merging, is there a more elegant way to do this? Could this thus be removed from the phases?
        self.associteNodeToBestMeuMatches(self.nodes[gsm_item['id']])
        self.singletonTypeResolution(self.nodes[gsm_item['id']], gsm_json)

    # Phase 1.2
    def get_relationship_entities(self, grouped_nodes, gsm_json, gsm_item, has_relationship, norm_confidence,
                                  edge_relationship_label={'compound'}, deplete_phi=True):
        gsm_id_json = {row['id']: row for row in gsm_json}  # Associate ID to key value
        edges_to_remove = []
        for idx, edge in enumerate(gsm_item['phi']):
            is_current_edge_relationship = edge['containment'] in edge_relationship_label
            if is_current_edge_relationship:
                has_relationship = True
                # TODO: Patch: skipping if missing (but this has to be fixed)
                if not edge['score']['child'] in self.nodes:
                    continue
                child_node = self.nodes[edge['score']['child']]  ## Giacomo: BUG, child node might be missing.
                grouped_nodes.append(child_node)
                norm_confidence *= child_node.confidence

                child_gsm_item = gsm_id_json[edge['score']['child']]
                if len(child_gsm_item['phi']) > 0:
                    # Remove current compound edge
                    edges_to_remove.append(idx)
                    self.get_relationship_entities(grouped_nodes, gsm_json, child_gsm_item, has_relationship,
                                                   norm_confidence, edge_relationship_label)
                else:
                    edges_to_remove.append(idx)

        # Remove compound edges as no longer needed
        if deplete_phi:
            gsm_item['phi'] = [i for j, i in enumerate(gsm_item['phi']) if j not in edges_to_remove]
        return has_relationship, norm_confidence

    # Phase 1.3
    def create_set_of_singletons(self, group_type, grouped_nodes, gsm_item, has_conj, has_multipleindobj, is_compound, has_compound_prt, norm_confidence, gsm_json, pos):
        new_node = None
        if has_conj:
            new_id = self.node_functions.fresh_id_and_add_to_node_map(gsm_item['id'])
            new_node = SetOfSingletons(
                id=new_id,
                type=group_type,
                entities=tuple(grouped_nodes),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                confidence=norm_confidence,
                root=any(map(is_kernel_in_props, grouped_nodes))
            )
        elif is_compound:
            grouped_nodes.insert(0, self.nodes[self.node_functions.get_node_id(gsm_item['id'])])
            new_id = self.node_functions.fresh_id_and_add_to_node_map(gsm_item['id'])
            new_node = SetOfSingletons(
                id=new_id,
                type=Grouping.GROUPING,
                entities=tuple(grouped_nodes),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                confidence=norm_confidence,
                root=any(map(is_kernel_in_props, grouped_nodes))
            )
        elif has_multipleindobj:
            new_node = SetOfSingletons(
                id=gsm_item['id'],
                type=Grouping.MULTIINDIRECT,
                entities=tuple(grouped_nodes),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                confidence=norm_confidence,
                root=any(map(is_kernel_in_props, grouped_nodes))
            )
        elif has_compound_prt:
            grouped_nodes.insert(0, self.nodes[self.node_functions.get_node_id(gsm_item['id'])])
            new_id = self.node_functions.fresh_id_and_add_to_node_map(gsm_item['id'])

            sorted_entities = sorted(grouped_nodes, key=lambda x: float(dict(x.properties)['pos']))
            sorted_entity_names = list(map(getattr, sorted_entities, repeat('named_entity')))

            all_types = list(map(getattr, sorted_entities, repeat('type')))
            specific_type = self.services.getHOnK().most_specific_type(all_types)
            name = " ".join(sorted_entity_names)

            new_node = Singleton(
                id=new_id,
                named_entity=name,
                properties=frozenset(gsm_item['properties'].items()),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                type=specific_type,
                confidence=1
            )

        if new_node is not None:
            self.nodes = self.node_functions.add_to_nodes_at_pos(self.nodes, new_node, pos)

        node_to_resolve = self.nodes[self.node_functions.get_node_id(gsm_item['id'])]
        if isinstance(node_to_resolve, SetOfSingletons):
            self.resolveSpecificSetOfSingletonsNERs(node_to_resolve)
            self.checkForSpecificNegation(gsm_json, self.nodes[self.node_functions.get_node_id(gsm_item['id'])])

    # Phase 1.4
    def remove_duplicate_nodes(self):
        ids_to_remove = set()
        for idx, node in self.nodes.items():
            for jdx, another_node in self.nodes.items():
                if (idx != jdx and isinstance(node, SetOfSingletons) and
                        isinstance(another_node, SetOfSingletons) and
                        node.type == another_node.type):
                    if set(node.entities).issubset(set(another_node.entities)):
                        ids_to_remove.add(idx)
        for x in ids_to_remove:
            self.nodes.pop(x)

    # Phase 1.5
    def check_for_but(self, gsm_json):
        number_of_nodes = range(len(gsm_json))
        for row in number_of_nodes:
            gsm_item = gsm_json[row]
            but_gsm_item = None
            is_but = self.node_functions.is_node_but(gsm_item)  # Is the current node BUT

            # Check for if the current node has an EDGE that contains BUT
            if len(gsm_item['phi']) > 0:
                for edge in gsm_item['phi']:
                    but_gsm_item = self.node_functions.get_gsm_item_from_id(edge['score']['child'], gsm_json)
                    has_but = self.node_functions.is_node_but(but_gsm_item)

                    # Check if BUT has edges, if it does, then we will use those later
                    if len(but_gsm_item['phi']) > 0:
                        has_but = False

                    if has_but:
                        break
                    else:
                        but_gsm_item = None

            if is_but:  # If current node "BUT", then check if it has children
                norm_confidence = 1.0
                but_node = self.nodes[self.node_functions.get_node_id(gsm_item['id'])]
                grouped_nodes = []

                # Create group of all nodes attached to "but"
                for edge in gsm_item['phi']:
                    if 'neg' not in edge['containment']:
                        node = self.nodes[self.node_functions.get_node_id(edge['content'])]
                        grouped_nodes.append(node)
                        norm_confidence *= node.confidence

                self.create_but_group(but_node, grouped_nodes, gsm_item, gsm_json, norm_confidence)
            elif but_gsm_item is not None:  # This node has a BUT child, so therefore create group
                norm_confidence = 1.0
                parent_node = self.nodes[self.node_functions.get_node_id(gsm_item['id'])]
                grouped_nodes = [parent_node]
                self.create_but_group(parent_node, grouped_nodes, gsm_item, gsm_json, norm_confidence)

    def create_but_group(self, node, grouped_nodes, gsm_item, gsm_json, norm_confidence):
        # If node is a NOT and it equals the content of grouped nodes, then it is already a BUT and no need to do this again
        if isinstance(node, SetOfSingletons) and node.type == Grouping.NOT and node == grouped_nodes[0]:
            return
        if len(grouped_nodes) > 0:
            new_id = self.node_functions.fresh_id_and_add_to_node_map(gsm_item['id'])

            # Create an AND grouping from given child nodes
            new_but_node = SetOfSingletons(
                id=new_id,
                type=Grouping.AND,
                entities=tuple(grouped_nodes),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                confidence=norm_confidence
            )
            self.nodes[new_id] = new_but_node

            # If "but" has a negation, group the original grouped nodes in a NOT and then wrap again in an AND
            if self.checkForSpecificNegation(gsm_json, node, create_node=False):
                # Create new NOT node within nodes
                if grouped_nodes[0].id != gsm_item['id']:
                    new_new_id = self.node_functions.fresh_id_and_add_to_node_map(grouped_nodes[0].id)
                else:
                    new_new_id = self.node_functions.fresh_id()

                self.nodes[new_new_id] = SetOfSingletons(
                    id=new_new_id,
                    type=Grouping.NOT,
                    entities=tuple(grouped_nodes),
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    confidence=norm_confidence
                )

                self.nodes[new_id] = SetOfSingletons(
                    id=new_id,
                    type=Grouping.AND,
                    entities=[self.nodes[new_new_id]],
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    confidence=norm_confidence
                )
            # else:
            #     # Otherwise return the new node
            #     self.nodes[new_id] = new_but_node

    # Phase 2
    def associateNodeToMeuMatches(self, node):
        # If key is SetOfSingletons, loop over each Singleton and make association to type
        # Giacomo: FIX, but only if they are not logical predicates

        id_associations = [x.id for x in self.associations]
        if node.id not in id_associations:
            if isinstance(node, SetOfSingletons) and ((node.type == Grouping.NONE) or (node.type == Grouping.GROUPING)):
                for entity in node.entities:
                    if entity.id not in id_associations:
                        self.associateNodeToMeuMatches(entity)
            elif isinstance(node, Singleton):
                self.associteNodeToBestMeuMatches(node)

    # Phase 2.1
    def associteNodeToBestMeuMatches(self, item):
        # Loop over Stanza MEU and GSM result and evaluate overlapping words from chars
        for meu in self.meu_db_row.multi_entity_unit:
            start_meu = meu.start_char
            end_meu = meu.end_char
            start_graph = item.min
            end_graph = item.max
            # https://scicomp.stackexchange.com/questions/26258/the-easiest-way-to-find-intersection-of-two-intervals/26260#26260
            if start_graph > end_meu or start_meu > end_graph:
                continue
            else:
                if not (start_graph > end_meu or start_meu > end_graph):
                    self.meu_entities[item].add(meu)
                    self.associations.add(item)

    # Phase 3
    def singletonTypeResolution(self, item, gsm_json):
        if len(self.meu_entities[item]) > 0:
            best_item = None
            if item.type in {"PRONOUN", "PRP", "PRP$", "WP", "WP$"}:
                best_type = 'PRONOUN'
                best_score = 1
            elif item.type.startswith("JJ"):
                best_type = 'JJ'
                best_score = 1
            elif item.type == 'verb':
                best_type = 'verb'
                best_score = 1
            else:
                if item.type == '∃' or item.type.startswith("JJ") or item.type.startswith("IN") or item.type.startswith(
                        "NEG"):
                    best_score = item.confidence
                    best_item = item
                    best_type = item.type
                else:
                    # TODO: Fix typing information (e.g. Golden Gate Bridge has GPE (0.8) and ENTITY (1.0)
                    best_score = max(map(lambda y: y.confidence, self.meu_entities[item]))

                    # If the best_score is better than what we currently have for the Singleton
                    if item.confidence >= best_score and item.type.upper() in {'VERB', 'PERSON', 'DATE', 'GPE', 'LOC', 'ENTITY'}:
                        best_type = item.type.lower() if item.type == 'VERB' else item.type
                    else:
                        # TODO: min and max might not be correct when coming from an inherit edge
                        best_items = [
                            y for y in self.meu_entities[item]
                            if y.confidence == best_score
                        ]
                        if len(best_items) == 0:
                            return
                        if len(best_items) == 1:
                            best_item = best_items[0]
                            best_type = best_item.type
                        else:
                            best_types = list(set(map(lambda best_item: best_item.type, best_items)))
                            if len(best_types) == 1:
                                best_type = best_types[0]
                            ## TODO! type disambiguation, in future works, needs to take into account also the verb associated to it!
                            elif ("VERB" in best_types or "verb" in best_types) and (
                                    # If a node is marked with a det, never consider this as a verb
                                    'det' not in dict(item.properties)
                                    and
                                    # Prepositional objects are not verbs
                                    'case' not in dict(item.properties)
                                    and
                                    # TODO: This condition may need to be revised
                                    # 'on' is very unlikely to lead to a verb
                                    ('on' not in case_in_props(dict(item.properties), True))
                                    and
                                    ((
                                            # if a node has at least one ingoing edge and comes with a case-derived float attribute
                                            (len(self.node_functions.get_node_parents(item, gsm_json)) > 0 and any(case_in_props(self.node_functions.get_gsm_item_from_id(x, gsm_json)['properties']) for x in self.node_functions.get_node_parents(item, gsm_json)))
                                            or
                                            # If a node is the last occurring in the root/kernel of all kernels and has no ingoing edges
                                            (list(self.nodes)[-1] == self.node_functions.get_original_node_id(item.id, self.nodes) and is_kernel_in_props(item) and len(self.node_functions.get_node_parents(item, gsm_json)) == 0)
                                            or
                                            # TODO: Is this an acceptable condition?
                                            # If a node is last occurring, and has one parent which is a root connected by a compound edge, and no other ingoing edges
                                            (list(self.nodes)[-1] == self.node_functions.get_original_node_id(item.id, self.nodes) and len(self.node_functions.get_node_parents(item, gsm_json)) == 1 and self.node_functions.get_node_parents(item, gsm_json, True)[0][1] == "compound" and is_kernel_in_props(self.node_functions.get_gsm_item_from_id(self.node_functions.get_node_parents(item, gsm_json)[0], gsm_json)))
                                    ))
                            ):
                                best_type = "verb"
                            elif "PERSON" in best_types:
                                best_type = "PERSON"
                            elif "DATE" in best_types or "TIME" in best_types:
                                best_type = "DATE"
                            elif "GPE" in best_types:
                                best_type = "GPE"
                            elif "LOC" in best_types:
                                best_type = "LOC"
                            elif "ENTITY" in best_types:
                                best_type = "ENTITY"
                            else:
                                best_type = "None"
            from LaSSI.structures.internal_graph.EntityRelationship import Singleton
            self.nodes[item.id] = Singleton(
                id=item.id,
                named_entity=item.named_entity,  # TODO: Future work: best_item.monad if best_item is not None and not isinstance(best_item, Singleton) else item.named_entity TODO: Future work
                properties=item.properties,
                min=item.min,
                max=item.max,
                type=best_type,
                confidence=best_score
            )

    ## Phase 4
    def resolveGraphNERs(self):
        # TODO: This appears to be unnecessary
        # for key in self.nodes:
        #     node = self.nodes[key]
        #     # association = nbb[item]
        #     # best_score = association.confidence
        #     if not isinstance(node, SetOfSingletons):
        #         self.ConfidenceAndEntityExpand(key)
        #
        # for key in self.nodes:
        #     node = self.nodes[key]
        #     # association = nbb[item]
        #     # best_score = association.confidence
        #     if isinstance(node, SetOfSingletons):
        #         self.ConfidenceAndEntityExpand(key)

        # With types assigned, merge SetOfSingletons into Singleton
        for key in self.nodes:
            node = self.nodes[key]
            if isinstance(node, SetOfSingletons):
                from LaSSI.ner.MergeSetOfSingletons import GraphNER_withProperties
                if node.type == Grouping.MULTIINDIRECT:
                    # # If entities is only 1 item, then we can simply replace the item.id
                    # if len(node.entities) == 1:
                    #     entity = node.entities[0]
                    #     if isinstance(entity, SetOfSingletons):
                    #         self.nodes[node.id] = GraphNER_withProperties(entity, self.is_simplistic_rewriting, meu_db_row,
                    #                                                       honk, existentials)
                    #     else:
                    #         self.node_id_map[node.id] = entity.id
                    #         self.nodes[node.id] = self.ConfidenceAndEntityExpand(entity.id)
                    # # If more than 1 item, then we replace the entity.id for each orig of the 'multiindirectobj'
                    # else:
                    #     # nodes[item.id] = associate_to_container(item, nbb)
                    for entity in node.entities:
                        if isinstance(entity, SetOfSingletons):
                            self.nodes[entity.id] = GraphNER_withProperties(
                                entity,
                                self.is_simplistic_rewriting,
                                self.meu_db_row,
                                self.services.getHOnK(),
                                self.existentials
                            )
                        else:
                            self.node_functions.node_id_map[node.id] = entity.id
                            self.ConfidenceAndEntityExpand(entity.id)
                if node.type == Grouping.GROUPING:
                    self.nodes[key] = GraphNER_withProperties(
                        node,
                        self.is_simplistic_rewriting,
                        self.meu_db_row,
                        self.services.getHOnK(),
                        self.existentials
                    )

    def resolveSpecificSetOfSingletonsNERs(self, node_to_resolve):
        for node in node_to_resolve.entities:
            if not isinstance(node, SetOfSingletons):
                self.ConfidenceAndEntityExpand(node.id)

        if isinstance(node_to_resolve, SetOfSingletons):
            self.ConfidenceAndEntityExpand(node_to_resolve.id)

        from LaSSI.ner.MergeSetOfSingletons import GraphNER_withProperties
        if node_to_resolve.type == Grouping.MULTIINDIRECT:
            for entity in node_to_resolve.entities:
                if isinstance(entity, SetOfSingletons):
                    self.nodes[entity.id] = GraphNER_withProperties(
                        entity,
                        self.is_simplistic_rewriting,
                        self.meu_db_row,
                        self.services.getHOnK(),
                        self.existentials
                    )
                else:
                    self.node_functions.node_id_map[node_to_resolve.id] = entity.id
                    self.ConfidenceAndEntityExpand(entity.id)
        if node_to_resolve.type == Grouping.GROUPING:
            self.nodes[node_to_resolve.id] = GraphNER_withProperties(
                node_to_resolve,
                self.is_simplistic_rewriting,
                self.meu_db_row,
                self.services.getHOnK(),
                self.existentials
            )

    # Phase 4.1
    def ConfidenceAndEntityExpand(self, key):
        from LaSSI.structures.internal_graph.EntityRelationship import SetOfSingletons
        item = self.nodes[key]
        if isinstance(item, SetOfSingletons):
            confidence = 1.0
            new_set = list(map(lambda x: self.ConfidenceAndEntityExpand(x.id), item.entities))
            for entity in new_set:
                confidence *= entity.confidence
            set_item = SetOfSingletons(
                id=item.id,
                type=item.type,
                entities=tuple(new_set),
                min=item.min,
                max=item.max,
                confidence=confidence,
                root=item.root
            )
            self.nodes[key] = set_item
            return set_item
        else:
            if item.id not in self.nodes:
                self.nodes[item.id] = item
                return item
            else:
                return self.nodes[item.id]

    # Phase 5
    def checkForNegation(self, gsm_json):
        # Check for negation in node 'xi' and 'properties'
        grouped_nodes = []
        for row in range(len(gsm_json)):
            gsm_item = gsm_json[row]  # Find node in JSON so we can get child nodes
            key = gsm_item['id']
            node = self.nodes[self.node_functions.get_node_id(key)]  # Singleton node
            found_negation = False

            if node.type == Grouping.NOT or (node.type == Grouping.AND and node.entities[0].type == Grouping.NOT):
                return

            grouped_nodes.append(node)
            for edge in gsm_item['phi']:
                if edge['score']['child'] in self.nodes:  # Node might have been removed so check key exists
                    child = self.nodes[self.node_functions.get_node_id(edge['score']['child'])]
                    if (hasattr(child, "named_entity") and child.named_entity in self.negations) or child.type == 'NEG':
                        # grouped_nodes.append(child)
                        found_negation = True

            # Check if BUT has a property NEG
            if 'neg' in gsm_item['properties']:
                found_negation = True

            if found_negation:
                fresh_id = self.node_functions.fresh_id_and_add_to_node_map(key)
                self.nodes[fresh_id] = SetOfSingletons(
                    id=fresh_id,
                    type=Grouping.NOT,
                    entities=tuple(grouped_nodes),
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    confidence=1
                )

            grouped_nodes = []

    def checkForSpecificNegation(self, gsm_json, node, grouped_nodes=None, create_node=True):
        if node.type == Grouping.NOT and len(node.entities) == 1:
            node = node.entities[0]

        # if node.type == Grouping.NOT:
        #     return

        if isinstance(node, SetOfSingletons):
            # grouped_nodes = [node]
            new_entities = []
            for entity in node.entities:
                # self.checkForSpecificNegation(gsm_json, entity, grouped_nodes, create_node)
                new_entity = self.checkForSpecificNegation(gsm_json, entity, grouped_nodes, create_node)
                if isinstance(new_entity, SetOfSingletons):
                    new_entities.append(new_entity)
                else:
                    new_entities.append(entity)
            if len(new_entities) > 0:
                self.nodes[node.id] = SetOfSingletons(
                    id=node.id,
                    type=node.type,
                    entities=tuple(new_entities),
                    min=min(new_entities, key=lambda x: x.min).min,
                    max=max(new_entities, key=lambda x: x.max).max,
                    confidence=node.confidence,
                    root=node.root
                )
                node = self.nodes[node.id]
                return node

        if grouped_nodes is None:
            grouped_nodes = [node]

        # Check for negation in node 'xi' and 'properties'
        node_id = self.node_functions.get_original_node_id(node.id, self.nodes)
        gsm_item = self.node_functions.get_gsm_item_from_id(node_id, gsm_json)
        if gsm_item is not None:
            found_negation = False

            # Check if NOT is attached to BUT through an edge
            for edge in gsm_item['phi']:
                if edge['score']['child'] in self.nodes:  # Node might have been removed so check key exists
                    child = self.nodes[self.node_functions.get_node_id(edge['score']['child'])]
                    if (hasattr(child, "named_entity") and child.named_entity in self.negations) or child.type == 'NEG':
                        # grouped_nodes.append(child)
                        found_negation = True

            # Check if BUT has a property NEG
            if 'neg' in gsm_item['properties']:
                found_negation = True

            if found_negation:
                if create_node:
                    fresh_id = self.node_functions.fresh_id_and_add_to_node_map(node_id)
                    self.nodes[fresh_id] = SetOfSingletons(
                        id=fresh_id,
                        type=Grouping.NOT,
                        entities=tuple(grouped_nodes),
                        min=min(grouped_nodes, key=lambda x: x.min).min,
                        max=max(grouped_nodes, key=lambda x: x.max).max,
                        confidence=1
                    )
                    return self.nodes[fresh_id]
                else:
                    return True


    def constructIntermediateGraph(self, gsm_json, rejected_edges, non_verbs) -> Graph:
        self.edges = []

        # Add child node if 'amod' as properties of source node
        for row, gsm_item, edge in self.iterateOverEdges(gsm_json, rejected_edges):
            edge_label_name = edge['containment']  # Name of edge label
            if edge_label_name in {'appos'}:
                source_node = self.nodes[self.node_functions.get_node_id(edge['score']['parent'])]
                target_node = self.nodes[self.node_functions.get_node_id(edge['score']['child'])]

                if isinstance(target_node, SetOfSingletons):
                    current_grouped_nodes = list(target_node.entities)
                    current_grouped_nodes.insert(0, source_node)

                    self.nodes[source_node.id] = SetOfSingletons(
                        id=source_node.id,
                        type=target_node.type,
                        entities=tuple(current_grouped_nodes),
                        min=min(current_grouped_nodes, key=lambda x: x.min).min,
                        max=max(current_grouped_nodes, key=lambda x: x.max).max,
                        confidence=1,
                        root=target_node.root
                    )
            elif edge_label_name in {'amod', 'advmod', 'case'}:  # TODO: Is this 'amod' or just 'mod' as could have 'nmod' (e.g. (traffic)-[nmod]->(Newcastle)
                source_node = self.nodes[self.node_functions.get_node_id(edge['score']['parent'])]
                target_node = self.nodes[self.node_functions.get_node_id(edge['score']['child'])]
                temp_prop = dict(copy(source_node.properties))
                if isinstance(target_node, Singleton):
                    if edge_label_name != 'case':
                        type_key = self.node_functions.get_node_type(target_node)
                    else:
                        type_key = 'case'
                else:
                    from LaSSI.external_services.Services import Services
                    type_key = self.services.getHOnK().most_general_type(
                        map(lambda x: x.type, target_node.entities))

                if type_key != 'existential':
                    if type_key not in temp_prop:
                        temp_prop[type_key] = (target_node,)
                    elif not isinstance(temp_prop[type_key], list):
                        temp_prop[type_key] = (temp_prop[type_key], target_node)
                    else:
                        temp_prop[type_key] += (target_node,)

                    self.nodes[self.node_functions.get_node_id(edge['score']['parent'])] = Singleton(
                        id=source_node.id,
                        named_entity=source_node.named_entity,
                        properties=frozenset(temp_prop.items()),
                        min=source_node.min,
                        max=source_node.max,
                        type=source_node.type,  # TODO: This type is the ENUM case (i.e. 3 instead of NOT) so use dacite
                        confidence=source_node.confidence
                    )

        for row in range(len(gsm_json)):
            gsm_item = gsm_json[row]

            if len(gsm_item['phi']) == 0:
                source_node_id = gsm_item['id']
                target_node_id = None
                source_node = self.node_functions.resolve_node_id(source_node_id, self.nodes)
                if isinstance(source_node, Singleton):
                    source_properties = dict(source_node.properties)
                    if 'action' in source_properties:
                        edge_label_name = source_properties['action']
                        is_verb = is_label_verb(edge_label_name)
                        if is_verb:
                            self.create_edges(edge_label_name, gsm_item, non_verbs, source_node_id, target_node_id, gsm_json)

            for edge in gsm_item['phi']:
                source_node_id = edge['score']['parent']
                target_node_id = None

                # Check if child node has 'multipleindobj' type, to reassign edges from each 'orig'
                child_gsm_item = self.node_functions.get_gsm_item_from_id(edge['score']['child'], gsm_json)
                if 'multipleindobj' in child_gsm_item['ell']:
                    edge_label_name = edge['containment']  # Name of edge label

                    source_node_id = edge['score']['parent']

                    for child_edge in child_gsm_item['phi']:
                        if 'orig' in child_edge['containment']:
                            target_node_id = child_edge['score']['child']
                            self.create_edges(edge_label_name, gsm_item, non_verbs, source_node_id, target_node_id, gsm_json)

                # Make sure current edge is not in list of rejected edges, e.g. 'compound'
                if edge['containment'] not in rejected_edges:
                    if 'orig' not in edge['containment']:
                        edge_label_name = edge['containment']  # Name of edge label
                        if self.node_functions.get_node_id(edge['score']['child']) != target_node_id:
                            target_node_id = edge['score']['child']
                            self.create_edges(edge_label_name, gsm_item, non_verbs, source_node_id, target_node_id, gsm_json)

        # print(json.dumps(Graph(self.edges), cls=EnhancedJSONEncoder))
        # print("\nOG")
        # for edge in self.edges:
        #     print(
        #         f"[({edge.source.id}): {edge.source.get_node_string()}] --({'NOT(' if edge.isNegated else ''}{edge.edgeLabel.named_entity}{')' if edge.isNegated else ''})--> [({edge.target.id}):, {edge.target.get_node_string()}]")

        return Graph(self.edges)


    def iterateOverEdges(self, gsm_json, rejected_edges):
        for row in range(len(gsm_json)):
            gsm_item = gsm_json[row]
            for edge in gsm_item['phi']:
                # Make sure current edge is not in list of rejected edges, e.g. 'compound'
                if edge['containment'] not in rejected_edges:
                    if 'orig' not in edge['containment']:
                        yield row, gsm_item, edge

    def create_edges(self, edge_label_name, gsm_item, non_verbs, source_node_id, target_node_id, gsm_json):
        source_node = self.node_functions.resolve_node_id(source_node_id, self.nodes)
        target_node = self.node_functions.resolve_node_id(target_node_id, self.nodes)

        has_negations = does_string_have_negations(edge_label_name)

        # Giacomo: the former code was not able to handle: "does not play"
        query_words = edge_label_name.split()
        result_words = [word for word in query_words if word.lower() not in self.negations]
        edge_label_name = ' '.join(result_words)

        # Check if name of edge is in "non verbs"
        non_verb_set = {nv.strip() for nv in non_verbs}
        if edge_label_name in non_verb_set:
            edge_type = "non_verb"
        else:
            edge_type = "verb"

        node_min = -1
        node_max = -1
        if 'properties' in gsm_item:
            if 'begin' in gsm_item['properties']:
                node_min = gsm_item['properties']['begin']
            if 'end' in gsm_item['properties']:
                node_max = gsm_item['properties']['end']

        rejected_edges = {'amod', 'advmod', 'adv', 'neg', 'conj', 'cc', 'case', 'none', 'appos', 'compound_prt'}
        target_gsm_item = self.node_functions.get_gsm_item_from_id(self.node_functions.get_node_id(target_node.id), gsm_json)

        if self.nodes[self.node_functions.get_node_id(source_node_id)].type == Grouping.MULTIINDIRECT:
            # TODO: I don't think this logic is correct (e.g. The mouse is eaten by the cat)
            for node in self.nodes[self.node_functions.get_node_id(source_node_id)].entities:
                self.edges.append(Relationship(
                    source=source_node,
                    target=self.node_functions.resolve_node_id(node.id, self.nodes),
                    edgeLabel=Singleton(
                        id=gsm_item['id'],
                        named_entity=edge_label_name,
                        properties=frozenset(dict().items()),
                        min=node_min,
                        max=node_max,
                        type=edge_type,
                        confidence=self.nodes[gsm_item['id']].confidence),
                    isNegated=has_negations
                ))
        elif edge_label_name not in rejected_edges or (edge_label_name in rejected_edges and target_gsm_item is not None and len(target_gsm_item['phi']) > 0):
            self.edges.append(Relationship(
                source=source_node,
                target=target_node,
                edgeLabel=Singleton(
                    id=gsm_item['id'],
                    named_entity=edge_label_name,
                    properties=frozenset(dict().items()),
                    min=node_min,
                    max=node_max,
                    type=edge_type,
                    confidence=self.nodes[self.node_functions.get_node_id(gsm_item['id'])].confidence
                ),
                isNegated=has_negations
            ))