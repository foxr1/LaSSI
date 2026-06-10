__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2025, Oliver R. Fox"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"
__email__ = "ollie.fox5@gmail.com"
__status__ = "Production"

import re
import time
from collections import defaultdict
from copy import copy
from types import SimpleNamespace

import networkx as nx
import itertools

from LaSSI.ner.node_functions import create_existential_node, create_props_for_singleton, get_min_position
from LaSSI.ner.string_functions import lemmatize_verb, check_semi_modal, lemmatize_sentence, is_position_key
from LaSSI.structures import DependencyRoles
from LaSSI.structures.internal_graph.EntityRelationship import Relationship, Singleton, SetOfSingletons, Grouping
from LaSSI.external_services.Services import Services
from LaSSI.tests.benchmark import Benchmark

# @dataclass(order=True, frozen=True, eq=True)
# class Sentence(Singleton):
#     kernel: Relationship
#     properties: dict = field(default_factory=lambda: {
#         'time': List[NodeEntryPoint],
#         'loc': List[NodeEntryPoint]
#     })
#
#     # @classmethod
#     # def from_dict(cls, c):
#     #     return cls(kernel=Relationship.from_dict(c.get('kernel')),
#     #                properties={k: [deserialize_NodeEntryPoint(x) for x in v] for k, v in c.get('properties').items()}
#     #                )
copula_types = DependencyRoles.copula_complement_pos_tags()


def _node_data(node_entry):
    if isinstance(node_entry, dict):
        return node_entry.get('data')
    return node_entry


def replaceNamed(entity: Singleton, s: str) -> Singleton:
    return Singleton(id=entity.id,
                     named_entity=s,
                     properties=entity.properties,
                     min=entity.min,
                     max=entity.max,
                     type=entity.type,
                     confidence=entity.confidence)


def create_existential(G, nodes, node_functions):
    for key in nodes:
        node = nodes[key]['data']
        if isinstance(node, SetOfSingletons) and len(node.entities) == 1:
            node = node.entities[0]

        # node_props = dict(node.properties)
        if is_kernel_in_props(node) or len(nodes) == 1:
            nx.set_node_attributes(G, {key: node}, 'data')

            if node.type == 'verb':
                node_props = dict(node.properties)
                new_target = None
                if 'extra' in node_props:
                    # If node has an extra, make as target
                    new_target = node_props['extra'][0]
                    node_props.pop('extra')
                    node = node.update_node_props(node_props)

                ex_node = node_functions.create_existential_node(G)
                if new_target is not None:
                    G.add_node(new_target.id, data=new_target)
                    # G.add_edge(ex_node.id, new_target.id, label=node, isNegated=False)
                    return G, Relationship(
                        source=ex_node,
                        target=G.nodes[new_target.id]['data'],
                        edgeLabel=node,
                        isNegated=False
                    )
                else:
                    none_id = node_functions.fresh_id()
                    G.add_node(none_id, data=None)
                    # G.add_edge(ex_node.id, none_id, label=node, isNegated=False)
                    return G, Relationship(
                        source=ex_node,
                        target=G.nodes[none_id]['data'],
                        edgeLabel=node,
                        isNegated=False
                    )
                # return G, (new_id, none_id, 0)

            else:
                ex_node = node_functions.create_existential_node(G)

                if len(G.edges) == 0:
                    G.add_edge(key, ex_node.id, label=Singleton(
                        id=-1,
                        named_entity="is",
                        properties=frozenset(dict().items()),
                        min=-1,
                        max=-1,
                        type="verb",
                        confidence=-1
                    ), isNegated=False)
                # return G, (key, new_id, 0)
                return G, Relationship(
                    source=G.nodes[key]['data'],
                    target=ex_node,
                    edgeLabel=Singleton(
                        id=-1,
                        named_entity="is",
                        properties=frozenset(dict().items()),
                        min=-1,
                        max=-1,
                        type="verb",
                        confidence=-1
                    ), isNegated=False)
    return G, None


def create_cop(node, kernel, target_or_source):
    _tgt_before = kernel.target.named_entity if kernel.target else None
    _lbl = kernel.edgeLabel.named_entity if kernel.edgeLabel else None
    if target_or_source == 'target':
        # TODO: Ollie: Is it correct to say if target is None add to source otherwise add to target?
        if kernel.target is None:
            temp_prop = dict(copy(kernel.source.properties))
            temp_prop['cop'] = [node]

            new_source = Singleton(
                id=kernel.source.id,
                named_entity=kernel.source.named_entity,
                properties=create_props_for_singleton(temp_prop),
                min=kernel.source.min,
                max=kernel.source.max,
                type=kernel.source.type,
                confidence=kernel.source.confidence
            )
            kernel = Relationship(
                source=new_source,
                target=kernel.target,
                edgeLabel=kernel.edgeLabel,
                isNegated=kernel.isNegated
            )
        else:
            # if isinstance(node, SetOfSingletons):
            #     temp_prop = dict(copy(node.entities[0].properties))
            # else:
            temp_prop = dict(copy(kernel.target.properties))
            temp_prop['cop'] = [node]

            if isinstance(kernel.target, Singleton):
                new_target = Singleton(
                    id=kernel.target.id,
                    named_entity=kernel.target.named_entity,
                    properties=create_props_for_singleton(temp_prop),
                    min=kernel.target.min,
                    max=kernel.target.max,
                    type=kernel.target.type,
                    confidence=kernel.target.confidence
                )
            else:
                new_target = kernel.target

            # Only add the copula if it differs from the target name
            kernel_target = new_target
            if isinstance(kernel.target,
                          Singleton) and kernel.target is not None and kernel.target.named_entity == node.named_entity:
                kernel_target = kernel.target

            kernel = Relationship(
                source=kernel.source,
                target=kernel_target,
                edgeLabel=kernel.edgeLabel,
                isNegated=kernel.isNegated
            )
    else:
        temp_prop = dict(copy(kernel.source.properties))
        temp_prop['cop'] = [node]

        # Only add the copula if it differs from the target name
        kernel_target = kernel.target
        if isinstance(kernel.target,
                      Singleton) and kernel.target is not None and kernel.target.named_entity == node.named_entity:
            kernel_target = None

        new_source = Singleton(
            id=kernel.source.id,
            named_entity=kernel.source.named_entity,
            properties=create_props_for_singleton(temp_prop),
            min=kernel.source.min,
            max=kernel.source.max,
            type=kernel.source.type,
            confidence=kernel.source.confidence
        )
        kernel = Relationship(
            source=new_source,
            target=kernel_target,
            edgeLabel=kernel.edgeLabel,
            isNegated=kernel.isNegated
        )
    return kernel


_VERBLESS_POST_COLON_LABELS = frozenset({'dep', 'obl', 'obj', 'dobj', 'xcomp', 'ccomp'})


def _is_verbless_nominal_sentence(root_node, edges, root_sentence_id):
    if getattr(root_node, 'type', None) == 'verb':
        return False
    root_props = dict(root_node.properties) if hasattr(root_node, 'properties') else {}
    punct_vals = root_props.get('punct', [])
    if isinstance(punct_vals, str): punct_vals = [punct_vals]
    if ':' not in punct_vals:
        return False
    for edge in edges:
        if edge[0] == root_sentence_id and edge[3]['label'].named_entity in _VERBLESS_POST_COLON_LABELS:
            return True
    return False


def _build_implicit_be_kernel(root_node):
    be_label = Singleton(
        id=-1,
        named_entity="be",
        properties=frozenset(dict().items()),
        min=root_node.min,
        max=root_node.max,
        type="verb",
        confidence=1.0,
    )
    return Relationship(
        source=root_node,
        target=create_existential_node(),
        edgeLabel=be_label,
        isNegated=False,
    )


def _append_unique_syntactic_property(properties, key, value):
    existing = properties.get(key, [])
    if not isinstance(existing, list):
        existing = [existing]
    value_id = getattr(value, 'id', None)
    value_name = getattr(value, 'named_entity', None)
    for item in existing:
        if value_id is not None and getattr(item, 'id', None) == value_id:
            properties[key] = existing
            return
        if value_name is not None and getattr(item, 'named_entity', None) == value_name:
            properties[key] = existing
            return
        if item == value:
            properties[key] = existing
            return
    existing.append(value)
    properties[key] = existing


def _seed_implicit_nominal_properties(properties, root_node, edges, G, root_sentence_id):
    """Keep the syntactic evidence of a colon-style nominal clause available
    for later structural rewrites.

    The interpretation of these properties belongs to the rewrite layer.  This
    constructor only records the root and its direct dependents under ordinary
    syntactic keys so post-processing can decide whether they are status,
    lifecycle, or something else.
    """
    _append_unique_syntactic_property(properties, "noun", root_node)
    for edge in edges:
        if edge[0] != root_sentence_id:
            continue
        edge_label = edge[3].get('label')
        if edge_label is None:
            continue
        edge_target = G.nodes[edge[1]]['data']
        _append_unique_syntactic_property(properties, edge_label.named_entity, edge_target)


def create_sentence(G, edges, nodes, negations, root_sentence_id, found_preposition_labels, node_functions,
                    prev_loop_settings, acl_relcl_map):
    edges = list(edges)
    loop_settings = SimpleNamespace(shouldLoop=False, edgeForKernel=None, previousKernel=None)

    root_node = _node_data([node for node in nodes.items() if node[0] == root_sentence_id][0][1])
    if root_node.type == 'verb' and len(edges) <= 0:
        return G, create_edge_kernel(G, root_node, node_functions), loop_settings, acl_relcl_map
    elif len(edges) == 1:
        if edges[0][3]['label'].named_entity == 'adv':
            return G, root_node, loop_settings, acl_relcl_map
    elif len(edges) <= 0:
        return G, root_node, loop_settings, acl_relcl_map

    # nodes = {node[0]: node[1]['data'] for node in nodes}
    # edges = [Relationship(
    #         source=G.nodes[edge[0]]['data'],
    #         target=G.nodes[edge[1]]['data'],
    #         edgeLabel=edge[2]['label'],
    #         isNegated=edge[2]['isNegated']
    #     ) for edge in edges]

    # With graph created, make the 'Sentence' object
    kernel = None
    # Verbless nominal-sentence pattern: a non-verb root with a `:` punct marker
    # and at least one `dep` child (Stanza's generic dependency for post-colon
    # content). The colon is a zero-copula ("forecast for X: rain, wind, ..."
    # = "the forecast IS rain, wind, ..."). Pre-commit to be(root, ?) so
    # kernel selection can't grab a spurious downstream verb candidate
    # (e.g. Tyne via the `upon` case marker).
    implicit_verbless_nominal = _is_verbless_nominal_sentence(root_node, edges, root_sentence_id)
    if implicit_verbless_nominal:
        kernel = _build_implicit_be_kernel(root_node)
    else:
        G, kernel = assign_kernel(G, edges, kernel, negations, nodes, root_sentence_id, found_preposition_labels,
                                  node_functions)  # Phase 3.2
    kernel_nodes = set()
    properties = defaultdict(list)
    if implicit_verbless_nominal:
        _seed_implicit_nominal_properties(properties, root_node, edges, G, root_sentence_id)

    if kernel is not None:
        # Add relevant nodes to kernel_nodes, and check if source or target should be a pronoun
        if kernel.source is not None:
            kernel, kernel_nodes = analyse_kernel_node(kernel, kernel_nodes, "source")
        if kernel.target is not None:
            kernel, kernel_nodes = analyse_kernel_node(kernel, kernel_nodes, "target")

    # Create properties, rewrite relevant edges, check if we find a property that should be a kernel
    for edge in edges:
        edge_source = G.nodes[edge[0]]['data']
        edge_target = G.nodes[edge[1]]['data']
        edge_label = edge[3]['label']
        is_negated = edge[3]['isNegated']

        # Source
        kernel, properties, kernel_nodes = add_to_properties(
            kernel, edge_source, 'source', kernel_nodes, properties, negations, node_functions)

        # Target
        kernel, properties, kernel_nodes = add_to_properties(
            kernel, edge_target, 'target', kernel_nodes, properties, negations, node_functions)

        # Lemmatize edge name (verbs only — structural dependency labels like
        # `acl_relcl`/`nmod`/`obl` must not be passed through Stanza, which can
        # capitalize unknown tokens and break the dependency-role lookups below).
        if edge_label is not None and getattr(edge_label, 'type', None) == 'verb':
            lemmatized_name = lemmatize_verb(edge_label.named_entity)
            updated_edge = edge_label.update_name(lemmatized_name)
            G = update_edge(G, edge, updated_edge)
            edge_label = updated_edge

        # Add certain edges to be rewritten later
        edge_label_name = edge_label.named_entity
        if edge_label_name in DependencyRoles.nominal_modifier_edges():
            if edge_label_name in {'acl_relcl', 'acl'}:
                acl_relcl_map[edge_target.id] = edge_source

            # For acl edges: detect a participial sub-clause body (verb edge from the
            # synthetic acl head, e.g. "with the offender gaining access through a rear
            # entrance" — node 24's outgoing "gaining" edges carry the real verb).
            # Collect ALL verb-labeled out-edges so obliques (e.g. obl `entrance[through]`)
            # can be promoted onto the sub-kernel as properties instead of being dropped.
            participial_verb_edges = []
            participial_verb_edge = None
            if (
                    edge_label_name == 'acl' and
                    isinstance(edge_target, Singleton) and
                    edge_target.id in G.nodes
            ):
                for u, v, data in G.out_edges(edge_target.id, data=True):
                    lbl = data.get('label')
                    if lbl is not None and getattr(lbl, 'type', None) == 'verb':
                        participial_verb_edges.append((u, v, data, lbl))

                # Primary target: prefer a child without a case preposition (true obj),
                # else fall back to the first verb-out edge.
                for cand in participial_verb_edges:
                    cand_dst = G.nodes[cand[1]]['data']
                    if isinstance(cand_dst, Singleton) and not case_in_props(dict(cand_dst.properties)):
                        participial_verb_edge = cand
                        break
                if participial_verb_edge is None and participial_verb_edges:
                    participial_verb_edge = participial_verb_edges[0]

            # For acl_relcl edges: the target is the relcl verb itself (e.g. "requires"
            # in "station which requires scaffolding"). Reconstruct its predicate from
            # its own nsubj/obj children so downstream HOnK rules (e.g.
            # is_consumption → REQUIREMENT) can classify the sub-kernel by its verb
            # rather than seeing a bare verb Singleton with no arguments.
            relcl_sub_source = None
            relcl_sub_target = None
            if (
                    edge_label_name == 'acl_relcl' and
                    isinstance(edge_target, Singleton) and
                    getattr(edge_target, 'type', None) == 'verb' and
                    edge_target.id in G.nodes
            ):
                for _, v, data in G.out_edges(edge_target.id, data=True):
                    lbl = data.get('label')
                    lbl_name = getattr(lbl, 'named_entity', None) if lbl is not None else None
                    if lbl_name == 'nsubj' and relcl_sub_source is None:
                        relcl_sub_source = G.nodes[v]['data']
                    elif lbl_name in {'obj', 'dobj'} and relcl_sub_target is None:
                        relcl_sub_target = G.nodes[v]['data']

            # Skip the SENTENCE property only when the source already carries a case
            # preposition and the acl has no participial body (e.g. "due to X" alone).
            skip_sentence_property = (
                    edge_label_name == 'acl' and
                    isinstance(edge_source, Singleton) and
                    case_in_props(dict(edge_source.properties)) and
                    participial_verb_edge is None
            )

            if not skip_sentence_property:
                kernel_source = kernel.source
                kernel_target = kernel.target
                valid_nodes = [kernel_source, kernel_target]

                # If we have a participial verb sub-clause, rewrite the SENTENCE so that
                # the predicate is the participial verb (e.g. "gaining"), with the source
                # noun (e.g. "offender") as subject and the verb's object as target.
                # This avoids the acl_replacement step dropping the SENTENCE because its
                # target id is in acl_relcl_map, and forces the SENTENCE key (not the
                # verb-named edgeLabel) so filter_invalid_property_keys keeps it.
                add_props_type_key = None
                if participial_verb_edge is not None and edge_label_name == 'acl':
                    _, sub_target_id, _, sub_verb_label = participial_verb_edge
                    sub_target = G.nodes[sub_target_id]['data']

                    # Promote any remaining verb-out children of the participial head
                    # (typically obl/nmod with a case preposition) onto the sub-kernel
                    # as properties keyed by the child's node type, so logical
                    # rewriting can later classify them via the case prep instead of
                    # silently discarding them.
                    sub_properties = defaultdict(list)
                    for other in participial_verb_edges:
                        if other is participial_verb_edge:
                            continue
                        other_dst = G.nodes[other[1]]['data']
                        if other_dst is None:
                            continue
                        sub_type_key = node_functions.get_node_type(other_dst)
                        if other_dst not in sub_properties[sub_type_key]:
                            sub_properties[sub_type_key].append(other_dst)

                    edge_kernel = Singleton(
                        id=sub_verb_label.id,
                        named_entity="",
                        type="SENTENCE",
                        min=node_functions.get_min_from_nodes(valid_nodes),
                        max=node_functions.get_max_from_nodes(valid_nodes),
                        confidence=1,
                        kernel=Relationship(
                            source=remove_acl_relcl_relationship(edge_source),
                            target=remove_acl_relcl_relationship(sub_target),
                            edgeLabel=sub_verb_label,
                            isNegated=edge[3]['isNegated']
                        ),
                        properties=create_props_for_singleton(sub_properties),
                    )
                    add_props_type_key = 'SENTENCE'
                elif edge_label_name == 'acl_relcl' and (relcl_sub_source is not None or relcl_sub_target is not None):
                    # Rebuild "station which requires scaffolding" as a SENTENCE
                    # property whose edgeLabel is the relcl verb itself, with the
                    # verb's nsubj/obj resolved against the graph. Downstream HOnK
                    # classification (e.g. ConsumptionVerb → REQUIREMENT) keys off
                    # this edgeLabel, and `RequirementClauseSimplifierRule` then
                    # collapses the wh-pronoun source down to the bare object.
                    edge_kernel = Singleton(
                        id=edge_target.id,
                        named_entity="",
                        type="SENTENCE",
                        min=node_functions.get_min_from_nodes(valid_nodes),
                        max=node_functions.get_max_from_nodes(valid_nodes),
                        confidence=1,
                        kernel=Relationship(
                            source=remove_acl_relcl_relationship(relcl_sub_source) if relcl_sub_source is not None else None,
                            target=remove_acl_relcl_relationship(relcl_sub_target) if relcl_sub_target is not None else None,
                            edgeLabel=edge_target,
                            isNegated=edge[3]['isNegated']
                        ),
                        properties=frozenset(dict()),
                    )
                    add_props_type_key = 'SENTENCE'
                else:
                    # Rewrite edge as a kernel (relative clause / standard nominal modifier)
                    edge_kernel = Singleton(
                        id=edge_label.id,
                        named_entity="",
                        type="SENTENCE",
                        min=node_functions.get_min_from_nodes(valid_nodes),
                        max=node_functions.get_max_from_nodes(valid_nodes),
                        confidence=1,
                        kernel=Relationship(
                            source=remove_acl_relcl_relationship(edge_source),
                            target=remove_acl_relcl_relationship(edge_target),
                            edgeLabel=edge_label,
                            isNegated=edge[3]['isNegated']
                        ),
                        properties=frozenset(dict()),
                    )

                kernel, properties, kernel_nodes = add_to_properties(
                    kernel, edge_kernel, 'edgeLabel', kernel_nodes, properties, negations, node_functions,
                    type_key=add_props_type_key)

        # If we have an edge that is a verb and not already in the kernel nodes, use this as the "edge to loop", out the next iteration on the same root node
        from LaSSI.ner.structural_rewrites.base import is_canonical_copula
        kernel_edge_label = kernel.edgeLabel
        _src_is_acl_child = any(
            data.get('label') is not None and data['label'].named_entity in {'acl', 'acl_relcl'}
            for _, _, data in G.in_edges(edge[0], data=True)
        )
        if (edge_label is not None and
                edge_label.type == 'verb' and not is_canonical_copula(edge_label.named_entity) and
                kernel_edge_label is not None and
                kernel_edge_label.named_entity != edge_label_name and
                not is_node_in_kernel_nodes(edge_label, kernel_nodes) and
                edge != prev_loop_settings.edgeForKernel and
                not _src_is_acl_child):
            loop_settings = SimpleNamespace(
                shouldLoop=True,
                edgeForKernel=(edge[0], edge[1], edge[2], {'label': edge_label, 'isNegated': is_negated}),
                # edgeForKernel=Relationship(
                #     source=G.nodes[edge[0]]['data'],
                #     target=G.nodes[edge[1]]['data'],
                #     edgeLabel=edge_label,
                #     isNegated=is_negated
                # ),
                previousKernel=None
            )

    # If we have a kernel returned from the previous loop, add this to the properties
    if prev_loop_settings.previousKernel is not None:
        # Check which order the kernel should be:
        #   Should we add the NEW kernel as a property, or
        #    remain as the root kernel, based on the topological position of the nodes
        top_node_id_positions = {
            node_data.id: idx
            for idx, node_entry in enumerate(nodes.values())
            for node_data in [_node_data(node_entry)]
            if hasattr(node_data, 'id')
        }
        kernel_id_to_check = get_kernel_top_id(kernel, top_node_id_positions)
        returned_kernel_id_to_check = get_kernel_top_id(prev_loop_settings.previousKernel.kernel, top_node_id_positions)

        if (
                # TODO: PATCH kernel_id_to_check is None or returned_kernel_id_to_check is None or  ## GIACOMO: This is a patch. TODO: handle the case
                kernel_id_to_check is not None and (returned_kernel_id_to_check is None or
                                                    top_node_id_positions[kernel_id_to_check] < top_node_id_positions[
                                                        returned_kernel_id_to_check])
        ):
            kernel, properties, kernel_nodes = add_to_properties(
                prev_loop_settings.previousKernel.kernel,
                node_functions.convert_relationship_to_sentence(root_sentence_id, kernel),
                'target', kernel_nodes, properties, negations, node_functions
            )
        else:
            kernel, properties, kernel_nodes = add_to_properties(
                kernel, prev_loop_settings.previousKernel,
                'target', kernel_nodes, properties, negations, node_functions
            )

    properties_to_keep = defaultdict()
    new_kernel = None
    for key in properties:
        if key == 'verb':  # TODO: Check for "NOT" prop / SetOfSingletons
            verbs_to_keep = []
            for node in properties['verb']:
                if (not isinstance(node, SetOfSingletons) and
                        not 'mark' in node.properties and root_sentence_id == node.id):
                    new_kernel = node
                else:
                    verbs_to_keep.append(node)
            if len(verbs_to_keep) > 0:
                properties_to_keep[key] = verbs_to_keep
        else:
            properties_to_keep[key] = properties[key]

    final_kernel = node_functions.convert_relationship_to_sentence(root_sentence_id, kernel, properties_to_keep)

    if new_kernel is not None:
        valid_nodes = node_functions.get_valid_nodes(
            [kernel.source, kernel.target])  # Get all nodes that are not None else return -1
        final_kernel = Singleton(
            id=root_sentence_id,
            named_entity="",
            type="SENTENCE",
            min=node_functions.get_min_from_nodes(valid_nodes),
            max=node_functions.get_max_from_nodes(valid_nodes),
            confidence=1,  # TODO: Should this always be 1?
            kernel=Relationship(
                source=create_existential_node(),
                target=final_kernel,
                edgeLabel=new_kernel,
                isNegated=new_kernel.isNegated if hasattr(new_kernel, "isNegated") else False,
                # TODO: Check this is correctly negated
            ),
            properties=create_props_for_singleton(properties_to_keep),
        )
    if loop_settings.shouldLoop:
        loop_settings.previousKernel = final_kernel

    return G, final_kernel, loop_settings, acl_relcl_map


def update_edge(G, edge, updated_edge):
    nx.set_edge_attributes(G, {
        (edge[0], edge[1], edge[2]): {'label': updated_edge, 'isNegated': edge[3]['isNegated']}})
    return G


def remove_acl_relcl_relationship(node):
    if isinstance(node, SetOfSingletons):
        return node
    if node.kernel is not None and node.kernel.edgeLabel.named_entity in {'acl_relcl', 'acl'}:
        source_node_pos = get_min_position(node.kernel.source)
        target_node_pos = get_min_position(node.kernel.target)

        if target_node_pos is None or source_node_pos < target_node_pos:
            return node.kernel.source
        else:
            return node.kernel.target

    return node


def get_kernel_top_id(kernel, topological_node_id_positions):
    return kernel.source.id if kernel.source.id in topological_node_id_positions else kernel.target.id if kernel.target is not None and kernel.target.id in topological_node_id_positions else None


def analyse_kernel_node(kernel, kernel_nodes, kernel_node_type):
    if kernel_node_type == 'source':
        kernel_node = kernel.source
    else:
        kernel_node = kernel.target

    kernel_nodes = add_to_kernel_nodes(kernel_node, kernel_nodes)

    if isinstance(kernel_node, Singleton):
        # If source/target is 'det' or adjective AND a pronoun of any type
        if ((
                ('det' in dict(kernel_node.properties) and
                 dict(kernel_node.properties)['det'] == 'det') or
                kernel_node.type == 'JJ' or
                kernel_node.type == 'JJS'
        ) and len(
            Services.getInstance().getHOnK().getPronouns().intersection({kernel_node.named_entity.lower()})) != 0):
            kernel_node = Singleton(
                id=kernel_node.id,
                named_entity=kernel_node.named_entity,
                properties=kernel_node.properties,
                min=kernel_node.min,
                max=kernel_node.max,
                type="PRON",
                confidence=kernel_node.confidence
            )
            if kernel_node_type == 'source':
                kernel = kernel.update_vertex(kernel_node, "source")
            else:
                kernel = kernel.update_vertex(kernel_node, "target")

    return kernel, kernel_nodes


def add_to_kernel_nodes(node, kernel_nodes):
    # if is_node_in_kernel_nodes(node, kernel_nodes):
    #     return kernel_nodes

    if isinstance(node, SetOfSingletons):
        kernel_nodes.add(node)
        for entity in node.entities:
            kernel_nodes = add_to_kernel_nodes(entity, kernel_nodes)
    else:
        # Check if properties has any Singleton's and add to kernel nodes also
        for key in dict(node.properties):
            properties_key_ = dict(node.properties)[key]
            if not isinstance(properties_key_, str):
                if isinstance(properties_key_, Singleton):
                    kernel_nodes = add_to_kernel_nodes(properties_key_, kernel_nodes)
                elif properties_key_ is not None:
                    for prop_node in properties_key_:
                        if isinstance(prop_node, Singleton):
                            kernel_nodes = add_to_kernel_nodes(prop_node, kernel_nodes)
                        elif isinstance(prop_node, SetOfSingletons):
                            kernel_nodes.add(node)
                            for prop_entity in prop_node.entities:
                                kernel_nodes = add_to_kernel_nodes(prop_entity, kernel_nodes)

        if node.kernel is not None:
            if node.kernel.edgeLabel is not None:
                kernel_nodes = add_to_kernel_nodes(node.kernel.edgeLabel, kernel_nodes)
            if node.kernel.source is not None:
                kernel_nodes = add_to_kernel_nodes(node.kernel.source, kernel_nodes)
            if node.kernel.target is not None:
                kernel_nodes = add_to_kernel_nodes(node.kernel.target, kernel_nodes)
        # else:
        if (node.kernel is None or (node.kernel is not None and node.kernel.edgeLabel.named_entity not in {'acl_relcl',
                                                                                                           'acl'})) or node.type != 'SENTENCE':
            kernel_nodes.add(node)

    return kernel_nodes


def add_to_properties(kernel, node, source_or_target, kernel_nodes, properties, negations, node_functions,
                      type_key=None):
    lemma_kernel_edge_label_name = lemmatize_verb(kernel.edgeLabel.named_entity) if kernel.edgeLabel is not None else ""

    # Check if action matched edge label, and if it contains a negation, negate the kernel
    if isinstance(node, SetOfSingletons) and node.type == Grouping.NOT:
        is_negated = True
        if isinstance(node.entities[0], SetOfSingletons):
            return add_to_properties(kernel, node.entities[0], source_or_target, kernel_nodes, properties, negations,
                                     node_functions, type_key)
        node_props = dict(node.entities[0].properties)
        if 'action' in node_props and len(node_props['action']) > 0:
            # Get label from action prop
            lemma_node_edge_label_name = lemmatize_verb(node_props['action'])

            # Remove negation from label
            query_words = lemma_node_edge_label_name.split()
            result_words = [word for word in query_words if word.lower() not in negations]
            lemma_node_edge_label_name = ' '.join(result_words)

            # If lemmatized action == kernel edge, then negate
            if lemma_node_edge_label_name == lemma_kernel_edge_label_name:
                kernel = Relationship(
                    source=kernel.source,
                    target=kernel.target,
                    edgeLabel=kernel.edgeLabel,
                    isNegated=is_negated
                )
                return kernel, properties, kernel_nodes
        else:
            lemma_node_edge_label_name = node.entities[0].named_entity
    elif isinstance(node, Singleton):  # Get label name from action if present, otherwise just the node name
        node_props = dict(node.properties)
        if 'action' in node_props and len(node_props['action']) > 0:
            # Get label from action prop
            lemma_node_edge_label_name = lemmatize_verb(node_props['action'])
        else:
            lemma_node_edge_label_name = lemmatize_verb(node.named_entity)
    else:
        lemma_node_edge_label_name = None

    # Check for a SetOfSingletons, or if the node name or action label is not equal to the kernel edge label
    if (
            (isinstance(node, SetOfSingletons)) or
            (node is not None and lemma_node_edge_label_name != lemma_kernel_edge_label_name)
    ):
        if type_key is None:
            if source_or_target == 'edgeLabel':
                type_key = node.kernel.edgeLabel.named_entity
            else:
                type_key = node_functions.get_node_type(node)
        if (type_key in copula_types) and not is_node_in_kernel_nodes(node, kernel_nodes):
            kernel_nodes = add_to_kernel_nodes(node, kernel_nodes)
            kernel = create_cop(node, kernel, source_or_target)
        # NOT SetOfSingletons that didn't negate the kernel verb: preserve the wrapper as
        # a property re-keyed by the inner entity's type so logical rewriting can classify
        # it correctly (e.g. verb inner → TIME_STATUS:NOT(fixed reopening date)).
        if ('NOT' in type_key and
                isinstance(node, SetOfSingletons) and node.type == Grouping.NOT and node.entities):
            inner_type = node_functions.get_node_type(node.entities[0])
            if inner_type not in ('NOT', 'NEG', 'existential'):
                type_key = inner_type
                if not is_node_in_kernel_nodes(node, kernel_nodes) and node not in properties[type_key]:
                    kernel_nodes = add_to_kernel_nodes(node, kernel_nodes)
                    properties[type_key].append(node)
                return kernel, properties, kernel_nodes
        if 'NEG' not in type_key and 'NOT' not in type_key and 'existential' not in type_key:
            if (
                    (node.type == Grouping.MULTIINDIRECT) or
                    (
                            isinstance(node, Singleton) and
                            (node.named_entity == "but" or node.named_entity == "and")
                    ) or
                    (
                            isinstance(node, SetOfSingletons) and
                            node.type == Grouping.AND and 'NEG' in node_functions.get_node_type(node.entities[0]) and
                            len(node.entities) == 1
                    )
            ):
                return kernel, properties, kernel_nodes
            elif not is_node_in_kernel_nodes(node, kernel_nodes) and node not in properties[type_key]:
                if isinstance(node, SetOfSingletons):
                    for entity in node.entities:
                        if not is_node_in_kernel_nodes(entity, kernel_nodes):
                            kernel_nodes = add_to_kernel_nodes(entity, kernel_nodes)
                            properties[type_key].append(entity)
                else:
                    if 'actioned' in dict(node.properties) or 'action' in dict(node.properties):
                        node_props = dict(node.properties)
                        if 'actioned' in node_props and node.type != 'verb' and node.named_entity:
                            node_props['amod'] = node_props.pop('actioned')
                            node = node.update_node_props(node_props)
                        else:
                            node = rewrite_action_ed_node(node, node, negations)
                            type_key = 'SENTENCE'
                    kernel_nodes = add_to_kernel_nodes(node, kernel_nodes)
                    properties[type_key].append(node)
    return kernel, properties, kernel_nodes


# If IDs are not the same, check if the "begin" and "end" match as this means it should be the same node
def are_begin_end_equiv(check_node, kernel_node):
    check_props = dict(check_node.properties)
    kernel_props = dict(kernel_node.properties)

    check_begin = check_props['begin'] if 'begin' in check_props else check_node.min
    kernel_begin = kernel_props['begin'] if 'begin' in kernel_props else kernel_node.min
    check_end = check_props['end'] if 'end' in check_props else check_node.max
    kernel_end = kernel_props['end'] if 'end' in kernel_props else kernel_node.max

    return check_begin == kernel_begin and check_end == kernel_end


# TODO: Is this enough information to determine? As ID might differ but have all same properties etc.
#  min and max might differ despite being the same node (i.e. a MULTIINDIRECT might have larger min max)?
def is_node_in_kernel_nodes(check_node, kernel_nodes):
    for kernel_node in kernel_nodes:
        if isinstance(check_node, Singleton) and isinstance(kernel_node, Singleton) and check_node.kernel is None:
            if (check_node.named_entity == kernel_node.named_entity and
                (check_node.id == kernel_node.id or are_begin_end_equiv(check_node, kernel_node))
                    # and check_node.type == kernel_node.type and check_node.min == kernel_node.min and check_node.max == kernel_node.max
                    # check_node.named_entity == kernel_node.named_entity and
            ) or (
                    check_node.type == 'verb' and lemmatize_verb(check_node.named_entity) == lemmatize_verb(
                kernel_node.named_entity) and check_node.id == kernel_node.id
            ):
                return True
        elif isinstance(check_node, SetOfSingletons) and isinstance(kernel_node, SetOfSingletons):
            if check_node.entities == kernel_node.entities and check_node.type == kernel_node.type and check_node.min == kernel_node.min and check_node.max == kernel_node.max:
                return True
        # If both check_node and kernel_node are SENTENCEs
        elif (isinstance(check_node, Singleton) and isinstance(kernel_node, Singleton)
              and check_node.kernel is not None and kernel_node.kernel is not None):
            if (
                    ((
                             check_node.kernel.source is not None and kernel_node.kernel.source is not None and check_node.kernel.source.id == kernel_node.kernel.source.id) or (
                             check_node.kernel.source is None and kernel_node.kernel.source is None))
                    and
                    ((
                             check_node.kernel.target is not None and kernel_node.kernel.target is not None and check_node.kernel.target.id == kernel_node.kernel.target.id) or (
                             check_node.kernel.target is None and kernel_node.kernel.target is None))
                    and
                    ((
                             check_node.kernel.edgeLabel is not None and kernel_node.kernel.edgeLabel is not None and check_node.kernel.edgeLabel.id == kernel_node.kernel.edgeLabel.id) or (
                             check_node.kernel.edgeLabel is None and kernel_node.kernel.edgeLabel is None))
            ):
                # or (lemmatize_verb(check_node.kernel.edgeLabel.named_entity) == lemmatize_verb(
                #     kernel_node.kernel.edgeLabel.named_entity))
                return True
    return False


def assign_kernel(G, edges, kernel, negations, nodes, root_sentence_id, found_preposition_labels, node_functions):
    chosen_edge = None
    honk = Services.getInstance().getHOnK()
    transitive_verbs = honk.getTransitiveVerbs()
    rejected_verbs = honk.getRejectedVerbs()
    preposition_surface_forms = {str(p).strip().lower() for p in honk.getPrepositions() if p}
    status_state_noun_forms = {
        str(n).strip().lower()
        for n in (set(honk.getStatusNouns()) | set(honk.getServiceStateNouns()))
        if n
    }

    def is_valid_verb(node):
        if node is None: return False
        if node.type != "verb": return False

        # Grammatical Rules to determine if it is NOT a verb in this scenario
        props = dict(node.properties)

        # Rule 0: A preposition that the grammar mis-typed as a verb edge label
        # (e.g. a subject-attached locative "between South Gosforth and Benton"
        # lifted onto the edge) is a case/spatial marker, never the clausal verb.
        # Without this it can be picked as the kernel relation ahead of the real
        # root verb (e.g. "between(trains, ...)" instead of "run(trains, ...)").
        if node.named_entity is not None and node.named_entity.strip().lower() in preposition_surface_forms:
            return False

        # Rule 1: Verbs usually don't have determiners (e.g. "The staircases")
        if 'det' in props:
            return False

        # Rule 2: Objects of prepositions are rarely the main verb (e.g. "at University station")
        # Prepositions often add a 'case' property to the object
        if 'case' in props:
            return False

        # Rule 3: Compound entities (with 'extra' names) are usually nouns, not the main verb
        if 'extra' in props:
            return False

        # Rule 4: Check incoming dependency edges in the graph G
        # Only check if the node is actually part of the graph (edge labels might not be)
        if node.id in G:
            incoming = G.in_edges(node.id, data=True)
            for u, v, data in incoming:
                edge_lbl = data.get('label')
                if edge_lbl and edge_lbl.named_entity in DependencyRoles.verb_disqualifying_incoming_edges():
                    return False

        # Rule 5: Check if lemma is in rejected_verbs (lexical check)
        lemmas = lemmatize_sentence(node.named_entity)
        return len(rejected_verbs.intersection({lemmatize_verb(x) for x in lemmas})) == 0

    # Priority 0: structure-2 anomaly recovery. Normally the grammar emits the
    # main verb as an edge *label* (`subject -[verb]-> object`). Occasionally it
    # leaves the (active, transitive) root verb as a *node* that carries its own
    # `nsubj`/`obj` dependency edges and emits no verb-labelled edge at all. The
    # main loop below then promotes the verb node to the edge label, trips the
    # `source == edge_label` existential branch, and wrongly makes the verb's
    # SUBJECT the kernel target under an existential source (e.g.
    # "The roadworks involve replacement" -> involve(?existential, roadworks)
    # with `replacement` demoted to a property). Recover the intended
    # `verb(nsubj, obj)` here. The gate is deliberately tight so it cannot
    # disturb sentences that DO have a verb-labelled edge (they keep their
    # existing selection, e.g. "...recorded ... involved ...") or passives
    # (which expose an `nsubj` but no `obj`).
    root_verb_kernel = None
    if not any(is_valid_verb(e[3]['label']) for e in edges):
        root_data = G.nodes[root_sentence_id]['data'] if root_sentence_id in G.nodes else None
        if root_data is not None and is_valid_verb(root_data):
            nsubj_tgt = obj_tgt = obl_tgt = None
            obl_tgt_id = None
            obl_edge = None
            root_neg = False
            for e in edges:
                if e[0] != root_sentence_id:
                    continue
                lbl = e[3]['label'].named_entity
                if lbl == 'nsubj' and nsubj_tgt is None:
                    nsubj_tgt = G.nodes[e[1]]['data']
                    root_neg = root_neg or e[3]['isNegated']
                elif lbl in ('obj', 'dobj') and obj_tgt is None:
                    obj_tgt = G.nodes[e[1]]['data']
                    root_neg = root_neg or e[3]['isNegated']
                elif lbl == 'obl' and obl_tgt is None:
                    obl_tgt = G.nodes[e[1]]['data']
                    obl_tgt_id = e[1]
                    obl_edge = e

            # Fall back to an oblique complement when the (intransitive) root verb
            # exposes no direct object: "trains are running ON a timetable" ->
            # run(trains, timetable). Only when the obl object is a contentful noun
            # acting as the event's theme, NOT:
            #  - a location (a locative obl is spatial framing -> SPACE), or
            #  - a status/state noun ("remains UNDER investigation" is a predicative
            #    lifecycle state -> TIME_STATUS, not the kernel target).
            obl_is_status = (
                isinstance(obl_tgt, Singleton)
                and obl_tgt.named_entity is not None
                and obl_tgt.named_entity.strip().lower() in status_state_noun_forms
            )
            # Only a plain common/proper NOUN obl is a genuine theme object. Exclude
            # DATE/TIME obliques ("recorded in January 2026" -> TIME, not target),
            # verb obliques / reduced relatives ("involved a bicycle taken ..." ->
            # `taken` is a sub-clause, not the target), locations (-> SPACE), and
            # lifecycle status nouns ("remains under investigation" -> TIME_STATUS).
            if (obj_tgt is None and obl_tgt is not None
                    and not isinstance(obl_tgt, SetOfSingletons)
                    and not obl_is_status
                    and getattr(obl_tgt, 'type', None) == 'noun'):
                # Strip the obl's own case-marker preposition ("on") IN THE GRAPH so
                # that (a) the later case-stripping pass keeps it as the genuine target
                # rather than demoting it to a property and nulling the target, and
                # (b) the property loop does not re-route the same node into SPACE.
                for _k in list(dict(obl_tgt.properties).keys()):
                    if _k == 'case' or is_position_key(_k):
                        obl_tgt = obl_tgt.remove_prop(_k)
                if obl_tgt_id is not None and obl_tgt_id in G.nodes:
                    nx.set_node_attributes(G, {obl_tgt_id: obl_tgt}, 'data')
                obj_tgt = obl_tgt

            if (nsubj_tgt is not None and obj_tgt is not None
                    and getattr(nsubj_tgt, 'type', None) != 'existential'
                    and getattr(obj_tgt, 'type', None) != 'existential'
                    and nsubj_tgt is not obj_tgt):
                root_verb_kernel = Relationship(
                    source=nsubj_tgt, target=obj_tgt,
                    edgeLabel=root_data, isNegated=root_neg,
                )
                kernel = root_verb_kernel
                # The obl edge has been consumed as the kernel's verb→target
                # relation. Drop it from the shared edge list so the downstream
                # property loop does not ALSO reprocess it as a nominal modifier
                # (`obl` ∈ nominal_modifier_edges), which would re-attach the root
                # verb onto the target as a spurious specification/extra
                # (timetable → "timetable/running").
                if obl_edge is not None and obl_edge in edges:
                    try:
                        edges.remove(obl_edge)
                    except (ValueError, AttributeError):
                        pass

    # When several edges leave the root verb, the direct object (`obj`/`dobj`) is
    # the kernel target, NOT a subordinate adverbial/relative clause. The grammar
    # can emit the `advcl`/`acl` edge before the `obj` edge, so a naive first-match
    # would wrongly pick the modifier clause as the target (e.g. "operating a
    # timetable affecting trains" -> operate(?, affecting) with `timetable` lost).
    # Order candidate edges so a real object is preferred over a modifier clause.
    _subordinate_clause_labels = {'advcl', 'acl', 'acl_relcl'}
    _object_labels = {'obj', 'dobj'}
    _root_has_object = any(
        e[0] == root_sentence_id and e[3]['label'].named_entity in _object_labels
        for e in edges
    )

    def _edge_target_preference(edge):
        # Only intervene when the root verb has a genuine direct object: in that
        # case a subordinate adverbial/relative clause must NOT be picked as the
        # kernel target ahead of the object. A *stable* sort keeps every other
        # edge in its original relative order, so this is a no-op for sentences
        # without an obj+clause conflict (avoids disturbing unrelated cases).
        if (_root_has_object
                and edge[0] == root_sentence_id
                and edge[3]['label'].named_entity in _subordinate_clause_labels):
            return 1
        return 0

    ordered_edges = sorted(edges, key=_edge_target_preference)

    # Priority 1: Find an edge that is explicitly marked as 'kernel' or 'root'
    for edge in ordered_edges:
        source_data = G.nodes[edge[0]]['data']
        target_data = G.nodes[edge[1]]['data']
        edge_label = edge[3]['label']
        if (
                (is_valid_verb(edge_label) or is_valid_verb(source_data)) and
                (
                        'kernel' in edge_label.get_props() or 'root' in edge_label.get_props() or
                        'kernel' in source_data.get_props() or 'root' in source_data.get_props()
                ) and
                edge[0] == root_sentence_id and
                edge_label.named_entity not in {'aux'} and target_data.type != 'existential'  # TODO: Hacky fix??
        ):
            chosen_edge = edge
            break

    # Priority 2: Standard verb-based search if no priority edge found
    found_preposition_values = set(found_preposition_labels.values())
    if chosen_edge is None:
        for edge in ordered_edges:
            if (
                    is_valid_verb(edge[3]['label']) and
                    (
                            root_sentence_id in found_preposition_labels or
                            edge[3]['label'].named_entity not in found_preposition_values
                    ) and
                    edge[0] == root_sentence_id
            ):
                chosen_edge = edge
                break

    for edge in edges:
        source, target, _, edge_label = edge
        source = G.nodes[source]['data']
        target = G.nodes[target]['data']
        edge_label = edge_label['label']
        is_negated = edge[3]['isNegated']

        # If we have found a chosen edge OR we haven't but the source or edge label are verbs AND
        #  root in preposition labels or edge label not in prepositions
        if (
                root_verb_kernel is None
                and
                (
                        (
                                (is_valid_verb(edge_label) or is_valid_verb(source)) and
                                chosen_edge is None
                        ) or
                        chosen_edge is not None and chosen_edge == edge
                )
                and
                (
                        root_sentence_id in found_preposition_labels or
                        edge_label.named_entity not in found_preposition_values
                )
        ):
            # If edge label is NOT a verb, use the source instead
            edge_label = edge_label if is_valid_verb(edge_label) else source

            # If source is NOT semi-modal AND not in nodes then create existential for source
            edge_source = create_existential_node() if (
                    (
                            isinstance(source, Singleton) and
                            not check_semi_modal(source.named_entity) and
                            len([x for x in nodes.keys() if source.id == x]) == 0 and
                            source.type != 'existential'
                    ) or source == edge_label) else source

            # Check if "action" property in equal to edge label, and therefore redundant
            if (
                    isinstance(edge_source, Singleton) and
                    'action' in dict(edge_source.properties) and
                    lemmatize_verb(dict(edge_source.properties)['action']) == lemmatize_verb(edge_label.named_entity)
            ):
                edge_source = edge_source.remove_prop('action')

            # If edge is negated, we no longer need respective negation (if) in the edge label
            if is_negated:
                for name in negations:
                    # If edge label contains negations (no, not), remove them
                    if bool(re.search(rf"\b{re.escape(name)}\b", edge_label.named_entity)):
                        edge_label_name = edge_label.named_entity.replace(name, "").strip()
                        edge_label = edge_label.update_name(edge_label_name)
                        break

            # If NOT a transitive verb, remove target as target reflects direct object
            lemmas = lemmatize_sentence(edge_label.named_entity)
            if len(transitive_verbs.intersection({lemmatize_verb(x) for x in lemmas})) == 0:
                kernel = Relationship(
                    source=edge_source,
                    target=None,
                    edgeLabel=edge_label,
                    isNegated=is_negated
                )

                # Special Case: Copula rule for 'be'
                # lemmatize_verb strips AUX tokens so "are"→"", use a direct form-set check instead.
                _be_forms = Services.getInstance().getHOnK().getCopulaSurfaceForms()
                if edge_label.named_entity.lower().strip() in _be_forms and kernel.target is None:
                    source_props = dict(edge_source.properties)
                    if 'cop' in source_props:
                        # Move cop node to target (legacy path)
                        cop_node = source_props['cop']
                        if isinstance(cop_node, tuple):
                            cop_node = cop_node[0]  # Handle case where it might be a sequence

                        kernel = Relationship(
                            source=edge_source.remove_prop('cop'),
                            target=cop_node,
                            edgeLabel=edge_label,
                            isNegated=is_negated
                        )
                    elif isinstance(target, Singleton) and target.type in copula_types:
                        # Edge target IS the predicate complement (JJ/JJS/RB): keep it as target
                        kernel = Relationship(
                            source=edge_source,
                            target=target,
                            edgeLabel=edge_label,
                            isNegated=is_negated
                        )
                break
            else:
                kernel = Relationship(
                    source=edge_source,
                    target=target,
                    edgeLabel=edge_label,
                    isNegated=is_negated
                )

                # Special Case: Copula rule for 'be' (even if it was thought transitive or has a target)
                _be_forms = Services.getInstance().getHOnK().getCopulaSurfaceForms()
                if edge_label.named_entity.lower().strip() in _be_forms:
                    source_props = dict(edge_source.properties)
                    if 'cop' in source_props:
                        cop_node = source_props['cop']
                        if isinstance(cop_node, tuple):
                            cop_node = cop_node[0]
                        kernel = Relationship(
                            source=edge_source.remove_prop('cop'),
                            target=cop_node,
                            edgeLabel=edge_label,
                            isNegated=is_negated
                        )
                break

    # If kernel is none, look for existential within source and target properties
    if kernel is None:
        kernel = next(iter([node for node in itertools.chain.from_iterable(
            map(lambda x: [G.nodes[x[0]]['data'], G.nodes[x[1]]['data']], edges)) if
                            find_existential_in_properties(node)]), None)

    # If we cannot find existential, create it instead
    if kernel is None:
        n = len(edges)
        G, new_kernel = create_existential(G, nodes, node_functions)

        # If we cannot create it, just use the edge (unchanged)
        if n <= len(edges) and new_kernel is not None:
            kernel = new_kernel
        else:
            kernel = edges[-1]
            kernel = Relationship(
                source=G.nodes[kernel[0]]['data'],
                target=G.nodes[kernel[1]]['data'],
                edgeLabel=kernel[3]['label'],
                isNegated=kernel[3]['isNegated']
            )

    # Check if source or target have "case" property, if so remove it (to be added to properties of the kernel later)
    if case_in_props(kernel.source.get_props()):
        kernel = Relationship(
            source=create_existential_node(),
            target=kernel.target,
            edgeLabel=kernel.edgeLabel,
            isNegated=kernel.isNegated
        )
    if kernel.target is not None and case_in_props(kernel.target.get_props()):
        # Find the first occurring element by position in the SetOfSingletons
        #  Remove this from the SetOfSingletons, to be added to properties, use the first element as new target
        #  Unless target is NOT a SetOfSingletons, then just remove
        if isinstance(kernel.target, SetOfSingletons):
            chosen_target = None
            for node in kernel.target.entities:
                if chosen_target is None or chosen_target.min > node.min:
                    chosen_target = node

            chosen_new_entities = []
            for node in kernel.target.entities:
                if node != chosen_target:
                    chosen_new_entities.append(node)
            nodes[kernel.target.id] = kernel.target.update_entities(chosen_new_entities)

        kernel = Relationship(
            source=kernel.source,
            target=chosen_target if isinstance(kernel.target, SetOfSingletons) else None,
            edgeLabel=kernel.edgeLabel,
            isNegated=kernel.isNegated
        )

    # Remove edge label if not a verb
    if kernel.edgeLabel.type != "verb":
        kernel = Relationship(
            source=kernel.source,
            target=kernel.target,
            edgeLabel=None,
            isNegated=kernel.isNegated
        )

    result = kernel.update_vertex(kernel.edgeLabel.update_name(
        lemmatize_verb(kernel.edgeLabel.named_entity)) if kernel.edgeLabel is not None else None, "edgeLabel")
    return G, result


def find_existential_in_properties(node):
    new_kernel = None
    if isinstance(node, SetOfSingletons):
        for entity in node.entities:
            find_existential_in_properties(entity)
    else:
        for prop in node.properties:
            if prop[1] == '∃':
                new_kernel = node
                break
    return new_kernel


def case_in_props(node_props, return_props=False):
    if node_props is None:
        return [] if return_props else False

    # Ignore "by" as passive sentence: https://www.uc.utoronto.ca/passive-voice
    # Ignore "of" and "'s" as "possessive": https://en.m.wikipedia.org/wiki/English_possessive
    try:
        from LaSSI.ner.structural_rewrites.declarative import structural_lexical_set
        ignore_cases = structural_lexical_set("case_prepositions_ignored_for_kernel_targets")
    except Exception:
        ignore_cases = set()
    if not ignore_cases:
        ignore_cases = {'by', "'s", 'of', "’s"}
    found_cases = []

    for key in node_props:
        if key == 'case':
            case_values = node_props[key]
            if not isinstance(case_values, (list, tuple, set)):
                case_values = [case_values]
            case_values = [case for case in case_values if case not in ignore_cases]
            if not return_props:
                return bool(case_values)
            found_cases.extend(case_values)
            continue
        try:
            case_position = float(key)
            if node_props[key] not in ignore_cases:
                if not return_props:
                    return True
                else:
                    found_cases.append(node_props[key])
        except ValueError:
            continue

    if not return_props:
        return False
    else:
        return found_cases


def get_prepositions(node):
    found_prepositions = []
    node_props = dict(node.properties)
    _smart_apos = chr(0x2019)
    float_keyed = {}
    for key in node_props:
        if key in DependencyRoles.preposition_marker_labels():
            value = node_props[key]
            if isinstance(value, str):
                found_prepositions.append(value.lower().replace(_smart_apos, "'"))
        try:
            case_position = float(key)
            value = node_props[key]
            if not isinstance(value, str):
                continue
            val = value.lower().replace(_smart_apos, "'")
            found_prepositions.append(val)
            float_keyed[case_position] = val
        except (ValueError, AttributeError):
            continue

    if node.type in {"IN", "TO", "RB"}:
        found_prepositions.append(node.named_entity.lower())

    # Reconstruct compound prepositions from position-ordered marker tokens.
    # CoreNLP sometimes stores "on or near" as separate numeric properties
    # (6:on, 7:or, 8:near) without a `case:on`; logical_analysis.json expects
    # the compound surface form so the disjunctive SPACE type can fire.
    if len(float_keyed) >= 2:
        sorted_vals = [
            v for _, v in sorted(float_keyed.items())
            if re.search(r"[a-z]", v)
        ]
        if sorted_vals:
            unique_vals = [sorted_vals[0]]
            for v in sorted_vals[1:]:
                if v != unique_vals[-1]:
                    unique_vals.append(v)
            compound = " ".join(unique_vals)
            if compound and compound not in found_prepositions:
                found_prepositions.append(compound)
    return set(found_prepositions)


def create_edge_kernel(G, node, node_functions):
    nx.set_node_attributes(G, {node.id: node.update_name(lemmatize_verb(node.named_entity))}, 'data')
    node = G.nodes[node.id]['data']
    return Singleton(
        id=node.id,
        named_entity="",
        type="SENTENCE",
        min=node.min,
        max=node.max,
        confidence=1,
        kernel=Relationship(
            source=node_functions.create_existential_node(G),
            target=None,
            edgeLabel=node if node is not None else None,
            isNegated=False
        ),
        properties=frozenset(),  # TODO: Should this be empty?
    )


def is_kernel_in_props(node, check_jj=True):
    if isinstance(node, SetOfSingletons):
        # If the Set has a true root, return, otherwise check children for root
        if node.root:
            return node.root
        else:
            for entity in node.entities:
                return is_kernel_in_props(entity)

    if isinstance(node, Singleton):
        node_props = dict(node.properties)
        return (('kernel' in node_props or 'root' in node_props) and (
                'JJ' not in node.type and check_jj or not check_jj)) or 'verb' in node.type
    elif isinstance(node, dict) and hasattr(node, 'properties'):
        node_props = node['properties']
        return 'kernel' in node_props or 'root' in node_props
    else:
        return False

    # TODO: Do we need to check if the JJ is/not a verb?
    # ('JJ' not in x.type or ('JJ' in x.type and self.is_label_verb(x.named_entity))))


def find_action_ed_node_in_kernel(kernel):
    if isinstance(kernel, Singleton):
        if kernel.kernel is not None:
            if kernel.kernel.source is not None:
                props = kernel.kernel.source.get_props()
                if 'actioned' in props or 'action' in props:
                    return kernel.kernel.source

            if kernel.kernel.target is not None:
                props = kernel.kernel.target.get_props()
                if 'actioned' in props or 'action' in props:
                    return kernel.kernel.target

            if kernel.kernel.edgeLabel is not None:
                props = kernel.kernel.edgeLabel.get_props()
                if 'actioned' in props or 'action' in props:
                    return kernel.kernel.edgeLabel
        elif kernel is not None:
            if 'actioned' in dict(kernel.properties) or 'action' in dict(kernel.properties):
                return kernel
    return False


def rewrite_action_ed_node(node, action_ed_node, negations):
    # Get label from action prop
    action_ed_node_props = dict(action_ed_node.properties)
    actioned = 'actioned' in action_ed_node_props
    lemma_node_edge_label_name = lemmatize_verb(
        action_ed_node_props['actioned']) if actioned else lemmatize_verb(action_ed_node_props['action'])

    action_ed_node_props.pop('actioned' if actioned else 'action')
    action_ed_node = action_ed_node.update_node_props(action_ed_node_props)

    node_props = dict(node.properties)
    node_actioned = 'actioned' in node_props
    if 'actioned' in node_props or 'action' in node_props:
        node_props.pop('actioned' if node_actioned else 'action')
        node = node.update_node_props(node_props)
    # if node.id != action_ed_node.id:
    #     node_props = dict(node.properties)
    #     node_props.pop('actioned' if actioned else 'action')
    #     node = Singleton.update_node_props(node, node_props)

    # Remove negation from label
    query_words = lemma_node_edge_label_name.split()
    result_words = [word for word in query_words if word.lower() not in negations]
    refactored_lemma_node_edge_label_name = ' '.join(result_words)
    return Singleton(
        id=node.id,
        named_entity='',
        type='SENTENCE',
        min=node.min,
        max=node.max,
        confidence=1,
        kernel=Relationship(
            source=action_ed_node if not actioned else create_existential_node(),
            target=action_ed_node if actioned else None,
            edgeLabel=Singleton(
                id=-1,
                named_entity=refactored_lemma_node_edge_label_name,
                type='verb',
                min=node.min,
                max=node.max,
                confidence=1,
                kernel=None,
                properties=frozenset(),  # TODO: Keep properties
            ),
            isNegated=node.kernel.isNegated if node.kernel is not None else lemma_node_edge_label_name != refactored_lemma_node_edge_label_name,
        ),
        # properties=node.properties if node.id != action_ed_node.id else frozenset(),
        properties=node.properties,
    )
