__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

import itertools
import math
from collections import defaultdict

import networkx as nx

from LaSSI.external_services.Services import Services
from LaSSI.ner.MergeSetOfSingletons import merge_properties, GraphNER_withProperties, _promote_geo_suffixed_type
from LaSSI.ner.SemanticRoleRewriting import DependencyRoleRewriter
from LaSSI.ner.node_functions_X import NodeFunctions
from LaSSI.ner.string_functions import does_string_have_negations
from LaSSI.structures import DependencyRoles
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping, SetOfSingletons
from LaSSI.structures.kernels.Sentence import is_kernel_in_props, case_in_props


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

        # Phase 1 (Convert to graphs to NetworkX representation, and convert to Singletons)
        G = self.convertToGraph(gsm_json)

        # Phase 1.1 (Sanitise compound edges whose endpoints are non-adjacent in
        # surface position, almost always Stanza parse errors. See
        # `sanitiseCompoundChainsByPosition` for the rationale.)
        G = self.sanitiseCompoundChainsByPosition(G)

        # Phase 1.2 (Drop spurious cross-clause `appos` edges before orphan
        # reconnection runs.)
        G = self.dropSpuriousApposEdges(G)

        # Phase 1.25 (Reattach orphan components left disconnected by p3pass
        # Passive verbs only rebind one obl/nmod child to the new
        # subject S; remaining obl children become rootless components that would
        # otherwise be lost.)
        G = self.reconnectOrphanComponents(G)

        # Phase 1.5 (Re-assign 'multipleindobj' node children with parent)
        G = self.resolveMultipleInDobj(G)

        # Phase 2 (Resolve types for Singletons from meuDB)
        for node in G.nodes(data=True):
            nx.set_node_attributes(G, {
                node[0]: self.nodeTypeResolution(node[1]['data'], self.associateNodeToBestMeuMatch(node[1]['data']), G)
            }, 'data')

        # Phase 2.5 (Merge nodes based on multi-entity units)
        G = self.mergeMeuNodes(G)

        # Phase 3 (Pre-processing: Merging, adding properties etc.)
        G = self.preProcessGraph(G)

        # Phase 4 (Preserve dependency semantic roles, then create and resolve SetOfSingletons)
        G = self.mergeNodes(G)

        return G

    # Phase 1.1
    def convertToGraph(self, gsm_json):
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
        G = self.node_functions.sort_G(G)

        if self.shouldDrawGraphs:
            self.drawNetworkXGraph(G)

        return G

    # Phase 1.2
    def create_singleton(self, gsm_item):
        min_value = -1
        max_value = -1
        if len(gsm_item['xi']) > 0 and gsm_item['xi'][0] != '' and ((len(gsm_item['ell']) > 0 and gsm_item['ell'][0] != '∃') or (len(gsm_item['ell']) == 0)):  # TODO: Checking for ∃ might not be valid...
            name = gsm_item['xi'][0]

            if 'begin' in dict(gsm_item['properties'].items()):
                min_value = int(gsm_item['properties']['begin'])
                max_value = int(gsm_item['properties']['end'])
            node_type = gsm_item['ell'][0] if len(gsm_item['ell']) > 0 else "None"
            node_type = self.get_group_enum(gsm_item['properties']['conj']) if 'conj' in gsm_item['properties'] else node_type
        else:
            # xi might be empty if the node is invented, therefore existential
            name = "?" + str(self.existentials.increaseAndGetExistential())
            node_type = 'existential'

        # If we have "root" in "ell", add it to properties
        if len(gsm_item['ell']) > 1:
            gsm_item['properties']['kernel'] = gsm_item['ell'][1]
        elif gsm_item.get('properties', {}).get('root') == 'root':
            # `p3pass`/`p3` move the root marker into `properties.root` (rather
            # than appending it to `ell`) when they nominate a non-verb node
            # (e.g. the oblique target `area` in "near the Haymarket area") as
            # the new kernel head. Mirror it into `properties.kernel` so the
            # downstream `reconnectOrphanComponents` step (which keys off
            # `properties.kernel`) recognises this node and reattaches the
            # orphan TIME / TIME_STATUS subgraphs left behind by `del V`.
            gsm_item['properties']['kernel'] = 'root'
        if len(gsm_item['xi']) > 1 and 'subjpass' in gsm_item['xi'][1]:
            gsm_item['properties']['subjpass'] = gsm_item['xi'][1]

        # Add 'det' to properties
        if len(gsm_item['ell']) > 0 and 'det' in gsm_item['ell']:
            gsm_item['properties']['det'] = gsm_item['ell'][0]

        return Singleton(
            id=gsm_item['id'],
            named_entity=name,
            properties=frozenset(gsm_item['properties'].items()),
            min=min_value,
            max=max_value,
            type=node_type,
            confidence=1.0
        )

    # Phase 1.1
    def sanitiseCompoundChainsByPosition(self, G):
        """Repair `compound(X, Y)` edges whose endpoints are not surface-adjacent.

        In English (and most head-final compound languages), a compound modifier
        sits immediately before its head in surface form, so `|pos(X) - pos(Y)|`
        should be exactly 1.  When Stanza emits a compound edge whose endpoints
        are several tokens apart it is almost always a parse error: the modifier
        belongs to a different head than the parser inferred.

        Concrete example — "The Percy Street entrance to Monument Metro station
        is currently closed …":

            compound(station[pos 8], Percy[pos 2])     gap = 6  -- bad
            compound(station[pos 8], entrance[pos 4])  gap = 4  -- bad
            compound(entrance[pos 4], Street[pos 3])   gap = 1  -- fine
            compound(Metro[pos 7], Monument[pos 6])    gap = 1  -- fine

        For each suspect edge we either:
          * **Reroute** it to the modifier's immediate right-neighbour token
            (`pos = dst.pos + 1`) when that neighbour is a noun, since that is
            the head the modifier almost certainly meant to attach to; or
          * **Remove** it, letting the modifier subtree become an orphan that
            `reconnectOrphanComponents` (the next phase) re-attaches via nmod.

        Continuing the example:
          * `compound(station, Percy)`  -> Percy's pos+1 is Street (noun) -> reroute as
            `compound(Street, Percy)`.
          * `compound(station, entrance)` -> entrance's pos+1 is "to" (IN, not a noun)
            -> remove. `entrance` then heads an orphan subtree containing the
            corrected `Percy Street entrance` chain, which `reconnectOrphanComponents`
            re-attaches to the kernel.

        Threshold of 1 is intentional and not configurable: legitimate
        compounds in the dependency conventions used here are always adjacent.
        The check is restricted to edges literally labelled `compound`, so
        long-range `nmod`/`obl`/etc. are unaffected."""
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

        # Position -> node id lookup, for fast "right neighbour of pos N" queries.
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
                continue  # adjacent: legitimate compound, keep as-is

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
                            continue  # separated by a punctuation, keep as-is

            edges_to_remove.append((src, dst, key))

            # Attempt reroute via the modifier's immediate right-neighbour
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
            self.drawNetworkXGraph(G)

        return G

    def dropSpuriousApposEdges(self, G):
        """Remove `appos` edges that aren't real appositions.

        Stanza emits `appos(X, Y)` whenever two NPs sit next to each other
        across a comma, even when the comma is actually separating a fronted
        PP from the main clause. Example: in "Due to storm damage, the Percy
        Street entrance is currently closed", Stanza emits `appos(damage,
        entrance)` — but `damage` is the head of the fronted PP "Due to ..."
        (carries case markers `"Due"`, `"to"` and `"punct":","`), and
        `entrance` is the passive subject of `closed`, not an apposition of
        `damage`.

        Distinguishing this from a genuine locative apposition like "Edward
        Place, Newcastle" (where the source also has case markers + a comma
        because it sits in a PP) requires looking at the target. A spurious
        appos target heads its own noun phrase / clause and carries a `det`
        edge to a determiner ("the entrance"); a genuine appositive
        proper-noun ("Newcastle") does not. Depending on the upstream graph
        rewrite, that determiner may already be folded into the target node's
        properties rather than remaining as a `det` edge.

        Drop only when BOTH conditions hold:
          - source has case markers or a separating comma (`punct: ,`);
          - target has a determiner, either as a `det` property or edge.

        After dropping, the target subtree becomes an orphan component that
        `reconnectOrphanComponents` (Phase 1.25) reattaches to the kernel
        verb via `nmod`."""
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
        # TODO: Check if the concatenation of the node children with preposition for multiindobj is contained within HOnK, e.g. (side)<-(side side)->(side)->(by)
        # Hard coding for now...
        test_phrases = {'side by side'}

        multiindobj_nodes = [node for node in G.nodes(data=True) if node[1]['data'].type == 'multipleindobj']
        for node in multiindobj_nodes:
            names = []
            def _node_pos(child):
                if child is None:
                    return math.inf
                props = dict(child.properties) if hasattr(child, "properties") and child.properties is not None else {}
                if "pos" in props:
                    return float(props["pos"])
                if hasattr(child, "pos_f"):
                    return float(child.pos_f())
                return float(getattr(child, "min", math.inf))

            for child in sorted([G.nodes[edge[1]]['data'] for edge in G.out_edges(node[0], data=True)], key=_node_pos):
                child_pos = _node_pos(child)
                appended = False
                for key, value in dict(child.properties).items():
                    try:
                        preposition = float(key)
                        if preposition > child_pos:
                            names.append(child.named_entity)
                            names.append(value)
                            appended = True
                        else:
                            names.append(value)
                            names.append(child.named_entity)
                            appended = True
                    except ValueError:
                        pass
                if not appended:
                    names.append(child.named_entity)

            joined_name = ' '.join(names)
            if joined_name in test_phrases:
                old_sing = G.nodes[node[0]]['data']
                nx.set_node_attributes(G, {node[0]: Singleton(
                    id=node[0],
                    named_entity=joined_name,
                    properties=old_sing.properties,
                    min=old_sing.min,
                    max=old_sing.max,
                    type='ENTITY',
                    confidence=1
                )}, 'data')

                edges_to_remove = [edge for edge in G.out_edges(node[0], data=True)]
                for edge in edges_to_remove:
                    G.remove_edge(*edge[:2])

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

                # Propagate incoming edges
                for parent_edge in in_edges:
                    G.add_edge(parent_edge[0], child_id, label=parent_edge[2]['label'], isNegated=parent_edge[2]['isNegated'])

                # Propagate properties
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
            self.drawNetworkXGraph(G)

        return G

    # Phase 2.1
    def associateNodeToBestMeuMatch(self, item):
        meu_entities = []
        # Loop over Stanza MEU and Singleton to evaluate overlapping words from chars
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
                    meu_entities.append(meu)

        return meu_entities

    # Phase 2.5
    def mergeMeuNodes(self, G):
        if self.meu_db_row is None:
            return G

        # Identify MEUs that are more than one word (contain spaces)
        multi_word_meus = [meu for meu in self.meu_db_row.multi_entity_unit if " " in meu.text]
        
        # Sort by length descending to handle overlapping MEUs
        multi_word_meus.sort(key=lambda x: len(x.text), reverse=True)

        for meu in multi_word_meus:
            # Find all nodes that are contained within this MEU's character range
            meu_nodes_ids = []
            for node_id, node_data in G.nodes(data=True):
                singleton = node_data['data']
                if singleton.min >= meu.start_char and singleton.max <= meu.end_char:
                    meu_nodes_ids.append(node_id)
            
            if len(meu_nodes_ids) > 1:
                # SAFE MERGING: Only merge nodes if they are connected by an edge within the MEU
                # This ensures we don't skip over intermediate structure and cause cycles.
                
                # Sort by position
                meu_nodes_ids.sort(key=lambda nid: G.nodes[nid]['data'].min)
                
                # Use nx.contracted_nodes ONLY for adjacent nodes in the MEU
                # Only merge across lexical-cohesion edges, never across semantic argument edges.
                # Semantic edges like obl/nmod/acl introduce genuine structural relationships that
                # must survive as separate nodes in the graph (e.g. "open" → obl → "public").
                LEXICAL_MERGE_EDGES = DependencyRoles.lexical_merge_edges()

                def _has_lexical_edge(G, u, v):
                    for src, dst, data in G.edges(data=True):
                        if {src, dst} == {u, v}:
                            label = data.get('label', '')
                            label_name = label.named_entity if hasattr(label, 'named_entity') else str(label)
                            if label_name in LEXICAL_MERGE_EDGES:
                                return True
                    return False

                changed = True
                while changed:
                    changed = False
                    for i in range(len(meu_nodes_ids)):
                        for j in range(len(meu_nodes_ids)):
                            if i == j: continue
                            u_id = meu_nodes_ids[i]
                            v_id = meu_nodes_ids[j]
                            if u_id in G and v_id in G and _has_lexical_edge(G, u_id, v_id):
                                # Combine data manually before contraction to preserve information
                                u_data = G.nodes[u_id]['data']
                                v_data = G.nodes[v_id]['data']

                                # Decide on target (keep the one with lower min for stability)
                                target, source = (u_id, v_id) if u_data.min <= v_data.min else (v_id, u_id)

                                # Merge data
                                target_data = G.nodes[target]['data']
                                source_data = G.nodes[source]['data']

                                combined_props = dict(target_data.properties)
                                for k, v in dict(source_data.properties).items():
                                    if k not in combined_props:
                                        combined_props[k] = v

                                merged_min = min(target_data.min, source_data.min)
                                merged_max = max(target_data.max, source_data.max)
                                # Derive the surviving name from the sentence
                                # text over the combined span. Keeping only
                                # target_data.named_entity drops the source's
                                # surface form (e.g. for the compound
                                # "roof repairs" with target=roof, source=
                                # repairs we'd lose "repairs"). Slicing the
                                # original sentence is robust to chained
                                # merges (avoids the "Prince Terrace Albert"
                                # ordering problem) since the substring is
                                # already in surface order.
                                if (self.meu_db_row is not None
                                        and isinstance(getattr(self.meu_db_row, 'first_sentence', None), str)
                                        and 0 <= merged_min < merged_max <= len(self.meu_db_row.first_sentence)):
                                    merged_name = self.meu_db_row.first_sentence[merged_min:merged_max]
                                else:
                                    merged_name = target_data.named_entity

                                new_singleton = Singleton(
                                    id=target_data.id,
                                    named_entity=merged_name,  # may still be corrected by meu.text rename below
                                    properties=frozenset(combined_props.items()),
                                    min=merged_min,
                                    max=merged_max,
                                    type=meu.type if (meu.type != "None" and (target_data.min == meu.start_char or source_data.min == meu.start_char)) else target_data.type,
                                    confidence=meu.confidence
                                )
                                new_singleton = _promote_geo_suffixed_type(new_singleton, self.meu_db_row, self.honk)

                                G = nx.contracted_nodes(G, target, source, self_loops=False)
                                nx.set_node_attributes(G, {target: new_singleton}, 'data')
                                G.nodes[target].pop('contraction', None)
                                changed = True
                                break
                        if changed: break

            # After merging (if any), correct the name of the surviving node.
            # Incremental concatenation can produce wrong word order when the merge
            # order does not follow sentence position (e.g. "Prince Terrace Albert"
            # instead of "Prince Albert Terrace").  meu.text is always built from
            # tokens in sentence order, so it is the authoritative surface form.
            #
            # Guard: only rename when the surviving node's character range EXACTLY
            # matches the MEU's full span (both start and end).  Two failure cases
            # this prevents:
            #   • End mismatch — e.g. "are" in "University station are" has no graph
            #     node, so surviving.max=36 != meu.end_char=40 → no rename.
            #   • Start mismatch — e.g. a "takes place" MEU (start=92) whose only
            #     in-range node is "place" (min=98) must not be renamed to
            #     "takes place"; surviving.min=98 != meu.start_char=92 → no rename.
            # Moving this block outside `if len > 1:` also lets it fire for the
            # single-node case: after a previous MEU merged University+station into
            # one node (min=18, max=36), the "University station" MEU finds that
            # node alone and correctly renames it.
            remaining = [nid for nid in meu_nodes_ids if nid in G]
            if len(remaining) == 1:
                surviving = G.nodes[remaining[0]]['data']
                # Don't let a "None"-typed MEU overwrite a node that has already
                # been renamed by a better-typed MEU with the same span.
                type_ok = (meu.type != "None" or
                           surviving.type in {"None", "existential", "noun"})
                if (isinstance(surviving, Singleton) and
                        surviving.named_entity != meu.text and
                        surviving.min == meu.start_char and
                        surviving.max == meu.end_char and
                        type_ok):
                    renamed_singleton = Singleton(
                        id=surviving.id,
                        named_entity=meu.text,
                        properties=surviving.properties,
                        min=surviving.min,
                        max=surviving.max,
                        type=surviving.type,
                        confidence=surviving.confidence,
                    )
                    renamed_singleton = _promote_geo_suffixed_type(renamed_singleton, self.meu_db_row, self.honk)
                    nx.set_node_attributes(G, {remaining[0]: renamed_singleton}, 'data')

        return G

    # Phase 2.2
    def nodeTypeResolution(self, item, meu_entities, G):
        if len(meu_entities) > 0:
            best_item = None
            item_type = item.type.name if isinstance(item.type, Grouping) else str(item.type)
            if item_type in {"PRONOUN", "PRP", "PRP$", "WP", "WP$"}:
                best_type = 'PRONOUN'
                best_score = 1
            elif item_type in {'HYPH'}:
                best_type = 'HYPH'
                best_score = 1
            elif item_type.startswith("JJ"):
                best_type = 'JJ'
                best_score = 1
            elif item_type == 'verb':
                best_type = 'verb'
                best_score = 1
            else:
                if item_type == '∃' or item_type.startswith("JJ") or item_type.startswith("IN") or item_type.startswith(
                        "NEG") or item_type == "RB":
                    best_score = item.confidence
                    best_item = item
                    best_type = item_type
                else:
                    # TODO: Fix typing information (e.g. Golden Gate Bridge has GPE (0.8) and ENTITY (1.0)
                    best_score = max(map(lambda y: y.confidence, meu_entities))

                    # If the best_score is better than what we currently have for the Singleton
                    if item.confidence >= best_score and item_type.upper() in {'VERB', 'PERSON', 'DATE', 'GPE', 'LOC', 'ENTITY'}:
                        best_type = item_type.lower() if item_type == 'VERB' else item_type
                    else:
                        # TODO: min and max might not be correct when coming from an inherit edge
                        best_items = [
                            y for y in meu_entities
                            if y.confidence == best_score
                        ]
                        if len(best_items) == 0:
                            return item
                        if len(best_items) == 1:
                            best_item = best_items[0]
                            # Apply the same disambiguation conditions as the multi-item case:
                            # A node with a case preposition (float key) or passive-subject marker
                            # is a noun/location, not a verb, even if meuDB says verb at top confidence.
                            if best_item.type in ("VERB", "verb") and (
                                'subjpass' in dict(item.properties) or
                                case_in_props(dict(item.properties))
                            ):
                                non_verb_items = sorted(
                                    [y for y in meu_entities if y.type not in ("VERB", "verb")],
                                    key=lambda y: y.confidence,
                                    reverse=True
                                )
                                if non_verb_items:
                                    best_item = non_verb_items[0]
                                    best_type = non_verb_items[0].type
                                else:
                                    best_type = item_type
                            else:
                                best_type = best_item.type
                        else:
                            best_types = list(set(map(lambda best_item: best_item.type, best_items)))
                            if len(best_types) == 1:
                                # Apply disambiguation before committing to a single-type result.
                                # Nodes with a passive-subject marker or case preposition are nouns, not verbs.
                                if best_types[0] in ("VERB", "verb") and (
                                    'subjpass' in dict(item.properties) or
                                    case_in_props(dict(item.properties))
                                ):
                                    non_verb_items = sorted(
                                        [y for y in meu_entities if y.type not in ("VERB", "verb")],
                                        key=lambda y: y.confidence,
                                        reverse=True
                                    )
                                    best_type = non_verb_items[0].type if non_verb_items else item_type
                                else:
                                    best_type = best_types[0]
                            ## TODO! type disambiguation, in future works, needs to take into account also the verb associated to it!
                            elif ("VERB" in best_types or "verb" in best_types) and (
                                # If a node is marked with a det, never consider this as a verb
                                'det' not in dict(item.properties)
                                and
                                # OBL nodes from passive constructions carry subjpass - these are nouns not verbs
                                'subjpass' not in dict(item.properties)
                                and
                                # TODO: This condition may need to be revised
                                # 'on' is very unlikely to lead to a verb
                                ('on' not in case_in_props(dict(item.properties), True))
                                and
                                ((
                                    # If a node has at least one ingoing edge and comes with a case-derived float attribute
                                    (
                                        len(G.in_edges(item.id)) > 0 and
                                        any(case_in_props(dict(G.nodes[x]['data'].properties)) for x in [edge[0] for edge in G.in_edges(item.id)])
                                    )
                                    or
                                    # If a node is the last occurring in the root/kernel of all kernels and has no ingoing edges
                                    (
                                        len(G.in_edges(item.id)) == 0 and
                                        is_kernel_in_props(item) # TODO: Check if this matches previous condition
                                    )
                                    or
                                    # TODO: Is this an acceptable condition?
                                    # If a node is last occurring, and has one parent which is a root connected by a compound edge, and no other ingoing edges
                                    (
                                        len(G.in_edges(item.id)) == 1 and
                                        list(G.in_edges(item.id, data=True))[0][2]['label'].named_entity == "compound" and
                                        is_kernel_in_props(dict(G.nodes[list(G.in_edges(item.id, data=True))[0][1]]['data'].properties))
                                    )
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
            # Universal compound-head override: any node that itself heads a compound
            # chain (outgoing `compound` edge to a modifier) or has already absorbed
            # one (carries an `extra` property after merging) is a noun-noun compound
            # head, not a verb — regardless of which disambiguation branch above
            # produced the verdict.
            if str(best_type).lower() == 'verb':
                if (any(e[2]['label'].named_entity == 'compound' for e in G.out_edges(item.id, data=True))
                        or 'extra' in dict(item.properties)):
                    non_verb_items = sorted(
                        [y for y in meu_entities if str(y.type).lower() != 'verb'],
                        key=lambda y: y.confidence,
                        reverse=True
                    )
                    best_type = non_verb_items[0].type if non_verb_items else 'noun'
            return Singleton(
                id=item.id,
                named_entity=item.named_entity,  # TODO: Future work: best_item.monad if best_item is not None and not isinstance(best_item, Singleton) else item.named_entity TODO: Future work
                properties=item.properties,
                min=item.min,
                max=item.max,
                type=best_type,
                confidence=best_score
            )
        else:
            return item

    # Phase 3
    def preProcessGraph(self, G):
        nodes_to_remove = []
        edges_to_remove = []

        # Replace personal pronoun nodes (POS=PRP, lemma in HOnK PersonalPronoun set)
        # with existentials so downstream FOL doesn't carry the surface "We"/"They".
        personal_pronouns = self.honk.getPersonalPronouns()
        if personal_pronouns:
            for node_id, node_data in list(G.nodes(data=True)):
                singleton = node_data.get('data')
                if not isinstance(singleton, Singleton):
                    continue
                raw_type = getattr(singleton, 'type', '')
                if not isinstance(raw_type, str):
                    continue
                node_type = (raw_type or '').upper()
                if node_type not in {'PRP', 'PRP$', 'PRONOUN'}:
                    continue
                props = dict(singleton.properties) if singleton.properties is not None else {}
                lemma = (props.get('lemma') or singleton.named_entity or '').lower()
                if lemma not in personal_pronouns:
                    continue
                preserved_props = {}
                for key in ('pos', 'begin', 'end', 'kernel', 'root', 'subjpass'):
                    if key in props:
                        preserved_props[key] = props[key]
                new_singleton = Singleton(
                    id=singleton.id,
                    named_entity="?" + str(self.existentials.increaseAndGetExistential()),
                    properties=frozenset(preserved_props.items()),
                    min=singleton.min,
                    max=singleton.max,
                    type='existential',
                    confidence=singleton.confidence,
                )
                nx.set_node_attributes(G, {node_id: new_singleton}, 'data')

        for hyph_node in [node[1]['data'] for node in G.nodes(data=True) if node[1]['data'].type == 'HYPH']:
            first_word = next((node for node in G.nodes(data=True) if node[1]['data'].max == hyph_node.min), None)
            second_word = next((node for node in G.nodes(data=True) if node[1]['data'].min == hyph_node.max), None)

            if first_word and second_word:
                # HYPH could be '-' or '/', so use node's name
                nx.set_node_attributes(G, {first_word[0]: first_word[1]['data'].update_name(f"{first_word[1]['data'].named_entity}{hyph_node.named_entity}{second_word[1]['data'].named_entity}")}, 'data')
                nodes_to_remove.extend([hyph_node.id, second_word[0]])

                for node in [n[1]['data'] for n in G.nodes(data=True) if dict(second_word[1]['data'].properties)['pos'] in dict(n[1]['data'].properties)]:
                    nx.set_node_attributes(G, {node.id: node.remove_prop(dict(second_word[1]['data'].properties)['pos'])}, 'data')

        for edge in G.edges(data=True):
            # If any node already marked for removal is in the edge, skip
            if any([x for x in nodes_to_remove if x in edge]):
                continue

            source, target, edge_label = edge
            source = G.nodes[source]
            target = G.nodes[target]
            edge_label = edge_label['label'].named_entity

            if 'inherit_edge' in edge_label:
                source_props = source['data'].get_props()
                target_props = target['data'].get_props()
                if not target['data'].type in source_props:
                    new_properties = merge_properties(source_props, target_props, {'begin', 'end', 'pos'})
                    nx.set_node_attributes(G, {edge[0]: source['data'].update_node_props(new_properties)}, 'data')
                if edge[1] not in nodes_to_remove:
                    nodes_to_remove.append(edge[1])
            elif edge_label in {'mark', 'punct', 'amod', 'advmod', 'case', 'adv'}: # and ('IN' in target['data'].type or 'TO' in target['data'].type):
                if isinstance(target['data'], Singleton):
                    type_key = self.node_functions.get_node_type(target['data']) if edge_label != 'case' else 'case'
                else:
                    type_key = self.honk.most_general_type(
                        map(lambda x: x.type, target['data'].entities))

                if type_key != 'existential':
                    target_props = target['data'].get_props()
                    source_props = source['data'].get_props()
                    new_properties = merge_properties(source_props, target_props, {'begin', 'end', 'pos'})
                    nx.set_node_attributes(G, {edge[0]: source['data'].update_node_props(new_properties)}, 'data')
                    nx.set_node_attributes(G, {edge[0]: source['data'].add_property(edge_label, target['data'].get_name())}, 'data')
                    if edge[1] not in nodes_to_remove:
                        edges_to_remove.append(edge)
                    # For case edges, propagate the target's position key and any conj/cc children
                    # so that compound prepositions like "on or near" accumulate on the source node.
                    if edge_label == 'case':
                        # Add position-keyed property for the case target itself
                        case_target_pos = dict(G.nodes[edge[1]]['data'].properties).get('pos')
                        if case_target_pos is not None:
                            nx.set_node_attributes(G, {edge[0]: G.nodes[edge[0]]['data'].add_property(case_target_pos, G.nodes[edge[1]]['data'].get_name())}, 'data')
                        for conj_edge in list(G.out_edges(edge[1], data=True)):
                            if conj_edge[2]['label'].named_entity == 'conj' and conj_edge[1] in G.nodes:
                                conj_node = G.nodes[conj_edge[1]]['data']
                                conj_pos = dict(conj_node.properties).get('pos')
                                if conj_pos is not None:
                                    nx.set_node_attributes(G, {edge[0]: G.nodes[edge[0]]['data'].add_property(conj_pos, conj_node.get_name())}, 'data')
                                for cc_edge in list(G.out_edges(conj_edge[1], data=True)):
                                    if cc_edge[2]['label'].named_entity == 'cc' and cc_edge[1] in G.nodes:
                                        cc_node = G.nodes[cc_edge[1]]['data']
                                        cc_pos = dict(cc_node.properties).get('pos')
                                        if cc_pos is not None:
                                            nx.set_node_attributes(G, {edge[0]: G.nodes[edge[0]]['data'].add_property(cc_pos, cc_node.get_name())}, 'data')
                                        if cc_edge[1] not in nodes_to_remove:
                                            nodes_to_remove.append(cc_edge[1])
                                if conj_edge[1] not in nodes_to_remove:
                                    nodes_to_remove.append(conj_edge[1])
            elif edge_label in {'cop'} and target['data'].type.lower() == 'verb':
                G.add_edge(edge[1], edge[0], label=target['data'], isNegated=edge[2]['isNegated'])
                # Only mark the cop existential as kernel root if no other node already
                # carries a 'kernel'/'root' marker.  Otherwise the cop existential becomes
                # a competing top-level root and (via topo sort id-ordering) can hijack
                # `final_kernel`, producing `be(?, None)` instead of the actual main verb
                # kernel (e.g. `record(?, burglary)`).
                has_existing_root = any(
                    G.nodes[n].get('data') is not None and
                    hasattr(G.nodes[n]['data'], 'properties') and
                    n != edge[1] and
                    ('kernel' in dict(G.nodes[n]['data'].properties) or 'root' in dict(G.nodes[n]['data'].properties))
                    for n in G.nodes
                )
                new_existential = self.node_functions.create_existential_node(G, edge[1])
                if not has_existing_root:
                    new_existential = new_existential.add_property('kernel', 'root')
                nx.set_node_attributes(G, {edge[1]: new_existential}, 'data')
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            G.remove_edge(*edge[:2])

        for node in nodes_to_remove:
            G = self.remove_node(G, node)

        if self.shouldDrawGraphs:
            self.drawNetworkXGraph(G)

        return G

    # Keep only 'data' part of node, remove anything like 'type' or 'contraction' that might remain
    def keepDataKey(self, G, node_id):
        node_attributes = G.nodes[node_id]

        keys_to_remove = [key for key in node_attributes if key != 'data']
        for key in keys_to_remove:
            del node_attributes[key]

    # Phase 4
    def mergeNodes(self, G):
        nodes_to_remove = []

        # Merge 'compound_prt' to Singleton
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

            # Identify what type of merge is happening
            nx.set_node_attributes(G, {
                edge[0]: G.nodes[edge[0]]['data'].type if edge[2]['label'].named_entity in ['orig'] else edge[2]['label'].named_entity
            }, 'type')

            if G.nodes[edge[0]]['type'] == Grouping.NONE:
                nx.set_node_attributes(G, {
                    edge[0]: Grouping.AND if 'conj' in G.nodes[edge[0]]['data'].get_props() else Grouping.NONE
                }, 'type')

            # Merge child into parent node
            if edge[0] in G and edge[1] in G:
                if self.dependency_role_rewriter.preserve_dependency_role(G, edge):
                    if edge[1] not in nodes_to_remove:
                        nodes_to_remove.append(edge[1])
                    continue

                G = nx.contracted_nodes(G, edge[0], edge[1], self_loops=False)

                # Create SetOfSingleton nodes - conditions to wait before resolving contracted nodes
                if ((
                    # If the next edge's source is NOT equal to current source AND edge's source is not equal to next edge's target
                    idx + 1 < len(merge_edges) and merge_edges[idx + 1][0] != edge[0] and merge_edges[idx + 1][1] != edge[0] or
                        # [OR:] If the next edge's source IS equal to current source and edge labels are different
                        (
                            idx + 1 < len(merge_edges) and merge_edges[idx + 1][0] == edge[0] and
                            merge_edges[idx + 1][2]['label'].named_entity != edge[2]['label'].named_entity
                        )
                # [OR:] This is the last edge in the array
                ) or idx == len(merge_edges) - 1):
                    node = G.nodes[edge[0]]
                    node_type = node['type']

                    # Append recursively through all node 'contractions'
                    # TODO: NOTE: Currently excluding children with 'conj' as a property as likely they are accounted for by the 'orig' nodes
                    # grouped_nodes = (lambda f: f(f, node))(lambda f, node: ([node['data']] if 'data' in node and hasattr(node['data'], 'type') and not isinstance(node['data'].type, Grouping) else []) + [item for sub_node in node.get('contraction', {}).values() for item in f(f, sub_node)])

                    grouped_nodes = (lambda f: f(f, node))(lambda f, node: ([node['data']] if 'data' in node and hasattr(node['data'], 'type') and not (isinstance(node['data'], Singleton) and 'conj' in dict(node['data'].properties)) else []) + [item for sub_node in node.get('contraction', {}).values() for item in f(f, sub_node)])

                    # Absorb all properties and merge (only Singletons have .properties;
                    # SetOfSingletons can appear in the contraction tree but cannot be merged here)
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
                            node_type = self.get_group_enum(G.nodes[cc_node]['data'].named_entity)  # TODO: What if AND/OR, is this still AND?
                            nodes_to_remove.append(cc_node)
                        else:
                            node_type = Grouping.AND
                    elif node['type'] == 'existential':
                        # If type is 'existential', we are unlikely wanting a group, and as properties have all been merged, we can just select 'root' node from group
                        gid = grouped_nodes[0].id
                        grouped_nodes = [G.nodes[gid]['data'] if gid in G.nodes else grouped_nodes[0]]
                    else:
                        # Get the Grouping type from the node, if we have a string, then find the type, if this is NONE, then just use the node_type if it is already a Grouping otherwise it is GROUPING
                        node_type = self.get_group_enum(node_type) if not isinstance(node_type, Grouping) and self.get_group_enum(node_type) is not Grouping.NONE else node_type if isinstance(node_type, Grouping) else Grouping.GROUPING

                    norm_confidence = 1.0

                    # Create the SetOfSingletons
                    if len(grouped_nodes) > 1:
                        new_node = SetOfSingletons(
                            id=node['data'].id,
                            type=node_type,
                            entities=tuple(grouped_nodes),
                            min=min(grouped_nodes, key=lambda x: x.min).min,
                            max=max(grouped_nodes, key=lambda x: x.max).max,
                            confidence=norm_confidence * math.prod([node.confidence for node in grouped_nodes]),
                            root=any(map(is_kernel_in_props, grouped_nodes))# or is_kernel_in_props(node['data'])
                        )

                        # Resolve to Singleton if type GROUPING
                        new_node = GraphNER_withProperties(
                            new_node,
                            self.is_simplistic_rewriting,
                            self.meu_db_row,
                            self.services.getHOnK(),
                            self.existentials
                        ) if node_type == Grouping.GROUPING else new_node
                    else:
                        new_node = grouped_nodes[0]

                    if new_node.id not in list(G.nodes):
                        G.add_node(new_node.id, data=new_node)
                    else:
                        nx.set_node_attributes(G, {node['data'].id: new_node}, 'data')
                    self.keepDataKey(G, node['data'].id)

                    # Re-check node type for new Singleton (meaning might have changed, i.e. compound_prt merge etc.)
                    if isinstance(new_node, Singleton):
                        nx.set_node_attributes(G, {node['data'].id: self.nodeTypeResolution(new_node, self.associateNodeToBestMeuMatch(new_node), G)}, 'data')

                    # Check if this new Singleton has a BUT parent node
                    for parent_id in [n for n in [edge[0] for edge in G.in_edges(node['data'].id)] if (isinstance(G.nodes[n]['data'], Singleton) and G.nodes[n]['data'].named_entity == 'but') or (isinstance(G.nodes[n]['data'], SetOfSingletons) and G.nodes[n]['data'].entities[0].named_entity == 'but') and n not in nodes_to_remove]:
                        nodes_to_remove.append(node['data'].id)

                        # If BUT node has a negation, negate the newly created group
                        negation_nodes = [
                            e for e in G.out_edges(
                                [n for n in [edge[0] for edge in G.in_edges(node['data'].id)]
                                 if (isinstance(G.nodes[n]['data'], Singleton) and G.nodes[n]['data'].named_entity == 'but') or (isinstance(G.nodes[n]['data'], SetOfSingletons) and G.nodes[n]['data'].entities[0].named_entity == 'but')][0], data=True
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

                    # Check if newly grouped node has a negation, to ensure it's properly negated before perhaps being grouped further
                    self.check_for_negations(G, nodes_to_remove)

        # Final check for negations incase there are nodes that weren't grouped that need to be negated
        self.check_for_negations(G, nodes_to_remove)

        G.remove_nodes_from(nodes_to_remove)

        # Remove isolated nodes, as long as it is not a 'root' node, and there is more than one node in the graph.
        # Nodes with case prepositions (e.g. TIME nodes with 'in') must be preserved for reconnection below.
        G.remove_nodes_from([node for node in nx.isolates(G) if (
            isinstance(G.nodes[node]['data'], Singleton) and
            'kernel' not in dict(G.nodes[node]['data'].properties) and
            not case_in_props(dict(G.nodes[node]['data'].properties))
        )]) if len(G.nodes()) > 1 else None

        # Reconnect isolated nodes that carry a case preposition to the root/kernel node.
        # These can arise when a passive construction has multiple obl children but the grammar
        # rule only captures one (the other obl then loses its parent edge when V is deleted).
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
                    # Get the innermost Singleton to inspect its preposition properties
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
            self.drawNetworkXGraph(G)

        return G

    def check_for_negations(self, G, nodes_to_remove):
        for e in [n for n in [edge for edge in G.edges(data=True)] if
                  G.nodes[n[1]]['data'].type == 'NEG' and n[1] not in nodes_to_remove]:
            node = G.nodes[e[0]]['data']
            neg_node = G.nodes[e[1]]['data']
            nodes_to_remove.append(e[1])

            # When the negated node is an AND group and the neg word has a valid position,
            # only wrap entities that appear after the negation word in NOT.
            # This handles "X but not Y" → AND(X, NOT(Y)) rather than NOT(AND(X, Y)).
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

    def get_group_enum(self, name):
        if 'and' in name or 'but' in name or 'appos' in name:  # TODO: 'appos' assumption?
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

    # Removes a node while ensuring any paths are reconnected from parent to removed child node
    def remove_node(self, G, node):
        for parent in [edge for edge in G.in_edges(node, data=True)]:
            for child in [edge for edge in G.out_edges(node, data=True)]:
                G.add_edge(parent[0], child[1], label=child[2]['label'], isNegated=child[2]['isNegated'])

        G.remove_node(node)

        return G

    # Add any edges from source to child's children to ensure all paths are kept
    def remove_edge(self, G, edge):
        for child in G.out_edges(edge[1], data=True):
            G.add_edge(edge[0], child[1], label=child[2]['label'], isNegated=child[2]['isNegated'])

        G.remove_edge(*edge[:2])

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
