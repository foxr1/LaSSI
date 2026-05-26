import math
import networkx as nx
import json
import os
from LaSSI.structures.internal_graph.EntityRelationship import Singleton
from LaSSI.structures.kernels.Sentence import case_in_props

class GraphSanitiser:
    def __init__(self, shouldDrawGraphs=False, honk=None):
        self.shouldDrawGraphs = shouldDrawGraphs
        self.honk = honk


    def sanitise(self, G):
        G = self.flip_measurement_compounds(G)
        G = self.sanitiseCompoundChainsByPosition(G)
        G = self.rewriteParentheticalApposToCompound(G)
        G = self.dropSpuriousApposEdges(G)
        G = self.reconnectOrphanComponents(G)
        G = self.resolveMultipleInDobj(G)
        G = self.promoteApposHeadAboveMisparsedRoot(G)
        return G

    def promoteApposHeadAboveMisparsedRoot(self, G):
        """Repair Stanford's occasional root misanalysis on `Met Office
        forecast for X: ..." sentences.

        Compare the two GSMs for "Met Office forecast for Newcastle..." in
        weather_002 vs weather_006: weather_002 roots correctly on
        `forecast` (with `Met`/`Office` as compound modifiers and the
        appositional colon-list hanging off it), whereas weather_006 roots
        on `Office`, with `forecast` reached via a `dep` edge through a
        `multipleindobj`. The two analyses are semantically the same
        sentence and should produce structurally equivalent kernels, but
        the misparsed version ends up with `Met Office` as the kernel
        source and `forecast` stranded in the predicate target — breaking
        cross-sentence comparison.

        Pattern (post-`resolveMultipleInDobj`):
          - root R is a noun
          - R has at least one outgoing `compound` edge (the modifier
            tokens, e.g. `Met`)
          - R has an outgoing `dep` edge to a noun N that itself carries
            an `appos` chain (the real semantic head of the noun-phrase)

        Rewrite:
          1. Move `kernel='root'` from R to N so the downstream kernel
             builder uses N as the source-side head.
          2. Add a single `compound` edge from N to R. NodeMerger's
             compound pass first contracts Met into Office via the
             existing Office→compound→Met edge, then contracts Office
             (already carrying Met) into N via the new edge. The
             finalize step's `grouped_nodes` lambda recursively unfolds
             nested contractions, so GraphNER receives [N, Office, Met]
             and picks "Met Office" as the head with N (forecast) as
             extra — exactly as weather_002 does.

             We deliberately do NOT mirror R's compound children
             directly onto N (e.g. N→compound→Met). Adding redundant
             same-label edges with the same source confuses NodeMerger's
             finalize trigger: it walks `merge_edges` looking for a label
             change to know when a compound chain ends, and an extra
             N→compound→Met that gets skipped (because Met is already
             gone by then) lets the chain run off the end of the list
             without ever finalizing.
          3. Re-point R's remaining out-edges (e.g. `dep`→Z<dot> timestamp)
             onto N so the colon-list tail sits under the new root.
          4. If N has a `dep` edge to a common-noun head H (e.g.
             `shower`), MOVE N's appos chain onto H — i.e. each
             N→appos→X becomes H→appos→X and the original edge is
             dropped. Why: after step (2) the compound merge turns N
             into the Singleton `Met Office[extra:forecast]`. If the
             appos edges stayed on N, the subsequent appos contraction
             in NodeMerger would re-wrap N in an `AND` SetOfSingletons
             — turning the kernel source itself into a group, which is
             wrong. Moving the chain onto H mirrors weather_002 (where
             `rain` is the appos source feeding the AND target while
             `forecast` is a clean Singleton root), and N→dep→H stays
             intact as the attachment from root to target.
        """
        root_id = None
        for nid in G.nodes:
            data = G.nodes[nid].get('data')
            if data is None or not hasattr(data, 'properties'):
                continue
            if dict(data.properties).get('kernel') == 'root':
                root_id = nid
                break
        if root_id is None:
            return G

        root_data = G.nodes[root_id].get('data')
        if not isinstance(root_data, Singleton):
            return G
        if getattr(root_data, 'type', None) != 'noun':
            return G

        def _pos_of(node):
            if node is None or not hasattr(node, 'properties'):
                return None
            raw = dict(node.properties).get('pos')
            if raw is None:
                return None
            try:
                return int(float(raw))
            except (TypeError, ValueError):
                return None

        root_pos = _pos_of(root_data)

        # Collect root's compound children and dep candidates.
        # Gate dep candidates on positional adjacency (root.pos + 1):
        # weather_002 has a correctly-rooted `forecast` (pos 3) whose dep
        # children (rain pos 10 etc.) are positionally far away — we must
        # NOT re-root those. The misparse signature we're targeting is
        # specifically the case where Stanford rooted on the *middle* of
        # an "X Y Z" noun chain (Met-Office-forecast), so the real
        # rightmost head sits at exactly root.pos + 1.
        root_compound_children = []
        dep_candidates = []
        for _src, dst, edata in G.out_edges(root_id, data=True):
            label = edata.get('label')
            if label is None:
                continue
            name = label.named_entity
            if name == 'compound':
                root_compound_children.append(dst)
            elif name == 'dep':
                dst_data = G.nodes[dst].get('data')
                if not isinstance(dst_data, Singleton) or dst_data.type != 'noun':
                    continue
                dst_pos = _pos_of(dst_data)
                if dst_pos is None or root_pos is None or dst_pos != root_pos + 1:
                    continue
                has_appos = any(
                    e.get('label') is not None and e['label'].named_entity == 'appos'
                    for _s, _d, e in G.out_edges(dst, data=True)
                )
                if has_appos:
                    dep_candidates.append(dst)
        if not root_compound_children or not dep_candidates:
            return G

        new_root_id = dep_candidates[0]
        new_root_data = G.nodes[new_root_id].get('data')

        # (1) Move root marker.
        new_root_props = dict(new_root_data.properties)
        new_root_props['kernel'] = 'root'
        nx.set_node_attributes(
            G, {new_root_id: new_root_data.update_node_props(new_root_props)}, 'data'
        )
        old_root_props = dict(root_data.properties)
        old_root_props.pop('kernel', None)
        nx.set_node_attributes(
            G, {root_id: root_data.update_node_props(old_root_props)}, 'data'
        )
        # Refresh local refs after property updates.
        new_root_data = G.nodes[new_root_id]['data']
        root_data = G.nodes[root_id]['data']

        def _compound_label():
            return Singleton(
                id=new_root_id,
                named_entity='compound',
                properties=frozenset(),
                min=new_root_data.min,
                max=new_root_data.max,
                type='non_verb',
                confidence=1.0,
            )

        # (2) Add a single N→compound→R edge. The existing R→compound→X
        # chain stays put; NodeMerger contracts X into R, then R (with X
        # nested) into N, and finalize unfolds the lot via recursion.
        if not any(
            e['label'].named_entity == 'compound'
            for _s, _d, e in G.out_edges(new_root_id, data=True)
            if _d == root_id
        ):
            G.add_edge(new_root_id, root_id, label=_compound_label(), isNegated=False)

        # (3) Re-point R's remaining out-edges (everything except R's own
        # compound children, which stay so the nested compound merge fires
        # correctly) onto N. Removes the R→dep→N edge in passing.
        edges_to_move = []
        for src, dst, key, edata in list(G.out_edges(root_id, keys=True, data=True)):
            label_name = edata['label'].named_entity
            if label_name == 'compound':
                continue
            edges_to_move.append((src, dst, key, edata))
        for src, dst, key, edata in edges_to_move:
            if G.has_edge(src, dst, key):
                G.remove_edge(src, dst, key)
            if dst == new_root_id:
                # The R→dep→N edge that motivated the rewrite — no longer
                # needed once N is the root.
                continue
            G.add_edge(new_root_id, dst, label=edata['label'], isNegated=edata.get('isNegated', False))

        # (4) Find a common-noun dep target of N (the semantic head H of
        # the appositional list, e.g. `shower`) and MOVE N's appos chain
        # onto H. Leaves N as a clean Singleton after compound merge,
        # while H becomes the appos source whose AND group is the kernel
        # target.
        head_dst = None
        for _src, dst, edata in list(G.out_edges(new_root_id, data=True)):
            if edata['label'].named_entity != 'dep':
                continue
            dst_data = G.nodes[dst].get('data')
            if not isinstance(dst_data, Singleton):
                continue
            if dst_data.type != 'noun':
                continue
            spec = dict(dst_data.properties).get('specification', '')
            if spec != 'common':
                # Only common nouns — leaves proper-noun timestamps etc.
                # under `dep` so they become TIME context downstream.
                continue
            head_dst = dst
            break

        if head_dst is not None:
            appos_to_move = []
            for src, dst, key, edata in list(G.out_edges(new_root_id, keys=True, data=True)):
                if edata['label'].named_entity != 'appos':
                    continue
                appos_to_move.append((src, dst, key, edata))
            for src, dst, key, edata in appos_to_move:
                if not G.has_edge(src, dst, key):
                    continue
                G.remove_edge(src, dst, key)
                # Skip self-loops if H itself appears as one of N's appos targets.
                if dst == head_dst:
                    continue
                G.add_edge(head_dst, dst, label=edata['label'], isNegated=edata.get('isNegated', False))

        if self.shouldDrawGraphs:
            self._draw_graph(G)

        return G

    def rewriteParentheticalApposToCompound(self, G):
        # "Light rain shower (day), 11.28°C, ..." — Stanford analyses the
        # parenthetical "(day)" as an `appos` of "shower".  NodeMerger then
        # treats `appos` as an AND grouping (sibling of `shower` in the
        # kernel target), producing `AND:[rain shower, day]` at the
        # predicate-properties level.  The parenthetical is semantically a
        # parenthetical *modifier* of the head, not a co-referential
        # apposition — so rewrite the edge label from `appos` to `compound`
        # which NodeMerger folds into a GROUPING and GraphNER_withProperties
        # then absorbs as an `extra` on the surviving head node.
        #
        # Signal: dst is a Singleton whose `punct` property equals "(" — set
        # by the GSM grammar when it inherits the opening-paren punct onto
        # the parenthesised noun.
        edges_to_rewrite = []
        for src, dst, key, data in G.edges(data=True, keys=True):
            label = data.get('label')
            if label is None or label.named_entity != 'appos':
                continue
            dst_data = G.nodes[dst].get('data')
            if not isinstance(dst_data, Singleton):
                continue
            dst_props = dict(dst_data.properties) if dst_data.properties else {}
            if dst_props.get('punct') != '(':
                continue
            edges_to_rewrite.append((src, dst, key, data))

        for src, dst, key, data in edges_to_rewrite:
            if not G.has_edge(src, dst, key):
                continue
            old_label = data['label']
            new_label = Singleton(
                id=old_label.id,
                named_entity='compound',
                properties=old_label.properties,
                min=old_label.min,
                max=old_label.max,
                type=old_label.type,
                confidence=old_label.confidence,
            )
            G.remove_edge(src, dst, key)
            G.add_edge(src, dst, label=new_label, isNegated=data.get('isNegated', False))

        if self.shouldDrawGraphs:
            self._draw_graph(G)

        return G

    def flip_measurement_compounds(self, G):
        # `wind 3.09 mph` as mph(NN) -compound-> wind, -nummod-> 3.09:
        # the unit is the head and the entity is a modifier. Rewrite so the
        # entity becomes the surviving node and the unit moves onto it as a
        # `measurement` property like "3.09 mph".
        units_of_measure = {u.lower() for u in self.honk.getUnitsOfMeasure()} if self.honk else set()
        if not units_of_measure:
            return G
        for edge in list(G.edges(data=True)):
            if edge[2]['label'].named_entity != 'compound':
                continue
            head_id, entity_id = edge[0], edge[1]
            if head_id not in G or entity_id not in G:
                continue
            head = G.nodes[head_id]['data']
            if not isinstance(head, Singleton):
                continue
            head_name = (head.named_entity or '').strip()
            if head_name.lower() not in units_of_measure:
                continue
            entity = G.nodes[entity_id]['data']
            if not isinstance(entity, Singleton):
                continue
            head_props = dict(head.properties)
            value = head_props.get('nummod')
            measurement = f"{value} {head_name}".strip() if value is not None else head_name
            entity_new = entity.add_property('measurement', measurement)
            nx.set_node_attributes(G, {entity_id: entity_new}, 'data')
            for src, _dst, _key, data in list(G.in_edges(head_id, keys=True, data=True)):
                if src == entity_id:
                    continue
                G.add_edge(src, entity_id, label=data['label'], isNegated=data.get('isNegated', False))
            for _src, dst, _key, data in list(G.out_edges(head_id, keys=True, data=True)):
                if dst == entity_id:
                    continue
                if data['label'].named_entity in {'nummod', 'compound'}:
                    continue
                G.add_edge(entity_id, dst, label=data['label'], isNegated=data.get('isNegated', False))
            G.remove_node(head_id)
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

            # Preserve compound edges into the head of a hyphenated chain
            # (e.g. `chance -compound-> one` in "one-in-three chance"): the head
            # carries an inherit_edge to a HYPH and will be renamed to the full
            # hyphenated form by GraphPreprocessor._reconstruct_hyphenated_chains
            # downstream, after which NodeMerger can fold it back into the parent.
            dst_has_hyph_inherit = False
            for _src2, hyph_dst, edata2 in G.out_edges(dst, data=True):
                lbl2 = edata2.get('label')
                lbl2_name = lbl2.named_entity if hasattr(lbl2, 'named_entity') else str(lbl2)
                if lbl2_name != 'inherit_edge':
                    continue
                hyph_node = G.nodes[hyph_dst].get('data')
                if hyph_node is not None and getattr(hyph_node, 'type', None) == 'HYPH':
                    dst_has_hyph_inherit = True
                    break
            if dst_has_hyph_inherit:
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
