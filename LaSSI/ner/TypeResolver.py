import networkx as nx
import json
import os
from LaSSI.ner.MergeSetOfSingletons import _promote_geo_suffixed_type
from LaSSI.structures import DependencyRoles
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping
from LaSSI.structures.kernels.Sentence import is_kernel_in_props, case_in_props

class TypeResolver:
    def __init__(self, meu_db_row, honk):
        self.meu_db_row = meu_db_row
        self.honk = honk
        self._load_type_resolution_rules()

    def _load_type_resolution_rules(self):
        self.type_rules = {}
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'raw_data', 'type_resolution_rules.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.type_rules = json.load(f)
        else:
            self.type_rules = {
                "exact_matches": {},
                "startswith_matches": {},
                "confidence_based_exact_matches": [],
                "confidence_based_startswith_matches": [],
                "meu_types": []
            }

    def resolve(self, G):
        # Phase 2 (Resolve types for Singletons from meuDB)
        for node in G.nodes(data=True):
            nx.set_node_attributes(G, {
                node[0]: self.nodeTypeResolution(node[1]['data'], self.associateNodeToBestMeuMatch(node[1]['data']), G)
            }, 'data')

        # Phase 2.5 (Merge nodes based on multi-entity units)
        G = self.mergeMeuNodes(G)
        return G

    def associateNodeToBestMeuMatch(self, item):
        meu_entities = []
        if not self.meu_db_row:
            return meu_entities
            
        for meu in self.meu_db_row.multi_entity_unit:
            start_meu = meu.start_char
            end_meu = meu.end_char
            start_graph = item.min
            end_graph = item.max
            if start_graph > end_meu or start_meu > end_graph:
                continue
            else:
                if not (start_graph > end_meu or start_meu > end_graph):
                    meu_entities.append(meu)
        return meu_entities

    def mergeMeuNodes(self, G):
        if self.meu_db_row is None:
            return G

        multi_word_meus = [meu for meu in self.meu_db_row.multi_entity_unit if " " in meu.text]
        multi_word_meus.sort(key=lambda x: len(x.text), reverse=True)

        for meu in multi_word_meus:
            meu_nodes_ids = []
            for node_id, node_data in G.nodes(data=True):
                singleton = node_data['data']
                if singleton.min >= meu.start_char and singleton.max <= meu.end_char:
                    meu_nodes_ids.append(node_id)
            
            if len(meu_nodes_ids) > 1:
                meu_nodes_ids.sort(key=lambda nid: G.nodes[nid]['data'].min)
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
                                u_data = G.nodes[u_id]['data']
                                v_data = G.nodes[v_id]['data']

                                target, source = (u_id, v_id) if u_data.min <= v_data.min else (v_id, u_id)
                                target_data = G.nodes[target]['data']
                                source_data = G.nodes[source]['data']

                                combined_props = dict(target_data.properties)
                                for k, v in dict(source_data.properties).items():
                                    if k not in combined_props:
                                        combined_props[k] = v

                                merged_min = min(target_data.min, source_data.min)
                                merged_max = max(target_data.max, source_data.max)
                                if (self.meu_db_row is not None
                                        and isinstance(getattr(self.meu_db_row, 'first_sentence', None), str)
                                        and 0 <= merged_min < merged_max <= len(self.meu_db_row.first_sentence)):
                                    merged_name = self.meu_db_row.first_sentence[merged_min:merged_max]
                                else:
                                    merged_name = target_data.named_entity

                                new_singleton = Singleton(
                                    id=target_data.id,
                                    named_entity=merged_name,
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

            remaining = [nid for nid in meu_nodes_ids if nid in G]
            if len(remaining) == 1:
                surviving = G.nodes[remaining[0]]['data']
                type_ok = (meu.type != "None" or surviving.type in {"None", "existential", "noun"})
                if (isinstance(surviving, Singleton) and
                        surviving.named_entity != meu.text and
                        len(meu.text) > len(surviving.named_entity) and
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

    def nodeTypeResolution(self, item, meu_entities, G):
        if len(meu_entities) > 0:
            best_item = None
            item_type = item.type.name if isinstance(item.type, Grouping) else str(item.type)
            
            # Use json configuration for simple exact matches
            if item_type in self.type_rules.get("exact_matches", {}):
                best_type = self.type_rules["exact_matches"][item_type]
                best_score = 1
            else:
                # Use json configuration for simple startswith matches
                matched_start = next((v for k, v in self.type_rules.get("startswith_matches", {}).items() if item_type.startswith(k)), None)
                if matched_start:
                    best_type = matched_start
                    best_score = 1
                else:
                    if item_type in self.type_rules.get("confidence_based_exact_matches", []) or \
                       any(item_type.startswith(prefix) for prefix in self.type_rules.get("confidence_based_startswith_matches", [])):
                        best_score = item.confidence
                        best_item = item
                        best_type = item_type
                    else:
                        best_score = max(map(lambda y: y.confidence, meu_entities))

                        if item.confidence >= best_score and item_type.upper() in self.type_rules.get("meu_types", []):
                            best_type = item_type.lower() if item_type == 'VERB' else item_type
                        else:
                            best_items = [y for y in meu_entities if y.confidence == best_score]
                            if len(best_items) == 0:
                                return item
                            if len(best_items) == 1:
                                best_item = best_items[0]
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
                                elif ("VERB" in best_types or "verb" in best_types) and (
                                        'det' not in dict(item.properties) and
                                        'subjpass' not in dict(item.properties) and
                                        'nsubj' not in dict(item.properties) and
                                        'obj' not in dict(item.properties) and
                                        not any(e[2]['label'].named_entity in ('compound', 'amod') for e in
                                                G.in_edges(item.id, data=True)) and
                                        ('on' not in case_in_props(dict(item.properties), True)) and
                                        ((
                                                (len(G.in_edges(item.id)) > 0 and any(case_in_props(dict(G.nodes[x]['data'].properties)) for x in [edge[0] for edge in G.in_edges(item.id)])) or
                                                (len(G.in_edges(item.id)) == 0 and is_kernel_in_props(item)) or
                                                (len(G.in_edges(item.id)) == 1 and list(G.in_edges(item.id, data=True))[0][2]['label'].named_entity == "compound" and is_kernel_in_props(dict(G.nodes[list(G.in_edges(item.id, data=True))[0][1]]['data'].properties)))
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
                                elif "NOUN" in best_types or "noun" in best_types:
                                    best_type = "noun"
                                elif "ENTITY" in best_types:
                                    best_type = "ENTITY"
                                else:
                                    best_type = "None"
                
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
                named_entity=item.named_entity,
                properties=item.properties,
                min=item.min,
                max=item.max,
                type=best_type,
                confidence=best_score
            )
        else:
            return item
