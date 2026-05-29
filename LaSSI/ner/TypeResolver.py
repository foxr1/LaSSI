import networkx as nx
import json
import os
import re
from LaSSI.ner.MergeSetOfSingletons import _promote_geo_suffixed_type
from LaSSI.structures import DependencyRoles
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping
from LaSSI.structures.kernels.Sentence import is_kernel_in_props, case_in_props

def _filter_spurious_meus(meu_db_row, honk):
    """Drop three classes of false-positive MEU.

    1. SUTime-style DATE MEUs whose original surface text is a HOnK
       ``TimeNoun`` (e.g. a month name like "March") that is immediately
       followed by a proper-noun token in the same sentence (e.g. "March
       Road" → SUTime emits DATE "2026-03"). Almost always a toponym
       false positive: the time-bearing noun is being used as a name.

       The vocabulary is taken from the ontology (``honk.getTemporalNouns()``)
       so that adding a new time noun to the TTL automatically widens the
       filter; nothing is hard-coded here.

    2. Short all-caps GPE / LOC / FAC MEUs (≤ 2 chars, e.g. "SI", "LP",
       "PE"). HOnK fuzzy-matches these to country codes / GeoNames stubs
       with confidence 1.0, drowning out the noun-typed MEU. The same span
       still keeps its noun-typed MEU entry, so dropping the GPE/LOC ones
       just removes the misclassification.

    3. **Confidence-dominated** GPE / LOC / FAC MEUs: a geo-typed match
       at span S whose confidence is strictly lower than any noun-typed
       match at the same span. Fuzzy matches like "Roadworks → Road
       Forks (LOC, 0.8)" lose to the exact noun match "Roadworks →
       roadworks (noun, 1.0)"; dropping them prevents the geo type from
       being picked up later when the node is folded into a Grouping by
       ``GraphNER_withProperties`` (whose ``most_specific_type`` prefers
       GPE/LOC over noun once the type is exposed).

    4. **Tie-broken-by-noun-morphology** GPE / LOC / FAC MEUs: when the
       geo match and the noun match are equally confident at the same
       span, look at how many *distinct noun monads* HOnK produced for
       the span. A genuinely ambiguous common English word like
       "Possession" has rich morphological derivations in the lexicon
       (possessions, possessive, possessing, possessor, repossession,
       …) so the span gets many noun-typed entries; in that case the
       noun reading is the right one and the LOC/GPE is a gazetteer
       collision on the surface form. A real proper noun like
       "Newcastle" only attracts a handful of fuzzy case-/spelling-
       variants ("newcastle", "New Castle", "snowcastle"), nothing
       like a morphological family, so the GPE/LOC is preserved.
       Threshold ≥ 5 distinct noun monads gates the tie-break.

    Returns a new MeuDB-like object with the spurious entries removed; the
    input is not mutated."""
    if meu_db_row is None or not getattr(meu_db_row, 'multi_entity_unit', None):
        return meu_db_row
    sentence = getattr(meu_db_row, 'first_sentence', None) or ''
    if not sentence:
        return meu_db_row

    # Pull TimeNoun vocabulary from HOnK once, normalised. Empty set if
    # HOnK is unavailable — the DATE-versus-toponym filter just becomes
    # a no-op in that case.
    try:
        time_nouns = {str(n).lower() for n in (honk.getTemporalNouns() if honk else set())}
    except Exception:
        time_nouns = set()

    # Pre-index the max noun confidence and the set of distinct noun
    # monads at each (start, end) span. The confidence-dominance check
    # (rule 3) uses the max; the morphological-richness tie-break
    # (rule 4) uses the cardinality of the monad set.
    max_noun_conf_at_span = {}
    noun_monads_at_span = {}
    for meu in meu_db_row.multi_entity_unit:
        if str(meu.type).lower() == 'noun':
            key = (meu.start_char, meu.end_char)
            if meu.confidence > max_noun_conf_at_span.get(key, -1.0):
                max_noun_conf_at_span[key] = meu.confidence
            monad = (meu.monad or '').strip().lower()
            if monad:
                noun_monads_at_span.setdefault(key, set()).add(monad)

    kept = []
    for meu in meu_db_row.multi_entity_unit:
        meu_type = str(meu.type).upper()
        if meu_type == 'DATE' and time_nouns:
            surface = sentence[meu.start_char:meu.end_char] if 0 <= meu.start_char < meu.end_char <= len(sentence) else ''
            surface_clean = surface.strip().lower()
            if surface_clean in time_nouns:
                # Look ahead in the sentence for a proper-noun follow-up
                # that isn't itself a time noun ("March April" stays).
                tail = sentence[meu.end_char:meu.end_char + 40]
                m = re.match(r"\s+([A-Z][a-zA-Z]+)", tail)
                if m and m.group(1).lower() not in time_nouns:
                    # Likely a toponym like "March Road". Drop the MEU.
                    continue
        if meu_type in {'GPE', 'LOC', 'FAC'}:
            text = (meu.text or '').strip()
            # Gazetteer/fuzzy matches on bare numeric or symbolic spans are
            # noise for geo typing. Keeping them lets a quantified metric
            # inherit a place type from one token and later be rewritten as
            # SPACE.
            if text and not any(ch.isalpha() for ch in text):
                continue
            # ≤ 2-char all-caps surface form: nearly always a chemistry /
            # engineering abbreviation (SI, LP, PE, NaCl-style) that HOnK
            # has mis-matched to a country code or place stub. The same
            # token still has a noun-typed MEU; this only drops the
            # geographic noise.
            if 0 < len(text) <= 2 and text.isupper():
                continue
            span_key = (meu.start_char, meu.end_char)
            best_noun_at_span = max_noun_conf_at_span.get(span_key, -1.0)
            # Rule 3: strictly dominated by a noun match — drop.
            if best_noun_at_span > meu.confidence:
                continue
            # Rule 4: tied with a noun match that has morphological
            # richness (a common English word with many derivations in
            # the lexicon) — drop. Proper nouns like "Newcastle" only
            # produce a few fuzzy variants and stay above the threshold.
            if best_noun_at_span == meu.confidence:
                monads = noun_monads_at_span.get(span_key, set())
                if len(monads) >= 5:
                    continue
        kept.append(meu)
    if len(kept) == len(meu_db_row.multi_entity_unit):
        return meu_db_row
    # Shallow-clone the MeuDB with the filtered list.
    try:
        from copy import copy as _copy
        new_row = _copy(meu_db_row)
        new_row.multi_entity_unit = kept
        return new_row
    except Exception:
        return meu_db_row


class TypeResolver:
    def __init__(self, meu_db_row, honk):
        self.honk = honk
        self.meu_db_row = _filter_spurious_meus(meu_db_row, honk)
        self._load_type_resolution_rules()

    def chunk_role(self):
        """Structural role assigned by :mod:`LaSSI.ner.ChunkProfiler` upstream
        (``ACTION``, ``STATUS``, ``HEADER``, ``REPORT_HEADER``, ``ATTRIBUTE``,
        ``TIME_RANGE``, ``PROSE``, ``CONTEXT``), or ``None`` when the row was
        not chunked (e.g. ``SentenceRepresentation.FullText`` mode). Type
        resolution can consult this hint to bias resolution away from
        unsuitable classes — e.g. an offence-style head noun inside a
        ``STATUS`` chunk should not collapse to ``Location``."""
        return getattr(self.meu_db_row, 'chunk_role', None)

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

    def _is_part_of_multiword_toponym(self, item):
        # True if the node's character span is strictly contained within a
        # longer MEU entry of type GPE/LOC. Catches cases like "Tyne" inside
        # "Newcastle upon Tyne" — HOnK already emits the multi-word entry as
        # GPE 1.0, so the single-token child shouldn't be promoted to a verb
        # by the equal-confidence tiebreak below.
        if self.meu_db_row is None:
            return False
        item_min, item_max = item.min, item.max
        if item_min is None or item_max is None or item_min < 0:
            return False
        item_len = item_max - item_min
        for meu in self.meu_db_row.multi_entity_unit:
            if str(meu.type).upper() not in {'GPE', 'LOC'}:
                continue
            meu_len = meu.end_char - meu.start_char
            if meu_len <= item_len:
                continue
            if meu.start_char <= item_min and item_max <= meu.end_char:
                return True
        return False

    def _is_weather_condition_noun(self, item):
        # True if the node's lemma/surface form matches a HOnK
        # WeatherConditionNoun (rain, precipitation, cloud, wind, snow, …).
        # These nouns commonly attach to a case marker ("of rain", "with
        # rain") which would otherwise let the equal-confidence tiebreak
        # below promote them to verb-typed. Weather *condition* nouns are
        # never the semantic verb of a clause, so block that promotion.
        weather_nouns = self.honk.getWeatherConditionNouns() if self.honk else set()
        if not weather_nouns:
            return False
        candidates = {item.named_entity}
        props = dict(getattr(item, 'properties', frozenset())) if getattr(item, 'properties', None) is not None else {}
        if isinstance(props.get('lemma'), str):
            candidates.add(props['lemma'])
        candidates_lc = {str(c).lower() for c in candidates if c}
        ontology_lc = {str(w).lower() for w in weather_nouns if w}
        return bool(candidates_lc & ontology_lc)

    def mergeMeuNodes(self, G):
        if self.meu_db_row is None:
            return G

        # Also fold no-space MEUs that the tokenizer split into chunks — e.g.
        # ISO 8601 datetimes ("2026-04-14T14:00Z") which CoreNLP breaks at the
        # `T` and dashes. The `len(meu_nodes_ids) > 1` check below is the real
        # gate; single-token MEUs are no-ops regardless.
        multi_word_meus = [
            meu for meu in self.meu_db_row.multi_entity_unit
            if " " in meu.text or re.search(r"\d[-:T]\d", meu.text)
        ]
        multi_word_meus.sort(key=lambda x: (-len(x.text), str(x.type) == "None", -x.confidence))

        for meu in multi_word_meus:
            # For no-space MEUs the tokenizer may have glued a trailing
            # sentence-final punctuation onto the last token (e.g.
            # "Z<dot>" for "Z." after "...T14:00Z."). Accept any node whose
            # start lies inside the MEU even if its end pokes past.
            no_space_special = " " not in meu.text
            meu_nodes_ids = []
            for node_id, node_data in G.nodes(data=True):
                singleton = node_data['data']
                if singleton.min >= meu.start_char and singleton.max <= meu.end_char:
                    meu_nodes_ids.append(node_id)
                elif (no_space_special
                      and meu.start_char <= singleton.min < meu.end_char):
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

            # Multi-token MEU fallback for cases the lexical-merge loop above
            # can't reach because the joining edge isn't one of
            # fixed/mwe/orig/compound/flat. Two patterns:
            #   - ISO 8601 datetimes ("2026-04-14T14:00Z") whose fragments are
            #     wired via `nummod`/`case`.
            #   - Multi-token toponyms ("Newcastle upon Tyne") joined by
            #     `nmod` + `case:upon` — HOnK ships these as a single GPE MEU
            #     so the span is unambiguous.
            # In both cases fold the leftover nodes into the leftmost so the
            # merged node carries the MEU's name and type.
            is_iso_datetime = " " not in meu.text and re.search(r"\d[-:T]\d", meu.text)
            # Gate toponym fallback on high confidence: HOnK ships noisy LOC
            # matches like "Light rain" at ~0.8 from fuzzy lookups that should
            # not fold their tokens into one location.  Also drop MEUs whose
            # first token is a determiner — Stanza occasionally tags spans
            # like "the Percy Street" as LOC 1.0, but the determiner belongs
            # to whatever noun the location modifies (here, "entrance"), not
            # to the location itself.  Folding those would consume the `the`
            # node and break the determiner attachment downstream.
            first_token = meu.text.split(" ", 1)[0].lower() if meu.text else ""
            is_multi_token_toponym = (
                " " in meu.text
                and str(meu.type).upper() in {'GPE', 'LOC'}
                and meu.confidence >= 0.95
                and first_token not in {"the", "a", "an"}
            )
            if len(remaining) > 1 and (is_iso_datetime or is_multi_token_toponym):
                remaining.sort(key=lambda nid: G.nodes[nid]['data'].min)
                head_id = remaining[0]
                head_data = G.nodes[head_id]['data']
                combined_props = dict(head_data.properties)
                for nid in remaining[1:]:
                    if nid not in G:
                        continue
                    for k, v in dict(G.nodes[nid]['data'].properties).items():
                        if k not in combined_props:
                            combined_props[k] = v
                    G = nx.contracted_nodes(G, head_id, nid, self_loops=False)
                    G.nodes[head_id].pop('contraction', None)
                merged_singleton = Singleton(
                    id=head_data.id,
                    named_entity=meu.text,
                    properties=frozenset(combined_props.items()),
                    min=meu.start_char,
                    max=meu.end_char,
                    type=meu.type,
                    confidence=meu.confidence,
                )
                nx.set_node_attributes(G, {head_id: merged_singleton}, 'data')
                remaining = [head_id]

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
                                        'amod' not in dict(item.properties) and
                                        'subjpass' not in dict(item.properties) and
                                        'nsubj' not in dict(item.properties) and
                                        'obj' not in dict(item.properties) and
                                        not any(e[2]['label'].named_entity in ('compound', 'amod') for e in
                                                G.in_edges(item.id, data=True)) and
                                        # HOnK grouped this token into a longer toponym (e.g.
                                        # "Tyne" inside "Newcastle upon Tyne"); don't override
                                        # that with a verb reading from the same equal-confidence
                                        # candidate set.
                                        not self._is_part_of_multiword_toponym(item) and
                                        # HOnK marks this lemma as a weather-condition
                                        # noun (rain, precipitation, cloud, ...) — don't
                                        # let it become a verb just because it picked up
                                        # a case marker like "of rain" / "with rain".
                                        not self._is_weather_condition_noun(item) and
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
