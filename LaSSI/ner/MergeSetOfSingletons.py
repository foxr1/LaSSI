from collections import defaultdict
from itertools import repeat

from LaSSI.ner.node_functions import create_props_for_singleton
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons


def score_from_meu(min_value, max_value, node_type, meu_db_row, honk):
    # max_value = max_value
    matched_meus = []

    for meu in meu_db_row.multi_entity_unit:
        # Support both MeuDBEntry (object with attributes) and namedtuple
        if hasattr(meu, 'start_char'):
            start_meu = meu.start_char
            end_meu = meu.end_char
            m_type = meu.type
            m_conf = meu.confidence
        else:
            start_meu, end_meu, m_type, m_conf = meu[:4]

        if min_value == start_meu and end_meu == max_value:
            # TODO: mgu or its opposite...
            if (honk.most_specific_type([node_type, m_type]) == m_type or
                    node_type == m_type or
                    node_type == "None"):
                matched_meus.append(meu)

    if len(matched_meus) == 0:
        return 0, "None"
    else:
        # Support both MeuDBEntry and namedtuple/dict
        def get_conf(x):
            return x.confidence if hasattr(x, 'confidence') else x[3]
        def get_type(x):
            return x.type if hasattr(x, 'type') else x[2]

        max_score = max(map(get_conf, matched_meus))
        return max_score, honk.most_specific_type(
            list(map(get_type, filter(lambda x: get_conf(x) == max_score, matched_meus))))


_GEO_TYPES_FOR_PROMOTION = frozenset({"GPE", "LOC", "FAC", "SPACE"})


def _promote_geo_suffixed_type(merged_node, meu_db_row, honk):
    """Promote `merged_node.type` to a geo type when its name ends in a
    HOnK GeoSuffixNoun and the prefix has a higher-specificity geo match in
    the meuDB.

    Why: the multi-word span (e.g. "Haymarket area") may not match a known
    proper place in the meuDB on its own, so the chosen MEU classifies it as
    `noun`; meanwhile the prefix alone ("Haymarket") matches as `LOC`.
    Without this promotion the merged Singleton keeps the noun type, which
    causes `make_arg` (in rewrite_kernels) to lowercase the named_entity and
    prevents downstream similarity comparisons from recognising it as the
    same place as a plain-prefix reference in another sentence.
    """
    if merged_node is None:
        return merged_node
    current_type = (merged_node.type or "").upper()
    if current_type in _GEO_TYPES_FOR_PROMOTION:
        return merged_node
    name = merged_node.named_entity or ""
    words = name.strip().split()
    if len(words) < 2:
        return merged_node
    suffix_set = honk.getGeoSuffixNouns()
    if not suffix_set or words[-1].lower() not in suffix_set:
        return merged_node
    prefix = " ".join(words[:-1]).strip()
    if not prefix:
        return merged_node

    best_geo_type = None
    best_conf = -1.0
    for meu in meu_db_row.multi_entity_unit:
        if hasattr(meu, "text"):
            meu_text = meu.text
            meu_type = meu.type
            meu_conf = meu.confidence
        else:
            try:
                meu_text = meu[4]
                meu_type = meu[2]
                meu_conf = meu[3]
            except (IndexError, TypeError):
                continue
        if not meu_text or meu_text.strip() != prefix:
            continue
        if (meu_type or "").upper() not in {"GPE", "LOC", "FAC"}:
            continue
        if meu_conf > best_conf:
            best_conf = meu_conf
            best_geo_type = meu_type

    if best_geo_type is None:
        return merged_node
    return merged_node.update_type(best_geo_type)


def _governor_is_locationish(governor, honk):
    """True when the compound governor itself denotes a place — its type is
    already geo (LOC/GPE/FAC) or its surface/lemma is a HOnK location/facility/
    route/access-point noun (station, area, road, …). Used to decide whether a
    named-entity geo MODIFIER may keep the head: "University station" (governor
    `station` is a facility) keeps the geo head, but "Fern Drive roadworks"
    (governor `roadworks` is an event noun) does not."""
    if governor is None:
        return False
    if str(getattr(governor, 'type', '') or '').upper() in {'LOC', 'GPE', 'FAC'}:
        return True
    if honk is None:
        return False
    # Word-level membership: a multi-word governor like "Metro station" is a
    # place when ANY of its tokens is a location/facility noun ("station"), not
    # only when the whole surface matches. Exact-match would wrongly fire the
    # demotion for "Haymarket Metro station" (governor "Metro station").
    names = set()
    full = str(governor.named_entity or '').lower()
    if full:
        names.add(full)
        names.update(full.split())
    props = dict(getattr(governor, 'properties', frozenset()) or {})
    if isinstance(props.get('lemma'), str):
        names.add(props['lemma'].lower())
        names.update(props['lemma'].lower().split())
    loc_sets = set()
    for getter in ('getLocationNouns', 'getFacilityNouns', 'getRouteNouns', 'getAccessPointNouns'):
        try:
            loc_sets |= {str(x).lower() for x in (getattr(honk, getter)() or set())}
        except Exception:
            pass
    return bool(names & loc_sets)


def GraphNER_withProperties(node, is_simplistic_rewriting, meu_db_row, honk, existentials, head_hint=None):
    from LaSSI.structures.internal_graph.EntityRelationship import Singleton
    from LaSSI.utils.allChunks import allChunks
    import numpy

    chosen_entity = None
    norm_confidence = 1
    fusion_properties = dict()

    # ### PATCH: skipping SetOfSingletons that might be within the collection. TODO: these should be included
    # resulting_entities = list(filter(lambda x: not isinstance(x, SetOfSingletons), node.entities))
    #
    # # Sort entities based on word position to keep correct order
    # ## GIACOMO: BUG: SetOfSingletons do not have properties. It might happen that this is provided as part of the GraphNerWithProperties.
    # sorted_entities = sorted(resulting_entities, key=lambda x: float(dict(x.properties)['pos']))

    # Sort entities based on word position to keep correct order
    from LaSSI.structures.internal_graph.EntityRelationship import Grouping
    flattened_entities = []
    has_negation = False
    for entity in node.entities:
        if isinstance(entity, SetOfSingletons):
            flattened_entities.extend(entity.entities)
            if entity.type == Grouping.NOT:
                has_negation = True
        else:
            flattened_entities.append(entity)

    sorted_entities = sorted(flattened_entities, key=lambda x: (x.min_f(), x.pos_f()))

    sorted_entity_names = list(map(getattr, sorted_entities, repeat('named_entity')))
    d = dict(zip(range(len(sorted_entity_names)), sorted_entity_names))  # dictionary for storing the replacing elements
    resolved_d = []

    layered_alternatives = defaultdict(list)
    for x in allChunks(list(d.keys())):
        layered_alternatives[len(x)].append(x)

    for layer in layered_alternatives.values():
        max_score = -1
        alternatives = []
        for x in layer:
            # if all(y in set(map(lambda z: z[0], itertools.chain(map(lambda t: (t, ) if isinstance(t, int) else t, d.keys())))) for y in x):
            exp = " ".join(map(lambda z: sorted_entity_names[z], x))
            min_value = min(map(lambda z: sorted_entities[z].min, x))
            max_value = max(map(lambda z: sorted_entities[z].max, x))

            all_types = [sorted_entities[z].type for z in x]
            non_verb_types = [t for t in all_types if t.lower() != 'verb']
            specific_type = honk.most_specific_type(non_verb_types if non_verb_types else all_types)

            # TODO: Is this okay to do? This is done because min/max no match in MEU, but VERB is important to keep...
            if specific_type == "VERB":
                candidate_meu_score, candidate_meu_type = 1.0, "VERB"
            else:
                candidate_meu_score, candidate_meu_type = score_from_meu(min_value, max_value, specific_type,
                                                                         meu_db_row, honk)
            all_meu_score_prod = numpy.prod(list(map(lambda z: sorted_entities[z].confidence, x)))

            # DECISION LOGIC:
            # 1. Candidate must have score >= product of parts OR it must improve the type specification
            # 2. ALSO, we should NOT merge if the merged type is LESS specific than one of the parts' types
            #    (e.g. merging GPE into LOC).
            
            is_type_compatible = ((specific_type == candidate_meu_type) or 
                                (honk.most_specific_type([specific_type, candidate_meu_type]) == candidate_meu_type))
            
            # Loss of specification check: if any part has a more specific type than the candidate, reject merge
            has_loss_of_spec = any(honk.most_specific_type([candidate_meu_type, t]) == t and t != candidate_meu_type for t in all_types if t != "None")

            if not has_loss_of_spec:
                if (
                        ((candidate_meu_score >= all_meu_score_prod)
                         or
                         ((specific_type != candidate_meu_type) and is_type_compatible))
                        or
                        (len(resolved_d) > 0 and all(candidate_meu_score >= subarray[1] for subarray in resolved_d) and (
                                specific_type != candidate_meu_type) and (all(
                            honk.most_specific_type([subarray[2], candidate_meu_type]) == candidate_meu_type for
                            subarray in resolved_d)))  # Check if current score is greater than previous resolutions
                ):
                    if candidate_meu_score > max_score:
                        alternatives = [(x, all_meu_score_prod, exp, candidate_meu_type)]
                        max_score = candidate_meu_score

        if len(alternatives) > 0:
            alternatives.sort(key=lambda x: x[1])
            d = dict(zip(range(len(sorted_entity_names)), sorted_entity_names))
            candidate_delete = set()
            x = alternatives[-1]
            for k, v in d.items():
                if isinstance(k, int):
                    if k in x[0]:
                        candidate_delete.add(k)
                    elif isinstance(k, tuple):
                        if len(set(x[0]).intersection(set(k))) > 0:
                            candidate_delete.add(k)
            for z in candidate_delete:
                d.pop(z)
            d[x[0]] = x[2]
            resolved_d.append([d, x[1], x[3]])  # [d, confidence_score, type]

    # print(resolved_d)
    # If resolved_d has > 1 elements, there are multiple resolutions with equal confidence score
    if len(resolved_d) > 1:
        # Therefore find the resolution with the most entities
        highest_num_of_entities = 0
        for current_d in resolved_d:
            # TODO: Check length of key instead of entity_name
            for entity_name in list(current_d[0].values()):
                current_num_of_entities = len(entity_name.split())
                if current_num_of_entities > highest_num_of_entities:
                    highest_num_of_entities = current_num_of_entities
                    d = current_d[0]
    elif len(resolved_d) == 1:
        d = resolved_d[0][0]

    # print(d)
    # print("OK")

    extra_name = ""
    extra_min = None
    extra_max = None
    extra_props = None

    extra_names_list = []
    extra_types_collected = []

    # Track which names are current head candidates from the resolution d
    head_names = set(d.values())

    for entity in sorted_entities:
        norm_confidence *= entity.confidence
        fusion_properties = merge_properties(fusion_properties, entity.get_props())

        # HEAD SELECTION LOGIC:
        # Pick the best entity from sorted_entities that is present as a head in resolution d.
        # Use hierarchical type comparison to find the most specific/important entity.
        if entity.named_entity in head_names:
            is_better_type = (chosen_entity is None or
                             (honk.most_specific_type([chosen_entity.type, entity.type]) == entity.type and
                              entity.type != chosen_entity.type))

            # Verbs are always strong candidates if nothing more specific is found
            is_verb_fallback = (entity.type.lower() == "verb" and (chosen_entity is None or chosen_entity.type.lower() != "verb"))

            if is_better_type or is_verb_fallback:
                chosen_entity = entity

    # Refined head correction (compound merges only — `head_hint` is the
    # dependency governor). The type-specificity choice above lets a named-entity
    # geo MODIFIER usurp the head: in "Fern Drive roadworks" the LOC "Fern Drive"
    # outranks the governor "roadworks", inverting the subject. Correct it ONLY
    # when the chosen head is geo AND the governor is NOT itself a place
    # (`_governor_is_locationish`) — so genuinely location-headed compounds like
    # "University station" (governor `station` is a facility noun) are untouched,
    # which is what kept the type-specificity behaviour load-bearing for the gold
    # transport/crime cases. The displaced geo falls to the normal `extra` slot.
    if (head_hint is not None and chosen_entity is not None
            and str(chosen_entity.type or '').upper() in {'LOC', 'GPE', 'FAC'}):
        governor = next(
            (e for e in sorted_entities
             if e.named_entity in head_names
             and (e.id == getattr(head_hint, 'id', None)
                  or e.named_entity == getattr(head_hint, 'named_entity', None))),
            None,
        )
        if (governor is not None and governor is not chosen_entity
                and str(governor.type or '').lower() != 'verb'
                and not _governor_is_locationish(governor, honk)):
            chosen_entity = governor

    # Build extra_names_list from everything that wasn't chosen as head
    for entity in sorted_entities:
        if entity == chosen_entity:
            continue
            
        extra_names_list.append(entity.named_entity)
        extra_min = entity.min if extra_min is None else extra_min if extra_min < entity.min else entity.min
        extra_max =  entity.max if extra_max is None else extra_max if extra_max > entity.max else entity.max

        # Carry the entity's classification so the extra Singleton inherits it
        # (rather than the previous hardcoded 'None' fallback).  We collect
        # every non-chosen entity's type and reduce to the most specific one
        # below, since `extra_name` may be assembled from words across several
        # of them.
        if entity.type and entity.type != "None":
            extra_types_collected.append(entity.type)

        # Only keep "core" properties, as other properties will be added to "chosen entity" instead
        entity_props = {k: v for k, v in entity.get_props().items() if k in {'begin', 'end', 'number', 'pos', 'specification'}}
        extra_props = entity_props if extra_props is None else merge_properties(entity_props, extra_props)
    
    chosen_words = set(chosen_entity.named_entity.split()) if chosen_entity else set()
    unique_words = []
    seen_words = set()
    for entity in sorted_entities:
        for word in entity.named_entity.split():
            if word not in seen_words and word not in chosen_words:
                unique_words.append(word)
                seen_words.add(word)
    
    extra_name = " ".join(unique_words).strip()
    extra_type = honk.most_specific_type(extra_types_collected) if extra_types_collected else "None"

    if norm_confidence > candidate_meu_score:
        candidate_meu_score = norm_confidence

    if is_simplistic_rewriting:
        new_properties = {
            "specification": "none",
            "begin": str(sorted_entities[0].min),
            "end": str(sorted_entities[len(sorted_entities) - 1].max),
            "pos": str(sorted_entities[0].pos_f()),
            "number": "none"
        }

        new_properties = new_properties | fusion_properties

        # In simplistic rewriting, we join all entities into one string
        final_simplistic_name = " ".join(e.named_entity for e in sorted_entities)

        merged_node = Singleton(
            id=node.id,
            named_entity=final_simplistic_name,
            properties=frozenset(new_properties.items()),
            min=sorted_entities[0].min,
            max=sorted_entities[len(sorted_entities) - 1].max,
            type=chosen_entity.type if chosen_entity else "ENTITY",
            confidence=norm_confidence
        )
    elif chosen_entity is None:  # Not simplistic
        sing_type = 'None'
        min_value = sorted_entities[0].min
        max_value = sorted_entities[len(sorted_entities) - 1].max

        if extra_name != '':
            name = extra_name
            extra_name = ''
        else:
            sing_type = 'existential'
            name = "?" + str(existentials.increaseAndGetExistential())

        # New properties for ? object
        new_properties = {
            "specification": "none",
            "begin": str(min_value),
            "end": str(max_value),
            "pos": str(sorted_entities[0].pos_f()),
            "number": "none"
        }
        if extra_name != '':
            new_properties['extra'] = [generate_extra_singleton(extra_name, extra_min, extra_max, extra_props, extra_type)]
        new_properties = merge_properties(fusion_properties, new_properties)

        # Get score and type for newly created Singleton
        concat_candidate_meu_score, concat_candidate_meu_type = (
            score_from_meu(min_value, max_value, sing_type, meu_db_row, honk))

        candidate_meu_type = honk.most_specific_type([concat_candidate_meu_type, candidate_meu_type])

        merged_node = Singleton(
            id=node.id,
            named_entity=name,
            properties=create_props_for_singleton(new_properties),
            min=min_value,
            max=max_value,
            type=candidate_meu_type,
            confidence=candidate_meu_score
        )
    elif chosen_entity is not None:  # Not simplistic and found chosen entity
        # Convert back from frozenset to append new "extra" attribute
        new_properties = merge_properties(fusion_properties, chosen_entity.get_props())
        if extra_name != '':
            new_properties['extra'] = [generate_extra_singleton(extra_name, extra_min, extra_max, extra_props, extra_type)]

        merged_node = Singleton(
            id=node.id,
            named_entity=chosen_entity.named_entity,
            properties=create_props_for_singleton(new_properties),
            min=sorted_entities[0].min,
            max=sorted_entities[len(sorted_entities) - 1].max,
            type=chosen_entity.type,
            confidence=norm_confidence
        )
    else:
        print("Error")
        merged_node = None

    merged_node = _promote_geo_suffixed_type(merged_node, meu_db_row, honk)

    if has_negation and merged_node is not None:
        from LaSSI.structures.kernels.Sentence import is_kernel_in_props
        merged_node = SetOfSingletons(
            id=node.id,
            type=Grouping.NOT,
            entities=tuple([merged_node]),
            min=merged_node.min,
            max=merged_node.max,
            confidence=merged_node.confidence,
            root=is_kernel_in_props(merged_node)
        )

    return merged_node


def generate_extra_singleton(extra_name, extra_min, extra_max, extra_props, extra_type='None'):
    return Singleton(
        id=-1,
        named_entity=extra_name,
        properties=create_props_for_singleton(extra_props),
        min=extra_min,
        max=extra_max,
        type=extra_type,
        confidence=1
    )


def merge_multiway_static_properties(orig_props, new_props):
    d = defaultdict(list)
    for k,v in orig_props.items() if isinstance(orig_props, dict) else orig_props:
        if isinstance(v, tuple) or isinstance(v, list):
            for x in v:
                d[k].append(x)
        else:
            d[k].append(v)
    for k,v in new_props:
        if isinstance(v, tuple) or isinstance(v, list):
            for x in v:
                d[k].append(x)
        else:
            d[k].append(v)
    return dict(d)

def merge_properties(orig_props, new_props, ignore_values=None):
    for key, new_value in new_props.items():
        if ignore_values is None or key not in ignore_values:
            if key in orig_props:
                if key == 'begin' or key == 'pos':
                    orig_props[key] = str(min(float(orig_props[key]), float(new_value)))
                elif key == 'end':
                    orig_props[key] = str(max(float(orig_props[key]), float(new_value)))
                elif key == 'extra':
                    # Merge extra singletons and deduplicate by name (since id=-1 for extra)
                    existing_extras = orig_props[key] if isinstance(orig_props[key], list) else [orig_props[key]]
                    new_extras = new_value if isinstance(new_value, list) else [new_value]
                    
                    # Deduplicate by name and ID
                    combined = existing_extras + new_extras
                    unique_extras = []
                    seen_names = set()
                    for node in combined:
                        if hasattr(node, 'named_entity') and node.named_entity not in seen_names:
                            unique_extras.append(node)
                            seen_names.add(node.named_entity)
                        elif not hasattr(node, 'named_entity') and node not in unique_extras:
                            unique_extras.append(node)
                    orig_props[key] = unique_extras
            else:
                orig_props[key] = new_value
        elif ignore_values is None:
            orig_props[key] = new_value
        elif key not in orig_props:
            orig_props[key] = new_value
    return orig_props
