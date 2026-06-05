__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

"""Structural / graph rewrite primitives — the declarative engine's escape hatch.

Each function here backs one named premise or consequence in `declarative.py`
that cannot be expressed as pure premise/consequence data because it needs
bespoke node surgery (split a modifier into an AND sibling, merge an adjacent
quantity, swap a head with an `extra`, promote property buckets into a target)
or the dependency graph (`ctx.matchers.G`: acl-relcl target recovery, nmod
`extra` recovery). New rules should prefer the general vocabulary and only reach
here when structure genuinely cannot express the rewrite."""

from LaSSI.ner.structural_rewrites.base import (
    append_unique_property_value,
    as_list,
    copy_props,
    is_copula_surface,
    is_date_like,
    property_values,
    replace_kernel,
    safe_lemmatize_verb,
    walk_kernel,
)
from LaSSI.ner.structural_rewrites.predicates import (
    create_existential,
    extras_in,
    is_content_node,
    is_negated_lifecycle_value,
    is_promotable_target,
    matches_class,
    wrap_entities,
)
from LaSSI.ner.node_functions import create_props_for_singleton
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    SetOfSingletons,
    Singleton,
)
from LaSSI.utils.datetime_canon import canonicalize_datetime_string


# ---- PromoteToTarget ------------------------------------------------------

def match_promotable_property_target(kernel, properties, ctx):
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return None
    target = kernel.kernel.target
    if not (isinstance(target, Singleton) and target.type == "existential"):
        return None

    props = dict(kernel.properties)
    collected = []
    consumed_keys = []
    for key in properties:
        value = props.get(key)
        if value is None:
            continue
        collected.extend(as_list(value))
        consumed_keys.append(key)
    if not collected:
        return None

    unique = []
    seen_ids = set()
    extras_seen = set()
    for item in collected:
        item_id = getattr(item, "id", None)
        if item_id is not None and item_id in seen_ids:
            continue
        if item_id is not None:
            seen_ids.add(item_id)
        extras_seen.update(extras_in(item))
        unique.append(item)
    unique = [
        item for item in unique
        if getattr(item, "id", None) is None or getattr(item, "id", None) not in extras_seen
    ]
    # A negated lifecycle value (e.g. "no suspect") must not be promoted as a
    # *sole* target — it belongs in a status property. But when it co-occurs
    # with genuinely promotable content (e.g. "clear skies and no rain"), the
    # negation is part of the coordinated predicate and must be retained;
    # dropping it silently loses the negation (and any downstream contradiction).
    # So we exclude negated values only when nothing promotable remains.
    promotable = [item for item in unique if is_promotable_target(item, ctx)]
    if not promotable:
        return None
    promotable_ids = {id(item) for item in promotable}
    kept = [
        item for item in unique
        if id(item) in promotable_ids or is_negated_lifecycle_value(item, ctx)
    ]
    if len(kept) == 1:
        promoted = kept[0]
    else:
        promoted = wrap_entities(kernel.id, kept, Grouping.AND)
    return {"property_keys": consumed_keys, "promoted_target": promoted}


def apply_promote_values_to_target(kernel, bindings):
    props = copy_props(kernel)
    for key in bindings["property_keys"]:
        props.pop(key, None)
    return replace_kernel(kernel, target=bindings["promoted_target"]).update_node_props(props)


# ---- SplitModifier --------------------------------------------------------

def split_modifier_node(node, config, ctx):
    if isinstance(node, SetOfSingletons):
        changed = False
        entities = []
        for entity in node.entities:
            entity_changed, entity_new = split_modifier_node(entity, config, ctx)
            changed = changed or entity_changed
            if (
                entity_changed
                and isinstance(entity_new, SetOfSingletons)
                and entity_new.type == node.type == Grouping.AND
            ):
                entities.extend(entity_new.entities)
            else:
                entities.append(entity_new)
        return (True, node.update_entities(entities)) if changed else (False, node)

    if not isinstance(node, Singleton):
        return False, node
    props = dict(node.properties) if node.properties else {}
    modifier_key = config.get("modifier_property", "amod")
    modifiers = as_list(props.get(modifier_key))
    valid_modifiers = [
        value for value in modifiers
        if isinstance(value, str)
        and matches_class(value, config["modifier_class"], ctx)
    ]
    if not valid_modifiers or not matches_class(node, config["head_class"], ctx):
        return False, node

    cleaned_props = dict(props)
    remaining_modifiers = [value for value in modifiers if value not in valid_modifiers]
    if remaining_modifiers:
        cleaned_props[modifier_key] = tuple(remaining_modifiers)
    else:
        cleaned_props.pop(modifier_key, None)

    lemmas = as_list(cleaned_props.get("lemma"))
    modifier_lowers = {value.lower() for value in valid_modifiers}
    lemmas = [lemma for lemma in lemmas if isinstance(lemma, str) and lemma.lower() not in modifier_lowers]
    if lemmas:
        cleaned_props["lemma"] = lemmas[0] if len(lemmas) == 1 else lemmas
    else:
        cleaned_props.pop("lemma", None)

    cleaned = node.update_node_props(cleaned_props)
    entities = [cleaned]
    for value in valid_modifiers:
        entities.append(Singleton(
            id=-(abs(node.id) + 100000 + len(entities)),
            named_entity=value,
            properties=create_props_for_singleton({}),
            min=node.min,
            max=node.max,
            type=config.get("sibling_type", "JJ"),
            confidence=node.confidence,
        ))
    return True, SetOfSingletons(
        id=node.id,
        type=Grouping.AND,
        entities=tuple(entities),
        min=node.min,
        max=node.max,
        confidence=node.confidence,
        root=False,
    )


# ---- MergeAdjacent (quantity) --------------------------------------------

def _pos_of(node):
    if not isinstance(node, Singleton):
        return None
    raw = dict(node.properties).get("pos")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _is_quantified_noun(node, ctx):
    return (
        isinstance(node, Singleton)
        and "nummod" in dict(node.properties)
        and is_content_node(node, ctx)
    )


def and_target(kernel):
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return None
    target = kernel.kernel.target
    return target if isinstance(target, SetOfSingletons) and target.type == Grouping.AND else None


def match_adjacent_quantity_merge(kernel, quantity_key, ctx):
    target = and_target(kernel)
    if target is None:
        return None
    quantity = dict(kernel.properties).get(quantity_key)
    if quantity is None:
        return None
    candidates = [item for item in as_list(quantity) if _is_quantified_noun(item, ctx)]
    if not candidates:
        return None

    and_entities = list(target.entities)
    actions = []
    for q in candidates:
        q_pos = _pos_of(q)
        if q_pos is None:
            continue
        head_idx = None
        for idx, entity in enumerate(and_entities):
            if not isinstance(entity, Singleton) or getattr(entity, "type", None) != "noun":
                continue
            entity_props = dict(entity.properties)
            if "nummod" in entity_props or "measurement" in entity_props:
                continue
            entity_pos = _pos_of(entity)
            if entity_pos is not None and q_pos + 1 == entity_pos:
                head_idx = idx
                break
        actions.append({"q": q, "kind": "merge" if head_idx is not None else "lift", "head_idx": head_idx})
    return {"quantity_actions": actions, "quantity_key": quantity_key} if actions else None


def apply_adjacent_quantity_merge(kernel, bindings):
    target = and_target(kernel)
    and_entities = list(target.entities)
    consumed_q_ids = set()
    for action in bindings["quantity_actions"]:
        q = action["q"]
        consumed_q_ids.add(id(q))
        if action["kind"] == "merge":
            head_idx = action["head_idx"]
            head = and_entities[head_idx]
            head_props = dict(head.properties)
            q_props = dict(q.properties)
            if "nummod" in q_props:
                head_props["nummod"] = q_props["nummod"]
            new_name = f"{q.named_entity} {head.named_entity}".strip()
            merged_node = head.update_name(new_name).update_node_props(head_props)
            and_entities[head_idx] = Singleton(
                id=merged_node.id,
                named_entity=merged_node.named_entity,
                properties=merged_node.properties,
                min=min(head.min, q.min),
                max=max(head.max, q.max),
                type=merged_node.type,
                confidence=merged_node.confidence,
                kernel=merged_node.kernel,
            )
        else:
            and_entities.append(q)

    props = copy_props(kernel)
    quantity_key = bindings["quantity_key"]
    quantity_items = as_list(props.get(quantity_key))
    remaining = [item for item in quantity_items if id(item) not in consumed_q_ids]
    if remaining:
        props[quantity_key] = remaining
    else:
        props.pop(quantity_key, None)
    return replace_kernel(kernel, target=target.update_entities(and_entities)).update_node_props(props)


# ---- SwapHeadWithExtra ----------------------------------------------------

def has_head_swap_candidate(node, config, ctx):
    if isinstance(node, SetOfSingletons):
        return any(has_head_swap_candidate(entity, config, ctx) for entity in node.entities)
    if not isinstance(node, Singleton):
        return False
    if node.kernel is not None:
        if has_head_swap_candidate(node.kernel.source, config, ctx):
            return True
        if has_head_swap_candidate(node.kernel.target, config, ctx):
            return True
    if _is_head_swap_candidate(node, config, ctx):
        return True
    for value in dict(node.properties).values():
        for item in as_list(value):
            if has_head_swap_candidate(item, config, ctx):
                return True
    return False


def _is_head_swap_candidate(node, config, ctx):
    if not isinstance(node, Singleton) or not node.named_entity:
        return False
    target_class = config["extra_class"]
    if matches_class(node, target_class, ctx):
        return False
    for extra in as_list(dict(node.properties).get(config.get("extra_property", "extra"))):
        if isinstance(extra, Singleton) and matches_class(extra, target_class, ctx):
            return True
    return False


def rewrite_head_with_extra_class(node, config, ctx):
    if isinstance(node, SetOfSingletons):
        entities = [rewrite_head_with_extra_class(entity, config, ctx) for entity in node.entities]
        return node.update_entities(entities) if any(a is not b for a, b in zip(node.entities, entities)) else node
    if not isinstance(node, Singleton):
        return node
    rewritten = node
    if node.kernel is not None:
        source = rewrite_head_with_extra_class(node.kernel.source, config, ctx) if node.kernel.source is not None else None
        target = rewrite_head_with_extra_class(node.kernel.target, config, ctx) if node.kernel.target is not None else None
        if source is not node.kernel.source or target is not node.kernel.target:
            rewritten = replace_kernel(rewritten, source=source, target=target)

    props = copy_props(rewritten)
    changed = False
    for key, value in list(props.items()):
        items = as_list(value)
        new_items = [rewrite_head_with_extra_class(item, config, ctx) for item in items]
        if any(a is not b for a, b in zip(items, new_items)):
            props[key] = new_items if isinstance(value, (list, tuple)) else new_items[0]
            changed = True
    if changed:
        rewritten = rewritten.update_node_props(props)
    if _is_head_swap_candidate(rewritten, config, ctx):
        return _swap_head_with_extra(rewritten, config, ctx)
    return rewritten


def _swap_head_with_extra(node, config, ctx):
    props = copy_props(node)
    extra_key = config.get("extra_property", "extra")
    extras = as_list(props.get(extra_key))
    target_idx = None
    for idx, extra in enumerate(extras):
        if isinstance(extra, Singleton) and matches_class(extra, config["extra_class"], ctx):
            target_idx = idx
            break
    if target_idx is None:
        return node
    promoted = extras[target_idx]
    promoted_props = dict(promoted.properties)
    relation_props = {}
    for key, value in promoted_props.items():
        try:
            float(str(key))
        except (TypeError, ValueError):
            continue
        relation_props[key] = value

    metric_props = {key: value for key, value in promoted_props.items() if key not in relation_props}
    extras[target_idx] = promoted.update_name(node.named_entity).update_node_props(metric_props)
    original_extra = dict(node.properties).get(extra_key)
    if isinstance(original_extra, tuple):
        props[extra_key] = tuple(extras)
    elif isinstance(original_extra, list):
        props[extra_key] = extras
    else:
        props[extra_key] = extras[0]
    props.update(relation_props)
    return node.update_node_props(props).update_type(promoted.type).update_name(promoted.named_entity)


# ---- Graph escape hatch (needs ctx.matchers.G) ----------------------------

def _referenced_ids(node):
    ids = set()
    matcher = None
    if isinstance(node, tuple) and len(node) == 2 and hasattr(node[1], "reachable_ids"):
        node, matcher = node
    if matcher is not None and hasattr(matcher, "reachable_ids"):
        return matcher.reachable_ids(node)
    if isinstance(node, Singleton):
        ids.add(node.id)
        for value in dict(node.properties or {}).values():
            ids.update(_referenced_ids(value))
        if node.kernel is not None:
            ids.update(_referenced_ids(node.kernel.source))
            ids.update(_referenced_ids(node.kernel.target))
            ids.update(_referenced_ids(node.kernel.edgeLabel))
    elif isinstance(node, SetOfSingletons):
        ids.add(node.id)
        for entity in node.entities:
            ids.update(_referenced_ids(entity))
    elif isinstance(node, (list, tuple)):
        for item in node:
            ids.update(_referenced_ids(item))
    return ids


def _has_extra(props, candidate):
    for item in as_list(props.get("extra")):
        if isinstance(item, Singleton) and (
            item.id == candidate.id or item.named_entity == candidate.named_entity
        ):
            return True
    return False


def _nmod_children(entity, ctx):
    from LaSSI.structures import DependencyRoles
    from LaSSI.structures.kernels.SentenceX import get_prepositions

    matcher = getattr(ctx, "matchers", None)
    labels = DependencyRoles.nominal_modifier_edges()
    if matcher is not None and hasattr(matcher, "dependency_children"):
        candidates = matcher.dependency_children(entity, labels)
    else:
        graph = getattr(matcher, "G", None)
        candidates = []
        if graph is not None and isinstance(entity, Singleton) and entity.id in graph:
            for _, child_id, data in graph.out_edges(entity.id, data=True):
                label = data.get("label")
                if getattr(label, "named_entity", None) not in labels:
                    continue
                if child_id in graph:
                    child = graph.nodes[child_id].get("data")
                    if child is not None:
                        candidates.append(child)
    out = []
    for child in candidates:
        if not isinstance(child, Singleton):
            continue
        if not is_content_node(child, ctx):
            continue
        if not get_prepositions(child):
            continue
        out.append(child)
    return out


def match_graph_nmod_extra_candidates(kernel, ctx):
    target = and_target(kernel)
    if target is None:
        return None
    target_ids = _referenced_ids((target, getattr(ctx, "matchers", None)))
    actions = []
    for idx, entity in enumerate(target.entities):
        if not isinstance(entity, Singleton):
            continue
        props = dict(entity.properties)
        for child in _nmod_children(entity, ctx):
            if child.id in target_ids or _has_extra(props, child):
                continue
            actions.append((idx, child))
    return {"actions": actions} if actions else None


def apply_graph_nmod_extra_candidates(kernel, bindings):
    target = kernel.kernel.target
    entities = list(target.entities)
    for idx, child in bindings["actions"]:
        entity = entities[idx]
        props = copy_props(entity)
        extras = as_list(props.get("extra"))
        if not _has_extra(props, child):
            extras.append(child)
        props["extra"] = extras
        entities[idx] = entity.update_node_props(props)
    return replace_kernel(kernel, target=target.update_entities(entities))


def _graph_has_dependency_edge(ctx, source, target, labels):
    if not (isinstance(source, Singleton) and isinstance(target, Singleton)):
        return False
    graph = getattr(getattr(ctx, "matchers", None), "G", None)
    if graph is None:
        return False
    labels = {str(label).lower() for label in labels}

    if source.id in graph:
        for _, child_id, data in graph.out_edges(source.id, data=True):
            label = data.get("label")
            if str(getattr(label, "named_entity", "")).lower() not in labels:
                continue
            child = graph.nodes[child_id].get("data") if child_id in graph else None
            if (
                child_id == target.id
                or (isinstance(child, Singleton) and child.id == target.id)
                or (isinstance(child, Singleton) and child.named_entity == target.named_entity)
            ):
                return True

    # Node contraction can leave the relevant Singletons inside a grouped node
    # while preserving edge labels elsewhere. Scan as a conservative fallback.
    for src_id, dst_id, data in graph.edges(data=True):
        label = data.get("label")
        if str(getattr(label, "named_entity", "")).lower() not in labels:
            continue
        src = graph.nodes[src_id].get("data") if src_id in graph else None
        dst = graph.nodes[dst_id].get("data") if dst_id in graph else None
        if _referenced_ids(src) & {source.id} and _referenced_ids(dst) & {target.id}:
            return True
    return False


def match_graph_compound_classifier_head_candidates(kernel, classifier_classes, ctx):
    target = and_target(kernel)
    if target is None:
        return None
    entities = list(target.entities)
    actions = []
    for head_idx, head in enumerate(entities):
        if not isinstance(head, Singleton):
            continue
        if not any(matches_class(head, cls, ctx, kernel=kernel) for cls in classifier_classes):
            continue
        for child_idx, child in enumerate(entities):
            if child_idx == head_idx or not isinstance(child, Singleton):
                continue
            if not is_content_node(child, ctx):
                continue
            if _graph_has_dependency_edge(ctx, head, child, {"compound"}):
                actions.append((head_idx, child_idx))
                break
    return {"compound_classifier_actions": actions} if actions else None


def apply_graph_compound_classifier_head_candidates(kernel, bindings):
    target = kernel.kernel.target
    entities = list(target.entities)
    remove_indices = set()
    for head_idx, child_idx in bindings["compound_classifier_actions"]:
        if head_idx in remove_indices:
            continue
        head = entities[head_idx]
        child = entities[child_idx]
        props = copy_props(child)
        extras = as_list(props.get("extra"))
        if not _has_extra(props, head):
            extras.append(head)
        props["extra"] = extras
        entities[child_idx] = child.update_node_props(props)
        remove_indices.add(head_idx)

    kept = [entity for idx, entity in enumerate(entities) if idx not in remove_indices]
    if len(kept) == 1:
        return replace_kernel(kernel, target=kept[0])
    return replace_kernel(kernel, target=target.update_entities(kept))


def match_graph_acl_recovered_target(kernel, ctx):
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return None
    rel = kernel.kernel
    if rel.target is not None:
        return None
    source = rel.source
    if not (isinstance(source, Singleton) and source.type == "existential"):
        return None
    if not isinstance(rel.edgeLabel, Singleton):
        return None
    graph = getattr(getattr(ctx, "matchers", None), "G", None)
    if graph is None or source.id not in graph.nodes:
        return None
    verb_target = _find_verb_edge_target(graph, source.id)
    if verb_target is None:
        return None
    nmod_target = _find_nmod_via_relcl(graph, verb_target)
    if nmod_target is None:
        return None
    return {"verb_target": verb_target, "nmod_target": nmod_target}


def apply_graph_acl_recovered_target(kernel, bindings, ctx):
    props = copy_props(kernel)
    verb_target = bindings["verb_target"]
    if isinstance(verb_target, Singleton) and matches_class(verb_target, "LocationLike", ctx, kernel=kernel):
        append_unique_property_value(props, "SPACE", verb_target)
    return replace_kernel(kernel, target=bindings["nmod_target"]).update_node_props(props)


def _find_verb_edge_target(graph, source_id):
    for _, target_id, data in graph.out_edges(source_id, data=True):
        label = data.get("label")
        if not isinstance(label, Singleton):
            continue
        if (label.type or "").lower() != "verb":
            continue
        target_node = graph.nodes[target_id].get("data")
        if target_node is not None:
            return target_node
    return None


def _find_nmod_via_relcl(graph, verb_target):
    if not isinstance(verb_target, Singleton) or verb_target.id not in graph.nodes:
        return None
    relcl_targets = []
    for _, child_id, data in graph.out_edges(verb_target.id, data=True):
        label = data.get("label")
        if not isinstance(label, Singleton):
            continue
        if (label.named_entity or "").lower() in {"acl_relcl", "acl"}:
            child = graph.nodes[child_id].get("data")
            if child is not None:
                relcl_targets.append(child)
    for relcl_target in relcl_targets:
        if not isinstance(relcl_target, Singleton) or relcl_target.id not in graph.nodes:
            continue
        for _, nmod_child_id, data in graph.out_edges(relcl_target.id, data=True):
            label = data.get("label")
            if not isinstance(label, Singleton):
                continue
            if (label.named_entity or "").lower() in {"nmod", "obj", "iobj", "dobj"}:
                nmod_child = graph.nodes[nmod_child_id].get("data")
                if isinstance(nmod_child, SetOfSingletons) and nmod_child.type == Grouping.AND:
                    return nmod_child
                if isinstance(nmod_child, Singleton) and "verb" not in (nmod_child.type or "").lower():
                    return nmod_child
    return None


# ---- StripLeadingCopula / Flatten / redundant-time ------------------------

def strip_leading_copula_aux(name, ctx):
    if not name:
        return None
    parts = [part for part in str(name).split() if part]
    if len(parts) < 2:
        return None
    leading = 0
    for part in parts:
        if is_copula_surface(part, ctx):
            leading += 1
            continue
        break
    if leading == 0 or leading >= len(parts):
        return None
    remaining = parts[leading:]
    return " ".join(
        [safe_lemmatize_verb(remaining[0]).lower()] + [part.lower() for part in remaining[1:]]
    )


def flatten_same_group(kernel, groups):
    def transform(node):
        if not (isinstance(node, SetOfSingletons) and node.type in groups):
            return node
        entities = []
        changed = False
        for entity in node.entities:
            if isinstance(entity, SetOfSingletons) and entity.type == node.type:
                entities.extend(entity.entities)
                changed = True
            else:
                entities.append(entity)
        return node.update_entities(entities) if changed else node

    return walk_kernel(kernel, transform)


def canonical_time_names(value):
    names = set()
    for item in as_list(value):
        if not is_date_like(item):
            continue
        canonical = canonicalize_datetime_string(item.named_entity)
        names.add(canonical or item.named_entity)
    return names


def is_duplicate_time(entity, time_names):
    if not is_date_like(entity):
        return False
    canonical = canonicalize_datetime_string(entity.named_entity)
    return (canonical or entity.named_entity) in time_names


# ---- LiftConjunctContext --------------------------------------------------
#
# Circumstantial context (SPACE / TIME) that the parser stranded *inside* an
# AND-target conjunct belongs at the kernel level, where it can be merged into a
# single list alongside any context the kernel already carries. Mirrors the
# single-noun-target behaviour for conjoined subjects (e.g. "criminal damage and
# arson recorded ... Newcastle" must not leave SPACE:Newcastle on `damage`).

_CONJUNCT_CONTEXT_KEYS = ("SPACE", "TIME")


def and_conjunct_has_context_property(kernel):
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return False
    target = kernel.kernel.target
    if not (isinstance(target, SetOfSingletons) and target.type == Grouping.AND):
        return False
    for entity in target.entities:
        if not isinstance(entity, Singleton):
            continue
        entity_props = dict(entity.properties)
        if any(entity_props.get(key) for key in _CONJUNCT_CONTEXT_KEYS):
            return True
    return False


def lift_conjunct_context_properties(kernel):
    target = kernel.kernel.target
    kernel_props = copy_props(kernel)
    new_entities = []
    changed = False
    for entity in target.entities:
        if not isinstance(entity, Singleton):
            new_entities.append(entity)
            continue
        entity_props = copy_props(entity)
        entity_changed = False
        for key in _CONJUNCT_CONTEXT_KEYS:
            values = entity_props.get(key)
            if not values:
                continue
            for value in as_list(values):
                append_unique_property_value(kernel_props, key, value)
            entity_props.pop(key, None)
            entity_changed = True
        if entity_changed:
            new_entities.append(entity.update_node_props(entity_props))
            changed = True
        else:
            new_entities.append(entity)
    if not changed:
        return kernel
    new_target = target.update_entities(new_entities)
    return replace_kernel(kernel, target=new_target).update_node_props(kernel_props)


# ---- DropTemporalFromSpace ------------------------------------------------
#
# A DATE / SUTime node is never spatial; when one leaks into a SPACE list
# (common in result/conclusion-clause sentences where it also lands in TIME),
# strip it from SPACE at every kernel level. TIME already carries it.


def space_has_temporal_entries(kernel):
    found = {"hit": False}

    def _check(node):
        if isinstance(node, Singleton):
            for value in property_values(dict(node.properties), "SPACE"):
                if is_date_like(value):
                    found["hit"] = True
                    break
        return node

    walk_kernel(kernel, _check)
    return found["hit"]


def drop_temporal_from_space(kernel):
    def _strip(node):
        if not isinstance(node, Singleton):
            return node
        space = property_values(dict(node.properties), "SPACE")
        if not space:
            return node
        kept = [value for value in space if not is_date_like(value)]
        if len(kept) == len(space):
            return node
        props = copy_props(node)
        if kept:
            props["SPACE"] = kept
        else:
            props.pop("SPACE", None)
        return node.update_node_props(props)

    return walk_kernel(kernel, _strip)


# ---- PromoteSourceReducedRelative -----------------------------------------
#
# The kernel SOURCE slot is occupied by a (participial) verb. That is always a
# structural anomaly — a subject/source is nominal, never a verb — produced when
# a reduced-relative clause ("the offence [recorded near Z] is awaiting W") gets
# its acl participle hoisted into the subject slot. The participle is the real
# event predicate, so promote it to the kernel edge and demote the surface
# matrix predication. Two shapes are handled, both keyed purely on the verb
# being where an argument should be (NOT on which verb it is):
#   * a single verb leaf inside a subject AND ("X and Y recorded ... involved W")
#     -> promote over the remaining noun conjuncts, matching record(?, AND(nouns));
#   * a bare verb Singleton in the source slot ("[offence] recorded ... is
#     awaiting W") -> promote over an existential; the real object is the subject
#     NP scattered into SPECIFICATION, which `specification_and_to_target` lifts.
#
# Where the demoted matrix complement lands IS a semantic decision driven by the
# matrix verb's class: a HOnK state/lifecycle verb ("is awaiting / pending /
# remaining") makes its complement a status, so it goes to TIME_STATUS with the
# matrix verb folded onto the status entity's `type` (mirroring the verbless
# "Awaiting court outcome" form); any other matrix verb demotes to SPECIFICATION.
# Trigger = structural (verb-agnostic); routing = verb-class.


def _verb_leaf_in_and(node):
    if not (isinstance(node, SetOfSingletons) and node.type == Grouping.AND):
        return None, None
    verbs = [
        e for e in node.entities
        if isinstance(e, Singleton) and str(getattr(e, "type", "")).lower() == "verb"
    ]
    nouns = [e for e in node.entities if e not in verbs]
    if len(verbs) != 1 or not nouns:
        return None, None
    return verbs[0], nouns


def _source_verb_leaf(node):
    """Return (verb_leaf, remaining_nouns) when the source slot holds a verb.

    A bare verb Singleton yields no remaining nouns; a verb leaf inside a
    subject AND yields the other (noun) conjuncts. Returns (None, None) when
    the source is a legitimate nominal argument.
    """
    if isinstance(node, Singleton) and str(getattr(node, "type", "")).lower() == "verb":
        return node, []
    return _verb_leaf_in_and(node)


def _status_verb_surface(edge_label, ctx):
    """The matrix edge's surface form if it is a HOnK state/lifecycle verb,
    else None. Surface (not lemma) is kept so the folded status type matches the
    verbless participial form (e.g. "awaiting")."""
    if not isinstance(edge_label, Singleton):
        return None
    services = getattr(ctx, "services", None)
    try:
        state_verbs = {str(v).lower() for v in (services.getHOnK().getStateVerbs() or set())}
    except Exception:
        return None
    if not state_verbs:
        return None
    name = edge_label.named_entity or ""
    if name.lower() in state_verbs or safe_lemmatize_verb(name).lower() in state_verbs:
        return name
    return None


def _fold_status_type(status_node, verb_surface):
    props = copy_props(status_node)
    existing = props.get("type")
    merged = list(existing) if isinstance(existing, (list, tuple)) else (
        [existing] if existing is not None else [])
    if verb_surface is not None and verb_surface not in merged:
        merged.append(verb_surface)
    if merged:
        props["type"] = tuple(merged) if len(merged) > 1 else merged[0]
    return status_node.update_node_props(props)


def _event_classifier_head(node, ctx):
    """If `node`'s surface ends in HOnK EventClassifierHeadNoun(s) ("possession
    offence", "... incidents"), return it with those trailing classifiers
    stripped ("possession"); else None. Used to recover the real head of a
    "<modifier> <head> <classifier>" compound."""
    if not isinstance(node, Singleton) or not node.named_entity:
        return None
    services = getattr(ctx, "services", None)
    try:
        classifiers = {str(c).lower() for c in (services.getHOnK().getEventClassifierHeadNouns() or set())}
    except Exception:
        return None
    if not classifiers:
        return None
    words = node.named_entity.split()
    n = len(words)
    while len(words) > 1 and words[-1].lower() in classifiers:
        words = words[:-1]
    return " ".join(words) if len(words) < n else None


def _recover_subject_object(props, ctx):
    """Pop the scattered subject NP from SPECIFICATION and reanalyse a
    "<modifier> <head> <classifier>" compound into <head>[extra:<modifier>].

    "A weapons possession offence ..." reaches the kernel as the content node
    ``weapons[extra:[possession offence, awaiting]]`` — modifier as head, the
    real head buried under a classifier ("offence"), plus a stray status-verb
    extra. This rebuilds ``possession[extra:weapons]`` (matching the canonical
    "Possession of weapons" form), dropping verb extras. Returns
    ``(object_node_or_None, status_verb_surface_or_None, props)``; ``props`` has
    the consumed SPECIFICATION entry removed."""
    spec = as_list(props.get("SPECIFICATION"))
    content = next((s for s in spec if is_content_node(s, ctx)), None)
    if content is None:
        return None, None, props
    remaining = [s for s in spec if s is not content]
    if remaining:
        props["SPECIFICATION"] = remaining
    else:
        props.pop("SPECIFICATION", None)

    extras = as_list(dict(content.properties).get("extra"))
    head_extra = None
    status_surface = None
    other_extras = []
    for e in extras:
        if isinstance(e, Singleton) and str(getattr(e, "type", "")).lower() == "verb":
            status_surface = status_surface or (e.named_entity or None)
            continue
        if head_extra is None and _event_classifier_head(e, ctx) is not None:
            head_extra = e
            continue
        other_extras.append(e)

    if head_extra is None:
        # No compound head to reanalyse — just strip the stray verb extras.
        node_props = copy_props(content)
        if other_extras:
            node_props["extra"] = other_extras
        else:
            node_props.pop("extra", None)
        return content.update_node_props(node_props), status_surface, props

    inner_props = copy_props(content)
    inner_props.pop("extra", None)
    inner = content.update_node_props(inner_props)  # the bare modifier ("weapons")
    head_props = copy_props(head_extra)
    head_props["extra"] = [inner] + other_extras
    obj = head_extra.update_name(_event_classifier_head(head_extra, ctx)).update_node_props(head_props)
    return obj, status_surface, props


def match_source_reduced_relative(kernel, ctx):
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return None
    rel = kernel.kernel
    if not isinstance(rel.edgeLabel, Singleton):
        return None
    verb_leaf, nouns = _source_verb_leaf(rel.source)
    if verb_leaf is None:
        return None
    # Bare-verb source ("[offence] recorded ... is awaiting W"): promoting the
    # reduced relative to root is only semantically valid when the matrix verb
    # is *light/status* — then the participle is the real event and the matrix
    # predication is mere status. With a contentful matrix verb (e.g. "...
    # involved a knife") the matrix predication is primary and must be kept, so
    # we leave that shape untouched. (The verb-leaf-in-AND shape keeps its
    # original verb-agnostic behaviour and supplies its own noun conjuncts.)
    if not nouns:
        if _status_verb_surface(rel.edgeLabel, ctx) is None:
            return None
        props = dict(kernel.properties)
        has_content = any(
            any(is_content_node(v, ctx) for v in as_list(props.get(k)))
            for k in ("SPECIFICATION", "TOPIC", "TOGETHERNESS")
        )
        if not has_content:
            return None
    return {
        "verb_leaf": verb_leaf,
        "remaining_nouns": nouns,
        "demote_node": rel.target,
    }


def apply_promote_source_reduced_relative(kernel, bindings, ctx):
    rel = kernel.kernel
    verb_leaf = bindings["verb_leaf"]
    nouns = bindings["remaining_nouns"]
    demote_node = bindings["demote_node"]

    new_edge = Singleton(
        id=verb_leaf.id,
        named_entity=safe_lemmatize_verb(verb_leaf.named_entity),
        properties=verb_leaf.properties,
        min=verb_leaf.min,
        max=verb_leaf.max,
        type="verb",
        confidence=verb_leaf.confidence,
        kernel=None,
    )
    props = copy_props(kernel)

    if nouns:
        # Verb-leaf-in-AND: the object nouns sit alongside the verb in the
        # source; the matrix complement demotes to SPECIFICATION (verb-agnostic).
        new_target = nouns[0] if len(nouns) == 1 else wrap_entities(
            getattr(rel.source, "id", kernel.id), nouns, Grouping.AND)
        if isinstance(demote_node, Singleton) and str(getattr(demote_node, "type", "")) != "existential":
            append_unique_property_value(props, "SPECIFICATION", demote_node)
    else:
        # Bare-verb status case: recover and reanalyse the scattered subject NP
        # as the target, and route the matrix complement to TIME_STATUS with the
        # status verb folded onto its type (status surface preferred from the
        # recovered NP's own verb extra, e.g. "awaiting").
        obj, status_surface, props = _recover_subject_object(props, ctx)
        new_target = obj if obj is not None else create_existential(ctx, kernel)
        props.pop("verb", None)  # drop the leftover copula marker ("verb:is")
        if isinstance(demote_node, Singleton) and str(getattr(demote_node, "type", "")) != "existential":
            surface = status_surface or _status_verb_surface(rel.edgeLabel, ctx)
            append_unique_property_value(
                props, "TIME_STATUS", _fold_status_type(demote_node, surface))

    rewritten = replace_kernel(
        kernel,
        source=create_existential(ctx, kernel),
        target=new_target,
        edge_label=new_edge,
    )
    return rewritten.update_node_props(props)


