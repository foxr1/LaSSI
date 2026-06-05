__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

"""The declarative structural-rewrite engine.

A rule is data: a ``premise`` (reusable predicate names -> value lists) and an
ordered ``consequence`` list (reusable structured ops), mirroring
``raw_data/logical_analysis.json``'s ``derivation_rules`` and the matcher in
``LaSSI/ner/SemanticRoleRewriting.py``. This module holds only the engine — the
premise/consequence *dispatch* and the small general ops (Move/Append/
ReplaceSlot/Drop). Class membership lives in ``predicates.py`` (-> HOnK/raw_data
via ``matches_class``); the bespoke structural/graph primitives live in
``consequence_primitives.py``. Config loading lives in ``config.py``.

Authoring a new rule is a data edit in ``structural_rewrites.json`` — never a
new Python function — unless it needs a genuinely new primitive."""

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    append_unique_property_value,
    as_list,
    copy_props,
    property_values,
    replace_kernel,
)
# Re-exported for external callers that historically import these from here
# (e.g. SentenceX, KernelOntologyMatchers, participial_predicate_promotion).
from LaSSI.ner.structural_rewrites.config import (  # noqa: F401
    load_structural_rewrite_config,
    structural_lexical_set,
)
from LaSSI.ner.structural_rewrites.predicates import (
    create_existential,
    edge_label,
    kernel_slot,
    matches_class,
    node_property_value_matches_class,
    wrap_entities,
)
from LaSSI.ner.structural_rewrites import consequence_primitives as prim
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    SetOfSingletons,
    Singleton,
)


def load_rules(path=None):
    """Build the complete ordered rule registry from the JSON config.

    Each entry is either a declarative ``premise -> consequence`` rule
    (``DeclarativeStructuralRewriteRule``) or an ``impl: "module:ClassName"``
    reference to a Python ``StructuralRewriteRule`` subclass for rewrites too
    bespoke to express as data. Registry order = array order (matters within a
    phase), so the JSON array is the single source of truth for the pipeline."""
    from importlib import import_module
    config = load_structural_rewrite_config(path)
    rules = []
    for rule_config in config.get("rules", []):
        if not rule_config.get("enabled", True):
            continue
        impl = rule_config.get("impl")
        if impl:
            module_name, _, class_name = impl.partition(":")
            module = import_module(f"LaSSI.ner.structural_rewrites.{module_name}")
            rules.append(getattr(module, class_name)())
        else:
            rules.append(DeclarativeStructuralRewriteRule(rule_config))
    return rules


def load_declarative_rules(path=None):
    """Only the data-backed (premise/consequence) rules, for keyed access in
    tests and lookups. Skips ``impl`` (Python-backed) entries; use ``load_rules``
    for the full ordered registry."""
    config = load_structural_rewrite_config(path)
    return [
        DeclarativeStructuralRewriteRule(rule_config)
        for rule_config in config.get("rules", [])
        if rule_config.get("enabled", True) and "impl" not in rule_config
    ]


class DeclarativeStructuralRewriteRule(StructuralRewriteRule):
    """A structural rewrite expressed entirely as data.

    Matching follows ``SemanticRoleRewriting.rule_matches``: **all** premises
    must hold, each premise ORs over its value list, and **no** ``not_premise``
    may hold. Premises that select nodes record them into ``bindings`` (e.g.
    ``matched:<KEY>``, plus the always-present ``source``/``target``/``edge``
    slot bindings) for the consequence stage to reference by name."""

    def __init__(self, config):
        self.config = dict(config)
        self.name = self.config["name"]
        self.phase = self.config["phase"]
        self.premise = self.config.get("premise", {})
        self.not_premise = self.config.get("not_premise", {})
        self.consequence = self.config.get("consequence", [])

    def matches(self, kernel, ctx):
        bindings = {}
        if isinstance(kernel, Singleton) and kernel.kernel is not None:
            bindings["source"] = kernel.kernel.source
            bindings["target"] = kernel.kernel.target
            bindings["edge"] = kernel.kernel.edgeLabel
        for name, values in self.premise.items():
            if not _eval_premise(name, as_list(values), kernel, ctx, bindings):
                return None
        for name, values in self.not_premise.items():
            if _eval_premise(name, as_list(values), kernel, ctx, dict(bindings)):
                return None
        return bindings

    def apply(self, kernel, bindings, ctx):
        rewritten = kernel
        for action in self.consequence:
            (name, params), = action.items()
            rewritten = _apply_consequence(name, params or {}, rewritten, bindings, ctx)
        return rewritten


# ---------------------------------------------------------------------------
# Premise vocabulary
#
# A small, reusable predicate set. Predicates that *select* nodes record them
# into `bindings` under canonical keys (`matched:<KEY>`, `prop:<KEY>`, or the
# rule-shared binding names the primitive helpers read). The names below the
# divider are the structural/graph escape hatch, backed by
# `consequence_primitives`; they should be rare.
# ---------------------------------------------------------------------------

_SLOT_FOR_MATCHED_BY = {
    "SourceMatchedBy": "source",
    "TargetMatchedBy": "target",
    "EdgeMatchedBy": "edge",
}


def _eval_premise(name, values, kernel, ctx, bindings):
    if name in _SLOT_FOR_MATCHED_BY:
        node = kernel_slot(kernel, _SLOT_FOR_MATCHED_BY[name])
        return any(matches_class(node, cls, ctx, kernel=kernel) for cls in values)
    if name == "SlotIsType":
        return any(_slot_is_type(kernel, spec) for spec in values)
    if name == "HasProperty":
        return isinstance(kernel, Singleton) and any(v in dict(kernel.properties) for v in values)
    if name == "PropertyValueCount":
        if not isinstance(kernel, Singleton):
            return False
        props = dict(kernel.properties)
        for spec in values:
            key, _, count = spec.partition(":")
            if len(property_values(props, key)) == int(count):
                return True
        return False
    if name == "PropertyMatchedBy":
        return _eval_property_matched_by(values, kernel, ctx, bindings)
    if name == "NodePropertyMatchedBy":
        return _eval_node_property_matched_by(values, kernel, ctx, bindings)
    if name == "PropertyHasPromotableContent":
        result = prim.match_promotable_property_target(kernel, values, ctx)
        if result is None:
            return False
        bindings.update(result)
        return True
    if name == "SlotHasConstructPreposition":
        return _eval_slot_has_construct_preposition(values, kernel, ctx, bindings)
    # ---- structural / graph escape-hatch premises ----
    if name == "EdgeHasLeadingCopulaAux":
        label = edge_label(kernel)
        if not isinstance(label, Singleton):
            return False
        new_name = prim.strip_leading_copula_aux(label.named_entity, ctx)
        if not new_name or new_name == label.named_entity:
            return False
        bindings["new_edge_name"] = new_name
        return True
    if name == "AndTargetRedundantTime":
        return _eval_and_target_redundant_time(values, kernel, ctx, bindings)
    if name == "AndConjunctHasContextProperty":
        return prim.and_conjunct_has_context_property(kernel)
    if name == "SpaceHasTemporalEntries":
        return prim.space_has_temporal_entries(kernel)
    if name == "SourceAndHasReducedRelativeVerb":
        result = prim.match_source_reduced_relative(kernel, ctx)
        if result is None:
            return False
        bindings.update(result)
        return True
    if name == "AdjacentQuantityMergeable":
        for key in values:
            result = prim.match_adjacent_quantity_merge(kernel, key, ctx)
            if result is not None:
                bindings.update(result)
                return True
        return False
    if name == "HasExtraOfClass":
        return any(
            prim.has_head_swap_candidate(kernel, {"extra_class": cls, "extra_property": "extra"}, ctx)
            for cls in values
        )
    if name == "GraphAclRecoverable":
        result = prim.match_graph_acl_recovered_target(kernel, ctx)
        if result is None:
            return False
        bindings.update(result)
        return True
    if name == "GraphNmodExtraCandidates":
        result = prim.match_graph_nmod_extra_candidates(kernel, ctx)
        if result is None:
            return False
        bindings.update(result)
        return True
    if name == "GraphCompoundClassifierHeadCandidates":
        result = prim.match_graph_compound_classifier_head_candidates(kernel, values, ctx)
        if result is None:
            return False
        bindings.update(result)
        return True
    raise ValueError(f"Unknown structural-rewrite premise {name!r}")


def _slot_is_type(kernel, spec):
    slot, _, type_name = spec.partition("=")
    node = kernel_slot(kernel, slot.strip())
    type_name = type_name.strip()
    if type_name == "existential":
        return isinstance(node, Singleton) and node.type == "existential"
    group = getattr(Grouping, type_name, None)
    if group is not None:
        return isinstance(node, SetOfSingletons) and node.type == group
    return isinstance(node, Singleton) and str(getattr(node, "type", "")) == type_name


def _eval_property_matched_by(values, kernel, ctx, bindings):
    if not isinstance(kernel, Singleton):
        return False
    props = dict(kernel.properties)
    matched_any = False
    for spec in values:
        key, _, cls = spec.partition(":")
        all_values = property_values(props, key)
        matched = [v for v in all_values if matches_class(v, cls, ctx, kernel=kernel)]
        if matched:
            bindings[f"matched:{key}"] = matched
            bindings[f"prop:{key}"] = all_values
            matched_any = True
    return matched_any


def _eval_node_property_matched_by(values, kernel, ctx, bindings):
    if not isinstance(kernel, Singleton):
        return False
    props = dict(kernel.properties)
    matched_any = False
    for spec in values:
        left, _, cls = spec.partition(":")
        key, _, node_property = left.partition(".")
        all_values = property_values(props, key)
        matched = [
            v for v in all_values
            if node_property_value_matches_class(v, node_property, cls, ctx)
        ]
        if matched:
            bindings[f"matched:{key}"] = matched
            bindings[f"prop:{key}"] = all_values
            matched_any = True
    return matched_any


def _eval_slot_has_construct_preposition(values, kernel, ctx, bindings):
    for spec in values:
        slot, _, construct_name = spec.partition(":")
        slot = slot.strip()
        construct_name = construct_name.strip().lower()
        if not slot or not construct_name:
            continue
        node = kernel_slot(kernel, slot)
        if not isinstance(node, Singleton):
            continue

        from LaSSI.structures.kernels.SentenceX import get_prepositions

        node_prepositions = {
            str(prep).strip().lower()
            for prep in get_prepositions(node)
            if prep
        }
        if not node_prepositions:
            continue

        matcher = getattr(ctx, "matchers", None)
        if matcher is None or not hasattr(matcher, "prepositions_for_construct"):
            continue
        construct_prepositions = {
            str(prep).strip().lower()
            for prep in matcher.prepositions_for_construct(construct_name)
            if prep
        }
        matched = node_prepositions & construct_prepositions
        if matched:
            bindings[f"preposition:{slot}:{construct_name}"] = tuple(sorted(matched))
            return True
    return False


def _eval_and_target_redundant_time(values, kernel, ctx, bindings):
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return False
    target = kernel.kernel.target
    if not (isinstance(target, SetOfSingletons) and target.type == Grouping.AND):
        return False
    for time_key in values:
        time_names = prim.canonical_time_names(dict(kernel.properties).get(time_key))
        if not time_names:
            continue
        kept = [entity for entity in target.entities if not prim.is_duplicate_time(entity, time_names)]
        if len(kept) != len(target.entities):
            bindings["kept_target_entities"] = kept
            return True
    return False


# ---------------------------------------------------------------------------
# Consequence vocabulary
#
# The general ops (`Move`, `Append`, `ReplaceSlot`, `Drop`, `PromoteToTarget`,
# `Flatten`) cover the bulk of rewrites; the rest delegate to the bespoke
# primitives in `consequence_primitives`. Node references in params resolve via
# `_resolve_ref` against the frozen match-time `bindings`, so consequence
# ordering never has to worry about an earlier op mutating a slot.
# ---------------------------------------------------------------------------


def _apply_consequence(name, params, kernel, bindings, ctx):
    if name == "Move":
        return _consequence_move(kernel, params, bindings, ctx)
    if name == "Append":
        return _consequence_append(kernel, params, bindings)
    if name == "ReplaceSlot":
        return _consequence_replace_slot(kernel, params, bindings, ctx)
    if name == "Drop":
        props = copy_props(kernel)
        props.pop(params["property"], None)
        return kernel.update_node_props(props)
    if name == "PromoteToTarget":
        return prim.apply_promote_values_to_target(kernel, bindings)
    if name == "Flatten":
        groups = {getattr(Grouping, g) for g in params.get("groups", ["AND", "OR"])}
        return prim.flatten_same_group(kernel, groups)
    # ---- structural / graph escape-hatch consequences ----
    if name == "StripLeadingCopula":
        label = edge_label(kernel)
        if not isinstance(label, Singleton):
            return kernel
        return kernel.update_kernel(label.update_name(bindings["new_edge_name"]), "edgeLabel")
    if name == "DropRedundantTime":
        target = kernel.kernel.target
        kept = bindings["kept_target_entities"]
        new_target = kept[0] if len(kept) == 1 else target.update_entities(kept)
        return replace_kernel(kernel, target=new_target)
    if name == "LiftConjunctContext":
        return prim.lift_conjunct_context_properties(kernel)
    if name == "DropTemporalFromSpace":
        return prim.drop_temporal_from_space(kernel)
    if name == "PromoteSourceReducedRelative":
        return prim.apply_promote_source_reduced_relative(kernel, bindings, ctx)
    if name == "SplitModifier":
        changed, new_target = prim.split_modifier_node(kernel.kernel.target, params, ctx)
        return replace_kernel(kernel, target=new_target) if changed else kernel
    if name == "MergeAdjacent":
        return prim.apply_adjacent_quantity_merge(kernel, bindings)
    if name == "SwapHeadWithExtra":
        return prim.rewrite_head_with_extra_class(kernel, params, ctx)
    if name == "GraphAclRecoveredTarget":
        return prim.apply_graph_acl_recovered_target(kernel, bindings, ctx)
    if name == "GraphNmodExtra":
        return prim.apply_graph_nmod_extra_candidates(kernel, bindings)
    if name == "GraphCompoundClassifierHead":
        return prim.apply_graph_compound_classifier_head_candidates(kernel, bindings)
    raise ValueError(f"Unknown structural-rewrite consequence {name!r}")


def _resolve_ref(ref, bindings):
    """Resolve a consequence node reference: a slot name (`source`/`target`/
    `edge`) or `binding:<NAME>`. Returns the match-time-frozen node."""
    if ref in ("source", "target", "edge"):
        return bindings.get(ref)
    if isinstance(ref, str) and ref.startswith("binding:"):
        return bindings.get(ref[len("binding:"):])
    return None


def _consequence_move(kernel, params, bindings, ctx):
    prop = params["property"]
    props = copy_props(kernel)
    all_values = property_values(props, prop)
    if params.get("select", "all") == "matched":
        moved = bindings.get(f"matched:{prop}", [])
    else:
        moved = list(all_values)
    if not moved:
        return kernel
    moved_obj_ids = {id(m) for m in moved}
    moved_node_ids = {getattr(m, "id", None) for m in moved if getattr(m, "id", None) is not None}
    remaining = [
        v for v in all_values
        if id(v) not in moved_obj_ids
        and (getattr(v, "id", None) is None or getattr(v, "id", None) not in moved_node_ids)
    ]
    if remaining:
        props[prop] = remaining
    else:
        props.pop(prop, None)
    if "intoProperty" in params:
        for m in moved:
            append_unique_property_value(props, params["intoProperty"], m)
        return kernel.update_node_props(props)
    if "intoSlot" in params:
        slot = params["intoSlot"]
        new_node = moved[0] if len(moved) == 1 else wrap_entities(kernel.id, moved, Grouping.AND)
        kwarg = "edge_label" if slot == "edge" else slot
        return replace_kernel(kernel, **{kwarg: new_node}).update_node_props(props)
    return kernel.update_node_props(props)


def _consequence_append(kernel, params, bindings):
    node = _resolve_ref(params["from"], bindings)
    if node is None:
        return kernel
    props = copy_props(kernel)
    append_unique_property_value(props, params["property"], node)
    return kernel.update_node_props(props)


def _consequence_replace_slot(kernel, params, bindings, ctx):
    slot = params["slot"]
    with_spec = params["with"]
    value = create_existential(ctx, kernel) if with_spec == "existential" else _resolve_ref(with_spec, bindings)
    kwarg = "edge_label" if slot == "edge" else slot
    return replace_kernel(kernel, **{kwarg: value})
