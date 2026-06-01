__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from LaSSI.ner.string_functions import lemmatize_verb
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    Relationship,
    SetOfSingletons,
    Singleton,
)


# Pipeline anchor points where structural rewrites may fire. The
# `KernelPostProcessor.run()` pipeline calls `apply_phase(kernel, <phase>)`
# at each of these points; a rule with `phase=<phase>` runs there.
#
# When adding a new phase, also add the corresponding `apply_phase` call in
# `KernelPostProcessor.run()` — the pipeline order is intentionally explicit
# rather than auto-derived.
PIPELINE_PHASES = frozenset({
    "post_hoist_pre_dedupe",   # after empty-kernel hoist, before duplicate removal
    "post_logical_rewrite",   # after rewrite_properties_logically, before nest_of_terms
    "post_nest_dedupe",       # after second dedupe pass, before cleanup_space_property
    "post_cleanup",           # after cleanup_space_property, before filter_invalid_property_keys
})


@dataclass
class RewriteContext:
    node_functions: Any
    services: Any
    matchers: Any
    trace: bool = False
    trace_log: Optional[List[Dict[str, Any]]] = None


class StructuralRewriteRule:
    """A pattern-and-rewrite over a final kernel.

    To add a new rule:
      1. Subclass and set `name` (identifier) and `phase` (one of PIPELINE_PHASES).
      2. Implement `matches(kernel, ctx)` — return a bindings dict if the
         pattern fires, else None.
      3. Implement `apply(kernel, bindings, ctx)` — return the rewritten kernel.
      4. Register the rule in `LaSSI/ner/structural_rewrites/__init__.py`.

    The split exists so `matches` is side-effect-free (cheap to retry / log)
    while `apply` builds the new kernel using bindings that `matches` produced."""

    name: str = ""
    phase: str = ""

    def matches(self, kernel, ctx: RewriteContext) -> Optional[Dict]:
        raise NotImplementedError

    def apply(self, kernel, bindings: Dict, ctx: RewriteContext):
        raise NotImplementedError

    def run(self, kernel, ctx: RewriteContext):
        bindings = self.matches(kernel, ctx)
        if bindings is None:
            return kernel
        return self.apply(kernel, bindings, ctx)


class RuleRegistry:
    """Holds StructuralRewriteRule instances indexed by phase."""

    def __init__(self, rules: Optional[List[StructuralRewriteRule]] = None):
        self._rules: List[StructuralRewriteRule] = []
        for rule in (rules or []):
            self.register(rule)

    def register(self, rule: StructuralRewriteRule):
        if not rule.name:
            raise ValueError(f"Rule {type(rule).__name__} must set `name`")
        if rule.phase not in PIPELINE_PHASES:
            raise ValueError(
                f"Rule {rule.name!r} has unknown phase {rule.phase!r}; "
                f"valid phases are {sorted(PIPELINE_PHASES)}"
            )
        self._rules.append(rule)

    def rules_for_phase(self, phase: str) -> List[StructuralRewriteRule]:
        return [r for r in self._rules if r.phase == phase]

    def apply_phase(self, kernel, phase: str, ctx: RewriteContext):
        for rule in self.rules_for_phase(phase):
            if not getattr(ctx, "trace", False):
                kernel = rule.run(kernel, ctx)
                continue
            bindings = rule.matches(kernel, ctx)
            if ctx.trace_log is None:
                ctx.trace_log = []
            before = _compact_kernel_string(kernel)
            if bindings is None:
                ctx.trace_log.append({
                    "phase": phase,
                    "rule": rule.name,
                    "matched": False,
                })
                continue
            rewritten = rule.apply(kernel, bindings, ctx)
            ctx.trace_log.append({
                "phase": phase,
                "rule": rule.name,
                "matched": True,
                "before": before,
                "after": _compact_kernel_string(rewritten),
            })
            kernel = rewritten
        return kernel


def _compact_kernel_string(kernel, max_len: int = 500) -> str:
    try:
        text = kernel.to_string() if hasattr(kernel, "to_string") else str(kernel)
    except Exception as exc:
        text = f"<to_string error: {exc!r}>"
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def as_list(value) -> list:
    """Return *value* as a list while treating None as empty."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def copy_props(node_or_props) -> dict:
    """Copy a property mapping, normalising tuple values to mutable lists."""
    if isinstance(node_or_props, dict):
        props = node_or_props
    else:
        props = dict(getattr(node_or_props, "properties", {}) or {})
    return {
        key: list(value) if isinstance(value, (list, tuple)) else value
        for key, value in props.items()
    }


def freeze_props(props: dict) -> frozenset:
    """Convert mutable list-valued properties to the hashable stored shape."""
    return frozenset({
        tuple(k) if isinstance(k, list) else k:
        tuple(v) if isinstance(v, list) else v
        for k, v in props.items()
    }.items())


def replace_kernel(kernel, *, source=None, target=None, edge_label=None):
    """Return *kernel* with selected relationship slots replaced."""
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return kernel
    relation = Relationship(
        source=kernel.kernel.source if source is None else source,
        target=kernel.kernel.target if target is None else target,
        edgeLabel=kernel.kernel.edgeLabel if edge_label is None else edge_label,
        isNegated=kernel.kernel.isNegated,
    )
    return Singleton(
        id=kernel.id,
        named_entity=kernel.named_entity,
        properties=kernel.properties,
        min=kernel.min,
        max=kernel.max,
        type=kernel.type,
        confidence=kernel.confidence,
        kernel=relation,
    )


def append_unique_property_value(props: dict, key: str, node):
    """Append `node` to `props[key]` (a list) unless an entry with matching
    `id` or `named_entity` is already present. Used by rules that grow a
    multi-valued property like SPACE / CAUSATION."""
    values = props.get(key, [])
    if values is None:
        values = []
    if not isinstance(values, list):
        values = list(values) if isinstance(values, (list, tuple)) else [values]
    node_id = getattr(node, "id", None)
    node_name = getattr(node, "named_entity", None)
    for existing in values:
        if node_id is not None and getattr(existing, "id", None) == node_id:
            props[key] = values
            return
        if node_name is not None and getattr(existing, "named_entity", None) == node_name:
            props[key] = values
            return
    values.append(node)
    props[key] = values


def append_unique(props: dict, key: str, node):
    append_unique_property_value(props, key, node)


def dedupe_by_id_or_name(values: list) -> list:
    deduped = []
    seen_ids = set()
    seen_names = set()
    for value in values:
        value_id = getattr(value, "id", None)
        value_name = getattr(value, "named_entity", None)
        if value_id is not None and value_id in seen_ids:
            continue
        if value_name is not None and value_name in seen_names:
            continue
        deduped.append(value)
        if value_id is not None:
            seen_ids.add(value_id)
        if value_name is not None:
            seen_names.add(value_name)
    return deduped


def property_values(props: dict, key: str) -> list:
    """Read `props[key]` as a list, regardless of whether it was stored as
    a scalar, list, tuple, or absent."""
    return as_list(props.get(key, []))


def is_date_like(node) -> bool:
    return isinstance(node, Singleton) and str(getattr(node, "type", "")).upper() in {
        "DATE", "TIME", "SUTIME",
    }


def is_location_like(node, ctx: RewriteContext) -> bool:
    matcher = getattr(ctx, "matchers", None)
    if matcher is not None and hasattr(matcher, "is_location_like"):
        return matcher.is_location_like(node)
    if isinstance(node, SetOfSingletons):
        return any(is_location_like(entity, ctx) for entity in node.entities)
    return isinstance(node, Singleton) and getattr(node, "type", None) in {"GPE", "LOC", "FAC"}


def _copula_surface_forms_lower(ctx: Optional[RewriteContext] = None) -> set:
    """Lowercased copula surface forms from HOnK (single source of truth).
    Falls back to the Services singleton when no RewriteContext is supplied."""
    try:
        services = ctx.services if ctx is not None else None
        if services is None:
            from LaSSI.external_services.Services import Services
            services = Services.getInstance()
        copula_forms = services.getHOnK().getCopulaSurfaceForms() or set()
    except Exception:
        copula_forms = set()
    return {str(form).lower() for form in copula_forms}


def is_copula_surface(name, ctx: Optional[RewriteContext] = None) -> bool:
    """True when *every* whitespace token of `name` is a copula surface form
    (e.g. "is", "be", "are"). Use for edge labels that are *purely* copular."""
    if not name:
        return False
    parts = [part for part in str(name).split() if part]
    if not parts:
        return False
    copula_lower = _copula_surface_forms_lower(ctx) or {"be"}
    return all(
        part.lower() in copula_lower or safe_lemmatize_verb(part).lower() in copula_lower
        for part in parts
    )


def contains_copula_surface(name, ctx: Optional[RewriteContext] = None) -> bool:
    """True when *any* whitespace token of `name` is a copula surface form.
    Distinct from `is_copula_surface` (which requires all tokens): use this to
    spot a copula-headed periphrastic edge label such as "being carried out"."""
    if not name:
        return False
    parts = [part for part in str(name).split() if part]
    copula_lower = _copula_surface_forms_lower(ctx)
    if not copula_lower:
        return safe_lemmatize_verb(str(name)).lower() == "be"
    return any(
        part.lower() in copula_lower or safe_lemmatize_verb(part).lower() in copula_lower
        for part in parts
    )


def safe_lemmatize_verb(edge_label_name):
    try:
        return lemmatize_verb(edge_label_name)
    except Exception:
        return edge_label_name


def lemmatise_verb_phrase(name, particle=None):
    """Lemmatise a (possibly phrasal) verb surface form to its canonical edge
    label. The head verb is lemmatised and lowercased; trailing particles are
    kept lowercased ("being carried out" -> "carry out"). For a single-token
    verb, an optional `particle` (e.g. from a `compound_prt` property) is
    appended. Shared by the passive-progressive rewrites."""
    if not name:
        return name
    parts = [p for p in str(name).split() if p]
    if not parts:
        return name
    if len(parts) > 1:
        return " ".join([safe_lemmatize_verb(parts[0]).lower()] + [p.lower() for p in parts[1:]])
    lemma = safe_lemmatize_verb(parts[0]).lower()
    return f"{lemma} {particle}".strip() if particle else lemma


def promote_property_to_target(kernel, property_keys, *, ctx=None, excluded_types=None):
    """Lift values from selected property buckets into an existential target."""
    if not isinstance(kernel, Singleton) or kernel.kernel is None:
        return kernel
    target = kernel.kernel.target
    if not (isinstance(target, Singleton) and target.type == "existential"):
        return kernel
    excluded_types = set(excluded_types or ())
    props = copy_props(kernel)
    collected = []
    consumed = []
    for key in property_keys:
        values = property_values(props, key)
        if not values:
            continue
        for value in values:
            value_type = str(getattr(value, "type", "") or "")
            if value_type in excluded_types:
                continue
            collected.append(value)
        consumed.append(key)
    collected = dedupe_by_id_or_name(collected)
    if not collected:
        return kernel
    for key in consumed:
        props.pop(key, None)
    if len(collected) == 1:
        new_target = collected[0]
    else:
        mins = [getattr(value, "min", 0) for value in collected]
        maxs = [getattr(value, "max", 0) for value in collected]
        new_target = SetOfSingletons(
            id=kernel.id,
            type=Grouping.AND,
            entities=tuple(collected),
            min=min(mins) if mins else 0,
            max=max(maxs) if maxs else 0,
            confidence=1.0,
        )
    return replace_kernel(kernel, target=new_target).update_node_props(props)


def move_property_values(props: dict, source_key: str, target_key: str, predicate):
    """Move matching values from one property bucket to another."""
    source_values = property_values(props, source_key)
    if not source_values:
        return False
    kept = []
    moved = []
    for value in source_values:
        if predicate(value):
            moved.append(value)
        else:
            kept.append(value)
    if not moved:
        return False
    if kept:
        props[source_key] = kept
    else:
        props.pop(source_key, None)
    for value in moved:
        append_unique_property_value(props, target_key, value)
    return True


def walk_kernel(node, transform):
    """Recursively walk kernel structures, applying *transform* bottom-up."""
    if isinstance(node, SetOfSingletons):
        entities = [walk_kernel(entity, transform) for entity in node.entities]
        if any(a is not b for a, b in zip(node.entities, entities)):
            node = node.update_entities(entities)
        return transform(node)
    if isinstance(node, Singleton):
        rewritten = node
        if node.kernel is not None:
            source = walk_kernel(node.kernel.source, transform) if node.kernel.source is not None else None
            target = walk_kernel(node.kernel.target, transform) if node.kernel.target is not None else None
            edge = walk_kernel(node.kernel.edgeLabel, transform) if node.kernel.edgeLabel is not None else None
            if source is not node.kernel.source or target is not node.kernel.target or edge is not node.kernel.edgeLabel:
                rewritten = replace_kernel(rewritten, source=source, target=target, edge_label=edge)
        props = copy_props(rewritten)
        changed = False
        for key, value in list(props.items()):
            if isinstance(value, list):
                new_value = [walk_kernel(item, transform) for item in value]
                if any(a is not b for a, b in zip(value, new_value)):
                    props[key] = new_value
                    changed = True
            else:
                new_value = walk_kernel(value, transform)
                if new_value is not value:
                    props[key] = new_value
                    changed = True
        if changed:
            rewritten = rewritten.update_node_props(props)
        return transform(rewritten)
    return transform(node)
