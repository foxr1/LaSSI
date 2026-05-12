__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Pipeline anchor points where structural rewrites may fire. The
# `KernelPostProcessor.run()` pipeline calls `apply_phase(kernel, <phase>)`
# at each of these points; a rule with `phase=<phase>` runs there.
#
# When adding a new phase, also add the corresponding `apply_phase` call in
# `KernelPostProcessor.run()` — the pipeline order is intentionally explicit
# rather than auto-derived.
PIPELINE_PHASES = frozenset({
    "post_logical_rewrite",   # after rewrite_properties_logically, before nest_of_terms
    "post_nest_dedupe",       # after second dedupe pass, before cleanup_space_property
    "post_cleanup",           # after cleanup_space_property, before filter_invalid_property_keys
})


@dataclass
class RewriteContext:
    node_functions: Any
    services: Any
    matchers: Any


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
            kernel = rule.run(kernel, ctx)
        return kernel


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


def property_values(props: dict, key: str) -> list:
    """Read `props[key]` as a list, regardless of whether it was stored as
    a scalar, list, tuple, or absent."""
    value = props.get(key, [])
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
