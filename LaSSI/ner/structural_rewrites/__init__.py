"""Structural rewrite rules over final kernels.

Each rule declares a `phase` (one of `PIPELINE_PHASES`) saying when in
`KernelPostProcessor.run()` it fires. `KernelPostProcessor` calls
`registry.apply_phase(kernel, <phase>, ctx)` at each pipeline anchor.

THE PREFERRED WAY TO ADD A RULE IS DATA, NOT PYTHON. Rules live in
`raw_data/structural_rewrites.json` as `premise -> consequence` structures (the
same shape as `logical_analysis.json`'s `derivation_rules`), interpreted by
`DeclarativeStructuralRewriteRule` in `declarative.py`:

  1. Add an object to the `rules` array with a `name`, a `phase`, a `premise`
     map (reusable predicate names -> value lists; ALL must hold, each ORs over
     its values) and an ordered `consequence` list (reusable structured ops).
     Vocabulary classes (StateVerb, LocationLike, OccurrenceVerb, ...) resolve
     via `KernelOntologyMatchers.matches_class` against HOnK / raw_data — put
     new lexical/class membership there, never as a string list in Python.
  2. See `declarative.py` for the full predicate and consequence vocabulary,
     including the structural/graph escape-hatch ops (use these sparingly).

Only drop to a Python `StructuralRewriteRule` subclass when a rewrite truly
cannot be expressed as premise/consequence (heavy graph traversal, multi-pass
property surgery). In that case:
  1. Create a module here, subclass `StructuralRewriteRule`, set `name`/`phase`,
     implement `matches`/`apply`, reusing the shared primitives in `base.py`.
  2. Register it below in `default_registry()`.

If a rule needs a new phase, add it to `PIPELINE_PHASES` in `base.py` AND add
the corresponding `apply_phase` call in `KernelPostProcessor.run()`."""

from LaSSI.ner.structural_rewrites.base import (
    PIPELINE_PHASES,
    RewriteContext,
    RuleRegistry,
    StructuralRewriteRule,
    append_unique_property_value,
    property_values,
)
from LaSSI.ner.structural_rewrites.declarative import (
    DeclarativeStructuralRewriteRule,
    load_declarative_rules,
    load_rules,
)


def default_registry() -> RuleRegistry:
    """The ordered rule registry, built entirely from
    `raw_data/structural_rewrites.json` (declarative rules inline, Python rules
    via `impl` references). There is intentionally no rule list here — to add,
    reorder, enable, or disable a rule, edit the JSON."""
    return RuleRegistry(load_rules())


__all__ = [
    "PIPELINE_PHASES",
    "RewriteContext",
    "RuleRegistry",
    "StructuralRewriteRule",
    "DeclarativeStructuralRewriteRule",
    "append_unique_property_value",
    "default_registry",
    "load_declarative_rules",
    "load_rules",
    "property_values",
]
