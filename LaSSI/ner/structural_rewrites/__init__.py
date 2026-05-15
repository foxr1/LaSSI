"""Structural rewrite rules over final kernels.

Each rule is a `StructuralRewriteRule` subclass with a `phase` declaring when
in `KernelPostProcessor.run()` it fires. Rules are registered in
`default_registry()` below; `KernelPostProcessor` calls
`registry.apply_phase(kernel, <phase>, ctx)` at each pipeline anchor.

To add a new rule:
  1. Create a new module here (e.g. `passive_subject_promotion.py`).
  2. Subclass `StructuralRewriteRule` — set `name`, `phase`, and implement
     `matches` and `apply`.
  3. Register the rule below in `default_registry()`.
  4. If the rule needs a phase that does not yet exist, add it to
     `PIPELINE_PHASES` in `base.py` AND add the corresponding
     `apply_phase` call in `KernelPostProcessor.run()` at the right point."""

from LaSSI.ner.structural_rewrites.base import (
    PIPELINE_PHASES,
    RewriteContext,
    RuleRegistry,
    StructuralRewriteRule,
    append_unique_property_value,
    property_values,
)
from LaSSI.ner.structural_rewrites.state_cause_access_point import StateCauseAccessPointRule
from LaSSI.ner.structural_rewrites.stative_eventive_swap import StativeEventiveSwapRule
from LaSSI.ner.structural_rewrites.specification_into_space import FoldSpecificationIntoSpaceRule
from LaSSI.ner.structural_rewrites.requirement_simplifier import RequirementClauseSimplifierRule
from LaSSI.ner.structural_rewrites.passive_progressive import PassiveProgressiveRewrite
from LaSSI.ner.structural_rewrites.existential_passive_and_promote import ExistentialPassiveAndPromoteRule
from LaSSI.ner.structural_rewrites.relcl_target_lift import RelclTargetLiftRule
from LaSSI.ner.structural_rewrites.strip_aux_from_edge import StripAuxFromEdgeLabelRule
from LaSSI.ner.structural_rewrites.singleton_and_location_to_space import SingletonAndLocationToSpaceRule
from LaSSI.ner.structural_rewrites.participial_collapse import ParticipialCollapseRule
from LaSSI.ner.structural_rewrites.after_occurrence_context_to_causation import AfterOccurrenceContextToCausationRule


def default_registry() -> RuleRegistry:
    return RuleRegistry([
        StateCauseAccessPointRule(),
        StativeEventiveSwapRule(),
        FoldSpecificationIntoSpaceRule(),
        RelclTargetLiftRule(),
        RequirementClauseSimplifierRule(),
        PassiveProgressiveRewrite(),
        ExistentialPassiveAndPromoteRule(),
        StripAuxFromEdgeLabelRule(),
        SingletonAndLocationToSpaceRule(),
        ParticipialCollapseRule(),
        AfterOccurrenceContextToCausationRule(),
    ])


__all__ = [
    "PIPELINE_PHASES",
    "RewriteContext",
    "RuleRegistry",
    "StructuralRewriteRule",
    "StateCauseAccessPointRule",
    "StativeEventiveSwapRule",
    "FoldSpecificationIntoSpaceRule",
    "RequirementClauseSimplifierRule",
    "PassiveProgressiveRewrite",
    "ExistentialPassiveAndPromoteRule",
    "RelclTargetLiftRule",
    "StripAuxFromEdgeLabelRule",
    "SingletonAndLocationToSpaceRule",
    "ParticipialCollapseRule",
    "AfterOccurrenceContextToCausationRule",
    "append_unique_property_value",
    "default_registry",
    "property_values",
]
