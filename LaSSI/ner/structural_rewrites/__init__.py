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
from LaSSI.ner.structural_rewrites.participial_sub_kernel_preserve import ParticipialSubKernelPreserveRule
from LaSSI.ner.structural_rewrites.after_occurrence_context_to_causation import AfterOccurrenceContextToCausationRule
from LaSSI.ner.structural_rewrites.specification_and_to_target import SpecificationAndToTargetRule
from LaSSI.ner.structural_rewrites.auxiliary_periphrasis_promotion import AuxiliaryPeriphrasisPromotionRule
from LaSSI.ner.structural_rewrites.time_canonicalisation import (
    DropRedundantTimeTargetRule,
    TimeCanonicalisationRule,
)
from LaSSI.ner.structural_rewrites.lifecycle_subject_promotion import LifecycleSubjectPromotionRule
from LaSSI.ner.structural_rewrites.flatten_nested_and import FlattenNestedAndRule
from LaSSI.ner.structural_rewrites.weather_amod_condition import WeatherAmodConditionRule
from LaSSI.ner.structural_rewrites.quantity_compound_merge import QuantityCompoundMergeRule
from LaSSI.ner.structural_rewrites.quantity_redundant_drop import QuantityRedundantDropRule
from LaSSI.ner.structural_rewrites.and_nmod_extra import AndNmodExtraRule
from LaSSI.ner.structural_rewrites.participial_predicate_promotion import ParticipialPredicatePromotionRule
from LaSSI.ner.structural_rewrites.lifecycle_property_promotion import (
    LifecyclePropertyPromotionRule,
    LifecyclePropertyPromotionLateRule,
)


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
        ParticipialSubKernelPreserveRule(),
        ParticipialPredicatePromotionRule(),
        LifecyclePropertyPromotionRule(),
        LifecyclePropertyPromotionLateRule(),
        AfterOccurrenceContextToCausationRule(),
        AuxiliaryPeriphrasisPromotionRule(),
        SpecificationAndToTargetRule(),
        TimeCanonicalisationRule(),
        DropRedundantTimeTargetRule(),
        WeatherAmodConditionRule(),
        QuantityCompoundMergeRule(),
        QuantityRedundantDropRule(),
        AndNmodExtraRule(),
        LifecycleSubjectPromotionRule(),
        FlattenNestedAndRule(),
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
    "ParticipialSubKernelPreserveRule",
    "AfterOccurrenceContextToCausationRule",
    "SpecificationAndToTargetRule",
    "AuxiliaryPeriphrasisPromotionRule",
    "TimeCanonicalisationRule",
    "DropRedundantTimeTargetRule",
    "WeatherAmodConditionRule",
    "QuantityCompoundMergeRule",
    "QuantityRedundantDropRule",
    "AndNmodExtraRule",
    "ParticipialPredicatePromotionRule",
    "LifecyclePropertyPromotionRule",
    "LifecycleSubjectPromotionRule",
    "FlattenNestedAndRule",
    "append_unique_property_value",
    "default_registry",
    "property_values",
]
