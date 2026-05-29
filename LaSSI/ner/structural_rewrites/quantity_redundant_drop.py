__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    SetOfSingletons,
    Singleton,
)


class QuantityRedundantDropRule(StructuralRewriteRule):
    """Drop the ``QUANTITY`` property when the kernel's target is already an
    AND group.

    By the time this rule fires, :class:`QuantityCompoundMergeRule` has
    lifted whatever quantified nouns it could into the AND target. Any
    leftover entries in ``QUANTITY`` (a) come from a NONE-wrapped
    :class:`SetOfSingletons` the upstream merge couldn't iterate, or
    (b) duplicate items already present as siblings of the AND target
    (or as their ``extra`` children).

    In every observed weather notice the ``QUANTITY`` payload either
    structurally mirrors or is fully encompassed by the AND target —
    keeping both is redundant noise that distracts from the kernel's
    real structure. This rule clears the property once the AND target
    is in place; rules that need quantitative info read it off the
    target group directly.

    The rule is intentionally conservative: it only fires when the
    target is the canonical ``AND`` :class:`Grouping`. Kernels whose
    target is a single Singleton (no AND group) keep their ``QUANTITY``
    untouched."""

    name = "quantity_redundant_drop"
    phase = "post_logical_rewrite"

    @staticmethod
    def _has_and_target(kernel):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return False
        tgt = kernel.kernel.target
        return isinstance(tgt, SetOfSingletons) and tgt.type == Grouping.AND

    def matches(self, kernel, ctx):
        if not self._has_and_target(kernel):
            return None
        if 'QUANTITY' not in dict(kernel.properties):
            return None
        return {}

    def apply(self, kernel, bindings, ctx):
        new_props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }
        new_props.pop('QUANTITY', None)
        return kernel.update_node_props(new_props)
