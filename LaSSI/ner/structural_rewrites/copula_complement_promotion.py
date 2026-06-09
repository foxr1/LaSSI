__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__maintainer__ = "Oliver R. Fox"

"""Promote a stranded copula complement (`cop`) into an empty target.

A copula whose predicate adjective is attached to the *subject* as a ``cop``
while the kernel target is empty — e.g. the causal-clause parse
``be(footbridge[cop: out of use], None)[CAUSATION: safety concerns]`` for "the
smaller footbridge is out of use due to safety concerns" — should read
``be(footbridge, out of use)``. Move the subject's ``cop`` to the target, and
drop any ``SPECIFICATION`` value that merely duplicates that same predicate (a
parser artefact of the copula complement being scattered across both slots so
that "out of use" ends up as both a ``cop`` and a ``SPECIFICATION`` head).

The guard is narrow — copula edge, empty target, a ``cop`` on the source — so
well-formed copulas (which already carry a real target) are untouched. This
keeps a causal "out of use" notice converging on the same kernel as its
plainer paraphrases instead of stranding the state on the subject.
"""

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    copy_props,
    first_value,
    is_copula_surface,
    property_values,
    replace_kernel,
)
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class CopulaComplementPromotionRule(StructuralRewriteRule):
    name = "copula_complement_promotion"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        rel = kernel.kernel
        if rel.target is not None:
            return None
        edge = rel.edgeLabel
        source = rel.source
        if not (isinstance(edge, Singleton) and is_copula_surface(edge.named_entity, ctx)):
            return None
        if not isinstance(source, Singleton):
            return None
        cop = first_value(dict(source.properties).get("cop"))
        if not isinstance(cop, Singleton):
            return None
        return {"source": source, "cop": cop}

    def apply(self, kernel, bindings, ctx):
        source = bindings["source"]
        cop = bindings["cop"]
        # Strip the cop off the subject and make it the kernel target.
        src_props = {k: v for k, v in dict(source.properties).items() if k != "cop"}
        kernel = replace_kernel(kernel, source=source.update_node_props(src_props), target=cop)

        # Drop a SPECIFICATION that merely duplicates the promoted predicate.
        kprops = copy_props(kernel)
        cop_name = (cop.named_entity or "").strip().lower()
        spec = property_values(kprops, "SPECIFICATION")
        if spec:
            remaining = [
                s for s in spec
                if not (isinstance(s, Singleton)
                        and (s.named_entity or "").strip().lower() == cop_name)
            ]
            if remaining:
                kprops["SPECIFICATION"] = remaining
            else:
                kprops.pop("SPECIFICATION", None)
        return kernel.update_node_props(kprops)
