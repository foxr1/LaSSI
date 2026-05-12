__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    append_unique_property_value,
    property_values,
)
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class StateCauseAccessPointRule(StructuralRewriteRule):
    """State-edge kernel + causal source + location target + SPACE access-point.

    Pattern (all must hold):
      * edgeLabel matches a HOnK StateVerb (or copula form).
      * source is a non-existential Singleton matching a HOnK CausativeVerb
        or StateNoun.
      * target is location-like (GPE/LOC/location-noun/facility/route).
      * SPACE property contains at least one entry that is an access-point
        noun.

    Rewrite:
      * Promote the access-point entry to be the kernel target.
      * Move the original target into SPACE.
      * Move the original source into CAUSATION.
      * Replace source with an existential.

    Example: a kernel like `be(open[CAUSATION:fire], University station)`
    with `SPACE:[public, station]` is rewritten to
    `be(?, public)[CAUSATION:fire, SPACE:[station]]`."""

    name = "state_cause_access_point_swap"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not (
                isinstance(kernel, Singleton) and
                kernel.kernel is not None and
                ctx.matchers.is_state_edge(kernel.kernel.edgeLabel) and
                isinstance(kernel.kernel.source, Singleton) and
                kernel.kernel.source.type != "existential" and
                ctx.matchers.is_causal_node(kernel, kernel.kernel.source)
        ):
            return None

        old_target = kernel.kernel.target
        if not ctx.matchers.is_location_like(old_target):
            return None

        space_values = property_values(dict(kernel.properties), "SPACE")
        access_space = next(
            (c for c in space_values if ctx.matchers.has_access_point(c)),
            None,
        )
        if access_space is None:
            return None

        return {
            "old_source": kernel.kernel.source,
            "old_target": old_target,
            "access_space": access_space,
            "space_values": space_values,
        }

    def apply(self, kernel, bindings, ctx):
        from LaSSI.ner.node_functions import create_existential_node

        access_space = bindings["access_space"]
        old_target = bindings["old_target"]
        old_source = bindings["old_source"]

        props = {
            key: list(value) if isinstance(value, (list, tuple)) else value
            for key, value in dict(kernel.properties).items()
        }
        props["SPACE"] = [
            v for v in bindings["space_values"]
            if getattr(v, "id", None) != getattr(access_space, "id", None)
        ]
        append_unique_property_value(props, "SPACE", old_target)
        append_unique_property_value(props, "CAUSATION", old_source)

        kernel = kernel.update_kernel(access_space, "target")
        kernel = kernel.update_kernel(create_existential_node(), "source")
        return kernel.update_node_props(props)
