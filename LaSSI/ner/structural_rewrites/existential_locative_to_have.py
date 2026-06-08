__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    append_unique_property_value,
    copy_props,
    freeze_props,
    is_copula_surface,
    is_location_like,
    property_values,
    replace_kernel,
)
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class ExistentialLocativeToHaveRule(StructuralRewriteRule):
    """Normalise an existential-there + locative kernel onto a possessive
    ``have(<facility>, <thing>)``.

    Pattern (e.g. transport_002 S1):
        "There is a temporary reduction in spaces available at NPCP while we
         carry out repairs ..."
      → be(?existential, repairs)[(SPACE: NPCP[LOC]),
                                   (SPECIFICATION: reduction[extra:spaces]),
                                   (CAUSATION: damage ...)]
    The grammar made the subordinate clause noun ("repairs") the copula
    target and stranded the real asserted entity (the SPECIFICATION head
    "reduction") in a property bag.

    Rewrite to the possessive shape the *paraphrase* "NPCP has a reduction in
    spaces" already produces (transport_002 S2):
        have(NPCP[LOC], reduction[extra:spaces])[(CAUSATION: damage, repairs)]
      - the locative SPACE node becomes the possessor-subject,
      - the SPECIFICATION head becomes the possessed target,
      - the stranded copula target ("repairs") is demoted to CAUSATION,
      - the SPACE / SPECIFICATION keys are consumed.

    Soundness gate: only fires when the SPACE filler is a genuine location
    (LOC/FAC/GPE). "there is X at <facility>" ≡ "<facility> has X" holds for
    locative possessors; it is NOT applied to animate/abstract framing, so
    "there is a man at the door" is left untouched."""

    name = "existential_locative_to_have"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not (isinstance(kernel, Singleton) and kernel.kernel is not None):
            return None
        rel = kernel.kernel
        edge = rel.edgeLabel
        if not (isinstance(edge, Singleton) and is_copula_surface(edge.named_entity, ctx)):
            return None
        # The copula subject must be the existential "there".
        if not (isinstance(rel.source, Singleton) and rel.source.type == "existential"):
            return None

        props = dict(kernel.properties)

        # Locative possessor: a single location-like SPACE filler.
        space_vals = property_values(props, "SPACE")
        locative = next((v for v in space_vals if is_location_like(v, ctx)), None)
        if locative is None:
            return None

        # Possessed thing: the SPECIFICATION head (a contentful Singleton).
        spec_vals = property_values(props, "SPECIFICATION")
        possessed = next(
            (v for v in spec_vals
             if isinstance(v, Singleton) and (v.type or "").lower() not in ("existential",)),
            None,
        )
        if possessed is None:
            return None

        return {
            "locative": locative,
            "possessed": possessed,
            "stranded_target": rel.target,
        }

    def apply(self, kernel, bindings, ctx):
        locative = bindings["locative"]
        possessed = bindings["possessed"]
        stranded = bindings["stranded_target"]

        props = copy_props(kernel.properties)

        # Consume the keys that have been promoted into the relationship.
        props.pop("SPACE", None)
        props.pop("SPECIFICATION", None)

        # Demote the stranded copula target (e.g. "repairs") to CAUSATION —
        # in "there is X while we carry out Z", Z is circumstantial cause, not
        # the asserted predicate. Skip an existential/None placeholder.
        if isinstance(stranded, Singleton) and (stranded.type or "").lower() != "existential":
            append_unique_property_value(props, "CAUSATION", stranded)

        have_edge = kernel.kernel.edgeLabel.update_name("have")
        new_kernel = replace_kernel(
            kernel, source=locative, target=possessed, edge_label=have_edge,
        )
        return new_kernel.update_node_props(props) if hasattr(new_kernel, "update_node_props") \
            else Singleton(
                id=new_kernel.id, named_entity=new_kernel.named_entity,
                properties=freeze_props(props), min=new_kernel.min, max=new_kernel.max,
                type=new_kernel.type, confidence=new_kernel.confidence, kernel=new_kernel.kernel,
            )
