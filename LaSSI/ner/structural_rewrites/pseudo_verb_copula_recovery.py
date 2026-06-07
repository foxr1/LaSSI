__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__maintainer__ = "Oliver R. Fox"

"""Recover a copula clause whose subject noun was mis-promoted to the root.

A scraped notice that prepends a bare noun-phrase headline to its body — e.g.
"Whitley Bay footbridge The smaller footbridge within the station is out of
use." — parses with the access-point noun "footbridge" promoted to the *root
edge* (a pseudo-verb), an existential subject carrying the real predicate as a
``cop``, and no target::

    footbridge⁽ᵛᵉʳᵇ⁾(?1[cop: out of use], None)
        [SPECIFICATION: station[extra: footbridge[smaller]], SPACE: Whitley Bay]

Rebuild the intended copula ``be(footbridge, out of use)``: take the subject
from the facility/access noun stranded under ``SPECIFICATION`` (the "smaller
footbridge"; falling back to the root label itself), promote the existential's
``cop`` to the target, and fold the locative ``SPECIFICATION`` head ("station")
into the existing ``SPACE`` entity as an ``extra`` — never leaving it as a
distinguishing ``SPECIFICATION`` key, which would otherwise cap the sentence's
similarity against a plainer "footbridge is out of use" paraphrase.

The firing guard is deliberately narrow and ontology-backed (root edge is an
``AccessPointLike`` noun mis-typed as a verb, existential subject with a ``cop``,
empty target), so it cannot disturb ordinary copula or eventive kernels.
"""

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    append_unique_property_value,
    copy_props,
    property_values,
    replace_kernel,
)
from LaSSI.ner.structural_rewrites.predicates import matches_class
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


class PseudoVerbCopulaRecoveryRule(StructuralRewriteRule):
    name = "pseudo_verb_copula_recovery"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        rel = kernel.kernel
        if rel.target is not None:
            return None
        edge = rel.edgeLabel
        source = rel.source
        # Root edge label is an access-point/facility noun used as a pseudo-verb.
        if not isinstance(edge, Singleton):
            return None
        if not matches_class(edge, "AccessPointLike", ctx, kernel=kernel):
            return None
        # Existential subject carrying the real predicate as a `cop`.
        if not (isinstance(source, Singleton) and source.type == "existential"):
            return None
        cop = _first(dict(source.properties).get("cop"))
        if not isinstance(cop, Singleton):
            return None
        return {"edge": edge, "cop": cop}

    def apply(self, kernel, bindings, ctx):
        edge = bindings["edge"]
        cop = bindings["cop"]
        kprops = copy_props(kernel)

        # Pull the real subject out of the locative SPECIFICATION head and fold
        # that head into SPACE as an extra.
        subject = None
        location_heads = []
        kept_spec = []
        for spec in property_values(kprops, "SPECIFICATION"):
            if not isinstance(spec, Singleton):
                kept_spec.append(spec)
                continue
            sp_props = copy_props(spec)
            facility_extras = [
                e for e in property_values(sp_props, "extra")
                if isinstance(e, Singleton)
                and matches_class(e, "AccessPointLike", ctx, kernel=kernel)
            ]
            if facility_extras and subject is None:
                subject = facility_extras[0]
            if facility_extras:
                remaining = [
                    e for e in property_values(sp_props, "extra")
                    if e not in facility_extras
                ]
                if remaining:
                    sp_props["extra"] = remaining
                else:
                    sp_props.pop("extra", None)
                spec = spec.update_node_props(sp_props)
            location_heads.append(spec)

        kprops.pop("SPECIFICATION", None)
        if kept_spec:
            kprops["SPECIFICATION"] = kept_spec

        # Fold locative SPECIFICATION heads into the primary SPACE entity.
        space_values = property_values(kprops, "SPACE")
        if space_values and isinstance(space_values[0], Singleton):
            primary = space_values[0]
            sp_props = copy_props(primary)
            for head in location_heads:
                append_unique_property_value(sp_props, "extra", head)
            kprops["SPACE"] = [primary.update_node_props(sp_props)] + space_values[1:]
        else:
            for head in location_heads:
                append_unique_property_value(kprops, "SPACE", head)

        # Subject fallback: the root label itself, read as a noun.
        if subject is None:
            subject = edge.update_type("noun")
        subj_props = {k: v for k, v in dict(subject.properties).items() if k != "kernel"}
        subject = subject.update_node_props(subj_props)

        be_edge = Singleton(
            id=edge.id,
            named_entity="be",
            properties=frozenset(),
            min=edge.min,
            max=edge.max,
            type="verb",
            confidence=edge.confidence,
        )
        kernel = replace_kernel(kernel, source=subject, target=cop, edge_label=be_edge)
        return kernel.update_node_props(kprops)
