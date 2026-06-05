__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__maintainer__ = "Oliver R. Fox"

"""Lift a geographic compound-modifier demoted to ``extra`` up to a kernel-level
``SPACE`` property.

Two shapes are handled:

* **Kernel argument extra** — "The Fern Drive roadworks involve X" merges
  ``Fern Drive`` (a LOC) into the head ``roadworks`` as an ``extra`` (there is
  no spatial preposition to route it to SPACE the way "roadworks ON Fern Drive"
  would). Semantically the geo modifier is the event location, so lift it.

* **Property-value extra (dedupe-only)** — a spatial PP that got mis-attached
  to a property value, e.g. ``SPECIFICATION: pipe replacement[(extra: Fern
  Drive[(4:on)])]`` in "... involving pipe replacement [on Fern Drive]", *when
  that same place is already a top-level SPACE entity on the kernel*. Such a
  buried geo is a redundant duplicate of the event SPACE, so it is simply
  removed. This is **purely subtractive** — it never adds to or changes SPACE —
  so it cannot introduce a spatial mismatch. (An earlier version that *lifted*
  property-value geos into SPACE regressed crime_003: at the post-logical phase
  a geo nested under another SPACE value got promoted to a separate SPACE
  entity, creating a false location mismatch with its paraphrase.)

Only explicit ``{LOC, GPE, FAC}`` extras are touched — deliberately narrower
than the matcher's ``is_location_like`` (which classifies e.g. "gas" as
location-like and would wrongly strip a real object's content). The kernel-
argument lift is gold-safe (no gold case has a geo extra on source/target);
the property-value step only removes a duplicate of a place already in SPACE.
"""

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    append_unique_property_value,
    copy_props,
    property_values,
    replace_kernel,
)
from LaSSI.structures.internal_graph.EntityRelationship import Singleton

# Strict named-entity geo types (narrower than matcher.is_location_like).
_GEO_TYPES = {"LOC", "GPE", "FAC"}


def _geo_extras(node):
    if not isinstance(node, Singleton):
        return []
    return [
        e for e in property_values(dict(node.properties), "extra")
        if isinstance(e, Singleton)
        and str(getattr(e, "type", "") or "").upper() in _GEO_TYPES
    ]


def _strip_geo_extras(node, geo):
    geo_ids = {id(g) for g in geo}
    props = copy_props(node)
    remaining = [e for e in property_values(props, "extra") if id(e) not in geo_ids]
    if remaining:
        props["extra"] = remaining
    else:
        props.pop("extra", None)
    return node.update_node_props(props)


class GeoExtraToSpaceRule(StructuralRewriteRule):
    name = "geo_extra_to_space"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        slot_moves = []
        for slot in ("source", "target"):
            node = getattr(kernel.kernel, slot)
            geo = _geo_extras(node)
            if geo:
                slot_moves.append((slot, node, geo))
        prop_moves = []
        kprops = dict(kernel.properties)
        for key in kprops:
            if key == "SPACE":
                continue
            for value in property_values(kprops, key):
                geo = _geo_extras(value)
                if geo:
                    prop_moves.append((key, value, geo))
        if not slot_moves and not prop_moves:
            return None
        return {"slot_moves": slot_moves, "prop_moves": prop_moves}

    def apply(self, kernel, bindings, ctx):
        kprops = copy_props(kernel)
        # 1. Lift geo extras off the kernel source/target into SPACE.
        for slot, node, geo in bindings["slot_moves"]:
            kernel = replace_kernel(kernel, **{slot: _strip_geo_extras(node, geo)})
            for g in geo:
                append_unique_property_value(kprops, "SPACE", g)
        # 2. Property-value extras: dedupe-only — drop a buried geo iff that
        #    same place is already a top-level SPACE entity (never add to SPACE).
        space_names = {
            (v.named_entity or "").strip().lower()
            for v in property_values(kprops, "SPACE")
            if isinstance(v, Singleton) and v.named_entity
        }
        for key, value, geo in bindings["prop_moves"]:
            removable = [
                g for g in geo
                if (g.named_entity or "").strip().lower() in space_names
            ]
            if not removable:
                continue
            kprops[key] = [
                _strip_geo_extras(value, removable) if x is value else x
                for x in property_values(kprops, key)
            ]
        return kernel.update_node_props(kprops)
