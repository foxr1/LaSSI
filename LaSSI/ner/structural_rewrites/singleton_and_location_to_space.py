__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    append_unique_property_value,
)
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class SingletonAndLocationToSpaceRule(StructuralRewriteRule):
    """A property bucket keyed by an ``AND`` Grouping carrying *exactly one*
    location-like entity is a structural artifact, not a real conjunction.

    Pattern:
      * ``kernel.properties['AND']`` exists and lists exactly one value.
      * That value is location-like per HOnK (GPE/LOC type, or any of the
        location/facility/route/access-point noun categories).

    Rewrite:
      * Move the entity to the ``SPACE`` bucket (deduped by id/name).
      * Remove the ``AND`` bucket.

    Origin: Stanza sometimes misparses an apposition-like link between two
    locations as ``conj``/``appos``, and ``mergeNodes`` then wraps the
    survivor in a SetOfSingletons that drops out under the ``AND`` key.  No
    HOnK preposition rule fires (the entity carries no ``case`` marker), so
    ``rewrite_properties_logically`` leaves it under ``AND``.  This rule
    fixes that downstream.

    The single-entity guard keeps the rule conservative: a genuine
    coordination of two locations (``Newcastle and Sunderland``) appears as
    a multi-entity AND and is left alone — its entities would still need
    their own preposition context to land in SPACE."""

    name = "singleton_and_location_to_space"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        props = dict(kernel.properties)
        and_value = props.get('AND')
        if and_value is None:
            return None
        and_list = list(and_value) if isinstance(and_value, (list, tuple)) else [and_value]
        if len(and_list) != 1:
            return None
        entity = and_list[0]
        if not ctx.matchers.is_location_like(entity):
            return None
        return {"entity": entity}

    def apply(self, kernel, bindings, ctx):
        entity = bindings["entity"]
        new_props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }
        new_props.pop('AND', None)
        append_unique_property_value(new_props, 'SPACE', entity)
        return kernel.update_node_props(new_props)
