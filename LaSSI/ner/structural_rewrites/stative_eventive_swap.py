__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from collections import defaultdict

from LaSSI.ner.string_functions import lemmatize_verb
from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class StativeEventiveSwapRule(StructuralRewriteRule):
    """Stative outer kernel + eventive SENTENCE child → swap.

    Pattern:
      * outer kernel's edgeLabel lemma is a HOnK StateVerb or copula form.
      * outer kernel has a SENTENCE property containing at least one
        Singleton kernel whose edgeLabel lemma is *not* stative.

    Rewrite:
      * Promote the matched eventive child as the new top-level kernel.
      * Merge any non-SENTENCE outer properties (e.g. TIME_STATUS) into the
        promoted kernel's properties, deduplicating by node id.

    The stative clause then surfaces as a property of the eventive verb
    rather than as the primary predicate."""

    name = "stative_eventive_swap"
    phase = "post_nest_dedupe"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        edge_label = kernel.kernel.edgeLabel
        if edge_label is None or edge_label.named_entity is None:
            return None
        verb_lemma = lemmatize_verb(edge_label.named_entity).lower()
        if not ctx.matchers.is_stative_lemma(verb_lemma):
            return None

        outer_props = dict(kernel.properties)
        sentence_value = outer_props.get('SENTENCE')
        if not sentence_value:
            return None
        sentence_list = (
            list(sentence_value) if isinstance(sentence_value, (list, tuple))
            else [sentence_value]
        )

        for candidate in sentence_list:
            if (isinstance(candidate, Singleton) and candidate.kernel is not None and
                    candidate.kernel.edgeLabel is not None):
                inner_lemma = lemmatize_verb(candidate.kernel.edgeLabel.named_entity).lower()
                if inner_lemma and not ctx.matchers.is_stative_lemma(inner_lemma):
                    return {"eventive": candidate, "outer_props": outer_props}
        return None

    def apply(self, kernel, bindings, ctx):
        eventive = bindings["eventive"]
        outer_props = bindings["outer_props"]

        new_props = defaultdict(list)
        for k, v in dict(eventive.properties).items():
            if isinstance(v, (list, tuple)):
                new_props[k] = list(v)
            else:
                new_props[k] = v

        for k, v in outer_props.items():
            if k == 'SENTENCE':
                continue
            items = list(v) if isinstance(v, (list, tuple)) else [v]
            existing = new_props.get(k)
            if existing is None or (isinstance(existing, list) and not existing):
                new_props[k] = items
                continue
            if not isinstance(existing, list):
                new_props[k] = [existing]
                existing = new_props[k]
            existing_ids = {x.id for x in existing if hasattr(x, 'id') and x.id is not None}
            for item in items:
                item_id = getattr(item, 'id', None)
                if item_id is None or item_id not in existing_ids:
                    existing.append(item)
                    if item_id is not None:
                        existing_ids.add(item_id)
        return eventive.update_node_props(new_props)
