__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    contains_copula_surface,
    freeze_props,
    lemmatise_verb_phrase,
)
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    Relationship,
    SetOfSingletons,
    Singleton,
)


class ExistentialPassiveAndPromoteRule(StructuralRewriteRule):
    """Promote a participial passive sub-verb that landed inside a kernel's
    `AND` property bag back into the kernel itself.

    Pattern: existential copular kernel produced by the grammar for sentences
    like "There are inspections and roof repairs being carried out at Y" —
    after the p4 guard prevents the auxpass acl-target from being lifted, the
    grammar leaves us with:

        be(?existential, None)[(SPACE: Y), (AND: [noun, ..., verb, ..., noun])]

    where the verb singleton is the participle ("carried out") and the
    surrounding entities in `AND` are the conceptual patient list.

    Rewrite: extract the verb from the AND list, lemmatise it (preserving any
    particle), and rebuild the kernel as

        carry_out(?existential, AND(noun, noun))[(SPACE: Y)]

    matching the `do(?, AND(...))[(SPACE:...)]` shape that p3pass already
    produces for the non-existential variant ("Inspections and roof repairs
    are being done at Y")."""

    name = "existential_passive_and_promote"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not (isinstance(kernel, Singleton) and kernel.kernel is not None):
            return None
        rel = kernel.kernel
        if not isinstance(rel.edgeLabel, Singleton):
            return None
        if not contains_copula_surface(rel.edgeLabel.named_entity):
            return None
        if rel.target is not None:
            if not (isinstance(rel.target, Singleton) and rel.target.type == 'existential'):
                return None

        props = dict(kernel.properties)
        and_value = props.get('AND')
        if and_value is None:
            return None
        and_entries = list(and_value) if isinstance(and_value, (list, tuple)) else [and_value]

        verb_entries = [
            (idx, e) for idx, e in enumerate(and_entries)
            if isinstance(e, Singleton) and (e.type or "").lower() == "verb"
            and not contains_copula_surface(e.named_entity)
        ]
        if not verb_entries:
            return None
        if len(verb_entries) > 1:
            return None  # ambiguous — leave alone

        verb_idx, verb_singleton = verb_entries[0]
        remaining = [e for i, e in enumerate(and_entries) if i != verb_idx]
        if len(remaining) < 2:
            return None  # need at least two patients to form an AND target

        return {
            "verb_singleton": verb_singleton,
            "patients": remaining,
        }

    def apply(self, kernel, bindings, ctx):
        verb_singleton = bindings["verb_singleton"]
        patients = bindings["patients"]

        new_verb_name = lemmatise_verb_phrase(verb_singleton.named_entity or "")
        new_edge_label = verb_singleton.update_name(new_verb_name)

        and_target = SetOfSingletons(
            id=-1,
            type=Grouping.AND,
            entities=tuple(patients),
            min=min((getattr(p, 'min', -1) for p in patients), default=-1),
            max=max((getattr(p, 'max', -1) for p in patients), default=-1),
            confidence=min(
                (getattr(p, 'confidence', 1.0) for p in patients), default=1.0
            ),
        )

        new_props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }
        new_props.pop('AND', None)

        new_relation = Relationship(
            source=kernel.kernel.source,
            target=and_target,
            edgeLabel=new_edge_label,
            isNegated=kernel.kernel.isNegated,
        )

        return Singleton(
            id=kernel.id,
            named_entity=kernel.named_entity,
            properties=freeze_props(new_props),
            min=kernel.min,
            max=kernel.max,
            type=kernel.type,
            confidence=kernel.confidence,
            kernel=new_relation,
        )


