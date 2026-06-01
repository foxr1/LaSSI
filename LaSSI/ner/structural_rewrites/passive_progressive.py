__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
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


class PassiveProgressiveRewrite(StructuralRewriteRule):
    """Rewrite `be(AND(X, Y), None)` to `carry_out(?, AND(X, Y))` when the
    AND-subject carries a passive progressive participial sub-clause (e.g.
    "There are inspections and roof repairs being carried out at ...").

    Pattern (all must hold):
      * `kernel.kernel.edgeLabel` lemma is a copula surface form (`be`).
      * `kernel.kernel.source` is a SetOfSingletons with `Grouping.AND`.
      * `kernel.kernel.target` is None or an existential Singleton.
      * One of the AND entities, OR the kernel itself, OR the SetOfSingletons
        carries an `acl` / `SENTENCE` sub-kernel whose edgeLabel is a
        non-copula verb (the past-participle "carried", "done", etc.).

    Rewrite:
      * Replace edgeLabel with the participle's phrasal form ("carry out" if
        the participle Singleton carries a `compound_prt` particle, else just
        the lemma).
      * Replace source with a fresh existential Singleton.
      * Replace target with the original AND SetOfSingletons.
      * Preserve all kernel-level properties (SPACE, TIME, etc.).
    """

    name = "passive_progressive_rewrite"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not (isinstance(kernel, Singleton) and kernel.kernel is not None):
            return None
        rel = kernel.kernel
        edge_label = rel.edgeLabel
        if not isinstance(edge_label, Singleton):
            return None
        if not contains_copula_surface(edge_label.named_entity):
            return None
        if not (isinstance(rel.source, SetOfSingletons) and rel.source.type == Grouping.AND):
            return None
        if not (rel.target is None or
                (isinstance(rel.target, Singleton) and rel.target.type == 'existential')):
            return None

        sub_kernel_singleton = None
        prop_key_consumed = None
        for entity in rel.source.entities:
            if isinstance(entity, Singleton):
                sub = _extract_passive_sub_kernel(entity)
                if sub is not None:
                    sub_kernel_singleton = sub
                    break
        if sub_kernel_singleton is None:
            sub_kernel_singleton = _extract_passive_sub_kernel(kernel)
        if sub_kernel_singleton is None:
            sub_kernel_singleton, prop_key_consumed = _extract_subkernel_from_kernel_props(
                kernel, rel.source
            )
        if sub_kernel_singleton is None:
            return None

        sub_kernel = sub_kernel_singleton.kernel
        if sub_kernel is None or sub_kernel.edgeLabel is None:
            return None
        sub_edge = sub_kernel.edgeLabel
        sub_edge_name = sub_edge.named_entity if isinstance(sub_edge, Singleton) else None
        if not sub_edge_name:
            return None
        if contains_copula_surface(sub_edge_name):
            return None

        return {
            "subjects": rel.source,
            "sub_edge": sub_edge,
            "sub_edge_name": sub_edge_name,
            "original_properties": dict(kernel.properties),
            "prop_key_consumed": prop_key_consumed,
            "sub_kernel_id": id(sub_kernel_singleton),
            "sub_kernel_target": sub_kernel.target,
        }

    def apply(self, kernel, bindings, ctx):
        from LaSSI.ner.node_functions import create_existential_node

        sub_edge = bindings["sub_edge"]
        sub_edge_name = bindings["sub_edge_name"]

        new_edge_name = lemmatise_verb_phrase(
            sub_edge_name, particle=self._extract_particle_from_props(sub_edge)
        )

        new_edge_label = (
            sub_edge.update_name(new_edge_name) if isinstance(sub_edge, Singleton)
            else Singleton(
                id=-1,
                named_entity=new_edge_name,
                properties=frozenset(),
                min=-1,
                max=-1,
                type='verb',
                confidence=1.0,
            )
        )

        new_source = create_existential_node()

        new_kernel_relation = Relationship(
            source=new_source,
            target=bindings["subjects"],
            edgeLabel=new_edge_label,
            isNegated=kernel.kernel.isNegated,
        )

        new_props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }
        prop_key = bindings.get("prop_key_consumed")
        sub_id = bindings.get("sub_kernel_id")
        sub_kernel_target = bindings.get("sub_kernel_target")

        if prop_key is not None and prop_key in new_props:
            existing = new_props[prop_key]
            if isinstance(existing, list):
                filtered = [v for v in existing if id(v) != sub_id]
                if filtered:
                    new_props[prop_key] = filtered
                else:
                    new_props.pop(prop_key, None)
            else:
                if id(existing) == sub_id:
                    new_props.pop(prop_key, None)

        if isinstance(sub_kernel_target, Singleton):
            try:
                from LaSSI.ner.KernelOntologyMatchers import KernelOntologyMatchers
                matcher = ctx.matchers if hasattr(ctx, "matchers") else KernelOntologyMatchers()
                if matcher.is_location_like(sub_kernel_target):
                    space_list = new_props.get("SPACE")
                    if space_list is None:
                        space_list = []
                    elif not isinstance(space_list, list):
                        space_list = [space_list]
                    existing_ids = {
                        getattr(x, 'id', None) for x in space_list if isinstance(x, Singleton)
                    }
                    if sub_kernel_target.id not in existing_ids:
                        space_list.append(sub_kernel_target)
                    new_props["SPACE"] = space_list
            except Exception:
                pass

        return Singleton(
            id=kernel.id,
            named_entity=kernel.named_entity,
            properties=freeze_props(new_props),
            min=kernel.min,
            max=kernel.max,
            type=kernel.type,
            confidence=kernel.confidence,
            kernel=new_kernel_relation,
        )

    @staticmethod
    def _extract_particle_from_props(sub_edge):
        if not isinstance(sub_edge, Singleton):
            return None
        props = dict(sub_edge.properties) if sub_edge.properties is not None else {}
        for key in ('compound_prt', 'prt', 'particle'):
            v = props.get(key)
            if v:
                return v if isinstance(v, str) else (v[0] if isinstance(v, (list, tuple)) and v else None)
        return None




def _extract_passive_sub_kernel(node):
    if not isinstance(node, Singleton):
        return None
    props = dict(node.properties) if node.properties is not None else {}
    for key in ('acl', 'SENTENCE'):
        value = props.get(key)
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for cand in candidates:
            if isinstance(cand, Singleton) and cand.kernel is not None:
                return cand
    return None


def _extract_subkernel_from_kernel_props(kernel, source_set_of_singletons):
    """Look in kernel's top-level properties (CAUSATION/SENTENCE) for an
    embedded sub-kernel whose source matches the AND-source. Returns
    (sub_kernel_singleton, prop_key) so the caller can later remove the
    consumed entry from properties."""
    if not isinstance(kernel, Singleton):
        return None, None
    props = dict(kernel.properties) if kernel.properties is not None else {}
    for key in ('CAUSATION', 'SENTENCE'):
        value = props.get(key)
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for cand in candidates:
            if not (isinstance(cand, Singleton) and cand.kernel is not None):
                continue
            sub_source = cand.kernel.source
            if not _sources_match(sub_source, source_set_of_singletons):
                continue
            sub_edge = cand.kernel.edgeLabel
            sub_edge_name = sub_edge.named_entity if isinstance(sub_edge, Singleton) else None
            if not sub_edge_name or contains_copula_surface(sub_edge_name):
                continue
            return cand, key
    return None, None


def _sources_match(a, b):
    if a is None or b is None:
        return False
    if isinstance(a, SetOfSingletons) and isinstance(b, SetOfSingletons):
        if a.type != b.type:
            return False
        a_ids = sorted(getattr(e, 'id', None) for e in a.entities)
        b_ids = sorted(getattr(e, 'id', None) for e in b.entities)
        return a_ids == b_ids
    if isinstance(a, Singleton) and isinstance(b, Singleton):
        return a.id == b.id
    return False
