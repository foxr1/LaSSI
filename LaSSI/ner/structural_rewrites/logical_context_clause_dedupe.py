__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule, copy_props
from LaSSI.structures.internal_graph.EntityRelationship import Singleton

# Only CAUSATION is targeted: a subordinate clause ("while repairs are carried
# out following intentional damage") lands here as a sub-kernel with no clean
# nominal head. Other logical-context keys (AIM_OBJECTIVE, TEMPORAL_CONTEXT, …)
# can legitimately carry clausal content whose object is NOT the right reduction
# (e.g. weather "with parakeets expected to fly over"), so they are left alone.
_LOGICAL_CONTEXT_KEYS = ("CAUSATION",)

def _is_sub_kernel(node):
    return isinstance(node, Singleton) and getattr(node, "kernel", None) is not None


def _circumstantial_subordinators(ctx):
    """Curated subordinating markers that flag a circumstantial clause whose verb
    is not a genuine cause predicate (so the clause reduces to its object noun).
    Sourced from raw_data/markers/circumstantial_subordinators.txt via HOnK — a
    deliberate subset of honk:SubordinatingConjunction (causal/conditional
    subordinators are excluded). Lowercased; empty when HOnK is unavailable, in
    which case this rule simply no-ops rather than over-collapsing."""
    try:
        markers = ctx.services.getHOnK().getCircumstantialSubordinators()
    except Exception:
        return frozenset()
    return frozenset(str(m).strip().lower() for m in (markers or ()))


def _is_subordinate_clause(sub_kernel_singleton, markers):
    """True iff the sub-kernel's edge is a subordinating circumstantial clause
    (e.g. edge label "while carry"), not a content predicate. Keeps this rule
    from collapsing meaningful clausal causes."""
    edge = getattr(sub_kernel_singleton.kernel, "edgeLabel", None)
    name = getattr(edge, "named_entity", None) if edge is not None else None
    if not name:
        return False
    return bool({tok.lower() for tok in str(name).split()} & markers)


def _clause_object(sub_kernel_singleton):
    """The clean nominal head of a clause sub-kernel: its target if that's a
    contentful noun, else its source. Returns None if neither qualifies."""
    rel = sub_kernel_singleton.kernel
    for cand in (getattr(rel, "target", None), getattr(rel, "source", None)):
        if (isinstance(cand, Singleton) and getattr(cand, "kernel", None) is None
                and (cand.type or "").lower() == "noun"
                and cand.named_entity):
            return cand
    return None


def _dedupe_by_id_name(values):
    out, seen = [], set()
    for v in values:
        key = (getattr(v, "id", None), getattr(v, "named_entity", None))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


class LogicalContextClauseDedupeRule(StructuralRewriteRule):
    """Reduce a clause sub-kernel inside a logical-context property to its clean
    nominal head.

    Pattern (transport_002 S2): the CAUSATION of
        have(NPCP, reduction)[(CAUSATION: while_carry(damage, repairs))]
    holds a clause sub-kernel ``while carry(damage, repairs)`` that has no clean
    nominal head, so it later decodes to a ``None`` eFOL constituent. That blocks
    the paraphrase
        "there is a reduction ... while we carry out repairs"  (S1, CAUSATION: repairs)
        "... has a reduction ... while repairs are carried out" (S2)
    from comparing as equivalent.

    Rewrite: replace each logical-context clause sub-kernel with its object noun
    (the clause's target, e.g. ``repairs``), then de-duplicate — so the property
    carries the cause as a plain noun matching the paraphrase, regardless of
    whether a redundant sibling noun also happened to be present (which varies
    run-to-run)."""

    name = "logical_context_clause_dedupe"
    phase = "post_logical_rewrite"

    def _reducible(self, v, markers):
        return (_is_sub_kernel(v) and _is_subordinate_clause(v, markers)
                and _clause_object(v) is not None)

    def matches(self, kernel, ctx):
        if not (isinstance(kernel, Singleton) and kernel.properties):
            return None
        markers = _circumstantial_subordinators(ctx)
        if not markers:
            return None
        props = dict(kernel.properties)
        reducible = {}
        for key in _LOGICAL_CONTEXT_KEYS:
            if key not in props:
                continue
            values = props[key]
            values = list(values) if isinstance(values, (list, tuple)) else [values]
            if any(self._reducible(v, markers) for v in values):
                reducible[key] = True
        return {"keys": list(reducible)} if reducible else None

    def apply(self, kernel, bindings, ctx):
        markers = _circumstantial_subordinators(ctx)
        props = copy_props(kernel.properties)
        for key in bindings["keys"]:
            values = props.get(key, [])
            values = list(values) if isinstance(values, (list, tuple)) else [values]
            rebuilt = []
            for v in values:
                if self._reducible(v, markers):
                    rebuilt.append(_clause_object(v))
                else:
                    rebuilt.append(v)
            rebuilt = _dedupe_by_id_name(rebuilt)
            if rebuilt:
                props[key] = rebuilt
            else:
                props.pop(key, None)
        return kernel.update_node_props(props)
