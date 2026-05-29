__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons


class ParticipialSubKernelPreserveRule(StructuralRewriteRule):
    """Preserve a participial verb sub-kernel as a sub-predicate under its
    logical role, but repair its subject and strip the duplicates the
    logical rewriter scattered across the outer property bag.

    Pattern: the outer kernel has its OWN real verb edge (so the participle
    is subordinate, not the main predicate), and one of its uppercase
    properties (e.g. CAUSATION) holds a Singleton whose own kernel is a
    verb predicate. The sub-kernel's source coincides with — or is the
    ``extra`` of — another node sitting in the outer property bag (e.g.
    ``individuals`` is the ``extra`` of ``group`` under SPECIFICATION).

    Rewrite:
      R1 — replace the sub-kernel's source with the *containing* outer node
           (head-NP restoration: ``cause(individuals, ...)`` becomes
           ``cause(group[extra:individuals], ...)``).
      R2 — inside the sub-kernel, relabel a CAUSATION whose values are
           non-verb nouns to SPECIFICATION (``damage to vehicles`` is
           nominal post-modification, not a causation chain).
      R3 — drop from every other outer property list any node whose id is
           now reachable inside the (rewritten) sub-kernel. That removes
           the duplicate ``SPECIFICATION:group``, ``OFTERM:vehicles`` and
           the bare ``damage`` sibling that ``rewrite_properties_logically``
           re-emitted alongside the predicate.
      R4 — fall out of (R3): the sub-kernel ends up as the sole value in
           its role's list."""

    name = "participial_sub_kernel_preserve"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        outer_edge = kernel.kernel.edgeLabel
        if not isinstance(outer_edge, Singleton):
            return None
        if (outer_edge.type or "").lower() != "verb":
            return None

        outer_props = dict(kernel.properties) if kernel.properties else {}
        candidates = []
        for key, value in outer_props.items():
            if not isinstance(key, str) or not key.isupper():
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            for idx, prop_node in enumerate(values):
                if not isinstance(prop_node, Singleton):
                    continue
                sub_kernel = prop_node.kernel
                if sub_kernel is None or sub_kernel.source is None:
                    continue
                edge_label = sub_kernel.edgeLabel
                if not isinstance(edge_label, Singleton):
                    continue
                if (edge_label.type or "").lower() != "verb":
                    continue
                src_id = getattr(sub_kernel.source, "id", None)
                if src_id is None:
                    continue
                host = self._find_host_in_props(
                    outer_props, src_id, skip_key=key, skip_idx=idx
                )
                if host is None:
                    continue
                candidates.append({
                    "prop_key": key,
                    "sub_singleton": prop_node,
                    "host_node": host["node"],
                    "needs_substitution": host["needs_substitution"],
                })
        if not candidates:
            return None
        return {"candidates": candidates}

    def apply(self, kernel, bindings, ctx):
        outer_props = {
            k: list(v) if isinstance(v, (list, tuple)) else [v]
            for k, v in dict(kernel.properties).items()
        }

        for c in bindings["candidates"]:
            sub = c["sub_singleton"]
            host_node = c["host_node"]

            # R1: head-NP restoration.
            new_sub = sub
            if c["needs_substitution"]:
                new_sub = sub.update_kernel(host_node, "source")

            # R2: nominal CAUSATION inside the sub-kernel → SPECIFICATION.
            new_sub = self._relabel_nominal_causation(new_sub)

            # Swap the sub-singleton in place inside its slot.
            slot = outer_props.get(c["prop_key"], [])
            sub_id = sub.id
            for j, n in enumerate(slot):
                if isinstance(n, Singleton) and n.id == sub_id:
                    slot[j] = new_sub
                    break
            outer_props[c["prop_key"]] = slot

            # R3: drop outer-bag duplicates now reachable inside new_sub.
            reached = self._collect_reachable_ids(new_sub)
            keep_id = new_sub.id
            for k in list(outer_props.keys()):
                outer_props[k] = [
                    n for n in outer_props[k]
                    if not (
                        isinstance(n, Singleton)
                        and n.id is not None
                        and n.id in reached
                        and n.id != keep_id
                    )
                ]

        cleaned = {k: v for k, v in outer_props.items() if v not in (None, [], ())}
        return kernel.update_node_props(cleaned)

    # ---- helpers ----

    @classmethod
    def _find_host_in_props(cls, outer_props, src_id, skip_key, skip_idx):
        for key, vals in outer_props.items():
            values = vals if isinstance(vals, (list, tuple)) else [vals]
            for idx, n in enumerate(values):
                if not isinstance(n, Singleton):
                    continue
                if key == skip_key and idx == skip_idx:
                    continue
                if n.id is not None and n.id == src_id:
                    return {"node": n, "needs_substitution": False}
                for ex in cls._extras_of(n):
                    if isinstance(ex, Singleton) and ex.id == src_id:
                        return {"node": n, "needs_substitution": True}
        return None

    @staticmethod
    def _extras_of(node):
        props = dict(node.properties) if node.properties else {}
        extras = props.get("extra")
        if extras is None:
            return []
        return list(extras) if isinstance(extras, (list, tuple)) else [extras]

    @classmethod
    def _collect_reachable_ids(cls, node, acc=None):
        if acc is None:
            acc = set()
        if isinstance(node, Singleton):
            if node.id is not None:
                acc.add(node.id)
            if node.kernel is not None:
                cls._collect_reachable_ids(node.kernel.source, acc)
                cls._collect_reachable_ids(node.kernel.target, acc)
            props = dict(node.properties) if node.properties else {}
            for v in props.values():
                vs = v if isinstance(v, (list, tuple)) else [v]
                for x in vs:
                    if isinstance(x, Singleton):
                        cls._collect_reachable_ids(x, acc)
                    elif isinstance(x, SetOfSingletons):
                        cls._collect_reachable_ids(x, acc)
        elif isinstance(node, SetOfSingletons):
            for e in node.entities:
                cls._collect_reachable_ids(e, acc)
        return acc

    @staticmethod
    def _relabel_nominal_causation(sub_singleton):
        props = dict(sub_singleton.properties) if sub_singleton.properties else {}
        cau_vals = props.get("CAUSATION")
        if cau_vals is None:
            return sub_singleton
        cau_list = list(cau_vals) if isinstance(cau_vals, (list, tuple)) else [cau_vals]
        if not cau_list:
            return sub_singleton
        all_nominal = all(
            isinstance(v, Singleton)
            and (v.type or "").lower() != "verb"
            and v.kernel is None
            for v in cau_list
        )
        if not all_nominal:
            return sub_singleton

        spec = props.get("SPECIFICATION")
        if spec is None:
            spec_list = []
        else:
            spec_list = list(spec) if isinstance(spec, (list, tuple)) else [spec]
        existing_ids = {getattr(s, "id", None) for s in spec_list}
        for v in cau_list:
            if v.id not in existing_ids:
                spec_list.append(v)
                existing_ids.add(v.id)

        props.pop("CAUSATION", None)
        props["SPECIFICATION"] = spec_list
        return sub_singleton.update_node_props(props)
