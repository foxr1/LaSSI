__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from collections import defaultdict

from LaSSI.ner.KernelLogicalRewriter import KernelLogicalRewriter
from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons


class ParticipialCollapseRule(StructuralRewriteRule):
    """Dissolve a reduced-relative / participial sub-kernel that survived
    `rewrite_properties_logically` so its objects and logical-construct
    children sit flat alongside the outer kernel's own properties.

    Pattern: an outer-kernel property (any uppercase key, or `SENTENCE`)
    holds a Singleton whose own kernel.source is the same entity as the
    outer kernel.source / target, or is nested inside their `extra` list.
    That's the shape produced when an acl participial like "recorded near X"
    (modifying `behaviour incident`) or "causing damage to vehicles"
    (modifying `individuals`) gets attached as a sub-kernel by
    `create_sentence`.

    Rewrite: drop the sub-kernel wrapper and re-classify its target plus
    every uppercase-keyed child through the same logical-rewriting rules
    that produced the outer property bag, so e.g. `record(behaviour,
    January)[SPACE: parking area]` under `SENTENCE` becomes
    `TIME:January[type:defined]` and `SPACE:parking area` at the outer
    kernel level, while `cause(individuals, damage)[CAUSATION:vehicles]`
    under `CAUSATION` flattens to `CAUSATION:[damage, vehicles]`.

    Non-verb singletons with `actioned: <ppl>` get normalized to
    `amod: <ppl>` on the way up — this matches what
    `add_to_properties` would have done if the node had entered the outer
    kernel through that path."""

    name = "participial_collapse"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None

        anchor_ids = self._collect_anchor_ids(kernel)
        if not anchor_ids:
            return None

        outer_edge = kernel.kernel.edgeLabel
        outer_is_verb = (
            isinstance(outer_edge, Singleton)
            and (outer_edge.type or "").lower() == "verb"
        )
        outer_lemma = self._verb_lemma(outer_edge) if outer_is_verb else None

        candidates = []
        for key, value in dict(kernel.properties).items():
            if not isinstance(key, str):
                continue
            if not (key.isupper() or key == "SENTENCE"):
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            for idx, prop_node in enumerate(values):
                if not isinstance(prop_node, Singleton):
                    continue
                sub_kernel = prop_node.kernel
                if sub_kernel is None or sub_kernel.source is None:
                    continue
                source = sub_kernel.source
                if not isinstance(source, Singleton):
                    continue
                if source.id not in anchor_ids:
                    continue
                # Guard: only collapse genuine acl-participial sub-kernels,
                # not e.g. copular complements. Their edgeLabel is a verb.
                edge_label = sub_kernel.edgeLabel
                if not isinstance(edge_label, Singleton):
                    continue
                if (edge_label.type or "").lower() != "verb":
                    continue
                # When the outer kernel has its own matrix verb whose lemma
                # differs from this sub-kernel's verb, the acl participle
                # carries the predicate identity we want at the top of the
                # formula (e.g. "incident recorded near X ... involved Y"
                # canonicalises to `record(incident)[..., CAUSATION:Y]`).
                # Dissolve as usual but also flag the candidate to swap the
                # outer kernel's edge-label verb for the sub-kernel's.
                swap_edge = False
                if outer_is_verb and outer_lemma is not None:
                    sub_lemma = self._verb_lemma(edge_label)
                    if sub_lemma is not None and sub_lemma != outer_lemma:
                        swap_edge = True
                candidates.append({
                    "prop_key": key,
                    "value_index": idx,
                    "sub_singleton": prop_node,
                    "swap_edge": swap_edge,
                })

        if not candidates:
            return None
        return {"candidates": candidates}

    def apply(self, kernel, bindings, ctx):
        rewriter = KernelLogicalRewriter(ctx.node_functions, ctx.matchers)
        new_props = {
            k: list(v) if isinstance(v, (list, tuple)) else [v]
            for k, v in dict(kernel.properties).items()
        }

        # Group candidates by prop_key so we can drop the right indices.
        consumed = defaultdict(set)
        for candidate in bindings["candidates"]:
            consumed[candidate["prop_key"]].add(candidate["value_index"])

        for prop_key, indices in consumed.items():
            existing = new_props.get(prop_key, [])
            new_props[prop_key] = [v for i, v in enumerate(existing) if i not in indices]

        new_edge_label = None
        for candidate in bindings["candidates"]:
            self._dissolve_sub_kernel(
                kernel, candidate["sub_singleton"], candidate["prop_key"],
                new_props, rewriter,
            )
            if candidate.get("swap_edge") and new_edge_label is None:
                new_edge_label = self._merge_edge_label(
                    kernel.kernel.edgeLabel,
                    candidate["sub_singleton"].kernel.edgeLabel,
                )

        # Drop now-empty property buckets.
        cleaned = {k: v for k, v in new_props.items() if v not in (None, [], ())}
        result = kernel.update_node_props(cleaned)
        if new_edge_label is not None:
            result = result.update_kernel(new_edge_label, "edgeLabel")
        return result

    @staticmethod
    def _merge_edge_label(outer_edge, sub_edge):
        """Produce a new edgeLabel that takes its verb identity from
        ``sub_edge`` (named_entity + ``lemma``) but keeps the outer edge's
        carrier properties (e.g. trailing ``punct``, position offsets) so
        sentence-level decoration stays attached to the surviving predicate.
        """
        if not isinstance(sub_edge, Singleton):
            return None
        if not isinstance(outer_edge, Singleton):
            return sub_edge
        merged_props = dict(outer_edge.properties) if outer_edge.properties else {}
        sub_props = dict(sub_edge.properties) if sub_edge.properties else {}
        # Replace identifying fields with sub's so position/lemma reflect the
        # surviving verb, keep outer's carrier properties (e.g. `punct`).
        for k in ("lemma", "xpos", "pos", "begin", "end"):
            if k in sub_props:
                merged_props[k] = sub_props[k]
            else:
                merged_props.pop(k, None)
        # Rebuild via update_node_props so properties become a hashable
        # frozenset (required by downstream set-membership checks).
        renamed = outer_edge.update_name(sub_edge.named_entity)
        return renamed.update_node_props(merged_props)

    # ---- helpers ----

    @staticmethod
    def _verb_lemma(edge_label):
        if not isinstance(edge_label, Singleton):
            return None
        props = dict(edge_label.properties) if edge_label.properties else {}
        lemma = props.get("lemma") or edge_label.named_entity
        return lemma.lower() if isinstance(lemma, str) else None

    @staticmethod
    def _collect_anchor_ids(kernel):
        anchors = set()
        for endpoint in (kernel.kernel.source, kernel.kernel.target):
            if isinstance(endpoint, Singleton):
                anchors.add(endpoint.id)
                extras = dict(endpoint.properties).get("extra", []) if endpoint.properties else []
                if isinstance(extras, (list, tuple)):
                    for extra in extras:
                        if isinstance(extra, Singleton):
                            anchors.add(extra.id)
                elif isinstance(extras, Singleton):
                    anchors.add(extras.id)
            elif isinstance(endpoint, SetOfSingletons):
                for entity in endpoint.entities:
                    if isinstance(entity, Singleton):
                        anchors.add(entity.id)
        return anchors

    def _dissolve_sub_kernel(self, outer_kernel, sub_singleton, default_key,
                             outer_props, rewriter):
        sub_kernel = sub_singleton.kernel

        # 1. Lift the sub-kernel's verb-edge target. Re-classify it via the
        # logical rewriter so a DATE/preposition like "January[12:in]" lands
        # in TIME rather than the inherited default_key.
        target = sub_kernel.target
        if target is not None:
            target = self._normalize_actioned_to_amod(target)
            if isinstance(target, Singleton) and target.kernel is None:
                classified = defaultdict(list)
                try:
                    classified = rewriter.rewrite_node_logically(
                        outer_kernel, target, classified, type_key=default_key,
                    )
                except Exception:
                    classified = defaultdict(list)
                    classified[default_key].append(target)
                for k, vs in classified.items():
                    for v in vs:
                        self._append_normalized(outer_props, k, v)
            else:
                self._append_normalized(outer_props, default_key, target)

        # 2. Promote any uppercase-keyed sub-properties of the sub-singleton.
        for sub_key, sub_value in dict(sub_singleton.properties).items():
            if not isinstance(sub_key, str) or not sub_key.isupper():
                continue
            values = sub_value if isinstance(sub_value, (list, tuple)) else [sub_value]
            for v in values:
                self._append_normalized(outer_props, sub_key, v)

    @staticmethod
    def _normalize_actioned_to_amod(node):
        if not isinstance(node, Singleton):
            return node
        props = dict(node.properties)
        if "actioned" in props and node.type != "verb" and node.named_entity:
            actioned_val = props.pop("actioned")
            if "amod" in props:
                existing = props["amod"]
                existing_list = list(existing) if isinstance(existing, (list, tuple)) else [existing]
                props["amod"] = tuple(existing_list + [actioned_val])
            else:
                props["amod"] = tuple([actioned_val])
            return node.update_node_props(props)
        return node

    def _append_normalized(self, props, key, value):
        value = self._normalize_actioned_to_amod(value)
        existing = props.setdefault(key, [])
        if not isinstance(existing, list):
            existing = list(existing) if isinstance(existing, (list, tuple)) else [existing]
            props[key] = existing
        new_id = getattr(value, "id", None)
        for e in existing:
            if new_id is not None and getattr(e, "id", None) == new_id:
                return
        existing.append(value)
