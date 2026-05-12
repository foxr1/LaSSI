__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import (
    Grouping,
    SetOfSingletons,
    Singleton,
)


class RelclTargetLiftRule(StructuralRewriteRule):
    """When a verb kernel has lost its direct object and the original obj
    landed under a relative clause's nmod (e.g. "We're carrying out
    inspections and roof repairs at Monkseaton station which requires
    scaffolding" — Stanza puts the conj `inspections AND repairs` on the
    relative pronoun's nmod), recover that nominal as the kernel target and
    promote the original verb-edge target to SPACE.

    Pattern (all must hold):
      * `kernel.kernel.target` is None.
      * `kernel.kernel.source` is an existential Singleton.
      * `kernel.kernel.edgeLabel` is a non-copula verb.
      * The graph (ctx.matchers.G) contains a verb-edge from kernel.source.id
        to some node V; V has an `acl_relcl` (or `acl`) child W; W has an
        `nmod` child N where N is a SetOfSingletons (AND) or a noun Singleton
        not already attached to the kernel.

    Rewrite:
      * Set kernel.target = N.
      * If V is location-like, append it to SPACE (deduped by id).
      * Leave existing properties otherwise untouched."""

    name = "relcl_target_lift"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        rel = kernel.kernel
        if rel.target is not None:
            return None
        source = rel.source
        if not (isinstance(source, Singleton) and source.type == 'existential'):
            return None
        edge_label = rel.edgeLabel
        if not isinstance(edge_label, Singleton):
            return None

        G = getattr(ctx.matchers, 'G', None)
        if G is None or source.id not in G.nodes:
            return None

        verb_target_node = self._find_verb_edge_target(G, source.id)
        if verb_target_node is None:
            return None

        nmod_node = self._find_nmod_via_relcl(G, verb_target_node)
        if nmod_node is None:
            return None

        return {
            "verb_target": verb_target_node,
            "nmod_target": nmod_node,
        }

    def apply(self, kernel, bindings, ctx):
        from LaSSI.ner.structural_rewrites.base import append_unique_property_value

        nmod_target = bindings["nmod_target"]
        verb_target = bindings["verb_target"]

        new_props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }
        if isinstance(verb_target, Singleton) and ctx.matchers.is_location_like(verb_target):
            append_unique_property_value(new_props, "SPACE", verb_target)

        kernel = kernel.update_node_props(new_props)
        kernel = kernel.update_kernel(nmod_target, "target")
        return kernel

    @staticmethod
    def _find_verb_edge_target(G, source_id):
        for _, target_id, data in G.out_edges(source_id, data=True):
            label = data.get('label')
            if not isinstance(label, Singleton):
                continue
            if (label.type or '').lower() != 'verb':
                continue
            target_node = G.nodes[target_id].get('data')
            if target_node is not None:
                return target_node
        return None

    @staticmethod
    def _find_nmod_via_relcl(G, verb_target):
        if not isinstance(verb_target, Singleton):
            return None
        if verb_target.id not in G.nodes:
            return None
        relcl_targets = []
        for _, child_id, data in G.out_edges(verb_target.id, data=True):
            label = data.get('label')
            if not isinstance(label, Singleton):
                continue
            if (label.named_entity or '').lower() in {'acl_relcl', 'acl'}:
                child = G.nodes[child_id].get('data')
                if child is not None:
                    relcl_targets.append(child)
        for relcl_target in relcl_targets:
            if not isinstance(relcl_target, Singleton):
                continue
            if relcl_target.id not in G.nodes:
                continue
            for _, nmod_child_id, data in G.out_edges(relcl_target.id, data=True):
                label = data.get('label')
                if not isinstance(label, Singleton):
                    continue
                if (label.named_entity or '').lower() in {'nmod', 'obj', 'iobj', 'dobj'}:
                    nmod_child = G.nodes[nmod_child_id].get('data')
                    if isinstance(nmod_child, SetOfSingletons) and nmod_child.type == Grouping.AND:
                        return nmod_child
                    if isinstance(nmod_child, Singleton) and 'verb' not in (nmod_child.type or '').lower():
                        return nmod_child
        return None
