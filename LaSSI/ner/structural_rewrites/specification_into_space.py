__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    property_values,
)
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class FoldSpecificationIntoSpaceRule(StructuralRewriteRule):
    """Fold a SPECIFICATION entry into the matching SPACE entry's `extra`.

    Pattern (all must hold):
      * Kernel has both SPACE and SPECIFICATION properties.
      * Some SPECIFICATION entry shares its `named_entity` with a SPACE entry.

    Rewrite:
      * Merge the SPECIFICATION entry's `extra` property into the SPACE
        entry's properties (deduplicating by id / named_entity).
      * Drop the matched SPECIFICATION entries.
      * Remove the SPECIFICATION key entirely if it becomes empty.

    Why: when a phrase like "near the Haymarket area of Newcastle" is parsed,
    the same entity ends up classified twice — once as SPACE (from "near") and
    once as SPECIFICATION (from "of"). The two views describe the same place;
    the SPECIFICATION's extra (e.g. Newcastle) belongs on the SPACE entry just
    like an apposition ("Haymarket, Newcastle") puts it there directly.
    Folding aligns sentences with different surface forms onto the same FOL
    shape so downstream similarity comparisons see them as equivalent.
    """

    name = "fold_specification_into_space"
    phase = "post_cleanup"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton):
            return None
        props = dict(kernel.properties)
        space_values = property_values(props, "SPACE")
        spec_values = property_values(props, "SPECIFICATION")
        if not space_values or not spec_values:
            return None

        space_by_name = {}
        for s in space_values:
            n = self._entity_name(s)
            if n:
                space_by_name.setdefault(n, []).append(s)

        pairs = []
        consumed_spec_indices = set()
        for i, spec in enumerate(spec_values):
            sn = self._entity_name(spec)
            if not sn or sn not in space_by_name:
                continue
            target_space = space_by_name[sn][0]
            pairs.append((target_space, spec))
            consumed_spec_indices.add(i)

        if not pairs:
            return None
        return {
            "pairs": pairs,
            "space_values": space_values,
            "spec_values": spec_values,
            "consumed_spec_indices": consumed_spec_indices,
        }

    def apply(self, kernel, bindings, ctx):
        from LaSSI.ner.MergeSetOfSingletons import merge_properties

        merged_space_by_id = {}
        for space, spec in bindings["pairs"]:
            new_space = self._merge_spec_into_space(space, spec, merge_properties)
            sid = id(space)
            merged_space_by_id[sid] = new_space

        new_space = []
        for s in bindings["space_values"]:
            new_space.append(merged_space_by_id.get(id(s), s))

        new_spec = [
            s for i, s in enumerate(bindings["spec_values"])
            if i not in bindings["consumed_spec_indices"]
        ]

        props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }
        props["SPACE"] = new_space
        if new_spec:
            props["SPECIFICATION"] = new_spec
        else:
            props.pop("SPECIFICATION", None)

        return kernel.update_node_props(props)

    @staticmethod
    def _entity_name(node):
        if isinstance(node, Singleton):
            return node.named_entity
        return None

    @staticmethod
    def _merge_spec_into_space(space, spec, merge_properties_fn):
        """Carry SPECIFICATION's `extra` into SPACE's properties.

        Deliberately narrow: we don't fold the spec entry's own `type` (which
        would clobber SPACE's type classification, e.g. `near place`) nor its
        case markers (already present on the SPACE entry). Only `extra` is
        moved across, since that's where the genuine new information sits.
        """
        if not isinstance(space, Singleton) or not isinstance(spec, Singleton):
            return space
        spec_props = dict(spec.properties)
        if "extra" not in spec_props:
            return space
        space_props = dict(space.properties)
        space_props = merge_properties_fn(space_props, {"extra": spec_props["extra"]})
        return space.update_node_props(space_props)
