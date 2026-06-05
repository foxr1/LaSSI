"""Promote a scattered imperative-action object into the kernel target.

Telegraphic notice actions like ``Abandon approx 11m of 6" SI`` and
``replace with approx 11m of LP PE`` reach the kernel as a targetless
predicate whose object NP has been scattered across construct buckets — the
causative/materialisation derivation routes it into ``CAUSATION`` /
``SPECIFICATION`` / ``TOGETHERNESS`` instead of the object slot, and the
measure (``11m``, ``6"``) ends up tangled with the material (``SI``, ``LP PE``)
at inconsistent nesting depths.

This rule reconstructs the intended shape::

    Abandon(?, SI[(MEASURE:[11m[(extra:approx)], 6"])])
    replace(?, LP PE[(MEASURE:11m[(extra:approx)])])

by partitioning the scattered nodes into the single **material** head (the
non-measure content noun → the patient) and the **measures** (number+unit
nodes → ``MEASURE`` on that head), then placing the material in the target
slot. It fires only for an action verb (``CausativeVerb`` /
``MaterialisationVerb``) with an empty/existential target *and* at least one
measure present, so it cannot disturb ordinary causation clauses.
"""

import re

from LaSSI.ner.structural_rewrites.base import (
    StructuralRewriteRule,
    copy_props,
    property_values,
    replace_kernel,
)
from LaSSI.structures.internal_graph.EntityRelationship import (
    SetOfSingletons,
    Singleton,
)

# A measure node's surface form begins with a digit (11m, 6, 2.8, 57).
_MEASURE_RE = re.compile(r"^\d")


class ActionObjectPromotionRule(StructuralRewriteRule):
    name = "action_object_promotion"
    phase = "post_logical_rewrite"

    _SCATTER_KEYS = ("CAUSATION", "SPECIFICATION", "TOGETHERNESS")
    _ACTION_CLASSES = ("CausativeVerb", "MaterialisationVerb")
    _MATERIAL_TYPES = {"noun", "gpe", "loc", "fac"}

    # ---- classification ------------------------------------------------

    @classmethod
    def _is_measure(cls, node):
        return (isinstance(node, Singleton)
                and bool(_MEASURE_RE.match((node.named_entity or "").strip())))

    @classmethod
    def _is_material(cls, node):
        if not isinstance(node, Singleton) or cls._is_measure(node):
            return False
        return (node.type or "").strip().lower() in cls._MATERIAL_TYPES

    # ---- matching ------------------------------------------------------

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return None
        edge = kernel.kernel.edgeLabel
        if not isinstance(edge, Singleton):
            return None
        matchers = getattr(ctx, "matchers", None)
        if matchers is None or not any(
            matchers.matches_class(edge, cls) for cls in self._ACTION_CLASSES
        ):
            return None
        target = kernel.kernel.target
        if not (target is None
                or (isinstance(target, Singleton) and target.type == "existential")):
            return None
        if not any(k in dict(kernel.properties) for k in self._SCATTER_KEYS):
            return None
        materials, measures = self._partition(kernel)
        if len(materials) != 1 or not measures:
            return None
        return {"material": materials[0], "measures": measures}

    def apply(self, kernel, bindings, ctx):
        material = bindings["material"]
        measures = bindings["measures"]
        props = copy_props(kernel)
        for key in self._SCATTER_KEYS:
            props.pop(key, None)
        mat_props = copy_props(material)
        existing = mat_props.get("MEASURE")
        existing = (list(existing) if isinstance(existing, (list, tuple))
                    else ([existing] if existing is not None else []))
        existing.extend(measures)
        mat_props["MEASURE"] = existing
        new_target = material.update_node_props(mat_props)
        return replace_kernel(kernel, target=new_target).update_node_props(props)

    # ---- partition -----------------------------------------------------

    def _partition(self, kernel):
        """Flat-collect all scattered nodes (and their nested ``extra``
        children) into ``(materials, measures)``, deduped by surface form.
        A measure keeps only its modifier extras (``approx``); nested
        material/measure extras are lifted out and classified in their own
        right."""
        materials, measures = [], []
        seen = set()

        def visit(node):
            if isinstance(node, SetOfSingletons):
                for entity in node.entities:
                    visit(entity)
                return
            if not isinstance(node, Singleton):
                return
            extras = property_values(dict(node.properties), "extra")
            modifier_extras = []
            for extra in extras:
                if self._is_measure(extra) or self._is_material(extra):
                    visit(extra)          # promote nested measure / material
                else:
                    modifier_extras.append(extra)  # keep approx etc. attached
            name = (node.named_entity or "").strip()
            if not name or name in seen:
                return
            np = copy_props(node)
            if modifier_extras:
                # Keep `extra` as a list: downstream FOL conversion
                # (rewrite_kernels.make_arg) indexes extra_val[0], so a bare
                # Singleton value is not acceptable.
                np["extra"] = modifier_extras
            else:
                np.pop("extra", None)
            rebuilt = node.update_node_props(np)
            if self._is_measure(node):
                seen.add(name)
                measures.append(rebuilt)
            elif self._is_material(node):
                seen.add(name)
                materials.append(rebuilt)

        for key in self._SCATTER_KEYS:
            for value in property_values(dict(kernel.properties), key):
                visit(value)
        return materials, measures
