__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.external_services.Services import Services
from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import Singleton

_RELATIVE_PRONOUNS = {"which", "that", "who", "whom"}


class RequirementClauseSimplifierRule(StructuralRewriteRule):
    """Collapse a relative-clause sub-kernel under a REQUIREMENT/CAUSATION
    property down to just the required object.

    Pattern: a kernel property keyed `REQUIREMENT` or `CAUSATION` whose value
    is a sub-kernel Singleton (Sentence) whose source is a relative pronoun
    (`which`, `that`, `who`, `whom`). The verb of the sub-kernel was matched
    by a HOnK ConsumptionVerb classifier (e.g. "require"), so the bound
    target — the actual required resource — is what we want to surface.

    Rewrite: replace the sub-kernel with its target Singleton, and force the
    property key to `REQUIREMENT` regardless of its starting key (some
    parses misclassify the original verb under CAUSATION via the `to`/`due`
    rule chain)."""

    name = "requirement_clause_simplifier"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not isinstance(kernel, Singleton):
            return None
        props = dict(kernel.properties)
        candidates = []
        for key in ("REQUIREMENT", "CAUSATION"):
            if key not in props:
                continue
            values = props[key]
            if not isinstance(values, (list, tuple)):
                values = [values]
            for idx, value in enumerate(values):
                sub_kernel = self._extract_sub_kernel(value)
                if sub_kernel is None:
                    continue
                source = sub_kernel.source
                if not isinstance(source, Singleton):
                    continue
                lemma = self._get_lemma(source)
                if lemma not in _get_relative_pronouns():
                    continue
                target = sub_kernel.target
                if target is None:
                    continue
                candidates.append({
                    "prop_key": key,
                    "value_index": idx,
                    "object": target,
                })
        if not candidates:
            return None
        return {"candidates": candidates}

    def apply(self, kernel, bindings, ctx):
        props = {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in dict(kernel.properties).items()
        }

        replacements_by_key = {}
        for candidate in bindings["candidates"]:
            replacements_by_key.setdefault(candidate["prop_key"], []).append(candidate)

        # Per the rule docstring: force the surviving objects under REQUIREMENT
        # regardless of the originating key (CAUSATION misclassifications, etc.)
        promoted = list(props.get("REQUIREMENT", [])) if not isinstance(props.get("REQUIREMENT", []), list) else list(props.get("REQUIREMENT", []))
        for key, candidates in replacements_by_key.items():
            existing = props.get(key, [])
            if not isinstance(existing, list):
                existing = [existing]
            else:
                existing = list(existing)
            consumed = set()
            for candidate in candidates:
                idx = candidate["value_index"]
                if idx >= len(existing):
                    continue
                consumed.add(idx)
                promoted.append(candidate["object"])
            kept = [v for i, v in enumerate(existing) if i not in consumed]
            if kept:
                props[key] = kept
            else:
                props.pop(key, None)
        if promoted:
            props["REQUIREMENT"] = promoted

        return kernel.update_node_props(props)

    @staticmethod
    def _extract_sub_kernel(value):
        if isinstance(value, Singleton) and value.kernel is not None:
            return value.kernel
        return None

    @staticmethod
    def _get_lemma(node):
        if not isinstance(node, Singleton):
            return ""
        props = dict(node.properties) if node.properties is not None else {}
        lemma = props.get("lemma") or node.named_entity or ""
        return str(lemma).lower()


def _get_relative_pronouns():
    try:
        honk = Services.getInstance().getHOnK()
        getter = getattr(honk, "getRelativePronouns", None)
        if callable(getter):
            return {str(p).lower() for p in getter() if p}
    except Exception:
        pass
    return _RELATIVE_PRONOUNS
