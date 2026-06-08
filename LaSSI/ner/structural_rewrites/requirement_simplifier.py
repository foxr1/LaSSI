__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.external_services.Services import Services
from LaSSI.ner.string_functions import lemmatize_verb
from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import Singleton

_RELATIVE_PRONOUNS = {"which", "that", "who", "whom"}


def _verb_in(verb, honk_list) -> bool:
    """Lemmatised membership test against a HOnK verb set (cf.
    HOnKLogicalRewriting.is_name_in_honk)."""
    candidates = {str(c).lower() for c in (lemmatize_verb(verb), verb) if c}
    return bool(candidates & {str(item).lower() for item in honk_list if item})


class RequirementClauseSimplifierRule(StructuralRewriteRule):
    """Collapse a relative-clause sub-kernel under a REQUIREMENT/CAUSATION
    property down to just the bound object.

    Pattern: a kernel property keyed `REQUIREMENT` or `CAUSATION` whose value
    is a sub-kernel Singleton (Sentence) whose source is a relative pronoun
    (`which`, `that`, `who`, `whom`). The bound target — the actual
    required/affected resource — is what we want to surface.

    Rewrite: replace the sub-kernel with its target Singleton, choosing the
    destination key from the sub-kernel's *verb sense* rather than its
    (often misclassified) starting key:
      - a ConsumptionVerb clause ("which requires scaffolding") → REQUIREMENT
      - a CausativeVerb clause ("that caused the reduction")     → CAUSATION
      - otherwise keep the originating key.
    This stops a causative relative clause being force-promoted to
    REQUIREMENT (e.g. transport_002 S4 "that caused the reduction")."""

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
                    "dest_key": self._dest_key_for_verb(sub_kernel, key),
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

        # Collapse each relative-clause sub-kernel to its bound object, routing
        # the surviving object to the key implied by the sub-kernel's verb sense
        # (see `_dest_key_for_verb`) rather than force-promoting to REQUIREMENT.
        promoted_by_dest = {}
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
                promoted_by_dest.setdefault(candidate["dest_key"], []).append(candidate["object"])
            kept = [v for i, v in enumerate(existing) if i not in consumed]
            if kept:
                props[key] = kept
            else:
                props.pop(key, None)

        for dest_key, objects in promoted_by_dest.items():
            base = props.get(dest_key, [])
            if not isinstance(base, list):
                base = [base]
            else:
                base = list(base)
            base.extend(objects)
            props[dest_key] = base

        return kernel.update_node_props(props)

    @staticmethod
    def _dest_key_for_verb(sub_kernel, original_key):
        """Pick the destination construct key from the relative clause's verb:
        ConsumptionVerb → REQUIREMENT, CausativeVerb → CAUSATION, else keep
        the original key. Resolves via HOnK's curated verb sets (no hardcoding)."""
        edge = getattr(sub_kernel, "edgeLabel", None)
        verb = getattr(edge, "named_entity", None) if edge is not None else None
        if not verb:
            return original_key
        try:
            honk = Services.getInstance().getHOnK()
            if _verb_in(verb, honk.getConsumptionVerbs()):
                return "REQUIREMENT"
            if _verb_in(verb, honk.getCausativeVerbs()):
                return "CAUSATION"
        except Exception:
            pass
        return original_key

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
