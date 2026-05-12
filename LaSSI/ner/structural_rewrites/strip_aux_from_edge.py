__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

from LaSSI.external_services.Services import Services
from LaSSI.ner.string_functions import lemmatize_verb
from LaSSI.ner.structural_rewrites.base import StructuralRewriteRule
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class StripAuxFromEdgeLabelRule(StructuralRewriteRule):
    """Strip leading copula auxiliary words from a kernel's edgeLabel.

    Pattern:
      * `kernel.kernel.edgeLabel` is a multi-word verb name whose first token
        is a HOnK copula surface form (e.g. "'re carrying out", "is being
        carried", "are doing").

    Rewrite:
      * Drop the leading copula token(s); keep the remaining lemmatised verb
        head + any particles. ("'re carrying out" → "carry out", "is being
        carried" → "carry").

    Stops as soon as a non-copula token appears, so phrasal verbs like "carry
    out" are preserved intact."""

    name = "strip_aux_from_edge_label"
    phase = "post_logical_rewrite"

    def matches(self, kernel, ctx):
        if not (isinstance(kernel, Singleton) and kernel.kernel is not None):
            return None
        edge_label = kernel.kernel.edgeLabel
        if not isinstance(edge_label, Singleton):
            return None
        edge_name = edge_label.named_entity or ""
        parts = [p for p in edge_name.split() if p]
        if len(parts) < 2:
            return None

        try:
            copula_forms = Services.getInstance().getHOnK().getCopulaSurfaceForms() or set()
        except Exception:
            copula_forms = set()
        copula_lower = {str(f).lower() for f in copula_forms}
        if not copula_lower:
            return None

        leading_copula = 0
        for part in parts:
            lemma = lemmatize_verb(part).lower()
            if part.lower() in copula_lower or lemma in copula_lower:
                leading_copula += 1
                continue
            break
        if leading_copula == 0 or leading_copula >= len(parts):
            return None

        remaining = parts[leading_copula:]
        new_name = " ".join(
            [lemmatize_verb(remaining[0]).lower()] + [p.lower() for p in remaining[1:]]
        )
        if new_name == edge_name:
            return None

        return {"new_name": new_name}

    def apply(self, kernel, bindings, ctx):
        edge_label = kernel.kernel.edgeLabel
        new_edge_label = edge_label.update_name(bindings["new_name"])
        return kernel.update_kernel(new_edge_label, "edgeLabel")
