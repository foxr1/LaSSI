from LaSSI.structures.extended_fol.Formulae import FVariable, FNot, FBinaryPredicate, FUnaryPredicate
from LaSSI.HOnK.TBox.ParaphraseManager import _canon_lookup_paraphrase_concept, _canon_paraphrase_member, _canon_numeric_member

_CANON_SYNTACTIC_PROPERTY_KEYS = frozenset({
    'punct', 'det', 'mark', 'lemma', 'number', 'specification',
    'kernel', 'root', 'subjpass', 'xpos', 'pos', 'begin', 'end',
    'conj', 'cc', 'asAll',
})

def _canon_is_positional_key(k):
    if isinstance(k, (int, float)):
        return True
    if isinstance(k, str):
        try:
            float(k)
            return True
        except (TypeError, ValueError):
            return False
    return False

def _canon_str(s):
    return s.lower() if isinstance(s, str) else s

def _canon_type(t):
    """Normalise the FVariable `type` tag for canonical equality."""
    if t is None:
        return None
    if isinstance(t, str):
        normalised = t.strip().lower()
        if normalised in ('', 'none', 'noun'):
            return 'noun'
        return normalised
    return t

def _canon_fvar_cop(cop_fvar, parent_concept):
    """Simplified canonicalisation for a cop FVariable."""
    if not isinstance(cop_fvar, FVariable):
        return cop_fvar
    name = _canon_str(cop_fvar.name)
    canonical = _canon_numeric_member(name, parent_concept)
    if canonical is None:
        canonical = _canon_paraphrase_member(name)
    if canonical is not None:
        name = canonical
    return FVariable(
        name=name,
        type=_canon_type(cop_fvar.type),
        specification=None,
        cop=None,
        id=None,
        properties=frozenset(),
        spec_negation=cop_fvar.spec_negation,
        meta=cop_fvar.meta,
        asAll=cop_fvar.asAll,
    )

def _canon_fvar_properties(props, parent_concept):
    """Strip syntactic-noise property keys and lift the first
    `cop`/`amod`/`nummod` magnitude value out."""
    if not props:
        return frozenset(), None
    magnitude = None
    out = {}
    for k, v in dict(props).items():
        if isinstance(k, str) and k in _CANON_SYNTACTIC_PROPERTY_KEYS:
            continue
        if _canon_is_positional_key(k):
            continue
        if k in ('cop', 'amod', 'nummod'):
            if magnitude is None:
                magnitude = v
            continue
        if isinstance(v, FVariable):
            out[k] = _canon_fvar(v)
        elif isinstance(v, str):
            out[k] = _canon_str(v)
        elif isinstance(v, (list, tuple)):
            new_items = []
            for item in v:
                if isinstance(item, FVariable):
                    new_items.append(_canon_fvar(item))
                elif isinstance(item, str):
                    new_items.append(_canon_str(item))
                else:
                    new_items.append(item)
            out[k] = type(v)(new_items)
        else:
            out[k] = v

    lifted = None
    if magnitude is not None:
        if isinstance(magnitude, FVariable):
            lifted = _canon_fvar_cop(magnitude, parent_concept)
        elif isinstance(magnitude, str):
            canonical = (_canon_numeric_member(magnitude, parent_concept)
                         or _canon_paraphrase_member(magnitude)
                         or _canon_str(magnitude))
            lifted = FVariable(
                name=canonical, type="JJ", specification=None, cop=None, id=None,
                properties=frozenset(), spec_negation=False, meta="FVariable", asAll=False,
            )
    return frozenset(out.items()), lifted

def _canon_fvar(fvar):
    if not isinstance(fvar, FVariable):
        return fvar
    name = _canon_str(fvar.name)
    spec = _canon_str(fvar.specification)

    original_concept = _canon_lookup_paraphrase_concept(name)

    if name and isinstance(spec, str) and spec:
        name = spec
        spec = None

    head_concept = _canon_lookup_paraphrase_concept(name)
    parent_concept_for_cop = head_concept or original_concept

    canonical_name = _canon_paraphrase_member(name)
    if canonical_name is not None:
        name = canonical_name

    new_cop = (_canon_fvar_cop(fvar.cop, parent_concept_for_cop)
               if fvar.cop is not None else None)

    new_props, lifted_magnitude = _canon_fvar_properties(fvar.properties, parent_concept_for_cop)
    if new_cop is None and lifted_magnitude is not None:
        new_cop = lifted_magnitude

    return FVariable(
        name=name,
        type=_canon_type(fvar.type),
        specification=spec,
        cop=new_cop,
        id=None,
        properties=new_props,
        spec_negation=fvar.spec_negation,
        meta=fvar.meta,
        asAll=fvar.asAll,
    )

def _canon_predicate_properties(props):
    """Canonicalise the predicate-level property bag."""
    if not props:
        return frozenset()
    out = {}
    for k, v in dict(props).items():
        if isinstance(k, str) and k in _CANON_SYNTACTIC_PROPERTY_KEYS:
            continue
        if _canon_is_positional_key(k):
            continue
        if isinstance(v, FVariable):
            out[k] = _canon_fvar(v)
        elif isinstance(v, (list, tuple)):
            new_items = []
            for item in v:
                if isinstance(item, FVariable):
                    new_items.append(_canon_fvar(item))
                else:
                    new_items.append(item)
            out[k] = type(v)(new_items)
        elif isinstance(v, str):
            out[k] = _canon_str(v)
        else:
            out[k] = v
    return frozenset(out.items())

def canonicalize_atom_for_paraphrase_expansion(atom):
    """Public entry point: return a paraphrase-canonical variant of an
    FOL atom suitable for injection into the atom's expansion set."""
    if isinstance(atom, FBinaryPredicate):
        return FBinaryPredicate(
            rel=_canon_str(atom.rel),
            src=_canon_fvar(atom.src),
            dst=_canon_fvar(atom.dst),
            score=atom.score,
            properties=_canon_predicate_properties(atom.properties),
        )
    if isinstance(atom, FUnaryPredicate):
        return FUnaryPredicate(
            rel=_canon_str(atom.rel),
            arg=_canon_fvar(atom.arg),
            score=atom.score,
            properties=_canon_predicate_properties(atom.properties),
        )
    if isinstance(atom, FNot):
        return FNot(canonicalize_atom_for_paraphrase_expansion(atom.arg))
    return atom
