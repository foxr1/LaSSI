import latextools

from LaSSI.structures.extended_fol.Formulae import Formula
from bs4 import Tag

def latex_rendering(strx, mathjax=True, separators=(r"$",r"$")):
    if mathjax:
        return separators[0] + strx + separators[1]
    latex_eq = latextools.render_snippet(
        separators[0] + strx + separators[1],
        commands=[latextools.cmd.all_math])
    svg_eq = latex_eq.as_svg()
    import base64
    encoded = base64.b64encode(str.encode(svg_eq.content))
    svg = 'data:image/svg+xml;base64,{}'.format(encoded.decode())
    img = Tag(name="img")
    img["src"] = svg
    # img["style"] = "width: 10vw; min-width: 50px;"
    return img

def latex_rendering_to_raster_file(strx, file):
    latex_eq = latextools.render_snippet(
        r'$' + str(strx) + '$',
        commands=[latextools.cmd.all_math])
    latex_eq.as_svg().save(file+".svg")
    # from drawsvg.raster import delay_import_cairo, Raster
    # cairosvg = delay_import_cairo()
    # from cairosvg.surface import Surface
    # Surface.device_units_per_user_units = 20
    # cairosvg.svg2png(bytestring=latex_eq.as_svg().content, write_to=file+".png", dpi=1200)
    # return Raster(None, png_file=file+".png")
    latex_eq.rasterize(file+".png")

def latex_formula_rendering(formula:Formula, mathjax=True):
    return latex_rendering(str(formula), mathjax=True)

def practicalInstance(self, class_, name):
    return isinstance(self, class_) or type(self).__name__ == name

def getAtomsWithNegations(self):
    """
    This function decomposes the formula (self) in its atomic constituents, returned as a bag of atoms
    """
    from LaSSI.structures.extended_fol.Formulae import FUnaryPredicate, FBinaryPredicate, FAnd, FNot, FOr
    if practicalInstance(self, FUnaryPredicate, "FUnaryPredicate") or practicalInstance(self, FBinaryPredicate, "FBinaryPredicate"):
        return {self}
    elif practicalInstance(self, FAnd, "FAnd") or practicalInstance(self, FOr, "FOr"):
        s = set()
        for x in self.args:
            s = s.union(getAtomsWithNegations(x))
        return s
    elif practicalInstance(self, FNot, "FNot"):
        return {self}     # getAtoms(self.arg)
    else:
        print("WARNING: cannot perform the atomization of a variable")
        return {}

def getAtoms(self):
    """
    This function decomposes the formula (self) in its atomic constituents, returned as a bag of atoms
    """
    from LaSSI.structures.extended_fol.Formulae import FUnaryPredicate, FBinaryPredicate, FAnd, FNot, FOr
    if practicalInstance(self, FUnaryPredicate, "FUnaryPredicate") or practicalInstance(self, FBinaryPredicate, "FBinaryPredicate"):
        return {self}
    elif practicalInstance(self, FAnd, "FAnd") or practicalInstance(self, FOr, "FOr"):
        s = set()
        for x in self.args:
            s = s.union(getAtomsWithNegations(x))
        return s
    elif practicalInstance(self, FNot, "FNot"):
        return getAtoms(self.arg)
    else:
        print("WARNING: cannot perform the atomization of a variable")
        return {}

from LaSSI.structures.extended_fol.Formulae import *

def _has_atomizable_content(self):
    if practicalInstance(self, FUnaryPredicate, "FUnaryPredicate") or practicalInstance(self, FBinaryPredicate,
                                                                                        "FBinaryPredicate"):
        return True
    if practicalInstance(self, FNot, "FNot"):
        return _has_atomizable_content(self.arg)
    if practicalInstance(self, FAnd, "FAnd") or practicalInstance(self, FOr, "FOr"):
        return any(_has_atomizable_content(x) for x in self.args)
    return False

def semantic(self, d: Dict[Formula, bool]):
    """
    This function decomposes the formula (self) in its atomic constituents, returned as a bag of atoms
    """
    if practicalInstance(self, FUnaryPredicate, "FUnaryPredicate") or practicalInstance(self, FBinaryPredicate,
                                                                                        "FBinaryPredicate"):
        assert self in d
        return d[self]
    elif practicalInstance(self, FAnd, "FAnd"):
        vals = [semantic(x, d) for x in self.args if _has_atomizable_content(x)]
        return min(vals) if vals else 1
    elif practicalInstance(self, FOr, "FOr"):
        vals = [semantic(x, d) for x in self.args if _has_atomizable_content(x)]
        return max(vals) if vals else 0
    elif practicalInstance(self, FNot, "FNot"):
        if self in d:
            return d[self]
        if not _has_atomizable_content(self.arg):
            return 1
        else:
            return 1 - semantic(self.arg, d)
    else:
        print("WARNING: cannot perform the atomization of a variable")
        return 1


def semantic_bdd(self, atom_to_bdd, manager):
    """
    Symbolic counterpart of `semantic`: builds a BDD node representing the
    truth of the formula `self` over the underlying boolean variables.

    `atom_to_bdd` maps atomic Formula objects (FUnaryPredicate, FBinaryPredicate,
    and FNot wrappers that have been pre-resolved) to BDD nodes.  FNot wrappers
    that are not present in the map fall through to a symbolic NOT applied to
    the recursion on `self.arg`, so negation does not require a separate
    boolean variable.
    """
    if practicalInstance(self, FUnaryPredicate, "FUnaryPredicate") or practicalInstance(self, FBinaryPredicate,
                                                                                        "FBinaryPredicate"):
        return atom_to_bdd[self]
    elif practicalInstance(self, FAnd, "FAnd"):
        result = manager.true
        for arg in self.args:
            if not _has_atomizable_content(arg):
                continue
            result = manager.apply('and', result, semantic_bdd(arg, atom_to_bdd, manager))
        return result
    elif practicalInstance(self, FOr, "FOr"):
        result = manager.false
        for arg in self.args:
            if not _has_atomizable_content(arg):
                continue
            result = manager.apply('or', result, semantic_bdd(arg, atom_to_bdd, manager))
        return result
    elif practicalInstance(self, FNot, "FNot"):
        if self in atom_to_bdd:
            return atom_to_bdd[self]
        if not _has_atomizable_content(self.arg):
            return manager.true
        return manager.apply('not', semantic_bdd(self.arg, atom_to_bdd, manager))
    else:
        print("WARNING: cannot perform the atomization of a variable")
        return manager.true
