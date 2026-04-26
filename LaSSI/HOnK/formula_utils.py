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

def semantic(self, d: Dict[Formula, bool]):
    """
    This function decomposes the formula (self) in its atomic constituents, returned as a bag of atoms
    """
    if practicalInstance(self, FUnaryPredicate, "FUnaryPredicate") or practicalInstance(self, FBinaryPredicate,
                                                                                        "FBinaryPredicate"):
        assert self in d
        return d[self]
    elif practicalInstance(self, FAnd, "FAnd"):
        return min(map(lambda x: semantic(x, d), self.args))
    elif practicalInstance(self, FOr, "FOr"):
        return max(map(lambda x: semantic(x, d), self.args))
    elif practicalInstance(self, FNot, "FNot"):
        if self in d:
            return d[self]
        else:
            return 1 - semantic(self.arg, d)
    else:
        print("WARNING: cannot perform the atomization of a variable")
        return 0