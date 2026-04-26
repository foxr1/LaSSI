from bs4 import Tag, BeautifulSoup
import plotly.express as px

from LaSSI.HOnK.formula_utils import latex_rendering
from LaSSI.structures.extended_fol.TabularCWASemantics import ExplainSentence


# latex_rendering : returns a string for rendering in mathjax

class List:
    def __init__(self, ordered=False):
        self.ool = Tag(name="ol" if ordered else "ul")

    def add(self, item, value=None):
        lli = Tag(name="li")
        if value is not None:
            lli["value"] = value
        lli.append(item)
        self.ool.append(lli)

    def get(self):
        return self.ool


class ReportBuilder:
    def __init__(self, explainer):
        self.html = Tag(name="html")
        self.head = Tag(name="head")
        mathjax = """
                    <script type="text/javascript" id="MathJax-script" async
              src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.0.0/es5/latest?tex-mml-chtml.js">
            </script>
            <script>
            MathJax = {
              tex: {
                inlineMath: [['$', '$']]
              }
            };
            </script>
            <script id="MathJax-script" async
              src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
            </script><script src="//d3js.org/d3.v7.min.js"></script>
        <script src="https://unpkg.com/@hpcc-js/wasm@2.20.0/dist/graphviz.umd.js"></script>
        <script src="https://unpkg.com/d3-graphviz@5.6.0/build/d3-graphviz.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.8/dist/umd/popper.min.js" integrity="sha384-I7E8VVD/ismYTF4hNIPjVp/Zjvgyol6VFvRkX/vR+Vc4jQkC+hVqc2pM8ODewa9r" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.6/dist/js/bootstrap.min.js" integrity="sha384-RuyvpeZCxMJCqVUGFI0Do1mQrods/hhxYlcVfGPOfQtPJh0JCw12tUAZ/Mv10S7D" crossorigin="anonymous"></script>
        """
        parser = BeautifulSoup(mathjax, 'html.parser')
        for x in list(parser.children):
            self.head.append(x)
        title = Tag(name="Title")
        title.append("LaSSI eXplainer")
        self.head.append(title)
        self.body = Tag(name="body")
        self._finalized = False
        from LaSSI.explainer.LaSSIExplainer import LaSSIExplainer
        self.explainer : LaSSIExplainer= explainer

    def build_implication_report(self, i: int, j:int):
        Ri_explanation = self.explainer.f.explain_sentence(i, True)
        self._append_sentence_explanation(i, Ri_explanation)
        Rj_explanation = self.explainer.f.explain_sentence(j, False)
        self._append_sentence_explanation(j, Rj_explanation)
        ConstImplExpl = self.explainer.f.explained_universal_truth(set(self.explainer.f.minimal_constituent_dict[i]),
                                                                   set(self.explainer.f.minimal_constituent_dict[j]))
        relevantColumns = list(set(Ri_explanation.table.columns).union(set(Rj_explanation.table.columns)))
        tableSemantics = ConstImplExpl.result.merge(Ri_explanation.table).merge(Rj_explanation.table)[relevantColumns].drop_duplicates()
        semantics = tableSemantics[["R" + str(j)]].prod(axis=1)
        Rj_holding = len(semantics)
        total = semantics.sum(axis=0)/Rj_holding if Rj_holding>0.0 else 0.0

        results = Tag(name="fieldset")
        results["id"] = f"results"
        results["style"] = """
        font-family: sans-serif; padding-top:10px;
	border:1px solid #666;
	border-radius:8px;
	box-shadow:0 0 10px #666;"""

        sign = ""
        start_decoratror = ""
        final_decorator = ""
        if total == 1.0:
            sign = " ⇒ "
        elif total == 0.0:
            start_decoratror = f"{i} ∧ ("
            sign = " ⇒ ¬ "
            final_decorator = ")"
        else:
            sign = " ? "

        legend = Tag(name="legend")
        legend["style"] = """    background: #1F497D;
        color: #fff;
        padding: 5px 10px ;
        border-radius: 5px;
        box-shadow: 0 0 0 5px #ddd;
        margin-left: 20px;"""
        legend.append(f"{start_decoratror} {i} {sign} {j} {final_decorator}\t")
        results.append(legend)

        ## Confidence value
        results.append(latex_rendering(f"\\mathtt{{confidence}}({i},{j})= {total}"))

        ## Constituent Motivation
        results.append(Tag(name="br"))
        results.append("Atom Motivation:")
        const_mot = List(False)
        for partial_result in ConstImplExpl.constituent_implication:
            const_mot.add(str(partial_result))
        results.append(const_mot.get())

        ## Tabular Intermediate Representation
        results.append("Possible World Combination:")
        results.append(self.style_local_table(tableSemantics, i, j))
        self.body.append(results)
        results.append(f"Possible Atom-Compatible Worlds: {Rj_holding}")
        results.append(Tag(name="br"))
        results.append(f"Worlds where consequence might world: {semantics.sum(axis=0)}")
        results.append(Tag(name="br"))


        # print(f"{i}~{j} := {total}")


    def _append_sentence_explanation(self, idx, tabular_explanation: ExplainSentence=None):
        self.body.append(self._textual_explanation(idx, self.explainer.explain_textual_sentence(idx), tabular_explanation))

    def style_local_table(self, df, *idx):
        new_columns = [None] * len(df.columns)
        for idx_, x in enumerate(df.columns):
            if x.startswith("R"):
                new_columns[idx_] = f"Sentence {x[1:]}"
            else:
                new_columns[idx_] = f"Atom {x}"
        display_df = df.rename(columns=dict(zip(df.columns,new_columns)))
        display_df.index.name = "Worlds"
        display_df.columns.name = "Truth"
        slice_ = [f'Sentence {x}' for x in idx]
        seen = set()
        slice_ = [x for x in slice_ if x not in seen and not seen.add(x)]
        obj = display_df[[x for x in display_df.columns if not x in slice_] + slice_].style.set_table_styles([
            {
                'selector': 'th:not(.index_name)',
                'props': 'background-color: #000066; color: white;'
            },
    {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'},
            {'selector':'table', 'props': "border-spacing:0; border-collapse: collapse;"}
], overwrite=False).set_properties(**{'background-color': '#ffffb3'}, subset=slice_)
        table_rendering = BeautifulSoup(obj.to_html())
        self.head.append(table_rendering.style)
        return table_rendering.table

    def _textual_explanation(self, idx, obj, tabular_explanation: ExplainSentence=None):
        from LaSSI.explainer.LaSSIExplainer import Provenance
        sentence = Tag(name="fieldset")
        sentence["id"] = f"sentence_{idx}"
        sentence["style"] = """
        font-family: sans-serif; padding-top:10px;
	border:1px solid #666;
	border-radius:8px;
	box-shadow:0 0 10px #666;"""
        # p = Tag(name="p")
        legend = Tag(name="legend")
        legend["style"] = """    background: #1F497D;
    color: #fff;
    padding: 5px 10px ;
    border-radius: 5px;
    box-shadow: 0 0 0 5px #ddd;
    margin-left: 20px;"""
        legend.append(f"Sentence №{idx}\t")
        sentence.append(legend)
        colour_index = 0
        colour_map = dict()
        for x in obj:
            if isinstance(x, str):
                sentence.append(x)
            else:
                for y in x:
                    if y.id not in colour_map:
                        colour = px.colors.qualitative.Plotly[colour_index]
                        colour_index+=1
                        colour_map[y.id] = colour
                    else:
                        colour = colour_map[y.id]
                    mark = Tag(name="mark")
                    mark["style"] = f"background: {colour}!important"
                    mark.append(y.value)
                    if y.id is not None:
                        sup = Tag(name="sup")
                        sup.append(str(y.id))
                        mark.append(sup)
                    sentence.append(mark)
        sentence.append(Tag(name="br"))
        sentence.append(latex_rendering(str(self.explainer.obj_list[idx])))
        if tabular_explanation is not None:
            sentence.append(Tag(name="br"))
            ## Adding atoms
            sentence.append("Atoms:")
            atom_list = List(True)
            for idx_, formula in tabular_explanation.atoms.items():
                atom_list.add(latex_rendering(str(formula)), idx_)
            sentence.append(atom_list.get())
            ## Adding the table
            sentence.append("Tabular Semantics:")
            sentence.append(self.style_local_table(tabular_explanation.table, idx))
        return sentence

    def finalize_document(self, filename):
        if not self._finalized:
            self.html.append(self.head)
            self.html.append(self.body)
            self.html.decode()
        with open(filename, "w") as f:
            f.write(self.html.prettify())