import json

import yaml


def load_and_html_print(yaml_file, json_file):
    sentences = []
    sentence_list = []
    with open(yaml_file, "r") as f:
        sentences = yaml.safe_load(f)
    assert isinstance(sentences, list)
    with open(json_file, "r") as f:
        from LaSSI.structures.extended_fol.Formulae import formula_from_dict
        sentence_list = [formula_from_dict(x) for x in json.load(f)]
    assert len(sentence_list) == len(sentences)
    from bs4 import Tag, BeautifulSoup
    html = Tag(name="html")
    mathjax = """
                <script type="text/javascript" id="MathJax-script" async
          src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.0.0/es5/latest?tex-mml-chtml.js">
        </script>
        <script>
        MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']]
          }
        };
        </script>
        <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
        </script><script src="//d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/@hpcc-js/wasm@2.20.0/dist/graphviz.umd.js"></script>
    <script src="https://unpkg.com/d3-graphviz@5.6.0/build/d3-graphviz.js"></script>"""
    parser = BeautifulSoup(mathjax)
    for x in list(parser.children):
        html.append(x)
    body = Tag(name="body")
    ol = Tag(name="ol")
    from LaSSI.HOnK.formula_utils import latex_rendering
    for text, formula in zip(sentences, sentence_list):
        li = Tag(name="li")
        li.append(text)
        li.append(latex_rendering(str(formula), separators=("\[", "\]")))
        ol.append(li)
    body.append(ol)
    html.append(body)
    with open("resultNCL2.html", "w") as f:
        f.write(html.prettify())






if __name__ == "__main__":
    load_and_html_print("/home/giacomo/projects/LaSSI/test_sentences/orig/newcastle_mdpi.yaml",
                        "../catabolites/newcastle_mdpi/logical_rewriting.json")