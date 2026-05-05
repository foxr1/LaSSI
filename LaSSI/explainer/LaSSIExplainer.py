import json
import os
from collections import defaultdict
from typing import Optional

import dash
import yaml
from intervaltree import IntervalTree, Interval

from LaSSI.Configuration import SentenceRepresentation

from LaSSI.LaSSI import LaSSI

from LaSSI.HOnK.HOnK import HOnKSingleton
from LaSSI.external_services.Services import Services
from dash import dcc, html
from pathlib import Path

from LaSSI.structures.extended_fol.Formulae import formula_from_dict

app = dash.Dash()
app.layout = html.Div([
    dcc.Markdown('$Area (m^{2})$', mathjax=True),
])
from dataclasses import dataclass

@dataclass(eq=True, frozen=True)
class Provenance:
    min: int
    max: int
    value: str
    id : Optional[int] = None

    def withId(self, idx):
        return Provenance(self.min, self.max, self.value, idx)

class LaSSIExplainer:
    fuzzyDBs = None

    @staticmethod
    def start_up_services(fuzzyDBs):
        if LaSSIExplainer.fuzzyDBs is None:
            Services.getInstance(lambda x: print(x))
            HOnKSingleton.instance()
            from LaSSI.external_services.utilities.DatabaseConfiguration import load_db_configuration
            LaSSIExplainer.fuzzyDBs = load_db_configuration(fuzzyDBs)
            HOnKSingleton.init("catabolites", LaSSIExplainer.fuzzyDBs.uname, LaSSIExplainer.fuzzyDBs.pw,
                                     LaSSIExplainer.fuzzyDBs.host, LaSSIExplainer.fuzzyDBs.port, False, "LaSSI/HOnK.ttl",
                                     rules_path="raw_data/logical_analysis.json")

    def __init__(self, dataset_name):
        from pathlib import Path
        from LaSSI.Configuration import SentenceRepresentation
        self.sentences = []
        self.provenance = None
        with open(dataset_name) as file:
            self.sentences = yaml.load(file, Loader=yaml.SafeLoader)
        self.full_transformation = SentenceRepresentation.Logical
        self.catabolites_dir = Path(dataset_name).stem
        self.catabolites_of_dataset = os.path.join("catabolites", self.catabolites_dir)
        self.logical_formulae_of_dataset = os.path.join(self.catabolites_of_dataset, "logical_rewriting.json")
        self.internals = os.path.join(self.catabolites_of_dataset, "internals.json")
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        TBoxReasoningSingleton.instance()
        # TODO: move the txt files to the resources
        if not os.path.exists(os.path.join(self.catabolites_of_dataset, str(self.full_transformation))):
            from pathlib import Path
            Path(os.path.join(self.catabolites_of_dataset, str(self.full_transformation))).mkdir(parents=True,
                                                                                                 exist_ok=True)
        kexp_pickle = os.path.join(self.catabolites_of_dataset, str(self.full_transformation), "_kexp.pickle")
        TBoxReasoningSingleton.init("query_impl.txt",
                                    "query_eq.txt",
                                    kexp_pickle)

        self.obj_list = []
        with open(self.logical_formulae_of_dataset) as file:
            self.obj_list = formula_from_dict(json.load(file), True)

        ## Get Main Matched Entities
        with open(self.internals) as f:
            json_data = json.load(f)
        from LaSSI.explainer.reconstruct_provenance import VisitSentence
        self.provenance = [None] * len(json_data)
        self.invProvenance = [None] * len(json_data)
        for idx, sentence in enumerate(json_data):
            s = VisitSentence()
            s.visit_sentence(sentence)
            self.provenance[idx] = {k: {Provenance(x[0], x[1], self.sentences[idx][x[0]:x[1]]) for x in v} for k, v in s.d.items()}
            Sinv = defaultdict(set)
            Sv = defaultdict(lambda: defaultdict(set))
            found_provenances = set()
            for prov in self.obj_list[idx].extract_provenance():
                if prov.id not in self.provenance[idx]:
                    continue
                found_provenances.add(prov.id)
                for x in self.provenance[idx][prov.id]:
                    if prov.name is not None and prov.name in x.value:
                        key ="name"
                        s = prov.name.strip()
                        if s == x.value:
                            Sv[prov.id][key].add(x)
                            Sinv[(x.min,x.max)].add(x.withId(prov.id))
                            continue
                        else:
                            idxW = x.value.find(s)
                            if (idxW != -1):
                                min = x.min+idxW
                                max = x.min+idxW+len(s)
                                w = Provenance(min, max, s)
                                Sv[prov.id][key].add(w)
                                Sinv[(w.min,w.max)].add(w.withId(prov.id))
                    if prov.specification is not None and prov.specification in x.value:
                        s = prov.specification.strip()
                        key ="specification"
                        if s == x.value:
                            Sv[prov.id][key].add(x)
                            Sinv[(x.min, x.max)].add(x.withId(prov.id))
                        else:
                            idxW = x.value.find(s)
                            if (idxW != -1):
                                min = x.min+idxW
                                max = x.min+idxW+len(s)
                                w = Provenance(min, max, s)
                                Sv[prov.id][key].add(w)
                                Sinv[(w.min, w.max)].add(w.withId(prov.id))
            for k, v in self.provenance[idx].items():
                if k not in found_provenances:
                    for w in v:
                        Sv[w.id]["name"].add(w)
                        Sinv[(w.min, w.max)].add(w)
            self.provenance[idx] = Sv
            self.invProvenance[idx] = Sinv

        self.obj_list_noid = []
        with open(self.logical_formulae_of_dataset) as file:
            self.obj_list_noid = formula_from_dict(json.load(file), False)
        from LaSSI.structures.extended_fol.TabularCWASemantics import TabularCWASemantics
        self.f = TabularCWASemantics(self.obj_list_noid,
                                         os.path.join(self.catabolites_of_dataset, str(self.full_transformation)))
        TBoxReasoningSingleton.instance().dump()

    def dump_explanation(self, i:int, j:int):
        from LaSSI.explainer.ReportBuilder import ReportBuilder
        rb = ReportBuilder(self)
        rb.build_implication_report(i, j)
        rb.finalize_document(f"explain_{i}_{j}.html")

    def explain_textual_sentence(self, idx)->list[str|Provenance]:
        sentence = self.sentences[idx]
        prov = self.invProvenance[idx]
        t = IntervalTree([Interval(0, len(sentence))])
        add_intervals = []
        for key in prov:
            t.chop(key[0], key[1])
            add_intervals.append(Interval(key[0], key[1]))
        t = list(t) + add_intervals
        t.sort(key=lambda i: i.begin)
        tree = IntervalTree(t)
        tree.split_overlaps()
        t = list(tree)
        t.sort(key=lambda i: i.begin)
        result = [None] * len(t)
        for idx, x in enumerate(t):
            if (x.begin,x.end) in prov:
                result[idx]= prov[(x.begin,x.end)]
            else:
                result[idx]= sentence[x.begin:x.end]
        return result

    def get_explanation(self, i:int, j:int):
        self.f.get_explained_id_similarity(i,j)


