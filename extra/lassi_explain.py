import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LaSSI.LaSSI import LaSSI
from LaSSI.Configuration import SentenceRepresentation
from LaSSI.explainer.LaSSIExplainer import LaSSIExplainer
from LaSSI.explainer.ReportBuilder import ReportBuilder

if __name__ == '__main__':
    ### Setting up the services as per the main LaSSI pipeline
    dataset_name = PROJECT_ROOT / "neet" / "evidence_cases" / "transport_005.yaml"
    fuzzyDBs = PROJECT_ROOT / "connection.yaml"
    LaSSIExplainer.start_up_services(fuzzyDBs)
    ## Running the full pipeline to retrieve the information using the ids.
    pipeline = LaSSI(dataset_name, fuzzyDBs, SentenceRepresentation.Logical, run_ex_post=False, useId=True)
    pipeline.run()
    ## Expanding using the IDs: this will blow up the number of the rules, but it will generate elements
    ## with provenance
    exp = LaSSIExplainer(dataset_name)
    exp.dump_explanation(0, 0)
    exp.dump_explanation(0, 1)
    exp.dump_explanation(1, 0)
