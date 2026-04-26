import os
import sys
import time

import yaml
from datasets import load_dataset

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.HOnK.HOnKRemote import HOnKRemoteSingleton
from LaSSI.LaSSI import LaSSI
from LaSSI.external_services.utilities.DatabaseConfiguration import load_db_configuration
from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import FuzzyStringMatchDatabase


def get_dataset(filename="test_sentences/commonsense/dataset.yaml", number=5):
    if not os.path.exists(filename):
        try:
            questions = load_dataset("tau/commonsense_qa")['train']['question'][:number]
            with open(filename, 'w') as f:
                yaml.dump(questions, f, default_flow_style=False, width=float('inf'))

        except Exception as e:
            print(e)
            return None

    return filename


if __name__ == '__main__':
    commonsense_qa = False

    if commonsense_qa:
        dataset_name = get_dataset()
    else:
        dataset_name = "test_sentences/smite/first_test.yaml"

    fuzzyDBs_file = "connection.yaml"

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        fuzzyDBs_file = sys.argv[2]

    endpoint = "http://132.226.131.196:7200/repositories/HOnK"
    print(f"Connecting to remote ontology server: {endpoint}")
    t0 = time.time()

    fuzzyDBs = load_db_configuration(fuzzyDBs_file)
    (FuzzyStringMatchDatabase
     .instance()
     .init(fuzzyDBs.db, fuzzyDBs.uname, fuzzyDBs.pw, fuzzyDBs.host, fuzzyDBs.port))
    HOnKRemoteSingleton.init_remote(endpoint)

    print(f"Remote ontology initialization completed in {time.time() - t0:.2f}s")

    pipeline = LaSSI(dataset_name, fuzzyDBs_file, SentenceRepresentation.Logical, disable_fuzzy_honk=False)
    pipeline.run()
    pipeline.close()
