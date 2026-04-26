import edge_mapping
import adjacency_list
from LaSSI.HOnK.conceptnet import parse_conceptnet_file, transitive_closure
import wiktionary_json_extract
import json
import csv
from config import config
from sqlitedict import SqliteDict

def from_db(db_name):
    db = SqliteDict(db_name)
    dict = {}
    for key, value in db.items():
        dict[key] = value
    db.close()
    return dict

def to_db(db_name, dict):
    db = SqliteDict(db_name)
    db.clear()
    for key, value in dict.items():
        db[key] = value
    db.commit()
    db.close()

def generate(conceptnet_path, wiktionary_path, load_from_db=False, test_limit=-1):
    adjacency = from_db("adjacency_list.db") if load_from_db else {}
    clusters = from_db("clusters.db") if load_from_db else {}
    # adjacency, clusters = SqliteDict("adjacency_list.db"), SqliteDict("clusters.db")

    if not load_from_db:
        adjacency_list.process_conceptnet_csv(conceptnet_path, adjacency)

        transitive_closure.build_closures(adjacency)
        transitive_closure.build_clusters(adjacency, clusters)

        to_db("adjacency_list.db", adjacency)
        to_db("clusters.db", clusters)

    def triplet_check(source, edge_label, target):
        return (not (source.startswith("Q") and source[1:].isdigit()) and
                not (target.startswith("Q") and target[1:].isdigit()))

    def modify_triplet(source, edge_label, target):
        source, target = transitive_closure.get_node(clusters, source), transitive_closure.get_node(clusters, target)
        edge_label = edge_mapping.get_edge(edge_label)
        return source.lower().replace(" ", "_"), edge_label, target.lower().replace(" ", "_")

    with open(config["result_file"], "w", encoding="utf-8", newline="") as tsv:
        wr = csv.writer(tsv, delimiter="\t")
        wr.writerow(["source", "relation", "target"])

        print("making result file")
        print("doing conceptnet")
        count = 0
        for triplet in parse_conceptnet_file.get_triplets(conceptnet_path, lang="en"):
            (source, edge_label, target) = triplet
            if not triplet_check(source, edge_label, target): continue
            source, edge, target = modify_triplet(source, edge_label, target)

            #wr.writerow((transitive_closure.get_node(source), edge_mapping.get_edge(relation), transitive_closure.get_node(target)))
            # wr.writerow((transitive_closure.get_node(clusters, source), edge_mapping.get_edge(edge_label),
            #              transitive_closure.get_node(clusters, target)))
            wr.writerow((source, edge, target))

            count += 1
            if count % 100000 == 0: print(count)
            if count == test_limit: break

        print("doing wiktionary")
        count = 0
        with open(wiktionary_path, 'r', encoding="utf-8") as file:
            # wik_json = json.load(file)
            for line in file:
                line = line.strip()
                if not line: continue

                entry = json.loads(line)

                count += 1
                if count % 100000 == 0: print(count)

                generator = wiktionary_json_extract.extract_information(entry, language_code="en")
                if not generator: continue

                for triplet in generator:
                    (source, edge_label, target) = triplet
                    if not triplet_check(source, edge_label, target): continue
                    # source = transitive_closure.get_node(clusters, source.split("#")[0])
                    # target = transitive_closure.get_node(clusters, target.split("#")[0])
                    source, edge, target = modify_triplet(source.split("#")[0], edge_label, target.split("#")[0])

                    # wr.writerow((source, edge_mapping.get_edge(edge_label), target))
                    wr.writerow((source, edge, target))

    edge_mapping.show_non_mapped_labels()
    # if type(adjacency) == SqliteDict:
    #     adjacency.close()
    #
    # if type(clusters) == SqliteDict:
    #     clusters.close()


if __name__ == '__main__':
    generate("./supporting_files/edges.csv", "./supporting_files/wiktionary_data.json", True)