from sqlitedict import SqliteDict

from LaSSI.Parmenides.conceptnet.parse_conceptnet_file import CompactRelation
import csv

def process_conceptnet_csv(csv_file, db):
    with open(csv_file, "r", encoding="utf-8") as tsv:
        edge_label_counts = {}
        english_count = 0
        rels = []

        print("doing adjacency list initialisation")
        count = 1
        for row in csv.reader(tsv, dialect="excel-tab"):
            r = CompactRelation(row)
            if r.rel != "ExternalURL" or (r.lang != "en"): continue ## gives a count of almost 420,000
            # if r.rel != "ExternalURL" or (r.langEnd != "en"): continue ## has a count of 0

            count += 1
            rels.append(r)
            if count % 10000 == 0:
                print(count)
                # print(r.rel)
                # #print(r.langStart)
                # #print(r.langEnd)
                # print(r.lang)
                # print(r.surfaceStart)
                # print(r.surfaceEnd)

            # concept_word = concept_row.split("/")[-1]
            # wiktionary_word = wiktionary_url_row.split("/")[5]

            # concept_key = f"con:{concept_word}"
            # wiktionary_key = f"wik:{wiktionary_word}"
            db[r.surfaceStart] = [r.surfaceEnd]

            # many conceptnet nodes could be linked to the same wiktionary node so i have to be appending to its adjacency list
            wik_list = db.get(r.surfaceEnd, [])
            wik_list.append(r.surfaceStart)
            db[r.surfaceEnd] = wik_list

    if type(db)==SqliteDict:
        db.commit()