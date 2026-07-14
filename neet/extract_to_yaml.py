import csv
import yaml
import os
from collections import defaultdict

def main():
    csv_path = 'neet/neet_v1.csv'
    out_dir = 'neet/evidence_cases'
    
    os.makedirs(out_dir, exist_ok=True)
    
    cases = defaultdict(dict)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row['item_id']
            if not item_id:
                continue
            
            # e.g. transport_001_S -> base_id = transport_001, suffix = S
            base_id = item_id[:-2]
            suffix = item_id[-1]
            
            if 'evidence' not in cases[base_id]:
                cases[base_id]['evidence'] = row['evidence_text']
                
            if suffix == 'S':
                cases[base_id]['supported'] = row['claim_text']
            elif suffix == 'R':
                cases[base_id]['refuted'] = row['claim_text']
            elif suffix == 'N':
                cases[base_id]['not_enough_evidence'] = row['claim_text']

    for base_id, data in cases.items():
        yaml_path = os.path.join(out_dir, f"{base_id}.yaml")
        
        # Structure as requested: 4 distinct sentences
        yaml_data = [
            data.get('evidence', ''),
            data.get('supported', ''),
            data.get('refuted', ''),
            data.get('not_enough_evidence', '')
        ]
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, width=float("inf"))

if __name__ == '__main__':
    main()
