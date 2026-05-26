import json
import os
import pandas as pd

def get_label_logical(v):
    if v == 1.0:
        return "Supported"
    elif v == 0.0:
        return "Refuted"
    elif v is None:
        return "Not Enough Evidence"
    elif 0.0 < v < 1.0:
        return "Not Enough Evidence"
    else:
        return "Unknown"

def get_label_transformer(v):
    if v is None:
        return "Unknown"
    if v < 0.2:
        return "Refuted"
    elif v > 0.8:
        return "Supported"
    else:
        return "Not Enough Evidence"

def main():
    csv_path = 'neet/neet_v1.csv'
    df = pd.read_csv(csv_path)

    catabolites_dir = 'catabolites'
    results = []
    suffix_map = {1: 'S', 2: 'R', 3: 'N'}

    if not os.path.exists(catabolites_dir):
        print("catabolites dir not found")
        return

    # To store dynamic models found
    model_names = set()
    
    for test_dir in os.listdir(catabolites_dir):
        test_path = os.path.join(catabolites_dir, test_dir)
        if not os.path.isdir(test_path):
            continue
            
        matrices = {}
        for f in os.listdir(test_path):
            if f.startswith('confusion_matrices_') and f.endswith('.json'):
                model_name = f[len('confusion_matrices_'):-len('.json')]
                model_names.add(model_name)
                with open(os.path.join(test_path, f), 'r') as file:
                    matrices[model_name] = json.load(file)
                    
        for c in [1, 2, 3]:
            item_id = f"{test_dir}_{suffix_map[c]}"
            row_data = {'item_id': item_id}
            
            for model_name in matrices:
                matrix = matrices[model_name]
                if matrix and c < len(matrix) and 0 < len(matrix[c]):
                    v = matrix[c][0]
                    row_data[f'{model_name}_value'] = v
                    if model_name == "Logical":
                        row_data[f'{model_name}_pred'] = get_label_logical(v)
                    else:
                        row_data[f'{model_name}_pred'] = get_label_transformer(v)
                else:
                    row_data[f'{model_name}_value'] = None
                    row_data[f'{model_name}_pred'] = "Unknown"
            
            results.append(row_data)

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No results found.")
        return
        
    merged = df.merge(res_df, on='item_id', how='inner')
    
    for model_name in model_names:
        if f'{model_name}_pred' in merged.columns:
            merged[f'{model_name}_correct'] = merged['label'] == merged[f'{model_name}_pred']

    for model_name in sorted(list(model_names)):
        col = f'{model_name}_correct'
        if col in merged.columns:
            print(f"Overall Accuracy ({model_name}):", merged[col].mean())
        
    for model_name in sorted(list(model_names)):
        col = f'{model_name}_correct'
        if col in merged.columns:
            print(f"\nAccuracy per domain ({model_name}):")
            print(merged.groupby('domain')[col].mean().to_string())
        
    for model_name in sorted(list(model_names)):
        col = f'{model_name}_correct'
        if col in merged.columns:
            print(f"\nAccuracy per claim_type ({model_name}):")
            print(merged.groupby('claim_type')[col].mean().to_string())
        
    print("\nDetailed results table:")
    cols = ['item_id', 'domain', 'claim_type', 'label']
    for model_name in sorted(list(model_names)):
        cols.extend([f'{model_name}_pred', f'{model_name}_correct', f'{model_name}_value'])
    
    # Filter cols to only those that exist
    cols = [c for c in cols if c in merged.columns]
    print(merged[cols].to_string())

    merged.to_csv('evaluation_output_compared.csv', index=False)

if __name__ == '__main__':
    main()
