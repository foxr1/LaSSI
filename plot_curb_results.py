import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from plotnine import *

def get_paradigm(model_name):
    model_name = model_name.lower()
    if 'minilm' in model_name or 'mpnet' in model_name:
        if 'nli' in model_name:
            return 'Cross-Encoders'
        return 'Sentence Transformers'
    elif 'deberta' in model_name or 'bge-reranker' in model_name:
        return 'Cross-Encoders'
    elif 'llama' in model_name or 'qwen' in model_name or 'gemma' in model_name:
        return 'Generative LLMs'
    elif 'logical' in model_name:
        return 'LaSSI (Ours)'
    else:
        return 'Other'

def plot_accuracy_by_mod_strategy(df, model_names):
    acc_records = []
    for model in model_names:
        paradigm = get_paradigm(model)
        col = f"{model}_correct"
        if col in df.columns:
            acc_by_mod = df.groupby('claim_type')[col].mean()
            for mod_strategy, acc in acc_by_mod.items():
                acc_records.append({
                    'Paradigm': paradigm,
                    'Model': model,
                    'Modification Strategy': mod_strategy.replace('_', ' ').title(),
                    'Accuracy': acc * 100
                })
    
    if not acc_records:
        print("No accuracy records found to plot.")
        return

    acc_df = pd.DataFrame(acc_records)
    avg_acc_df = acc_df.groupby(['Modification Strategy', 'Paradigm'])['Accuracy'].mean().reset_index()

    p = (
        ggplot(avg_acc_df, aes(x='Modification Strategy', y='Accuracy', fill='Paradigm'))
        + geom_bar(stat='identity', position='dodge')
        + scale_fill_brewer(type='qual', palette='Dark2')
        + labs(title='Accuracy by Modification Strategy across Model Paradigms',
               y='Accuracy (%)', x='Modification Strategy', fill='Model Paradigm')
        + theme_minimal()
        + theme(axis_text_x=element_text(angle=45, hjust=1),
                plot_title=element_text(face='bold', size=14),
                legend_position='right')
    )
    p.save('fig2_accuracy_by_modification.png', dpi=300, width=10, height=6)

def plot_confusion_matrices(df, model_names):
    representative_models = ['all-mpnet-base-v2', 'nli-deberta-v3-base', 'llama3.1:latest', 'Logical']
    present_models = [m for m in representative_models if m in model_names]
    
    if not present_models:
        present_models = sorted(list(model_names))[:4] 

    labels = ["Supported", "Refuted", "Not Enough Evidence"]
    
    if not present_models:
        print("No models available to plot confusion matrices.")
        return

    cm_records = []
    for model in present_models:
        true_labels = df['label'].fillna("Unknown").astype(str).str.title()
        pred_labels = df[f'{model}_pred'].fillna("Unknown").astype(str).str.title()
        
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        for i, actual in enumerate(labels):
            for j, predicted in enumerate(labels):
                cm_records.append({
                    'Model': model,
                    'Actual': actual,
                    'Predicted': predicted,
                    'Count': cm[i, j]
                })
                
    cm_df = pd.DataFrame(cm_records)
    
    # Reverse actual labels for standard heatmap layout
    cm_df['Actual'] = pd.Categorical(cm_df['Actual'], categories=reversed(labels), ordered=True)
    cm_df['Predicted'] = pd.Categorical(cm_df['Predicted'], categories=labels, ordered=True)

    p = (
        ggplot(cm_df, aes(x='Predicted', y='Actual', fill='Count'))
        + geom_tile(color='white')
        + geom_text(aes(label='Count'), color='black', size=10)
        + facet_wrap('~ Model', nrow=1)
        + scale_fill_gradient(low='white', high='#1f77b4')
        + labs(title='Confusion Matrices on CURB Test Split',
               x='Predicted Label', y='Gold Label')
        + theme_minimal()
        + theme(axis_text_x=element_text(angle=45, hjust=1),
                plot_title=element_text(face='bold', size=14),
                legend_position='none')
    )
    p.save('fig1_confusion_matrices.png', dpi=300, width=6 * len(present_models), height=5)

def main():
    try:
        df = pd.read_csv('evaluation_output_compared.csv')
    except FileNotFoundError:
        print("evaluation_output_compared.csv not found. Please run eval_results.py first.")
        return

    model_names = set([col.replace('_pred', '') for col in df.columns if col.endswith('_pred')])
    
    plot_accuracy_by_mod_strategy(df, model_names)
    plot_confusion_matrices(df, model_names)
    print("Successfully generated fig1_confusion_matrices.png and fig2_accuracy_by_modification.png.")

if __name__ == '__main__':
    main()
