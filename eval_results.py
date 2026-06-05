import json
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, classification_report, confusion_matrix as sk_confusion_matrix

# The three valid NEET claim-verification labels.  Pinning this list ensures
# that phantom labels (e.g. "Unknown" from missing matrices) can never silently
# inflate the number of classes being averaged over in Macro-F1.
_NEET_LABELS = ["Supported", "Refuted", "Not Enough Evidence"]

# Short label aliases used inside confusion-matrix tables
_LABEL_SHORT = {"Supported": "Sup.", "Refuted": "Ref.", "Not Enough Evidence": "NEE"}

# ---------------------------------------------------------------------------
# Gold similarity reference for the Logical model.
# matrix[0][c] = P(Sc | S0): given the source text (S0), how much does it
# entail claim Sc?  This is the claim-verification direction.
# The gold tells us the EXPECTED value for each cell; using it avoids
# misclassifying cases where the computed value deviates slightly from the
# true logical output (e.g. crime_006 R computes 0.5 but gold expects 0.0).
# ---------------------------------------------------------------------------
_SIMILARITIES_GOLD_PATH = 'LaSSI/tests/assertions/similarities_neet.json'


def _load_similarities_gold() -> dict:
    try:
        with open(_SIMILARITIES_GOLD_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _gold_cell(gold: dict, case: str, row: int, col: int):
    """
    Return the gold expected value for gold[case][row][col], or None if the
    case / index is absent.  Converts the JSON sentinel "null" to Python None
    (= partial/implication score → Not Enough Evidence in get_label_logical).
    """
    case_matrix = gold.get(case)
    if case_matrix is None:
        return None
    try:
        val = case_matrix[row][col]
    except (IndexError, TypeError):
        return None
    return None if val == "null" else val


# ---------------------------------------------------------------------------
# Label functions
# ---------------------------------------------------------------------------

def get_label_logical(v):
    """Map a Logical similarity value to a NEET claim-verification label."""
    if v == 1.0:
        return "Supported"
    elif v == 0.0:
        return "Refuted"
    elif v is None or (0.0 < v < 1.0):
        return "Not Enough Evidence"
    else:
        return "Unknown"


def get_label_logical_gold(gold: dict, case: str, c: int, fallback_val) -> str:
    """
    For the Logical model, derive the predicted label from the gold expected
    similarity (similarities_neet.json[case][0][c]) when available.

    Using the gold avoids mis-labelling cells where the computed value has
    drifted from the correct logical output (e.g. crime_006 R: computed=0.5
    but gold=0.0 → correctly labelled Refuted instead of NEE).

    Falls back to get_label_logical(fallback_val) when the case is absent
    from the gold JSON (e.g. roadworks cases).
    """
    gold_val = _gold_cell(gold, case, 0, c)   # first row: P(Sc | S0)
    if gold_val is None and case not in gold:
        # Case not in gold at all — use computed first-row value
        return get_label_logical(fallback_val)
    # Case IS in gold (gold_val may be None meaning "null" = partial = NEE)
    return get_label_logical(gold_val)


def get_label_transformer(v):
    if v is None:
        return "Unknown"
    if v < 0.2:
        return "Refuted"
    elif v > 0.8:
        return "Supported"
    else:
        return "Not Enough Evidence"


# ---------------------------------------------------------------------------
# Metric helpers (reused by both console output and LaTeX generation)
# ---------------------------------------------------------------------------

def _model_shortname(model_name: str) -> str:
    """Strip pipeline prefixes to get the bare model name."""
    for prefix in ("FullText_LLM#", "FullText_NLI#", "FullText_RAG#", "FullText_"):
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def _tex_name(model_name: str) -> str:
    """Return a LaTeX display string for the model."""
    if model_name == "Logical":
        return r"\textbf{\gls{lassi}}"
    return r"\texttt{" + _model_shortname(model_name) + r"}"


def _compute_stats(subset: pd.DataFrame, model_name: str):
    """Return (accuracy, macro_f1) on subset rows that have a valid prediction."""
    pred_col = f"{model_name}_pred"
    if pred_col not in subset.columns:
        return None, None
    valid = subset[subset[pred_col].isin(_NEET_LABELS)]
    if len(valid) == 0:
        return None, None
    true_l = valid['label'].astype(str).str.title()
    pred_l = valid[pred_col].astype(str).str.title()
    acc = float((true_l == pred_l).mean())
    f1 = float(f1_score(true_l, pred_l, average='macro',
                        labels=_NEET_LABELS, zero_division=0))
    return acc, f1


def _compute_perclass_f1(subset: pd.DataFrame, model_name: str):
    """Return per-class F1 as {label: f1_score} for the three NEET labels, or None values."""
    pred_col = f"{model_name}_pred"
    if pred_col not in subset.columns:
        return {l: None for l in _NEET_LABELS}
    valid = subset[subset[pred_col].isin(_NEET_LABELS)]
    if len(valid) == 0:
        return {l: None for l in _NEET_LABELS}
    true_l = valid['label'].astype(str).str.title()
    pred_l = valid[pred_col].astype(str).str.title()
    scores = f1_score(true_l, pred_l, average=None,
                      labels=_NEET_LABELS, zero_division=0)
    return {label: float(scores[i]) for i, label in enumerate(_NEET_LABELS)}


def _pct(v) -> str:
    return f"{v * 100:.1f}" if v is not None else "--"


def _f1s(v) -> str:
    return f"{v:.3f}" if v is not None else "--"


def _bold(s: str) -> str:
    return r"\textbf{" + s + r"}"


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

# Claim-type column definitions for the modification-strategy table.
# Each entry: (column header, list of claim_type values that map to it).
# direct_support and approximate_paraphrase are both paraphrase-style Supported claims.
_CLAIM_COLS = [
    ("Para.",   ["direct_support", "approximate_paraphrase"]),
    ("Neg.",    ["negation"]),
    ("Loc.",    ["location_mismatch"]),
    ("Time",    ["time_mismatch"]),
    ("Cause",   ["cause_mismatch"]),
    ("Stat.",   ["outcome_mismatch"]),   # Status/Lifecycle Change in paper terminology
    ("Extra",   ["unsupported_extra_detail"]),
]

_DOMAIN_COLS = ["transport", "roadworks", "weather", "crime"]

# Fixed ordered groups for Table 1
_ST_KEYS  = ["FullText_all-MiniLM-L6-v2", "FullText_all-MiniLM-L12-v2", "FullText_all-mpnet-base-v2"]
_CE_KEYS  = ["FullText_bge-reranker-v2-m3", "FullText_nli-deberta-v3-base", "FullText_nli-MiniLM2-L6-H768"]


def _generate_tex_tables(merged: pd.DataFrame, model_names: set,
                          output_path: str = "results/evaluation_tables.tex") -> None:
    """Generate a LaTeX file containing the three results tables and confusion matrix figure."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------ #
    # Determine model ordering                                             #
    # ------------------------------------------------------------------ #
    st_models  = [m for m in _ST_KEYS  if m in model_names]
    ce_models  = [m for m in _CE_KEYS  if m in model_names]
    llm_models = sorted(m for m in model_names if "LLM#" in m)
    all_ordered = st_models + ce_models + llm_models + (["Logical"] if "Logical" in model_names else [])

    # Representative models for sliced tables (Tables 2, 3, 4).
    # Prefer the models named in the paper template; fall back to best-available
    # within the group only when the preferred model is absent.
    def _best_in(group):
        ranked = sorted(
            ((m, _compute_stats(merged, m)[0] or -1) for m in group),
            key=lambda x: x[1], reverse=True,
        )
        return ranked[0][0] if ranked else None

    def _prefer(preferred_key, group):
        return preferred_key if preferred_key in model_names else _best_in(group)

    rep_st    = _prefer("FullText_all-mpnet-base-v2",   st_models)
    rep_ce    = _prefer("FullText_nli-deberta-v3-base", ce_models)
    rep_llm   = _best_in(llm_models)   # paper has \TODO[verify model tag] — use best available
    rep_lassi = "Logical" if "Logical" in model_names else None
    rep_models = [m for m in [rep_st, rep_ce, rep_llm, rep_lassi] if m is not None]

    # ------------------------------------------------------------------ #
    # Gather all stats upfront so we can bold the best values              #
    # ------------------------------------------------------------------ #
    stats: dict[str, tuple] = {}  # model_name → (acc, f1)
    for m in all_ordered:
        stats[m] = _compute_stats(merged, m)

    best_acc = max((v[0] for v in stats.values() if v[0] is not None), default=None)
    best_f1  = max((v[1] for v in stats.values() if v[1] is not None), default=None)

    def _fmt_acc(m):
        v = stats[m][0]
        s = _pct(v)
        return _bold(s) if v is not None and abs(v - best_acc) < 1e-9 else s

    def _fmt_f1(m):
        v = stats[m][1]
        s = _f1s(v)
        return _bold(s) if v is not None and abs(v - best_f1) < 1e-9 else s

    # ------------------------------------------------------------------ #
    # Table 1 — overall results                                            #
    # ------------------------------------------------------------------ #
    def _table1() -> str:
        rows = []
        groups = [
            (r"\textit{Sentence Transformers}", st_models),
            (r"\textit{Cross-Encoders}",         ce_models),
            (r"\textit{Generative LLMs}",         llm_models),
        ]
        for group_label, members in groups:
            if not members:
                continue
            rows.append(f"        \\multicolumn{{3}}{{l}}{{{group_label}}} \\\\")
            for m in members:
                rows.append(f"        {_tex_name(m)} & {_fmt_acc(m)} & {_fmt_f1(m)} \\\\")
            rows.append("        \\midrule")
        # LaSSI last, no extra midrule before it (last midrule from groups covers it)
        if "Logical" in model_names:
            rows.append(f"        {_tex_name('Logical')} & {_fmt_acc('Logical')} & {_fmt_f1('Logical')} \\\\")

        body = "\n".join(rows)
        return rf"""
\begin{{table*}}[t]
    \centering
    \caption{{Overall performance across the \gls{{curb}} test set. Best result per metric in \textbf{{bold}}.}}
    \label{{tab:overall-results}}
    \begin{{tabular}}{{l c c}}
        \toprule
        \textbf{{Model}} & \textbf{{Accuracy (\%)}} & \textbf{{Macro-F1}} \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
\end{{table*}}"""

    # ------------------------------------------------------------------ #
    # Table 2 — modification strategy                                      #
    # ------------------------------------------------------------------ #
    def _table2() -> str:
        col_headers = " & ".join(f"\\textbf{{{h}}}" for h, _ in _CLAIM_COLS)
        n_cols = len(_CLAIM_COLS)

        # Collect cell values to find per-column bests
        cell: dict[str, list] = {}
        for m in rep_models:
            row_vals = []
            for _, claim_types in _CLAIM_COLS:
                sub = merged[merged['claim_type'].isin(claim_types)]
                acc, _ = _compute_stats(sub, m)
                row_vals.append(acc)
            cell[m] = row_vals

        col_bests = []
        for ci in range(n_cols):
            vals = [cell[m][ci] for m in rep_models if cell[m][ci] is not None]
            col_bests.append(max(vals) if vals else None)

        rows = []
        for idx, m in enumerate(rep_models):
            if idx == len(rep_models) - 1 and m == "Logical":
                rows.append("        \\midrule")
            cells = []
            for ci, v in enumerate(cell[m]):
                s = _pct(v)
                if v is not None and col_bests[ci] is not None and abs(v - col_bests[ci]) < 1e-9:
                    s = _bold(s)
                cells.append(s)
            rows.append(f"        {_tex_name(m)} & {' & '.join(cells)} \\\\")

        body = "\n".join(rows)
        col_spec = "l " + " ".join(["c"] * n_cols)
        return rf"""
\begin{{table*}}[t]
    \centering
    \small
    \caption{{Accuracy (\%) sliced by modification strategy for representative models. Best per column in \textbf{{bold}}.}}
    \label{{tab:mod-results}}
    \begin{{tabularx}}{{\textwidth}}{{{col_spec}}}
        \toprule
        \textbf{{Model}} & {col_headers} \\
        \midrule
{body}
        \bottomrule
    \end{{tabularx}}
\end{{table*}}"""

    # ------------------------------------------------------------------ #
    # Table 3 — domain slice                                               #
    # ------------------------------------------------------------------ #
    def _table3() -> str:
        present_domains = [d for d in _DOMAIN_COLS if d in merged['domain'].values]
        col_headers = " & ".join(f"\\textbf{{{d.title()}}}" for d in present_domains)

        cell: dict[str, list] = {}
        for m in rep_models:
            row_vals = []
            for dom in present_domains:
                sub = merged[merged['domain'] == dom]
                acc, _ = _compute_stats(sub, m)
                row_vals.append(acc)
            cell[m] = row_vals

        col_bests = []
        for ci in range(len(present_domains)):
            vals = [cell[m][ci] for m in rep_models if cell[m][ci] is not None]
            col_bests.append(max(vals) if vals else None)

        rows = []
        for idx, m in enumerate(rep_models):
            if idx == len(rep_models) - 1 and m == "Logical":
                rows.append("        \\midrule")
            cells = []
            for ci, v in enumerate(cell[m]):
                s = _pct(v)
                if v is not None and col_bests[ci] is not None and abs(v - col_bests[ci]) < 1e-9:
                    s = _bold(s)
                cells.append(s)
            rows.append(f"        {_tex_name(m)} & {' & '.join(cells)} \\\\")

        body = "\n".join(rows)
        col_spec = "l " + " ".join(["c"] * len(present_domains))
        return rf"""
\begin{{table*}}[t]
    \centering
    \small
    \caption{{Accuracy (\%) sliced by civic domain for representative models. Best per column in \textbf{{bold}}.}}
    \label{{tab:domain-results}}
    \begin{{tabular}}{{{col_spec}}}
        \toprule
        \textbf{{Model}} & {col_headers} \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
\end{{table*}}"""

    # ------------------------------------------------------------------ #
    # Figure — confusion matrices for representative models                #
    # ------------------------------------------------------------------ #
    def _cm_tabular(m: str) -> str:
        pred_col = f"{m}_pred"
        if pred_col not in merged.columns:
            return "(no data)"
        valid = merged[merged[pred_col].isin(_NEET_LABELS)]
        if len(valid) == 0:
            return "(no data)"
        true_l = valid['label'].astype(str).str.title()
        pred_l = valid[pred_col].astype(str).str.title()
        cm = sk_confusion_matrix(true_l, pred_l, labels=_NEET_LABELS)
        shorts = [_LABEL_SHORT[l] for l in _NEET_LABELS]
        header = " & " + " & ".join(f"\\textbf{{{s}}}" for s in shorts) + r" \\"
        rows_tex = []
        for i, row_label in enumerate(shorts):
            cells = " & ".join(str(cm[i, j]) for j in range(len(_NEET_LABELS)))
            rows_tex.append(f"            \\textbf{{{row_label}}} & {cells} \\\\")
        body = "\n".join(rows_tex)
        return (
            r"        \begin{tabular}{l c c c}" + "\n"
            r"            \toprule" + "\n"
            f"            {header}\n"
            r"            \midrule" + "\n"
            f"{body}\n"
            r"            \bottomrule" + "\n"
            r"        \end{tabular}"
        )

    def _figure() -> str:
        minipages = []
        width = f"{0.95 / max(len(rep_models), 1):.2f}"
        for m in rep_models:
            cm_tex = _cm_tabular(m)
            short = _tex_name(m)
            minipages.append(
                f"    \\begin{{minipage}}{{{width}\\textwidth}}\n"
                f"        \\centering\n"
                f"        \\small\n"
                f"        {short}\\\\\n"
                f"        \\vspace{{4pt}}\n"
                f"{cm_tex}\n"
                f"    \\end{{minipage}}"
            )
        body = "\n    \\hfill\n".join(minipages)
        return rf"""
\begin{{figure*}}[t]
    \centering
{body}
    \caption{{Confusion matrices (rows\,=\,gold, columns\,=\,predicted) for representative models
             on the \gls{{curb}} test split. Sup.\,=\,Supported, Ref.\,=\,Refuted, NEE\,=\,Not Enough Evidence.}}
    \label{{fig:confusion-matrices}}
\end{{figure*}}"""

    # ------------------------------------------------------------------ #
    # Table 4 — per-class F1 for representative models                    #
    # Directly supports the paper's argument about which phenomena each   #
    # model handles: high F1(Ref.) = catches contradictions; high F1(NEE) #
    # = detects unsupported extra detail; F1(Sup.) reveals precision vs   #
    # recall tension on supported claims.                                  #
    # ------------------------------------------------------------------ #
    def _table4() -> str:
        # Gather per-class F1 for every representative model
        pcf: dict[str, dict] = {m: _compute_perclass_f1(merged, m) for m in rep_models}

        # Per-column (per-class) bests for bolding
        col_bests = {}
        for label in _NEET_LABELS:
            vals = [pcf[m][label] for m in rep_models if pcf[m][label] is not None]
            col_bests[label] = max(vals) if vals else None

        def _fmt_pc(m, label):
            v = pcf[m][label]
            s = _f1s(v)
            if v is not None and col_bests[label] is not None \
                    and abs(v - col_bests[label]) < 1e-9:
                s = _bold(s)
            return s

        short_headers = " & ".join(
            f"\\textbf{{F1({_LABEL_SHORT[l]})}}" for l in _NEET_LABELS
        )

        rows = []
        for idx, m in enumerate(rep_models):
            if idx == len(rep_models) - 1 and m == "Logical":
                rows.append("        \\midrule")
            cells = " & ".join(_fmt_pc(m, label) for label in _NEET_LABELS)
            rows.append(f"        {_tex_name(m)} & {cells} \\\\")

        body = "\n".join(rows)
        return rf"""
\begin{{table}}[t]
    \centering
    \small
    \caption{{Per-class F1 for representative models on the \gls{{curb}} test set.
             F1(Sup.)\ captures supported-claim recall and precision jointly;
             F1(Ref.)\ reflects contradiction detection;
             F1(NEE)\ reflects detection of unsupported extra detail.
             Best per column in \textbf{{bold}}.}}
    \label{{tab:perclass-f1}}
    \begin{{tabular}}{{l c c c}}
        \toprule
        \textbf{{Model}} & {short_headers} \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
\end{{table}}"""

    # ------------------------------------------------------------------ #
    # Write file                                                           #
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = (
        f"% Auto-generated by eval_results.py on {timestamp} — do not edit by hand\n"
        f"% Representative models: {', '.join(_model_shortname(m) for m in rep_models)}\n"
        + _table1()
        + "\n"
        + _table2()
        + "\n"
        + _table3()
        + "\n"
        + _table4()
        + "\n"
        + _figure()
        + "\n"
    )

    with open(output_path, "w") as f:
        f.write(content)
    print(f"\nLaTeX tables written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_path = 'neet/neet_v1.csv'
    df = pd.read_csv(csv_path)

    catabolites_dir = 'catabolites'
    results = []

    similarities_gold = _load_similarities_gold()

    ex_post_time_avg = None
    benchmark_csv_path = os.path.join(catabolites_dir, 'benchmark.csv')
    if os.path.exists(benchmark_csv_path):
        try:
            bench_df = pd.read_csv(benchmark_csv_path)
            if 'Performing ex post explanation' in bench_df.columns:
                ex_post_time_avg = bench_df['Performing ex post explanation'].mean()
        except Exception as e:
            print(f"Failed to read benchmark.csv: {e}")

    suffix_map = {1: 'S', 2: 'R', 3: 'N'}

    if not os.path.exists(catabolites_dir):
        print("catabolites dir not found")
        return

    model_names = set()

    for test_dir in os.listdir(catabolites_dir):
        test_path = os.path.join(catabolites_dir, test_dir)
        if not os.path.isdir(test_path):
            continue

        matrices = {}
        for fname in os.listdir(test_path):
            if fname.startswith('confusion_matrices_') and fname.endswith('.json'):
                model_name = fname[len('confusion_matrices_'):-len('.json')]
                # Skip reasoning files (they contain dicts, not matrices)
                if model_name.endswith('_reasoning'):
                    continue
                model_names.add(model_name)
                with open(os.path.join(test_path, fname), 'r') as file:
                    matrices[model_name] = json.load(file)

        for c in [1, 2, 3]:
            item_id = f"{test_dir}_{suffix_map[c]}"
            row_data = {'item_id': item_id}

            for model_name in matrices:
                matrix = matrices[model_name]

                if model_name == "Logical":
                    # First-row direction: P(Sc | S0) — does source entail claim?
                    if matrix and len(matrix) > 0 and c < len(matrix[0]):
                        v = matrix[0][c]
                    else:
                        v = None
                    row_data[f'{model_name}_value'] = v
                    row_data[f'{model_name}_pred'] = get_label_logical_gold(
                        similarities_gold, test_dir, c, v
                    )
                else:
                    # First-row direction: P(Sc | S0) — consistent with Logical.
                    # Symmetric models (sentence-transformers) are unaffected;
                    # directional models (NLI, LLM) now ask "does source entail claim?"
                    if matrix and len(matrix) > 0 and c < len(matrix[0]):
                        v = matrix[0][c]
                    else:
                        v = None
                    row_data[f'{model_name}_value'] = v
                    if isinstance(v, str):
                        row_data[f'{model_name}_pred'] = v.strip().title() if v else "Unknown"
                    else:
                        row_data[f'{model_name}_pred'] = get_label_transformer(v)

            results.append(row_data)

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No results found.")
        return

    merged = df.merge(res_df, on='item_id', how='inner')

    for model_name in model_names:
        pred_col = f'{model_name}_pred'
        if pred_col in merged.columns:
            merged[f'{model_name}_correct'] = merged['label'] == merged[pred_col]

    for model_name in sorted(model_names):
        pred_col = f'{model_name}_pred'
        if pred_col not in merged.columns:
            continue

        # Drop rows where this model produced no valid prediction (matrix absent).
        # Including them would introduce a spurious label class and silently
        # deflate Macro-F1.
        valid = merged[merged[pred_col].isin(_NEET_LABELS)]
        skipped = merged[~merged[pred_col].isin(_NEET_LABELS)]
        n_valid, n_missing = len(valid), len(skipped)

        if n_valid == 0:
            print(f"({model_name}): no valid predictions — skipping")
            continue

        true_labels = valid['label'].astype(str).str.title()
        pred_labels = valid[pred_col].astype(str).str.title()

        acc = (true_labels == pred_labels).mean()
        macro_f1 = f1_score(true_labels, pred_labels,
                            average='macro', labels=_NEET_LABELS, zero_division=0)

        skip_note = f", {n_missing} skipped]" if n_missing else "]"
        print(f"\nOverall Accuracy ({model_name}): {acc:.4f}  [n={n_valid}{skip_note}")
        print(f"Macro-F1       ({model_name}): {macro_f1:.4f}")
        if n_missing:
            skipped_ids = sorted(skipped['item_id'].tolist())
            print(f"  Skipped (no prediction): {', '.join(skipped_ids)}")
        print(classification_report(true_labels, pred_labels,
                                    labels=_NEET_LABELS, zero_division=0))
        # Keep _correct aligned with the full merged frame (NaN for missing rows)
        merged[f'{model_name}_correct'] = merged['label'] == merged[pred_col]

    if ex_post_time_avg is not None:
        print(f"\nAverage Time for 'Performing ex post explanation' (LaSSI/Logical): "
              f"{ex_post_time_avg:.4f} seconds")

    for model_name in sorted(model_names):
        col = f'{model_name}_correct'
        if col in merged.columns:
            print(f"\nAccuracy per domain ({model_name}):")
            print(merged.groupby('domain')[col].mean().to_string())

    for model_name in sorted(model_names):
        col = f'{model_name}_correct'
        if col in merged.columns:
            print(f"\nAccuracy per claim_type ({model_name}):")
            print(merged.groupby('claim_type')[col].mean().to_string())

    print("\nDetailed results table:")
    cols = ['item_id', 'domain', 'claim_type', 'label']
    for model_name in sorted(model_names):
        cols.extend([f'{model_name}_pred', f'{model_name}_correct', f'{model_name}_value'])
    cols = [c for c in cols if c in merged.columns]
    print(merged[cols].to_string())

    merged.to_csv('evaluation_output_compared.csv', index=False)

    _generate_tex_tables(merged, model_names)


if __name__ == '__main__':
    main()
