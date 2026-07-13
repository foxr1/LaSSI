"""Analysis + LaTeX/figure generation for the HOnK-grounding ablation (appendix).

Scans ``catabolites/<case>/confusion_matrices_FullText_{LLM#,LLMHOnK#}*.json``,
maps each cell to a NEET claim-verification label using the SAME helper the
``LLM#`` baseline uses in ``eval_results.py`` (``get_label_matrix_output``), and
emits, into ``results/appendix_honk/``:

  ablation_results.csv          per assertion x condition (value, pred, correct)
  ablation_summary.csv          per (model, condition): n, acc, macro-F1, F1/class
  tab_ablation_overall.tex      A1  model x condition -> Acc, MacroF1, F1(Sup/Ref/NEE), dMacroF1
  tab_ablation_claimtype.tex    A2  condition x claim_type accuracy (mean over models)
  tab_ablation_domain.tex       A3  condition x domain accuracy (mean over models)
  tab_qualitative.tex           A4  flipped examples mined from *_reasoning.json
  fig_macroF1_by_condition.{pdf,png}
  fig_perclass_f1.{pdf,png}

Run from the repo root with the project venv:
    ~/PycharmProjects/LaSSI/.venv/bin/python eval/eval_honk_grounding.py
"""
import argparse
import json
import os
import sys

# Make the repo root importable (this file lives in eval/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from sklearn.metrics import f1_score

from eval_results import (_NEET_LABELS, _load_similarities_gold,
                          get_label_matrix_output)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SOURCES = ("honk", "lifecycle", "paraphrase")
# canonical display order for conditions
_COND_ORDER = ["ungrounded", "conceptnet", "honk", "lifecycle", "honk+lifecycle",
               "honk+paraphrase", "lifecycle+paraphrase", "all"]
_COND_LABEL = {
    "ungrounded": r"LLM (ungrounded)",
    "conceptnet": r"+ConceptNet-only",
    "honk": r"+HOnK",
    "lifecycle": r"+Lifecycle",
    "honk+lifecycle": r"+HOnK+Lifecycle",
    "honk+paraphrase": r"+HOnK+Paraphrase",
    "lifecycle+paraphrase": r"+Lifecycle+Paraphrase",
    "all": r"+All three",
}
_SUFFIX_MAP = {1: "S", 2: "R", 3: "N"}
_F1_SHORT = {"Supported": "Sup.", "Refuted": "Ref.", "Not Enough Evidence": "NEE"}


# ---------------------------------------------------------------------------
# Model-id parsing
# ---------------------------------------------------------------------------

def _canonical_condition(sources_str: str) -> str:
    raw = {s for s in sources_str.split("+") if s}
    if raw == {"conceptnet"}:
        return "conceptnet"
    toks = sorted(raw & set(_SOURCES))
    if set(toks) == set(_SOURCES):
        return "all"
    return "+".join(toks) if toks else "ungrounded"


def parse_model_id(model_name: str):
    """('llama3.2:3b', 'honk+lifecycle') from a catabolites model filename stem,
    or None if it isn't one of our LLM/LLMHOnK conditions."""
    name = model_name
    if name.startswith("FullText_"):
        name = name[len("FullText_"):]
    if name.startswith("LLMHOnK#"):
        rest = name[len("LLMHOnK#"):]
        parts = rest.split("#")
        base = parts[0]
        sources = parts[1] if len(parts) > 1 and parts[1] else "honk+lifecycle+paraphrase"
        return base, _canonical_condition(sources)
    if name.startswith("LLM#"):
        return name[len("LLM#"):], "ungrounded"
    return None


# ---------------------------------------------------------------------------
# Scan catabolites -> long-form records
# ---------------------------------------------------------------------------

def gold_blind_label(v):
    """Map a model score to a NEET label without consulting the gold matrix.

    ``get_label_matrix_output`` (inherited from the Logical-matrix evaluation)
    upgrades a partial score to Supported when the gold cell itself licenses
    asymmetric entailment, i.e. the prediction consults the answer. For LLM
    outputs the score schema is {0, 0.5, 1}, so we instead map each value to
    the nearest valid score: > 0.75 -> Supported, < 0.25 -> Refuted, everything
    between -> Not Enough Evidence. Off-schema values emitted by weaker models
    (e.g. 0.8, 0.2) thus resolve to their nearest class instead of depending on
    the gold-aware partial branch. Ties at exactly 0.25/0.75 stay Not Enough
    Evidence.
    """
    if v is None:
        return "Unknown"
    if isinstance(v, str):
        label = v.strip().title()
        if label in _NEET_LABELS:
            return label
        try:
            v = float(label)
        except ValueError:
            return "Unknown"
    if v > 0.75:
        return "Supported"
    if v < 0.25:
        return "Refuted"
    return "Not Enough Evidence"


def collect_records(catabolites_dir, neet_csv, models_filter=None,
                    legacy_mapping=False):
    df = pd.read_csv(neet_csv)
    meta = df.set_index("item_id")[["domain", "claim_type", "label"]]
    expected = df.set_index("item_id")["label"].astype(str).str.title().to_dict()
    gold = _load_similarities_gold()

    records = []
    for case in sorted(os.listdir(catabolites_dir)):
        case_dir = os.path.join(catabolites_dir, case)
        if not os.path.isdir(case_dir):
            continue
        # Matrices now live in a per-case ``matrices/`` subfolder; fall back to the
        # case directory itself for older layouts.
        scan_dir = os.path.join(case_dir, "matrices")
        if not os.path.isdir(scan_dir):
            scan_dir = case_dir
        for fname in os.listdir(scan_dir):
            if not (fname.startswith("confusion_matrices_") and fname.endswith(".json")):
                continue
            stem = fname[len("confusion_matrices_"):-len(".json")]
            if stem.endswith("_reasoning"):
                continue
            parsed = parse_model_id(stem)
            if parsed is None:
                continue
            base_model, condition = parsed
            if models_filter and base_model not in models_filter:
                continue
            try:
                with open(os.path.join(scan_dir, fname)) as fh:
                    matrix = json.load(fh)
            except Exception:
                continue
            for c in (1, 2, 3):
                item_id = f"{case}_{_SUFFIX_MAP[c]}"
                if item_id not in expected:
                    continue
                v = matrix[0][c] if (matrix and len(matrix) > 0 and c < len(matrix[0])) else None
                if legacy_mapping:
                    pred = get_label_matrix_output(gold, case, c, v, expected.get(item_id))
                else:
                    pred = gold_blind_label(v)
                row = {"item_id": item_id, "case": case, "base_model": base_model,
                       "condition": condition, "value": v, "pred": pred}
                if item_id in meta.index:
                    m = meta.loc[item_id]
                    row["domain"] = m["domain"]
                    row["claim_type"] = m["claim_type"]
                    row["label"] = str(m["label"]).title()
                records.append(row)
    long_df = pd.DataFrame.from_records(records)
    if not long_df.empty:
        long_df["valid"] = long_df["pred"].isin(_NEET_LABELS)
        long_df["correct"] = long_df["valid"] & (long_df["pred"] == long_df["label"])
    return long_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarise(long_df):
    rows = []
    for (model, cond), g in long_df.groupby(["base_model", "condition"]):
        valid = g[g["valid"]]
        n = len(valid)
        if n == 0:
            continue
        true_l = valid["label"]
        pred_l = valid["pred"]
        acc = float((true_l == pred_l).mean())
        macro = float(f1_score(true_l, pred_l, average="macro",
                               labels=_NEET_LABELS, zero_division=0))
        perclass = f1_score(true_l, pred_l, average=None,
                            labels=_NEET_LABELS, zero_division=0)
        rows.append({
            "base_model": model, "condition": cond, "n": n,
            "accuracy": acc, "macro_f1": macro,
            "f1_supported": float(perclass[0]),
            "f1_refuted": float(perclass[1]),
            "f1_nee": float(perclass[2]),
        })
    s = pd.DataFrame(rows)
    if not s.empty:
        s["cond_rank"] = s["condition"].map(
            lambda c: _COND_ORDER.index(c) if c in _COND_ORDER else 99)
        s = s.sort_values(["base_model", "cond_rank"]).drop(columns="cond_rank")
    return s


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    return "".join(repl.get(ch, ch) for ch in str(s))


def _ordered_conditions(present):
    known = [c for c in _COND_ORDER if c in present]
    extra = sorted(c for c in present if c not in _COND_ORDER)
    return known + extra


def _colour_extrema(formatted, raw):
    """Given parallel lists of formatted strings and numeric values, return the
    formatted strings with the (block) maximum wrapped blue and minimum red.
    Higher is better for every column in the overall table. No-op on ties / when
    fewer than two distinct values."""
    vals = [v for v in raw if v is not None]
    if len(set(round(v, 6) for v in vals)) < 2:
        return list(formatted)
    hi, lo = max(vals), min(vals)
    out = []
    hi_done = lo_done = False
    for s, v in zip(formatted, raw):
        if v is not None and not hi_done and abs(v - hi) < 1e-9:
            out.append(r"\textcolor{blue}{" + s + "}"); hi_done = True
        elif v is not None and not lo_done and abs(v - lo) < 1e-9:
            out.append(r"\textcolor{red}{" + s + "}"); lo_done = True
        else:
            out.append(s)
    return out


def write_overall_table(summary, path):
    """A1: model x condition -> Acc, Macro-F1, F1(Sup/Ref), dMacroF1 vs ungrounded.
    Per model block, each numeric column's best value is blue and worst is red."""
    lines = [
        r"% Auto-generated by eval/eval_honk_grounding.py",
        r"\begin{table*}[p]",
        r"\centering",
        r"\caption{Ontology-grounding ablation on \gls{curb} claim verification "
        r"(90 evidence--claim pairs per condition). $\Delta$F1$_\mathrm{M}$ is the change "
        r"in macro-F1 against the ungrounded \gls{llm} of the same base model; the ungrounded "
        r"rows coincide with the \gls{llm} baselines in Table~\ref{tab:overall-results}. "
        r"F1(Ref.) is contradiction-detection F1. Per model, the best value in each column is "
        r"\textcolor{blue}{blue} and the worst \textcolor{red}{red}.}",
        r"\label{tab:honk-ablation-overall}",
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        r"Model & Grounding & Acc.\ (\%) & F1$_\mathrm{M}$ & "
        r"$\Delta$F1$_\mathrm{M}$ & F1(Sup.) & F1(Ref.) \\",
        r"\midrule",
    ]
    for model in sorted(summary["base_model"].unique()):
        sub = summary[summary["base_model"] == model]
        base = sub[sub["condition"] == "ungrounded"]
        base_macro = float(base["macro_f1"].iloc[0]) if len(base) else None
        conds = [c for c in _ordered_conditions(set(sub["condition"]))
                 if len(sub[sub["condition"] == c])]
        rows = [sub[sub["condition"] == c].iloc[0] for c in conds]
        # raw values + formatted strings, per column
        acc_raw = [100 * r["accuracy"] for r in rows]
        mac_raw = [r["macro_f1"] for r in rows]
        dlt_raw = [None if base_macro is None else r["macro_f1"] - base_macro for r in rows]
        sup_raw = [r["f1_supported"] for r in rows]
        ref_raw = [r["f1_refuted"] for r in rows]
        acc_f = _colour_extrema([f"{v:.1f}" for v in acc_raw], acc_raw)
        mac_f = _colour_extrema([f"{v:.3f}" for v in mac_raw], mac_raw)
        dlt_f = _colour_extrema(["" if d is None else f"{d:+.3f}" for d in dlt_raw], dlt_raw)
        sup_f = _colour_extrema([f"{v:.3f}" for v in sup_raw], sup_raw)
        ref_f = _colour_extrema([f"{v:.3f}" for v in ref_raw], ref_raw)
        for i, cond in enumerate(conds):
            model_cell = (r"\texttt{" + _tex_escape(model) + "}") if i == 0 else ""
            lines.append(
                f"{model_cell} & {_COND_LABEL.get(cond, _tex_escape(cond))} "
                f"& {acc_f[i]} & {mac_f[i]} & {dlt_f[i]} & {sup_f[i]} & {ref_f[i]} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# Compact column headers for the wide claim-type slice table.
_SLICE_HEADER_SHORT = {
    "approximate_paraphrase": "Paraphrase", "cause_mismatch": "Cause",
    "direct_support": "Support", "location_mismatch": "Location",
    "negation": "Negation", "outcome_mismatch": "Outcome",
    "time_mismatch": "Time", "unsupported_extra_detail": "Extra",
}


def _slice_table(long_df, by, path, caption, label):
    """Mean accuracy per (condition x `by`) averaged across models. Emitted as a
    full-width table* so the wide claim-type slice fits the text block."""
    valid = long_df[long_df["valid"]]
    pivot = (valid.groupby(["condition", by])["correct"].mean().unstack(by))
    conds = _ordered_conditions(set(pivot.index))
    pivot = pivot.reindex(conds)
    cols = list(pivot.columns)

    def _col_head(c):
        key = str(c)
        if key in _SLICE_HEADER_SHORT:
            return _SLICE_HEADER_SHORT[key]
        return _tex_escape(key.replace("_", " ").title())

    header = " & ".join([r"Grounding"] + [_col_head(c) for c in cols])
    caption = caption.rstrip() + r" Per column, the best grounding is " \
        r"\textcolor{blue}{blue} and the worst \textcolor{red}{red}."
    lines = [
        r"% Auto-generated by eval/eval_honk_grounding.py",
        r"\begin{table*}[p]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l" + " r" * len(cols) + "}",
        r"\toprule", header + r" \\", r"\midrule",
    ]
    # Per-column best (max) / worst (min) accuracy over the conditions.
    col_hi, col_lo = {}, {}
    for col in cols:
        nums = [round(100 * v) for v in (pivot.loc[c, col] for c in pivot.index)
                if not pd.isna(v)]
        if len(set(nums)) >= 2:
            col_hi[col], col_lo[col] = max(nums), min(nums)
    hi_done, lo_done = set(), set()
    for cond in pivot.index:
        cells = []
        for col in cols:
            v = pivot.loc[cond, col]
            if pd.isna(v):
                cells.append("--")
                continue
            disp = round(100 * v)
            s = f"{disp:d}"
            if col in col_hi and disp == col_hi[col] and col not in hi_done:
                s = r"\textcolor{blue}{" + s + "}"; hi_done.add(col)
            elif col in col_lo and disp == col_lo[col] and col not in lo_done:
                s = r"\textcolor{red}{" + s + "}"; lo_done.add(col)
            cells.append(s)
        lines.append(f"{_COND_LABEL.get(cond, _tex_escape(cond))} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def _reasoning_path(catabolites_dir, case, base_model, condition):
    if condition == "ungrounded":
        suffix = base_model.split("/")[-1]
    else:
        toks = sorted(condition.split("+")) if condition != "all" else list(_SOURCES)
        suffix = base_model.split("/")[-1] + "_" + "-".join(toks)
    fname = f"confusion_matrices_FullText_{suffix}_reasoning.json"
    in_matrices = os.path.join(catabolites_dir, case, "matrices", fname)
    if os.path.exists(in_matrices):
        return in_matrices
    return os.path.join(catabolites_dir, case, fname)


def _load_reasoning_cell(path, c):
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data["sentences"][0], data["sentences"][c], data["reasoning"][0][c]
    except Exception:
        return None, None, None


def write_qualitative_table(long_df, catabolites_dir, path, base_model=None,
                            grounded_cond="all", max_examples=3):
    """A4: examples where ungrounded is wrong and the grounded model is right."""
    if base_model is None:
        prefs = [m for m in ("qwen2.5:7b", "llama3.2:3b", "gemma4:e2b")
                 if m in set(long_df["base_model"])]
        base_model = prefs[0] if prefs else (sorted(long_df["base_model"].unique())[0]
                                             if not long_df.empty else None)
    lines = [r"% Auto-generated by eval/eval_honk_grounding.py"]
    if base_model is None:
        lines.append("% no data")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return

    ung = long_df[(long_df.base_model == base_model) & (long_df.condition == "ungrounded")]
    gnd = long_df[(long_df.base_model == base_model) & (long_df.condition == grounded_cond)]
    merged = ung.merge(gnd, on="item_id", suffixes=("_ung", "_gnd"))
    flips = merged[(~merged["correct_ung"]) & (merged["correct_gnd"])]

    lines += [
        rf"\begin{{table*}}[t]", r"\centering", r"\small",
        rf"\caption{{Qualitative flips for \texttt{{{_tex_escape(base_model)}}}: cases the "
        rf"ungrounded LLM gets wrong but the ontology-grounded model "
        rf"(\textit{{{_COND_LABEL.get(grounded_cond, grounded_cond)}}}) gets right.}}",
        r"\label{tab:honk-ablation-qualitative}",
        r"\begin{tabularx}{\textwidth}{l l l X}",
        r"\toprule",
        r"Item & Gold & Pred. & Grounded model's reasoning \\",
        r"\midrule",
    ]
    shown = 0
    for _, r in flips.iterrows():
        if shown >= max_examples:
            break
        item = r["item_id"]
        case = r["case_ung"]
        c = {v: k for k, v in _SUFFIX_MAP.items()}[item.rsplit("_", 1)[1]]
        rp = _reasoning_path(catabolites_dir, case, base_model, grounded_cond)
        prem, cons, reason = _load_reasoning_cell(rp, c)
        if reason is None:
            continue
        gold = r["label_ung"]
        pred = r["pred_gnd"]
        lines.append(
            f"{_tex_escape(item)} & {_tex_escape(_F1_SHORT.get(gold, gold))} "
            f"& {_tex_escape(_F1_SHORT.get(pred, pred))} & {_tex_escape(reason)} \\\\")
        lines.append(r"\addlinespace")
        shown += 1
    if shown == 0:
        lines.append(r"\multicolumn{4}{l}{\textit{No qualifying flips with reasoning files found.}} \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_macro_f1(summary, path_stem):
    models = sorted(summary["base_model"].unique())
    conds = _ordered_conditions(set(summary["condition"]))
    if not models or not conds:
        return
    import numpy as np
    x = np.arange(len(conds))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, m in enumerate(models):
        sub = summary[summary["base_model"] == m].set_index("condition")
        vals = [sub.loc[c, "macro_f1"] if c in sub.index else 0 for c in conds]
        ax.bar(x + i * w, vals, w, label=m)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels([_COND_LABEL.get(c, c) for c in conds], rotation=20, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 by grounding condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_stem + ".pdf")
    fig.savefig(path_stem + ".png", dpi=150)
    plt.close(fig)


def fig_perclass(summary, path_stem):
    import numpy as np
    base = summary[summary["condition"] == "ungrounded"]
    allc = summary[summary["condition"] == "all"]
    if base.empty or allc.empty:
        return
    classes = ["f1_supported", "f1_refuted", "f1_nee"]
    labels = ["F1(Sup.)", "F1(Ref.)", "F1(NEE)"]
    base_m = [base[c].mean() for c in classes]
    all_m = [allc[c].mean() for c in classes]
    x = np.arange(len(classes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, base_m, w, label="LLM (ungrounded)")
    ax.bar(x + w / 2, all_m, w, label="+All three")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 (mean over models)")
    ax.set_title("Per-class F1: ungrounded vs grounded")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_stem + ".pdf")
    fig.savefig(path_stem + ".png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="HOnK-grounding ablation analysis")
    ap.add_argument("--catabolites", default="catabolites")
    ap.add_argument("--neet-csv", default="neet/neet_v1.csv")
    ap.add_argument("--out", default="results/appendix_honk")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these base models")
    ap.add_argument("--legacy-mapping", action="store_true",
                    help="reproduce the original gold-aware partial-score "
                         "mapping (get_label_matrix_output) instead of the "
                         "gold-blind nearest-score rule")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    long_df = collect_records(args.catabolites, args.neet_csv, args.models,
                              legacy_mapping=args.legacy_mapping)
    if long_df.empty:
        print("No LLM#/LLMHOnK# matrices found under", args.catabolites)
        return 1

    long_csv = os.path.join(args.out, "ablation_results.csv")
    long_df.to_csv(long_csv, index=False)

    summary = summarise(long_df)
    sum_csv = os.path.join(args.out, "ablation_summary.csv")
    summary.to_csv(sum_csv, index=False)

    write_overall_table(summary, os.path.join(args.out, "tab_ablation_overall.tex"))
    _slice_table(long_df, "claim_type",
                 os.path.join(args.out, "tab_ablation_claimtype.tex"),
                 "Accuracy (\\%) by claim type, averaged over base models.",
                 "tab:honk-ablation-claimtype")
    _slice_table(long_df, "domain",
                 os.path.join(args.out, "tab_ablation_domain.tex"),
                 "Accuracy (\\%) by domain, averaged over base models.",
                 "tab:honk-ablation-domain")
    write_qualitative_table(long_df, args.catabolites,
                            os.path.join(args.out, "tab_qualitative.tex"))
    fig_macro_f1(summary, os.path.join(args.out, "fig_macroF1_by_condition"))
    fig_perclass(summary, os.path.join(args.out, "fig_perclass_f1"))

    print(f"Wrote outputs to {args.out}/")
    print(f"  configs found: {summary[['base_model','condition']].drop_duplicates().shape[0]}")
    print(f"  assertions:    {long_df['valid'].sum()} valid / {len(long_df)} total")
    print("\nSummary:")
    with pd.option_context("display.width", 140, "display.max_rows", None):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
