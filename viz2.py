"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import ast
import glob
import json
import os.path
import pathlib

import dash
from dash import dash_table
import dashvis
import flask
import dash_bootstrap_components as dbc
import pandas
from dash import Input, Output, dcc, html
from dashvis import DashNetwork

from LaSSI.viz import NestedTables


def convert_html_to_dash(html_code, dash_modules=None):
    """Convert standard html (as string) to Dash components.

    Looks into the list of dash_modules to find the right component (default to [html, dcc, dbc])."""
    from xml.etree import ElementTree

    if dash_modules is None:
        import dash_html_components as html
        import dash_core_components as dcc

        dash_modules = [html, dcc]
        try:
            import dash_bootstrap_components as dbc

            dash_modules.append(dbc)
        except ImportError:
            pass

    def find_component(name):
        for module in dash_modules:
            try:
                return getattr(module, name)
            except AttributeError:
                pass
        raise AttributeError(f"Could not find a dash widget for '{name}'")

    def parse_css(css):
        """Convert a style in ccs format to dictionary accepted by Dash"""
        return {k: v for style in css.strip(";").split(";") for k, v in [style.split(":")]}

    def parse_value(v):
        try:
            return ast.literal_eval(v)
        except (SyntaxError, ValueError):
            return v

    parsers = {"style": parse_css, "id": lambda x: x}

    def _convert(elem):
        comp = find_component(elem.tag.capitalize())
        children = [_convert(child) for child in elem]
        if not children:
            children = elem.text
        attribs = elem.attrib.copy()
        if "class" in attribs:
            attribs["className"] = attribs.pop("class")
        attribs = {k: parsers.get(k, parse_value)(v) for k, v in attribs.items()}

        return comp(children=children, **attribs)

    et = ElementTree.fromstring(html_code)

    return _convert(et)

# Capture the working directory once at startup so all path lookups are consistent
# even if uvicorn or any library changes os.getcwd() later.
_BASE_DIR = os.path.abspath(os.getcwd())

# app = dash.Dash()


def _analyze_atoms(rows: list, atom_cols: list, rc_j: str):
    """Derive atom-level conflict and blocking information from a truth table.

    Each row is a possible world where Sᵢ holds (rc_i is always 1).

    Returns:
        conflict_pairs  — list of (col_a, col_b): atom pairs that never
                          co-occur as 1 across any row (mutually exclusive).
        discriminating  — set of atom cols that are always 1 in supporting
                          rows (rc_j=1) but sometimes 0 in refuting rows
                          (rc_j=0).  These are the atoms whose absence blocks
                          Sⱼ from holding.
        blocking_per_row — parallel list to `rows`; each entry is the set
                           of discriminating atoms that are 0 in that row
                           (empty for supporting rows).
    """
    supporting_rows = [r for r in rows if r.get(rc_j) == 1]
    refuting_rows   = [r for r in rows if r.get(rc_j) == 0]

    # Atoms always 1 whenever Sⱼ holds
    always_1_when_j = {
        c for c in atom_cols
        if supporting_rows and all(r.get(c) == 1 for r in supporting_rows)
    }
    # Atoms sometimes 0 when Sⱼ doesn't hold
    sometimes_0_when_not_j = {
        c for c in atom_cols
        if any(r.get(c) == 0 for r in refuting_rows)
    }
    discriminating = always_1_when_j & sometimes_0_when_not_j

    # Conflict pairs: no row has both cols == 1
    conflict_pairs = []
    for idx_a, a in enumerate(atom_cols):
        for b in atom_cols[idx_a + 1:]:
            if not any(r.get(a) == 1 and r.get(b) == 1 for r in rows):
                conflict_pairs.append((a, b))

    blocking_per_row = [
        set() if row.get(rc_j) == 1
        else {c for c in discriminating if row.get(c) == 0}
        for row in rows
    ]

    return conflict_pairs, discriminating, blocking_per_row


def _classify_relationship(rows: list, rc_i: str, rc_j: str, conf: float,
                            conflict_pairs: list, alias: dict):
    """Return (verdict_label, colour, plain_English_explanation)."""
    if not rows:
        return ("Unknown", "#6c757d",
                "No satisfying worlds were found, so no relationship can be determined.")

    total      = len(rows)
    supporting = sum(1 for r in rows if r.get(rc_j) == 1)
    refuting   = total - supporting
    si, sj     = rc_i[1:], rc_j[1:]

    # Does any conflict pair involve the target atom (Sⱼ's own formula)?
    target_in_conflict = any(rc_j[1:] in (a, b) for (a, b) in conflict_pairs)

    if conf == 1.0:
        verdict, colour = "Implies", "#198754"
        explanation = (
            f"In every world where S{si} holds ({total}/{total}), "
            f"S{sj} also holds. All atomic propositions of S{sj} are "
            f"simultaneously satisfiable with those of S{si}."
        )
    elif conf == 0.0:
        if conflict_pairs:
            pair_labels = " and ".join(
                f"{alias[a]} ✗ {alias[b]}" for (a, b) in conflict_pairs
            )
            if target_in_conflict:
                verdict, colour = "Contradicts", "#dc3545"
                explanation = (
                    f"S{sj} cannot hold in any world where S{si} holds (0/{total}). "
                    f"The atomic propositions are mutually exclusive: {pair_labels}. "
                    f"These atoms were found to be logically incompatible by the ontology "
                    f"reasoner (e.g. antonym predicates, lifecycle-state partition clash, "
                    f"or named near-place divergence)."
                )
            else:
                verdict, colour = "Incompatible", "#e67e22"
                explanation = (
                    f"S{sj} never holds when S{si} holds (0/{total}). "
                    f"Some shared atoms are mutually exclusive ({pair_labels}), "
                    f"preventing S{sj}'s proposition from being satisfied."
                )
        else:
            verdict, colour = "Incompatible", "#e67e22"
            explanation = (
                f"S{sj} never holds when S{si} holds (0/{total}), "
                f"but no direct atom conflict was found. "
                f"S{sj} likely requires atoms that S{si} does not supply."
            )
    elif conf >= 0.75:
        verdict, colour = "Likely Implies", "#0d6efd"
        explanation = (
            f"S{sj} holds in {supporting}/{total} worlds where S{si} holds ({conf:.0%}). "
            f"{refuting} world{'s' if refuting != 1 else ''} refute it — "
            f"see the highlighted blocking atoms in the table below."
        )
    elif conf >= 0.25:
        verdict, colour = "Partial", "#fd7e14"
        explanation = (
            f"S{sj} holds in {supporting}/{total} worlds where S{si} holds ({conf:.0%}). "
            f"Some atom combinations support S{sj} and others block it — "
            f"blocking atoms are highlighted in the table below."
        )
    else:
        verdict, colour = "Weak / Indifferent", "#6c757d"
        explanation = (
            f"S{sj} holds in only {supporting}/{total} worlds where S{si} holds ({conf:.0%}). "
            f"The atomic propositions give little evidence that S{sj} follows from S{si}."
        )
    return verdict, colour, explanation


def _render_pairwise_table(entry: dict, lines: list) -> html.Div:
    """Render one directed pairwise entry (i→j) with:
    - Verdict banner with plain-English explanation
    - Conflict panel showing which atom pairs are mutually exclusive (with both formulas)
    - Atom glossary flagging discriminating atoms
    - Truth table with blocking atoms highlighted per row
    """
    atoms: dict = entry.get("atoms", {})
    rows: list  = entry.get("rows", [])
    si: int     = entry["i"]
    sj: int     = entry["j"]
    rc_i        = entry.get("result_col_i", f"R{si}")
    rc_j        = entry.get("result_col_j", f"R{sj}")
    conf: float = entry.get("confidence", 0.0)

    atom_cols   = sorted([k for k in atoms if k not in (rc_i, rc_j)], key=lambda k: int(k))
    result_cols = [c for c in (rc_i, rc_j) if rows and c in rows[0]]
    all_cols    = atom_cols + result_cols

    _subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    alias = {}
    import re
    for idx, c in enumerate(atom_cols):
        p_sub = f"P{str(idx + 1).translate(_subscripts)}"
        lat_c = atoms.get(c, c)
        is_neg = r'\neg' in lat_c
        m = re.search(r'\\(?:textit|textsf)\{([^}]+)\}', lat_c)
        if m:
            core = m.group(1)
            core = re.sub(r'\\[a-zA-Z]+', '', core)
            core = re.sub(r'[{}]', '', core).strip()
            if is_neg:
                core = f"¬{core}"
            alias[c] = f"{core} ({p_sub})"
        else:
            alias[c] = p_sub

    conflict_pairs, discriminating, blocking_per_row = _analyze_atoms(rows, atom_cols, rc_j)
    verdict, verdict_colour, explanation = _classify_relationship(
        rows, rc_i, rc_j, conf, conflict_pairs, alias)

    # ── Verdict banner ────────────────────────────────────────────────────────
    source_sentence = lines[si] if si < len(lines) else f"S{si}"
    target_sentence = lines[sj] if sj < len(lines) else f"S{sj}"

    banner = html.Div([
        html.Div([
            html.Span(f"S{si} → S{sj}", style={
                "fontWeight": "700", "fontSize": "1.05em", "marginRight": "0.8em"}),
            html.Span(verdict, style={
                "background": verdict_colour, "color": "#fff",
                "borderRadius": "4px", "padding": "2px 10px",
                "fontWeight": "600", "fontSize": "0.9em", "marginRight": "0.6em"}),
            html.Span(f"confidence: {conf:.3f}", style={"color": "#555", "fontSize": "0.88em"}),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                  "marginBottom": "0.4em"}),
        html.Div([
            html.Span(f"S{si}:", style={"fontWeight": "600", "marginRight": "0.3em",
                                        "whiteSpace": "nowrap"}),
            html.Span(source_sentence, style={"color": "#333", "fontSize": "0.9em"}),
        ], style={"marginBottom": "0.15em"}),
        html.Div([
            html.Span(f"S{sj}:", style={"fontWeight": "600", "marginRight": "0.3em",
                                        "whiteSpace": "nowrap"}),
            html.Span(target_sentence, style={"color": "#333", "fontSize": "0.9em"}),
        ], style={"marginBottom": "0.5em"}),
        html.Div(explanation, style={
            "background": "#f8f9fa", "borderLeft": f"4px solid {verdict_colour}",
            "padding": "6px 12px", "fontSize": "0.88em", "color": "#333",
            "borderRadius": "0 4px 4px 0", "marginBottom": "0.2em",
        }),
    ], style={
        "border": f"1px solid {verdict_colour}", "borderRadius": "6px",
        "padding": "0.75em 1em", "marginTop": "1.4em",
        "borderLeftWidth": "5px",
    })

    # ── Conflict panel ────────────────────────────────────────────────────────
    # Show each conflicting atom pair side-by-side so the user can read what
    # makes the two formulas logically incompatible.
    conflict_panel = None
    if conflict_pairs:
        pair_blocks = []
        for (a, b) in conflict_pairs:
            lat_a = atoms.get(a, a)
            lat_b = atoms.get(b, b)
            pair_blocks.append(html.Div([
                # Header row: Pₐ ✗ Pᵦ
                html.Div([
                    html.Span(alias[a], style={
                        "fontWeight": "700", "color": "#dc3545",
                        "fontSize": "0.95em", "marginRight": "0.4em"}),
                    html.Span("✗  never simultaneously true with  ", style={
                        "color": "#888", "fontSize": "0.82em",
                        "fontStyle": "italic", "margin": "0 0.2em"}),
                    html.Span(alias[b], style={
                        "fontWeight": "700", "color": "#dc3545",
                        "fontSize": "0.95em", "marginLeft": "0.4em"}),
                ], style={"marginBottom": "0.4em", "display": "flex",
                          "alignItems": "center", "flexWrap": "wrap"}),
                # Side-by-side formula display
                html.Div([
                    html.Div([
                        html.Div(alias[a], style={
                            "fontWeight": "700", "color": "#dc3545",
                            "fontSize": "0.85em", "marginBottom": "2px"}),
                        html.Div(f"\\({lat_a}\\)", style={
                            "fontSize": "0.82em", "lineHeight": "1.4",
                            "color": "#333"}),
                    ], style={
                        "flex": "1", "minWidth": "0",
                        "background": "#fff5f5",
                        "border": "1px solid #f5c2c7",
                        "borderRadius": "4px", "padding": "6px 10px",
                        "marginRight": "6px",
                    }),
                    html.Div([
                        html.Div(alias[b], style={
                            "fontWeight": "700", "color": "#dc3545",
                            "fontSize": "0.85em", "marginBottom": "2px"}),
                        html.Div(f"\\({lat_b}\\)", style={
                            "fontSize": "0.82em", "lineHeight": "1.4",
                            "color": "#333"}),
                    ], style={
                        "flex": "1", "minWidth": "0",
                        "background": "#fff5f5",
                        "border": "1px solid #f5c2c7",
                        "borderRadius": "4px", "padding": "6px 10px",
                    }),
                ], style={"display": "flex", "flexWrap": "wrap",
                          "gap": "6px", "alignItems": "stretch"}),
            ], style={"marginBottom": "0.8em"}))

        conflict_panel = html.Div([
            html.P(
                "Mutually exclusive atom pairs — these propositions cannot both be true "
                "in any possible world (the ontology reasoner found them incompatible, "
                "e.g. via antonym predicates, lifecycle-state partition clash, or "
                "near-place name divergence):",
                style={"fontWeight": "600", "fontSize": "0.85em",
                       "marginBottom": "0.5em", "color": "#333"},
            ),
            *pair_blocks,
        ], style={
            "background": "#fff8f8", "border": "1px solid #f5c2c7",
            "borderRadius": "6px", "padding": "0.75em 1em",
            "marginTop": "0.6em", "marginBottom": "0.7em",
        })

    # ── Atom glossary ─────────────────────────────────────────────────────────
    # Discriminating atoms (★) are those whose truth value determines whether
    # Sⱼ holds — always true in supporting worlds, sometimes false in refuting ones.
    glossary_rows = []
    for c in atom_cols:
        latex   = atoms.get(c, c)
        is_disc = c in discriminating
        in_conf = any(c in (a, b) for (a, b) in conflict_pairs)
        tag_els = []
        if in_conf:
            tag_els.append(html.Span(" ✗ conflicts", style={
                "background": "#dc3545", "color": "#fff",
                "borderRadius": "3px", "padding": "1px 5px",
                "fontSize": "0.72em", "marginLeft": "4px", "fontWeight": "600",
            }))
        if is_disc:
            tag_els.append(html.Span(" ★ discriminating", style={
                "background": "#fd7e14", "color": "#fff",
                "borderRadius": "3px", "padding": "1px 5px",
                "fontSize": "0.72em", "marginLeft": "4px", "fontWeight": "600",
            }))
        alias_cell = html.Td([alias[c]] + tag_els, style={
            "padding": "4px 10px", "fontWeight": "700",
            "whiteSpace": "nowrap", "verticalAlign": "top",
            "color": "#dc3545" if in_conf else ("#fd7e14" if is_disc else "#333"),
            "width": "8em",
        })
        glossary_rows.append(html.Tr([
            alias_cell,
            html.Td(f"\\({latex}\\)", style={
                "padding": "4px 10px", "fontSize": "0.85em", "lineHeight": "1.4"}),
        ]))
    glossary = None
    if glossary_rows:
        glossary = html.Div([
            html.P(
                "Atomic propositions  (★ = discriminating: determines whether Sⱼ holds; "
                "✗ = conflicts: never simultaneously satisfiable with another atom):",
                style={"fontWeight": "600", "fontSize": "0.82em",
                       "marginBottom": "4px", "color": "#333"},
            ),
            html.Table(glossary_rows, style={
                "borderCollapse": "collapse", "width": "100%",
                "background": "#fafafa", "border": "1px solid #e0e0e0",
                "borderRadius": "4px", "fontSize": "0.87em",
            }),
        ], style={"marginBottom": "0.7em"})

    # ── Truth table ───────────────────────────────────────────────────────────
    def th(col):
        base = {"padding": "5px 10px", "whiteSpace": "nowrap",
                "fontWeight": "bold", "textAlign": "center"}
        if col == rc_i:
            return html.Th(f"S{si} holds ✓", style={**base,
                "borderBottom": "2px solid #6ea8fe", "background": "#dce8ff",
                "fontStyle": "italic"})
        if col == rc_j:
            return html.Th(f"S{sj} holds?", style={**base,
                "borderBottom": "2px solid #6ea8fe", "background": "#dce8ff",
                "fontStyle": "italic"})
        # Mark discriminating atom columns with ★
        label = alias[col] + (" ★" if col in discriminating else
                               (" ✗" if any(col in (a, b) for (a, b) in conflict_pairs) else ""))
        return html.Th(label, style={**base,
            "borderBottom": "2px solid #aaa", "background": "#f5f5f5"})

    def cell_content_and_style(col, val, is_blocking):
        base = {"padding": "4px 10px", "textAlign": "center"}
        display = str(int(val)) if isinstance(val, (float, int)) else str(val)
        if col == rc_j:
            style = {**base, "background": "#d1e7dd" if val == 1 else "#f8d7da",
                     "fontWeight": "700"}
            return display, style
        if col == rc_i:
            return display, {**base, "background": "#dce8ff"}
        if is_blocking:
            # Blocking: this atom being 0 is why Sⱼ fails in this world
            style = {**base,
                     "background": "#842029", "color": "#fff",
                     "fontWeight": "700", "fontSize": "0.9em"}
            content = [display, html.Span(" ← blocks Sⱼ", style={
                "fontSize": "0.65em", "fontStyle": "italic",
                "display": "block", "fontWeight": "400",
                "whiteSpace": "nowrap", "color": "#ffc", "marginTop": "1px",
            })]
            return content, style
        return display, {**base, "background": "#eafaf1" if val == 1 else "#fdf2f2"}

    if not rows:
        body = html.Tbody([html.Tr([html.Td(
            "No satisfying worlds found.",
            colSpan=max(len(all_cols) + 1, 1),
            style={"fontStyle": "italic", "padding": "8px", "color": "#888"}
        )])])
    else:
        body_rows = []
        for row_idx, row in enumerate(rows):
            j_val    = row.get(rc_j, 0)
            blocking = blocking_per_row[row_idx]
            row_bg   = "#f6fff8" if j_val == 1 else "#fff8f8"
            row_label       = "✓ supports" if j_val == 1 else "✗ refutes"
            row_label_colour = "#198754"   if j_val == 1 else "#dc3545"
            cells = [html.Td(row_label, style={
                "padding": "4px 8px", "fontSize": "0.78em",
                "color": row_label_colour, "whiteSpace": "nowrap",
                "fontWeight": "600", "verticalAlign": "middle",
            })]
            for c in all_cols:
                v   = row.get(c, "")
                content, style = cell_content_and_style(c, v, c in blocking)
                cells.append(html.Td(content, style=style))
            body_rows.append(html.Tr(cells, style={"background": row_bg}))

        annot_th = html.Th("", style={"padding": "5px 8px", "background": "#f5f5f5",
                                       "borderBottom": "2px solid #aaa", "width": "5em"})
        header = html.Thead(html.Tr([annot_th] + [th(c) for c in all_cols]))
        body   = html.Tbody(body_rows)

    if rows:
        supporting = sum(1 for r in rows if r.get(rc_j) == 1)
        stat_line = html.P(
            f"{supporting} of {len(rows)} possible world{'s' if len(rows) != 1 else ''} "
            f"where S{si} holds also satisfy S{sj}.",
            style={"fontSize": "0.82em", "color": "#555", "marginBottom": "0.3em",
                   "fontStyle": "italic"},
        )
    else:
        stat_line = None

    table_div = html.Div([
        stat_line,
        html.Div([
            html.Table(
                [header, body] if rows else [
                    html.Thead(html.Tr([th(c) for c in all_cols])), body],
                style={"borderCollapse": "collapse", "fontSize": "0.87em", "width": "100%"},
            )
        ], style={"overflowX": "auto"}),
    ])

    children = [banner]
    if conflict_panel:
        children.append(conflict_panel)
    if glossary:
        children.append(glossary)
    children.append(table_div)
    return html.Div(children, style={"marginBottom": "0.5em"})

import re as _re
_SENTENCE_BOUNDARY_RE = _re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def _split_row_sentences(text: str) -> list:
    """Mirror SentenceLoader.split_yaml_row_sentences — split a YAML row into
    sub-sentences so we know how many graphs each row produced."""
    if not text or not text.strip():
        return [text] if text else []
    parts = _SENTENCE_BOUNDARY_RE.split(text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def _load_page_data(full_path: str):
    """Read all per-catabolite data fresh from disk on every call."""
    import plotly.graph_objects as go
    from LaSSI.structures.extended_fol.Formulae import formula_from_dict

    lines = []
    string_rep = os.path.join(full_path, "string_rep.txt")
    if os.path.exists(string_rep):
        with open(string_rep, "r") as f:
            for line in f:
                lines.append(line.split(" ⇒ ")[0].rstrip("\n"))

    # Build mapping from YAML-row index → list of graph directory indices.
    # Each row may produce N≥1 sub-sentences (graphs) after sentence splitting.
    graph_indices: list[list[int]] = []
    cursor = 0
    for line in lines:
        n = len(_split_row_sentences(line))
        graph_indices.append(list(range(cursor, cursor + n)))
        cursor += n

    LS = []
    lr_path = os.path.join(full_path, "logical_rewriting.json")
    if os.path.exists(lr_path):
        with open(lr_path, "r") as f:
            for logical in json.load(f):
                actual = formula_from_dict(logical)
                LS.append(html.Div(f'\\({actual}\\)',
                                   style={"fontSize": "1.4em", "margin": "1em 0"}))

    pairwise = {}
    pw_path = os.path.join(full_path, "SentenceRepresentation.Logical",
                           "pairwise_truth_tables.json")
    if os.path.exists(pw_path):
        with open(pw_path, "r") as f:
            for entry in json.load(f):
                pairwise[(entry["i"], entry["j"])] = entry

    local_meu = []
    meu_path = os.path.join(full_path, "meuDBs.json")
    if os.path.exists(meu_path):
        with open(meu_path, "r") as f:
            for x in json.load(f):
                local_meu.append(sorted(x["multi_entity_unit"],
                                        key=lambda x: x["confidence"], reverse=True))

    import textwrap

    def _wrap(s, width=40):
        """Wrap a sentence into Plotly-compatible multi-line tick label."""
        return "<br>".join(textwrap.wrap(s.strip(), width=width))

    tick_labels = [f"{i}. {_wrap(s)}" for i, s in enumerate(lines)]

    matrices = {}
    for matrix_file in glob.glob(os.path.join(full_path, "confusion_matrices_*.json")):
        mtype = pathlib.Path(matrix_file).name[19:-5]
        with open(matrix_file, "r") as f:
            z = json.load(f)
        n = len(z)
        labels = tick_labels[:n] if tick_labels else [str(i) for i in range(n)]
        z_text = [[str(round(v, 2)) for v in row] for row in z]
        cell_px = max(60, 600 // max(n, 1))
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="Viridis",
            colorbar={"thickness": 15},
        ))
        fig.update_layout(
            xaxis=dict(tickangle=-45, tickfont={"size": 11}, side="bottom"),
            yaxis=dict(autorange="reversed", tickfont={"size": 11}),
            margin=dict(l=300, b=300, t=60, r=40),
            height=max(500, n * cell_px + 360),
        )
        matrices[mtype] = fig

    return lines, LS, pairwise, local_meu, matrices, graph_indices


def create_dash_app(requests_pathname_prefix: str = None):
    server = flask.Flask(__name__)
    full_path = os.path.join(_BASE_DIR, "catabolites", requests_pathname_prefix)
    app = dash.Dash(__name__, server=server,
                    requests_pathname_prefix=f"/{requests_pathname_prefix}/",
                    external_stylesheets=[dbc.themes.BOOTSTRAP],
                    external_scripts=[
                        {"type": "text/javascript", "id": "MathJax-script-config",
                         "children": (
                             "window.MathJax = {"
                             "  tex: { packages: {'[+]': ['ams', 'boldsymbol']} },"
                             "  startup: { typeset: false }"
                             "};"
                         )},
                        {"src": "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js",
                         "id": "MathJax-script", "async": True},
                    ])
    SIDEBAR_STYLE = {
        "position": "fixed", "top": 0, "left": 0, "bottom": 0,
        "width": "16rem", "padding": "2rem 1rem",
        "background-color": "#f8f9fa", "overflowY": "auto",
    }
    CONTENT_STYLE = {
        "margin-left": "18rem", "margin-right": "2rem", "padding": "2rem 1rem",
    }
    _base = f"/{requests_pathname_prefix}"

    def make_layout():
        """Called by Dash on every fresh page load — reads navlinks from disk."""
        lines, _, _, _, _, _ = _load_page_data(full_path)
        navlinks = [dbc.NavLink("Home", href=f"{_base}/", active="exact")]
        for idx, sentence in enumerate(lines):
            navlinks.append(dbc.NavItem(
                dbc.NavLink(f"{idx}. {sentence}", href=f"{_base}/{idx}", active="exact")))
        sidebar = html.Div([
            html.H2("Sentences", className="display-4"),
            html.Hr(),
            html.P("The list of the sentences from the database", className="lead"),
            dbc.Nav(navlinks, vertical=True, pills=True),
        ], style=SIDEBAR_STYLE)
        content = html.Div(id="page-content", style=CONTENT_STYLE)
        return html.Div([dcc.Location(id="url"), sidebar, content,
                         html.Div(id="_mathjax-trigger", style={"display": "none"})])

    app.layout = make_layout

    app.clientside_callback(
        """
        function(children) {
            if (window.MathJax && window.MathJax.typesetPromise) {
                var el = document.getElementById('page-content');
                if (el) { window.MathJax.typesetPromise([el]); }
            }
            return '';
        }
        """,
        Output("_mathjax-trigger", "children"),
        Input("page-content", "children"),
    )

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        lines, LS, pairwise, local_meu, matrices, graph_indices = _load_page_data(full_path)
        try:
            val = int(pathname.rstrip("/").rsplit("/", 1)[-1])
            df = pandas.DataFrame(local_meu[val])
            from LaSSI.viz import NestedTables
            table = NestedTables.generate_morphism_html(
                os.path.join(full_path, "viz"), str(val))
            children = [
                html.H1("Original sentence"), html.Hr(),
                html.P(lines[val]),
                html.H1("Logical Representation"), html.Hr(),
                LS[val],
            ]
            pairwise_blocks = [
                _render_pairwise_table(pairwise[(val, j)], lines)
                for j in range(len(lines))
                if j != val and (val, j) in pairwise
            ]
            if pairwise_blocks:
                children += [
                    html.H1("Pairwise Truth Tables (ex post CWA)"), html.Hr(),
                    html.P(
                        "For each target sentence Sj, rows are the possible worlds where "
                        "the current sentence holds. Columns are the shared atomic "
                        "constituents; Sᵢ holds / Sⱼ holds show whether each sentence "
                        "is true in that world.",
                        style={"color": "#555", "fontSize": "0.9em"},
                    ),
                    *pairwise_blocks,
                ]
            _iframe_style = {
                "width": "100%", "height": "80vh",
                "border": "1px solid #dee2e6", "borderRadius": "4px",
                "marginBottom": "0.5em",
            }
            row_graph_idxs = graph_indices[val] if val < len(graph_indices) else [val]
            gsm_blocks = []
            for gidx in row_graph_idxs:
                viz_base = os.path.join(full_path, "viz", str(gidx))
                if not os.path.exists(os.path.join(viz_base, "input.json")):
                    continue
                sub_label = f" (sub-sentence {gidx})" if len(row_graph_idxs) > 1 else ""
                gsm_blocks += [
                    html.H2(f"Input graph{sub_label}"),
                    html.Iframe(src=f"/gsm/{requests_pathname_prefix}/{gidx}/input",
                                style=_iframe_style),
                    html.H2(f"Result graph{sub_label}"),
                    html.Iframe(src=f"/gsm/{requests_pathname_prefix}/{gidx}/result",
                                style=_iframe_style),
                ]
            if gsm_blocks:
                children += [html.H1("GSM Graphs"), html.Hr()] + gsm_blocks
            children += [
                html.H1("Morphisms"), table,
                html.H1("MeuDB"),
                dash_table.DataTable(local_meu[val],
                                     [{"name": i, "id": i} for i in df.columns]),
            ]
            return html.Div(children)
        except Exception:
            pass
        if pathname.rstrip("/") in ("/", _base):
            matrices_final = []
            for matrix_name, fig in matrices.items():
                matrices_final.append(html.H2(matrix_name))
                matrices_final.append(dcc.Graph(figure=fig))
            if matrices_final:
                matrices_final.insert(0, html.H1("Confusion Matrices"))
                return html.Div(matrices_final)
        return html.Div([html.P("Please select an item from the right bar.")],
                        className="p-3 bg-light rounded-3")

    return app

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse as FastAPIHTMLResponse

# ---------------------------------------------------------------------------
# GSM graph helpers – mirrors parsers/GSMExt.py from the datagram-db visualizer
# ---------------------------------------------------------------------------

def _gsm_to_vis_nodes(objects: list, removed_ids=None, inserted_ids=None) -> list:
    removed_ids = set(removed_ids or [])
    inserted_ids = set(inserted_ids or [])
    nodes = []
    for obj in objects:
        nid = obj["id"]
        ell = obj.get("ell", [])
        xi = obj.get("xi", [])
        label = (ell[0] if ell else "--") + "\n" + (xi[0] if xi else "--")
        node = {
            "id": nid,
            "label": label,
            "title": str(nid),
            "font": {"face": "Monospace", "align": "left"},
            "ell": ell,
            "xi": xi,
            "properties": obj.get("properties", {}),
        }
        if nid in removed_ids:
            node["color"] = {"background": "#f8d7da", "border": "#c00"}
        elif nid in inserted_ids:
            node["color"] = {"background": "#d2cceb", "border": "#7b5ea7"}
        nodes.append(node)
    return nodes


def _gsm_to_vis_edges(objects: list) -> list:
    edges = []
    for obj in objects:
        parent = obj["id"]
        by_label: dict = {}
        for phi in obj.get("phi", []):
            by_label.setdefault(phi.get("containment", ""), []).append(phi)
        for lbl, items in by_label.items():
            for i, phi in enumerate(items):
                edges.append({
                    "from": parent,
                    "to": phi["content"],
                    "label": lbl,
                    "font": {"align": "middle"},
                    "arrows": "to",
                    "instance": i,
                    "title": f"({parent},{lbl},{phi['content']},{i})",
                    "properties": phi.get("properties", {}),
                })
    return edges


_VIS_CDN = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"


def _gsm_graph_page(nodes_json: list, edges_json: list, title: str) -> str:
    """Return a self-contained HTML page that renders a vis.js network."""
    nodes_str = json.dumps(nodes_json)
    edges_str = json.dumps(edges_json)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  html,body{{margin:0;padding:0;font-family:monospace;height:100%;}}
  #net{{width:100%;height:82vh;border-bottom:1px solid #ccc;}}
  #info{{padding:8px 12px;font-size:0.82em;max-height:26vh;overflow-y:auto;
         background:#fafafa;}}
  table{{border-collapse:collapse;width:100%;}}
  th,td{{border:1px solid #ddd;padding:4px 8px;text-align:left;}}
  th{{background:#04AA6D;color:#fff;}}
  tr:nth-child(even){{background:#f2f2f2;}}
</style>
</head>
<body>
<div id="net"></div>
<div id="info"><i>Click a node or edge for details.</i></div>
<script src="{_VIS_CDN}"></script>
<script>
var nodesData={nodes_str};
var edgesData={edges_str};
var nodes=new vis.DataSet(nodesData);
var edges=new vis.DataSet(edgesData);
var net=new vis.Network(document.getElementById("net"),
  {{nodes:nodes,edges:edges}},
  {{interaction:{{hover:true}},nodes:{{shape:"dot"}}}});
function renderProps(props){{
  if(!props||Object.keys(props).length===0) return "<i>none</i>";
  return "<table><tr><th>Key</th><th>Value</th></tr>"
    +Object.entries(props).map(([k,v])=>"<tr><td>"+k+"</td><td>"+v+"</td></tr>").join("")
    +"</table>";
}}
net.on("click",function(p){{
  var info=document.getElementById("info");
  if(p.nodes.length===1){{
    var n=nodes.get(p.nodes[0]);
    info.innerHTML="<b>Node "+n.id+"</b><br>"
      +"<b>Labels:</b> "+(n.ell.length?n.ell.join(", "):"<i>none</i>")+"<br>"
      +"<b>Values:</b> "+(n.xi.length?n.xi.join(", "):"<i>none</i>")+"<br>"
      +"<b>Properties:</b><br>"+renderProps(n.properties);
  }}else if(p.edges.length===1){{
    var e=edges.get(p.edges[0]);
    info.innerHTML="<b>Edge</b> "+e.from+" &rarr;["+e.label+"]&rarr; "+e.to
      +" <i>(instance "+e.instance+")</i><br>"
      +"<b>Properties:</b><br>"+renderProps(e.properties);
  }}else{{
    info.innerHTML="<i>Click a node or edge for details.</i>";
  }}
}});
</script>
</body>
</html>"""


def _load_gsm_graph(catabolite: str, sent_idx: int, graph_type: str):
    """Load and convert a GSM graph to (nodes, edges) vis.js lists."""
    base = os.path.join(_BASE_DIR, "catabolites", catabolite, "viz", str(sent_idx))
    graph_file = os.path.join(base, f"{graph_type}.json")
    if not os.path.exists(graph_file):
        return None, None, None, graph_file
    with open(graph_file) as f:
        objects = json.load(f)
    removed, inserted = [], []
    removed_path = os.path.join(base, "removed.json")
    inserted_path = os.path.join(base, "inserted.json")
    if os.path.exists(removed_path):
        with open(removed_path) as f:
            removed = json.load(f)
    if os.path.exists(inserted_path):
        with open(inserted_path) as f:
            inserted = json.load(f)
    nodes = _gsm_to_vis_nodes(objects, removed, inserted)
    edges = _gsm_to_vis_edges(objects)
    return nodes, edges, removed, inserted


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest


app = FastAPI()


class _AutoMountMiddleware(BaseHTTPMiddleware):
    """On every request, scan for new catabolite directories and mount them."""
    async def dispatch(self, request: StarletteRequest, call_next):
        for name in _current_catabolites():
            _ensure_mounted(name)
        return await call_next(request)


@app.get("/gsm/{catabolite}/{sent_idx}/{graph_type}", response_class=FastAPIHTMLResponse)
def gsm_graph(catabolite: str, sent_idx: int, graph_type: str):
    nodes, edges, _, meta = _load_gsm_graph(catabolite, sent_idx, graph_type)
    if nodes is None:
        return FastAPIHTMLResponse(
            f"<html><body><p>Graph not found.</p><pre>{meta}</pre></body></html>",
            status_code=404,
        )
    title = f"{catabolite} / sentence {sent_idx} / {graph_type}"
    return FastAPIHTMLResponse(_gsm_graph_page(nodes, edges, title))


from starlette.routing import Mount

app.add_middleware(_AutoMountMiddleware)

_mounted_catabolites: set = set()


def _current_catabolites() -> list:
    """Read the catabolites directory on every call so new datasets appear immediately."""
    desktop = pathlib.Path(os.path.join(_BASE_DIR, "catabolites"))
    return sorted(item.name for item in desktop.iterdir() if item.is_dir())


def _ensure_mounted(name: str) -> None:
    """Mount a Dash app for *name* if it hasn't been mounted yet."""
    if name in _mounted_catabolites:
        return
    dash_app = create_dash_app(requests_pathname_prefix=name)
    app.router.routes.append(Mount(f"/{name}", WSGIMiddleware(dash_app.server)))
    _mounted_catabolites.add(name)


@app.get("/", response_class=FastAPIHTMLResponse)
def read_main():
    catabolites = _current_catabolites()
    for name in catabolites:
        _ensure_mounted(name)

    cards = ""
    for name in catabolites:
        full = os.path.join(_BASE_DIR, "catabolites", name)
        # Count sentences from string_rep.txt if present
        n_sentences = 0
        sr = os.path.join(full, "string_rep.txt")
        if os.path.exists(sr):
            with open(sr) as f:
                n_sentences = sum(1 for _ in f)
        badge = f'<span class="badge bg-secondary">{n_sentences} sentence{"s" if n_sentences != 1 else ""}</span>'
        cards += f"""
        <div class="col">
          <div class="card h-100 shadow-sm">
            <div class="card-body d-flex flex-column">
              <h5 class="card-title">{name}</h5>
              <p class="card-text">{badge}</p>
              <a href="/{name}/" class="btn btn-primary mt-auto">Open</a>
            </div>
          </div>
        </div>"""

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>LaSSI Visualiser</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
</head>
<body class="bg-light">
<div class="container py-5">
  <h1 class="mb-1">LaSSI Visualiser</h1>
  <p class="text-muted mb-4">Select a dataset to explore its sentences and logical representations.</p>
  <div class="row row-cols-1 row-cols-sm-2 row-cols-md-3 row-cols-lg-4 g-4">
    {cards if cards else '<p class="text-muted">No catabolite datasets found.</p>'}
  </div>
</div>
</body>
</html>"""
    return FastAPIHTMLResponse(html_page)


@app.get("/status")
def get_status():
    return {"status": "ok"}


# Mount all catabolites present at startup
for _name in _current_catabolites():
    _ensure_mounted(_name)

if __name__ == "__main__":
    # htmltbl = NestedTables.generate_morphism_html(os.path.join("/home/giacomo/projects/LaSSI/catabolites/alice_bob", "viz"),
    #                                     "0")
    # # table = convert_html_to_dash(htmltbl)
    # print(table)
    uvicorn.run(app, port=8000)

