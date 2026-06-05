import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas
from dash import html


_COLOURS = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
]


def _card(children, border="#dee2e6", background="#fff", extra_style=None):
    style = {
        "border": f"1px solid {border}",
        "borderRadius": "6px",
        "background": background,
        "padding": "0.9rem 1rem",
        "marginBottom": "1rem",
    }
    if extra_style:
        style.update(extra_style)
    return html.Div(children, style=style)


def _notice(message: str, kind: str = "warning"):
    colours = {
        "warning": ("#fff3cd", "#ffecb5", "#664d03"),
        "error": ("#f8d7da", "#f5c2c7", "#842029"),
        "info": ("#cff4fc", "#b6effb", "#055160"),
    }
    bg, border, text = colours.get(kind, colours["warning"])
    return _card(message, border=border, background=bg, extra_style={"color": text})


def _latex(value, block=False):
    text = str(value)
    component = html.Div if block else html.Span
    return component(
        f"\\({text}\\)",
        style={
            "fontSize": "0.95rem",
            "lineHeight": "1.55",
            "overflowWrap": "anywhere",
        },
    )


def _safe_int_sort(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def _display_value(value):
    if pandas.isna(value):
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _dataset_candidates(base_dir: Path, catabolite_name: str) -> Iterable[Path]:
    yield base_dir / "catabolites" / catabolite_name / "dataset_path.txt"
    yield base_dir / f"{catabolite_name}.yaml"
    yield base_dir / "neet" / "evidence_cases" / f"{catabolite_name}.yaml"
    yield base_dir / "test_sentences" / "evidence_cases" / f"{catabolite_name}.yaml"
    yield base_dir / "test_sentences" / "CURRENT" / f"{catabolite_name}.yaml"
    yield base_dir / "test_sentences" / f"{catabolite_name}.yaml"


@lru_cache(maxsize=64)
def resolve_dataset_path(base_dir: str, catabolite_name: str) -> str | None:
    base = Path(base_dir).resolve()
    for candidate in _dataset_candidates(base, catabolite_name):
        if candidate.name == "dataset_path.txt" and candidate.exists():
            path = Path(candidate.read_text().strip()).expanduser()
            if not path.is_absolute():
                path = base / path
            if path.exists() and path.stem == catabolite_name:
                return str(path)
        elif candidate.exists():
            return str(candidate)

    ignored = {".git", "catabolites", "cache", "results", "__pycache__"}
    for candidate in base.rglob(f"{catabolite_name}.yaml"):
        if ignored.intersection(candidate.relative_to(base).parts):
            continue
        return str(candidate)
    return None


def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@lru_cache(maxsize=8)
def _load_explainer_state(dataset_path: str, connection_path: str,
                          logical_mtime: float, internals_mtime: float,
                          string_rep_mtime: float):
    del logical_mtime, internals_mtime, string_rep_mtime
    from LaSSI.explainer.LaSSIExplainer import LaSSIExplainer
    from LaSSI.external_services.Services import Services

    try:
        Services.getInstance(lambda _: None)
    except Exception:
        pass

    try:
        return LaSSIExplainer(dataset_path), None
    except AssertionError:
        try:
            LaSSIExplainer.start_up_services(connection_path)
            return LaSSIExplainer(dataset_path), None
        except Exception as exc:
            return None, (
                "LaSSIExplainer needed HOnK/TBox services, but service startup "
                f"or retry failed. {type(exc).__name__}: {exc}"
            )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _get_explainer(catabolite_name: str, full_path: str, base_dir: str):
    dataset_path = resolve_dataset_path(base_dir, catabolite_name)
    if dataset_path is None:
        raise FileNotFoundError(
            f"Could not find the source YAML for catabolite '{catabolite_name}'."
        )
    connection_path = os.environ.get(
        "LASSI_CONNECTION",
        os.path.join(base_dir, "connection.yaml"),
    )
    logical_path = os.path.join(full_path, "logical_rewriting.json")
    internals_path = os.path.join(full_path, "internals.json")
    string_rep_path = os.path.join(full_path, "string_rep.txt")
    explainer, error = _load_explainer_state(
        dataset_path,
        connection_path,
        _mtime(logical_path),
        _mtime(internals_path),
        _mtime(string_rep_path),
    )
    if error:
        raise RuntimeError(error)
    return explainer


def _highlighted_sentence(explainer, idx: int):
    try:
        pieces = explainer.explain_textual_sentence(idx)
    except Exception:
        return html.Span(explainer.sentences[idx])

    children = []
    for piece in pieces:
        if isinstance(piece, str):
            children.append(html.Span(piece))
            continue

        spans = sorted(piece, key=lambda x: (x.min, x.max, x.id or -1, x.value))
        if not spans:
            continue
        value = spans[0].value
        ids = sorted({span.id for span in spans if span.id is not None})
        colour_key = ids[0] if ids else 0
        colour = _COLOURS[colour_key % len(_COLOURS)]
        children.append(html.Mark([
            value,
            html.Sup(",".join(str(x) for x in ids), style={"marginLeft": "2px"}) if ids else None,
        ], title=", ".join(f"logical constituent {x}" for x in ids), style={
            "background": colour,
            "padding": "0 2px",
            "borderRadius": "3px",
        }))
    return html.Span(children)


def _sentence_versions(explainer, idx: int):
    original_sentences = getattr(explainer, "original_sentences", explainer.sentences)
    original = original_sentences[idx] if idx < len(original_sentences) else explainer.sentences[idx]
    ingested = explainer.sentences[idx]
    return original, ingested


def _text_block(label: str, content, highlighted: bool = False):
    return html.Div([
        html.Div(label, style={
            "fontWeight": "700",
            "fontSize": "0.82rem",
            "color": "#495057",
            "marginBottom": "0.2rem",
        }),
        html.P(content, style={
            "marginBottom": "0",
            "lineHeight": "1.5",
            "fontFamily": "monospace" if not highlighted else "inherit",
            "fontSize": "0.9rem",
        }),
    ], style={
        "background": "#f8f9fa",
        "border": "1px solid #e9ecef",
        "borderRadius": "4px",
        "padding": "0.55rem 0.65rem",
        "marginBottom": "0.6rem",
    })


def _sentence_panel(explainer, idx: int, title: str, sentence_explanation=None):
    original, ingested = _sentence_versions(explainer, idx)
    highlight_note = (
        "Highlights mark logical constituents that have source-text spans. "
        "Predicate relation labels, such as verbs, are compared in the formula "
        "and constituent motivation, but are only highlighted when the cache "
        "retains a source span for them."
    )
    if original == ingested:
        sentence_children = [
            html.P(_highlighted_sentence(explainer, idx), style={"marginBottom": "0.7rem"}),
            _notice(highlight_note, "info"),
        ]
    else:
        sentence_children = [
            _notice(
                "The input sentence was normalised before logical rewriting. "
                "Highlights are aligned to the ingested pipeline text. "
                + highlight_note,
                "info",
            ),
            _text_block("Original input", original),
            _text_block("Ingested by pipeline", _highlighted_sentence(explainer, idx), True),
        ]

    children = [
        html.H4(title, style={"fontSize": "1rem", "marginBottom": "0.45rem"}),
        *sentence_children,
        html.Div(_latex(explainer.obj_list[idx], block=True), style={
            "background": "#f8f9fa",
            "border": "1px solid #e9ecef",
            "borderRadius": "4px",
            "padding": "0.6rem 0.75rem",
        }),
    ]
    if sentence_explanation is not None:
        children.extend([
            html.H5("Atoms", style={"fontSize": "0.9rem", "marginTop": "0.9rem"}),
            _atoms_table(sentence_explanation.atoms),
        ])
    return _card(children)


def _atoms_table(atoms: dict):
    if not atoms:
        return html.P("No atoms found.", style={"color": "#6c757d"})
    rows = []
    for atom_id, formula in sorted(atoms.items(), key=lambda x: _safe_int_sort(x[0])):
        rows.append(html.Tr([
            html.Th(str(atom_id), style={
                "width": "5rem",
                "padding": "0.35rem 0.5rem",
                "verticalAlign": "top",
                "background": "#f1f3f5",
            }),
            html.Td(_latex(formula), style={"padding": "0.35rem 0.5rem"}),
        ]))
    return html.Div(html.Table(rows, style={
        "width": "100%",
        "borderCollapse": "collapse",
        "fontSize": "0.88rem",
    }), style={"overflowX": "auto"})


def _verdict(confidence: float):
    if confidence == 1.0:
        return "Implies", "#198754"
    if confidence == 0.0:
        return "Does Not Imply", "#dc3545"
    if confidence >= 0.75:
        return "Mostly Implies", "#0d6efd"
    if confidence >= 0.25:
        return "Partial", "#fd7e14"
    return "Weak", "#6c757d"


def _truth_table(df, source_idx: int, target_idx: int):
    if df is None or df.empty:
        return _notice("No satisfying worlds were found for this pair.", "info")

    columns = list(df.columns)
    atom_cols = sorted([c for c in columns if not str(c).startswith("R")], key=_safe_int_sort)
    result_cols = [c for c in (f"R{source_idx}", f"R{target_idx}") if c in columns]
    other_cols = [c for c in columns if c not in atom_cols and c not in result_cols]
    ordered_cols = atom_cols + other_cols + result_cols

    def label(col):
        col = str(col)
        if col == f"R{source_idx}":
            return f"S{source_idx}"
        if col == f"R{target_idx}":
            return f"S{target_idx}"
        return f"Atom {col}"

    header = html.Thead(html.Tr([
        html.Th(label(col), style={
            "padding": "0.4rem 0.55rem",
            "textAlign": "center",
            "background": "#1f497d" if str(col).startswith("R") else "#f1f3f5",
            "color": "#fff" if str(col).startswith("R") else "#212529",
            "whiteSpace": "nowrap",
        })
        for col in ordered_cols
    ]))

    body_rows = []
    for _, row in df[ordered_cols].iterrows():
        supports = row.get(f"R{target_idx}", 0) == 1
        cells = []
        for col in ordered_cols:
            is_result = str(col).startswith("R")
            is_target = str(col) == f"R{target_idx}"
            value = row[col]
            background = "#fff"
            if is_target:
                background = "#d1e7dd" if value == 1 else "#f8d7da"
            elif is_result:
                background = "#dce8ff"
            elif value == 1:
                background = "#edf7ed"
            elif value == 0:
                background = "#fff5f5"
            cells.append(html.Td(_display_value(value), style={
                "padding": "0.35rem 0.55rem",
                "textAlign": "center",
                "background": background,
                "fontWeight": "700" if is_target else "400",
            }))
        body_rows.append(html.Tr(cells, style={
            "borderLeft": f"4px solid {'#198754' if supports else '#dc3545'}",
        }))

    return html.Div(html.Table([header, html.Tbody(body_rows)], style={
        "width": "100%",
        "borderCollapse": "collapse",
        "fontSize": "0.86rem",
    }), style={"overflowX": "auto"})


def _constituent_motivation(final_explanation):
    motivations = final_explanation.explained_joined_table.constituent_implication
    if not motivations:
        return html.P(
            "No separate atom-to-atom motivation was required for this pair.",
            style={"color": "#6c757d", "fontSize": "0.88rem"},
        )

    items = []
    for motivation in motivations:
        items.append(html.Li([
            html.Div(str(motivation), style={
                "fontWeight": "700",
                "marginBottom": "0.25rem",
            }),
            html.Div([
                html.Span("lhs: ", style={"fontWeight": "700"}),
                _latex(motivation.lhsF),
            ], style={"marginBottom": "0.2rem"}),
            html.Div([
                html.Span("rhs: ", style={"fontWeight": "700"}),
                _latex(motivation.rhsF),
            ]),
        ], style={"marginBottom": "0.8rem"}))
    return html.Ol(items, style={"paddingLeft": "1.2rem", "fontSize": "0.88rem"})


def _pair_panel(explainer, source_idx: int, target_idx: int):
    final_explanation = explainer.get_explanation(source_idx, target_idx)
    confidence = float(final_explanation.confidence)
    verdict, colour = _verdict(confidence)
    table = final_explanation.explained_joined_table_natural_joined_with_operands
    target_col = f"R{target_idx}"
    supporting = int(table[target_col].sum()) if target_col in table else 0
    total = len(table)

    return _card([
        html.Div([
            html.Span(f"S{source_idx} -> S{target_idx}", style={
                "fontWeight": "700",
                "fontSize": "1rem",
                "marginRight": "0.7rem",
            }),
            html.Span(verdict, style={
                "background": colour,
                "color": "#fff",
                "borderRadius": "4px",
                "padding": "0.12rem 0.55rem",
                "fontWeight": "700",
                "fontSize": "0.82rem",
                "marginRight": "0.6rem",
            }),
            html.Span(f"confidence {confidence:.3f}", style={"color": "#495057"}),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
        html.P(
            f"{supporting} of {total} possible world{'s' if total != 1 else ''} "
            f"where S{source_idx} holds also satisfy S{target_idx}.",
            style={"marginTop": "0.55rem", "color": "#495057", "fontSize": "0.9rem"},
        ),
        html.Div([
            html.Span(f"S{target_idx}: ", style={"fontWeight": "700"}),
            _highlighted_sentence(explainer, target_idx),
        ], style={
            "background": "#f8f9fa",
            "borderLeft": "4px solid #ced4da",
            "padding": "0.45rem 0.65rem",
            "fontSize": "0.9rem",
            "marginBottom": "0.75rem",
        }),
        html.H5("Constituent Motivation", style={"fontSize": "0.92rem", "marginTop": "0.8rem"}),
        _constituent_motivation(final_explanation),
        html.H5("Possible Worlds", style={"fontSize": "0.92rem", "marginTop": "0.8rem"}),
        _truth_table(table, source_idx, target_idx),
    ], border=colour)


def render_explanation_dashboard(catabolite_name: str, full_path: str, selected_idx: int,
                                 sentence_count: int, base_dir: str):
    try:
        explainer = _get_explainer(catabolite_name, full_path, base_dir)
    except Exception as exc:
        return _notice(
            "LaSSIExplainer could not be loaded for this catabolite. "
            f"{type(exc).__name__}: {exc}",
            "error",
        )

    if selected_idx >= len(explainer.sentences):
        return _notice("The selected sentence index is outside the explainer dataset.", "error")

    pair_panels = []
    for target_idx in range(sentence_count):
        if target_idx == selected_idx:
            continue
        try:
            pair_panels.append(_pair_panel(explainer, selected_idx, target_idx))
        except Exception as exc:
            pair_panels.append(_notice(
                f"Could not explain S{selected_idx} -> S{target_idx}. "
                f"{type(exc).__name__}: {exc}",
                "warning",
            ))

    return html.Div([
        _sentence_panel(explainer, selected_idx, f"Sentence S{selected_idx}"),
        html.H3("Directional Explanations", style={
            "fontSize": "1.25rem",
            "margin": "1.4rem 0 0.6rem",
        }),
        html.P(
            "Each block is computed from LaSSIExplainer's CWA explanation object, "
            "including the real constituent motivations and joined possible-world table.",
            style={"color": "#6c757d", "fontSize": "0.9rem"},
        ),
        *pair_panels,
    ])
