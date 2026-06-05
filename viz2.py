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
import glob
import json
import os.path
import pathlib

import dash
from dash import dash_table
import flask
import dash_bootstrap_components as dbc
import pandas
import yaml
from dash import Input, Output, dcc, html

from LaSSI.explainer.dashboard import render_explanation_dashboard

# Capture the working directory once at startup so all path lookups are consistent
# even if uvicorn or any library changes os.getcwd() later.
_BASE_DIR = os.path.abspath(os.getcwd())

# app = dash.Dash()

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


def _available_graph_indices(full_path: str) -> list[int]:
    viz_dir = os.path.join(full_path, "viz")
    if not os.path.isdir(viz_dir):
        return []
    indices = []
    for name in os.listdir(viz_dir):
        path = os.path.join(viz_dir, name)
        if name.isdigit() and os.path.isdir(path):
            indices.append(int(name))
    return sorted(indices)


def _normalise_graph_map(raw_map, row_count: int, graph_count: int) -> list[list[int]] | None:
    if isinstance(raw_map, dict):
        raw_map = raw_map.get("row_to_sub_indices")
    if not isinstance(raw_map, list) or len(raw_map) != row_count:
        return None
    result = []
    seen = set()
    for row in raw_map:
        if not isinstance(row, list):
            return None
        indices = []
        for value in row:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                return None
            if idx < 0:
                return None
            indices.append(idx)
            seen.add(idx)
        result.append(indices)
    if graph_count and any(idx >= graph_count for idx in seen):
        return None
    return result


def _row_graph_indices_from_cache(full_path: str, row_count: int, graph_count: int) -> list[list[int]] | None:
    mapping_path = os.path.join(full_path, "row_subsentence_map.json")
    if not os.path.exists(mapping_path):
        return None
    try:
        with open(mapping_path) as f:
            return _normalise_graph_map(json.load(f), row_count, graph_count)
    except Exception:
        return None


def _dataset_path_for_catabolite(full_path: str) -> str | None:
    dataset_marker = os.path.join(full_path, "dataset_path.txt")
    if not os.path.exists(dataset_marker):
        return None
    try:
        path = pathlib.Path(open(dataset_marker).read().strip()).expanduser()
    except Exception:
        return None
    if not path.is_absolute():
        path = pathlib.Path(_BASE_DIR) / path
    return str(path) if path.exists() else None


def _row_graph_indices_from_dataset(full_path: str, row_count: int, graph_count: int) -> list[list[int]] | None:
    dataset_path = _dataset_path_for_catabolite(full_path)
    if dataset_path is None:
        return None
    try:
        from LaSSI.phases.StructuredSentenceLoader import split_structured
        with open(dataset_path) as f:
            rows = yaml.load(f, Loader=yaml.SafeLoader)
    except Exception:
        return None
    if not isinstance(rows, list) or len(rows) != row_count:
        return None
    mapping = []
    cursor = 0
    for row in rows:
        chunks = split_structured(str(row))
        count = max(1, len(chunks))
        mapping.append(list(range(cursor, cursor + count)))
        cursor += count
    if graph_count and cursor != graph_count:
        return None
    return mapping


def _fallback_row_graph_indices(lines: list[str], graph_count: int) -> list[list[int]]:
    mapping = []
    cursor = 0
    for line in lines:
        n = len(_split_row_sentences(line))
        mapping.append(list(range(cursor, cursor + n)))
        cursor += n
    if graph_count and cursor != graph_count:
        return [[i] for i in range(len(lines))]
    return mapping


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

    # Build mapping from displayed YAML-row index → graph directory indices.
    # Graph/viz caches remain per expanded sub-sentence even after logical
    # outputs are merged back to one line per YAML row.
    graph_count = len(_available_graph_indices(full_path))
    graph_indices = (
        _row_graph_indices_from_cache(full_path, len(lines), graph_count)
        or _row_graph_indices_from_dataset(full_path, len(lines), graph_count)
        or _fallback_row_graph_indices(lines, graph_count)
    )

    LS = []
    lr_path = os.path.join(full_path, "logical_rewriting.json")
    if os.path.exists(lr_path):
        with open(lr_path, "r") as f:
            for logical in json.load(f):
                actual = formula_from_dict(logical)
                LS.append(html.Div(f'\\({actual}\\)',
                                   style={"fontSize": "1.4em", "margin": "1em 0"}))

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

    return lines, LS, local_meu, matrices, graph_indices


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
        "background-color": "#f8f9fa", "overflowY": "auto", "z-index": 10
    }
    CONTENT_STYLE = {
        "margin-left": "18rem", "margin-right": "2rem", "padding": "2rem 1rem",
    }
    _base = f"/{requests_pathname_prefix}"

    def make_layout():
        """Called by Dash on every fresh page load — reads navlinks from disk."""
        lines, _, _, _, _ = _load_page_data(full_path)
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
        content = html.Div(
            dcc.Loading(
                html.Div(id="page-content"),
                id="page-content-loading",
                type="circle",
                color="#1f497d",
                delay_show=200,
            ),
            style=CONTENT_STYLE,
        )
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
        lines, LS, local_meu, matrices, graph_indices = _load_page_data(full_path)
        try:
            val = int(pathname.rstrip("/").rsplit("/", 1)[-1])
            from LaSSI.viz import NestedTables
            row_graph_idxs = graph_indices[val] if val < len(graph_indices) else [val]
            if len(local_meu) == len(lines):
                meu_rows = local_meu[val]
            else:
                meu_rows = []
                for gidx in row_graph_idxs:
                    if gidx < len(local_meu):
                        meu_rows.extend(local_meu[gidx])
            df = pandas.DataFrame(meu_rows)
            children = [
                html.H1("Ingested sentence"), html.Hr(),
                html.P(lines[val]),
                html.H1("Logical Representation"), html.Hr(),
                LS[val],
            ]
            children += [
                html.H1("LaSSIExplainer"), html.Hr(),
                render_explanation_dashboard(
                    requests_pathname_prefix,
                    full_path,
                    val,
                    len(lines),
                    _BASE_DIR,
                ),
            ]
            _iframe_style = {
                "width": "100%", "height": "80vh",
                "border": "1px solid #dee2e6", "borderRadius": "4px",
                "marginBottom": "0.5em",
            }
            gsm_blocks = []
            for pos, gidx in enumerate(row_graph_idxs, start=1):
                viz_base = os.path.join(full_path, "viz", str(gidx))
                if not os.path.exists(os.path.join(viz_base, "input.json")):
                    continue
                sub_label = f" (chunk {pos}, graph {gidx})" if len(row_graph_idxs) > 1 else ""
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
            morphism_blocks = []
            for pos, gidx in enumerate(row_graph_idxs, start=1):
                viz_base = os.path.join(full_path, "viz", str(gidx))
                if not os.path.isdir(viz_base):
                    continue
                sub_label = f"Chunk {pos} / graph {gidx}" if len(row_graph_idxs) > 1 else f"Graph {gidx}"
                morphism_blocks.extend([
                    html.H2(sub_label),
                    NestedTables.generate_morphism_html(
                        os.path.join(full_path, "viz"), str(gidx)),
                ])
            if morphism_blocks:
                children += [html.H1("Morphisms"), *morphism_blocks]
            children += [
                html.H1("MeuDB"),
                dash_table.DataTable(meu_rows,
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
    uvicorn.run(app, port=8000)
