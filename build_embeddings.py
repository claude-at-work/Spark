"""
Chunk the round3 run logs by their internal structure, embed the chunks,
project to 3D, and write an interactive Plotly visualization.

Chunking strategy
-----------------
Each file is a JSON record of one model's response to the meta-prompt. The
responses are consistently organized into numbered markdown sections
(### 1. ..., ### 2. ...) where each section is one distinct observational
"facet" (technical architecture, visual hierarchy, semantic content, etc).
That internal segmentation is the structure that matters: it is how the
models themselves partitioned their own reading. So we chunk by those
headers. When a `thinking_trace` is present, it is chunked the same way
(falling back to double-newline paragraphs, since traces aren't always
header-structured).

Embedding
---------
No network calls, no model downloads: we use TF-IDF over a sublinear,
bigram vocabulary, reduce with TruncatedSVD to a dense 50-d space, then
project to 3D with t-SNE. That preserves local neighborhoods well, which
is what matters for "plot their associations."
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import plotly.graph_objects as go

ROOT = Path(__file__).parent
OUT_HTML = ROOT / "embeddings_3d.html"

HEADER_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    section: str          # header title or "(prelude)" / "(paragraph N)"
    source: str           # "response" or "thinking"
    model: str
    artifact: str         # "structured" | "flat"
    thinking_enabled: bool
    file: str             # basename
    chunk_id: int         # index within its source


def split_by_headers(text: str) -> list[tuple[str, str]]:
    """Return list of (section_title, body) pairs, splitting on markdown headers."""
    if not text:
        return []
    # Find all header positions
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        # No headers: fall back to double-newline paragraphs
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [(f"(paragraph {i+1})", p) for i, p in enumerate(parts)]

    sections: list[tuple[str, str]] = []
    # Prelude before first header
    if matches[0].start() > 0:
        prelude = text[: matches[0].start()].strip()
        if prelude:
            sections.append(("(prelude)", prelude))
    for i, m in enumerate(matches):
        title = re.sub(r"^#+\s+", "", m.group(0)).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    files = sorted(ROOT.glob("20260415_*_round3.json"))
    for path in files:
        rec = json.loads(path.read_text())
        base = path.name
        # Response chunks
        for i, (title, body) in enumerate(split_by_headers(rec.get("response") or "")):
            chunks.append(
                Chunk(
                    text=body,
                    section=title,
                    source="response",
                    model=rec["model"],
                    artifact=rec["artifact"],
                    thinking_enabled=bool(rec.get("thinking_enabled")),
                    file=base,
                    chunk_id=i,
                )
            )
        # Thinking-trace chunks (when present)
        trace = rec.get("thinking_trace")
        if trace:
            for i, (title, body) in enumerate(split_by_headers(trace)):
                chunks.append(
                    Chunk(
                        text=body,
                        section=title,
                        source="thinking",
                        model=rec["model"],
                        artifact=rec["artifact"],
                        thinking_enabled=bool(rec.get("thinking_enabled")),
                        file=base,
                        chunk_id=i,
                    )
                )
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
    )
    X = vec.fit_transform(texts)
    n_comp = min(50, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    Xd = svd.fit_transform(X)
    # L2-normalize so distances in SVD space approximate cosine
    norms = np.linalg.norm(Xd, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return Xd / norms


def project_3d(X: np.ndarray) -> np.ndarray:
    perplexity = max(5, min(30, (X.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,
        metric="cosine",
    )
    return tsne.fit_transform(X)


def knn_edges(X: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    edges: dict[tuple[int, int], float] = {}
    for i in range(X.shape[0]):
        for j_pos in range(1, k + 1):
            j = int(idx[i, j_pos])
            d = float(dist[i, j_pos])
            a, b = sorted((i, j))
            if (a, b) not in edges or edges[(a, b)] > d:
                edges[(a, b)] = d
    return [(a, b, d) for (a, b), d in edges.items()]


MODEL_COLORS = {
    "glm-5:cloud": "#e45756",
    "glm-5.1:cloud": "#f58518",
    "gemini-3-flash-preview:latest": "#4c78a8",
    "rnj-1:8b-cloud": "#54a24b",
}
ARTIFACT_SYMBOL = {"structured": "circle", "flat": "diamond"}


def short(text: str, n: int = 240) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def build_figure(chunks: list[Chunk], coords: np.ndarray, edges) -> go.Figure:
    fig = go.Figure()

    # Edges as a single line trace with None separators
    if edges:
        ex, ey, ez = [], [], []
        for a, b, _d in edges:
            ex += [coords[a, 0], coords[b, 0], None]
            ey += [coords[a, 1], coords[b, 1], None]
            ez += [coords[a, 2], coords[b, 2], None]
        fig.add_trace(
            go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(color="rgba(140,140,140,0.25)", width=1),
                hoverinfo="skip",
                name="k-NN associations",
                showlegend=True,
            )
        )

    # Group points by (model, artifact, thinking) so legend is tidy
    groups: dict[tuple, list[int]] = {}
    for i, c in enumerate(chunks):
        groups.setdefault((c.model, c.artifact, c.thinking_enabled), []).append(i)

    for (model, artifact, thinking), idxs in sorted(groups.items()):
        color = MODEL_COLORS.get(model, "#888888")
        symbol = ARTIFACT_SYMBOL.get(artifact, "square")
        # thinking runs get an outlined marker
        line = dict(color="white", width=1.5) if thinking else dict(color=color, width=0)
        name = f"{model} · {artifact}" + (" · thinking" if thinking else "")

        xs = coords[idxs, 0]
        ys = coords[idxs, 1]
        zs = coords[idxs, 2]
        hover = [
            f"<b>{chunks[i].model}</b> · {chunks[i].artifact}"
            f"{' · thinking' if chunks[i].thinking_enabled else ''}<br>"
            f"<i>{chunks[i].source} — {chunks[i].section}</i><br>"
            f"<span style='font-size:11px'>{short(chunks[i].text)}</span>"
            for i in idxs
        ]
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(
                    size=6 if not thinking else 7,
                    color=color,
                    symbol=symbol,
                    line=line,
                    opacity=0.92,
                ),
                name=name,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Round-3 chunk embeddings — structure-aware sections projected to 3D",
            x=0.02, xanchor="left",
        ),
        paper_bgcolor="#0a0a0a",
        font=dict(color="#c8c8c8", family="Courier New, monospace"),
        scene=dict(
            xaxis=dict(title="", backgroundcolor="#0a0a0a", gridcolor="#222",
                       zerolinecolor="#222", color="#888"),
            yaxis=dict(title="", backgroundcolor="#0a0a0a", gridcolor="#222",
                       zerolinecolor="#222", color="#888"),
            zaxis=dict(title="", backgroundcolor="#0a0a0a", gridcolor="#222",
                       zerolinecolor="#222", color="#888"),
            bgcolor="#0a0a0a",
        ),
        legend=dict(bgcolor="rgba(10,10,10,0.6)", bordercolor="#333", borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=820,
    )
    return fig


def main() -> None:
    chunks = load_chunks()
    print(f"loaded {len(chunks)} chunks from {len(set(c.file for c in chunks))} files")
    texts = [c.text for c in chunks]
    X = embed(texts)
    print(f"svd space: {X.shape}")
    coords = project_3d(X)
    edges = knn_edges(X, k=3)
    print(f"{len(edges)} k-NN edges")
    fig = build_figure(chunks, coords, edges)
    fig.write_html(OUT_HTML, include_plotlyjs="cdn", full_html=True)
    print(f"wrote {OUT_HTML}")

    # Also emit a small JSON index with the chunk metadata + coords, handy for
    # re-plotting later without re-running t-SNE.
    index = [
        {**asdict(c), "x": float(coords[i, 0]),
         "y": float(coords[i, 1]), "z": float(coords[i, 2])}
        for i, c in enumerate(chunks)
    ]
    (ROOT / "embeddings_index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {ROOT / 'embeddings_index.json'}")


if __name__ == "__main__":
    main()
