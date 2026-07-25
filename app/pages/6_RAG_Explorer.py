"""RAG Explorer — browse the Chroma collection: chunks, metadata, embeddings."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from advisor.config import settings  # noqa: E402
from advisor.rag.store import get_or_create_collection  # noqa: E402

from app.components.theme import BRAND_NAME, apply_theme  # noqa: E402

st.set_page_config(page_title=f"RAG Explorer · {BRAND_NAME}",
                    page_icon=":mag:", layout="wide")

apply_theme(page_key="RAG Explorer")

st.markdown(f'<div class="nw-hero-title">RAG Explorer</div>',
                unsafe_allow_html=True)
st.markdown(
    '<div class="nw-hero-sub">Browse the Chroma vector store used for '
    'retrieval-augmented Q&A.</div>',
    unsafe_allow_html=True,
)

collection = get_or_create_collection()
total = collection.count()

if total == 0:
    st.info("The Chroma collection is empty. Run `make index` to build it "
                "from `corpus/`.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Chunks indexed", f"{total:,}")
col2.metric("Embedding model", settings.embed_model_id)
col3.metric("Chroma path", str(settings.chroma_path))

st.divider()

# ------------------------- Browse chunks -------------------------
st.markdown("### Browse chunks")

all_data = collection.get(include=["documents", "metadatas", "embeddings"])
sources = sorted({(m or {}).get("source", "unknown") for m in all_data["metadatas"]})

f_col1, f_col2 = st.columns([1, 2])
with f_col1:
    source_filter = st.selectbox("Filter by source", ["All"] + sources)
with f_col2:
    text_filter = st.text_input("Filter by text (contains)", placeholder="e.g. IRA, deductible…")

rows = []
for i, (doc_id, doc, meta) in enumerate(
    zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
):
    meta = meta or {}
    src = meta.get("source", "unknown")
    if source_filter != "All" and src != source_filter:
        continue
    if text_filter and text_filter.lower() not in doc.lower():
        continue
    rows.append({
        "index": i,
        "id": doc_id,
        "source": src,
        "chunk": meta.get("chunk_index", meta.get("chunk", "")),
        "preview": doc[:180].replace("\n", " "),
    })

st.caption(f"Showing {len(rows)} of {total} chunks.")
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, height=360, hide_index=True)

st.divider()

# ------------------------- Inspect one chunk's embedding -------------------------
st.markdown("### Inspect a chunk's embedding")

if rows:
    options = {r["index"]: f'[{r["source"]}] {r["preview"][:80]}…' for r in rows}
    picked = st.selectbox(
        "Pick a chunk", options=list(options.keys()), format_func=lambda i: options[i],
    )
    doc = all_data["documents"][picked]
    meta = all_data["metadatas"][picked] or {}
    emb = all_data["embeddings"][picked]

    st.markdown("**Full chunk text**")
    st.text_area("chunk_text", value=doc, height=160, label_visibility="collapsed")

    st.markdown("**Metadata**")
    st.json(meta)

    m1, m2 = st.columns(2)
    m1.metric("Embedding dimensions", len(emb))
    m2.metric("Vector L2 norm", f"{sum(x * x for x in emb) ** 0.5:.4f}")

    st.markdown("**Embedding vector** (all dimensions)")
    st.bar_chart(pd.DataFrame({"value": emb}))
    with st.expander("Raw values"):
        st.code(", ".join(f"{x:.4f}" for x in emb), language="text")
else:
    st.info("No chunks match the current filters.")

st.divider()

# ------------------------- Semantic search test -------------------------
st.markdown("### Test semantic search")
st.caption("Runs the same embedding model used at query time and shows the "
            "nearest chunks by vector distance — this is what the ReAct "
            "advisor retrieves before answering.")

query = st.text_input("Query", placeholder="e.g. How much can I contribute to an IRA?")
n_results = st.slider("Number of results", 1, 10, 5)

if query:
    results = collection.query(query_texts=[query], n_results=n_results,
                                    include=["documents", "metadatas", "distances"])
    for rank, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ), start=1):
        meta = meta or {}
        with st.container(border=True):
            st.markdown(f"**#{rank}** · `{meta.get('source', 'unknown')}` · "
                            f"distance `{dist:.4f}`")
            st.write(doc[:400] + ("…" if len(doc) > 400 else ""))
