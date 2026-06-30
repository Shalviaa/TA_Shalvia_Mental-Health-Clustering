
from pathlib import Path
from collections import Counter
import re
import string

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KAMUS_DIR = DATA_DIR / "kamus"
VALIDASI_DIR = DATA_DIR / "validasi"

CLUSTER_PATH = DATA_DIR / "hasil_clustering_w2v_tuned_pca100.csv"
TUNING_PATH = DATA_DIR / "bigrams_w2v_tuning_results_pca100.csv"
VECTOR_PATH = DATA_DIR / "w2v_vectors_pca100.npz"

CLUSTER_INFO = {
    0: {
        "title": "Anxiety & Physical Symptoms",
        "short": "Kecemasan dan keluhan fisik",
        "description": "Klaster gejala terkait kecemasan dan keluhan fisik seperti napas sesak, jantung berdebar, lemas, dan pikiran negatif.",
        "keywords": ["cemas lebih", "sesak nafas", "cemas takut", "badan lemas", "jantung debar"],
        "color": "#159947",
        "soft": "#e8f8ee",
    },
    1: {
        "title": "Emotional Distress & Social Conflict",
        "short": "Distres emosional dan konflik sosial",
        "description": "Klaster tekanan emosional, keluh kesah, konflik interpersonal, sudut pandang, dan ekspresi emosi negatif.",
        "keywords": ["putus asa", "keluh kesah", "caci maki", "sudut pandang", "sesak nafas"],
        "color": "#2563eb",
        "soft": "#eaf1ff",
    },
    2: {
        "title": "Hopelessness & Crisis",
        "short": "Keputusasaan dan krisis",
        "description": "Klaster keputusasaan, kehilangan arah, keinginan pergi, dan indikasi krisis psikologis yang lebih kuat.",
        "keywords": ["putus asa", "sehat mental", "hilang arah", "keluh kesah", "ingin pergi"],
        "color": "#7c3aed",
        "soft": "#f2eafe",
    },
}

VALIDATION_METRICS = {
    "silhouette": 0.5536,
    "calinski": 11360.55,
    "davies": 0.8472,
}

RESEARCH_INFO = {
    "title": "Pengelompokan Gejala Gangguan Kesehatan Mental Berdasarkan Curahan Hati Pengguna di Media Sosial Menggunakan Algoritma K-Means Clustering",
    "author": "Shalvia Retno Salsabil",
    "student_id": "1202220027",
    "program": "S1 Sistem Informasi",
    "faculty": "Fakultas Rekayasa Industri",
    "university": "Universitas Telkom",
    "year": "2026",
}

st.set_page_config(
    page_title="Mental Health Clustering",
    page_icon="MH",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --navy: #06133a;
    --muted: #60708f;
    --blue: #2563eb;
    --line: #dbe3ef;
    --card: #ffffff;
    --bg: #f8fbff;
}
.stApp { background: linear-gradient(120deg, #fbfdff 0%, #f7faff 100%); color: var(--navy); }
[data-testid="stSidebar"] { background: #fbfcff; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
.block-container { padding-top: 2.1rem; max-width: 1280px; }
h1, h2, h3 { color: var(--navy); letter-spacing: 0; }
h1 { font-size: 2.55rem !important; line-height: 1.05; margin-bottom: .25rem; }
h2 { font-size: 1.45rem !important; }
textarea { border-radius: 10px !important; }
.card {
    background: rgba(255,255,255,.92);
    border: 1px solid var(--line);
    border-radius: 10px;
    box-shadow: 0 7px 22px rgba(18, 35, 70, .08);
    padding: 1.35rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: #fff;
    border: 1px solid #dfe7f3;
    border-radius: 10px;
    padding: 1rem 1.15rem;
    min-height: 108px;
}
.metric-label { color: #435173; font-size: .92rem; margin-bottom: .4rem; }
.metric-value { font-size: 1.55rem; font-weight: 800; color: #1e63e9; }
.cluster-row {
    display: grid;
    grid-template-columns: 52px minmax(240px, 1fr) minmax(260px, 1.4fr);
    gap: .8rem;
    align-items: center;
    border-top: 1px solid #e2e8f0;
    padding: .72rem 0;
}
.badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 800;
    width: 34px;
    height: 34px;
    border-radius: 999px;
}
.pill {
    display: inline-block;
    border-radius: 999px;
    padding: .42rem .78rem;
    margin: .22rem .28rem .22rem 0;
    background: #eef4ff;
    color: #145ce6;
    font-weight: 650;
    font-size: .92rem;
}
.small-muted { color: var(--muted); font-size: .96rem; }
.result-title { font-size: 1.9rem; font-weight: 850; margin-bottom: .1rem; }
.success-chip {
    display:inline-block; background:#16843c; color:#fff; padding:.34rem .8rem; border-radius:999px;
    font-weight:800; font-size:.88rem; margin-left:.6rem;
}
.info-box {
    border: 1px solid #b8cef8;
    background: #f7fbff;
    border-radius: 10px;
    padding: 1.2rem;
}
.sidebar-title { font-size:1.5rem; font-weight:850; line-height:1.1; margin-bottom:2rem; color:#06133a; }
.sidebar-logo { font-size:2.6rem; color:#7c3aed; margin-bottom:.5rem; }
hr { border: none; border-top: 1px solid #e0e7f0; margin: 1rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-logo">MH</div><div class="sidebar-title">Mental Health<br/>Clustering</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigasi",
        ["Input Teks", "Hasil Klastering", "Validasi Psikolog", "Tentang Penelitian"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="small-muted"><b>Tugas Akhir</b><br/>{RESEARCH_INFO["author"]}<br/>{RESEARCH_INFO["student_id"]}<br/><br/>Pemetaan teks ke klaster gejala kesehatan mental berdasarkan hasil penelitian.</div>',
        unsafe_allow_html=True,
    )
    return page


@st.cache_data(show_spinner=False)
def load_cluster_data():
    df = pd.read_csv(CLUSTER_PATH)
    df = df.dropna(subset=["lemmas"]).copy()
    df["cluster"] = df["cluster"].astype(int)
    df["tokens_lemma"] = df["lemmas"].astype(str).apply(str.split)
    return df


@st.cache_data(show_spinner=False)
def load_tuning_data():
    return pd.read_csv(TUNING_PATH)


@st.cache_resource(show_spinner=False)
def load_word_vectors():
    data = np.load(VECTOR_PATH, allow_pickle=True)
    words = data["words"].tolist()
    vectors = data["vectors"].astype("float32")
    vocab = {word: idx for idx, word in enumerate(words)}
    return vocab, vectors


@st.cache_data(show_spinner=False)
def load_kamus():
    master = {}
    for filename in ["kamusalay .csv", "slang_indo.csv", "tambahan_typo.csv"]:
        path = KAMUS_DIR / filename
        if path.exists():
            temp = pd.read_csv(path, header=None, encoding="utf-8").dropna()
            for _, row in temp.iterrows():
                master[str(row[0]).lower().strip()] = str(row[1]).lower().strip()
    stopwords_path = KAMUS_DIR / "custom_stopword.csv"
    stopwords = set()
    if stopwords_path.exists():
        sw = pd.read_csv(stopwords_path)
        col = "stopword" if "stopword" in sw.columns else sw.columns[0]
        stopwords = {str(x).strip() for x in sw[col].dropna() if len(str(x).split()) == 1}
    return master, stopwords


def normalize_repeated_letters(word):
    return re.sub(r"(.)\1{2,}", r"\1\1", word)


def preprocess_text(text):
    master, stopwords = load_kamus()
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    punc = string.punctuation.replace("-", "")
    text = text.translate(str.maketrans("", "", punc))
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    words = [normalize_repeated_letters(w) for w in text.split()]
    words = [master.get(w, w) for w in words]
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return " ".join(words), words


def document_vector(tokens, vocab, vectors):
    rows = [vectors[vocab[word]] for word in tokens if word in vocab]
    if not rows:
        return np.zeros(vectors.shape[1], dtype="float32")
    return np.mean(rows, axis=0)


@st.cache_data(show_spinner=False)
def compute_doc_vectors(_vectors_key="default"):
    df = load_cluster_data()
    vocab, word_vectors = load_word_vectors()
    doc_vectors = np.vstack([document_vector(tokens, vocab, word_vectors) for tokens in df["tokens_lemma"]])
    return doc_vectors


@st.cache_data(show_spinner=False)
def compute_centroids():
    df = load_cluster_data()
    vectors = compute_doc_vectors()
    centroids = {}
    for cluster_id in sorted(df["cluster"].unique()):
        centroids[int(cluster_id)] = vectors[df["cluster"].values == cluster_id].mean(axis=0)
    return centroids


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def keyword_fallback_score(tokens, cluster_id):
    joined = " ".join(tokens)
    score = 0
    for kw in CLUSTER_INFO[cluster_id]["keywords"]:
        if kw in joined:
            score += 1
    return score / max(len(CLUSTER_INFO[cluster_id]["keywords"]), 1)


def predict_cluster(text):
    vocab, word_vectors = load_word_vectors()
    normalized, tokens = preprocess_text(text)
    vec = document_vector(tokens, vocab, word_vectors)
    centroids = compute_centroids()
    raw_scores = {}
    for cluster_id, centroid in centroids.items():
        sim = cosine_similarity(vec, centroid)
        if sim == 0:
            sim = keyword_fallback_score(tokens, cluster_id)
        raw_scores[cluster_id] = sim
    best_cluster = max(raw_scores, key=raw_scores.get)
    sims = np.array([raw_scores[i] for i in sorted(raw_scores)])
    sims = sims - sims.min()
    if sims.sum() == 0:
        probs = np.ones_like(sims) / len(sims)
    else:
        probs = sims / sims.sum()
    score_map = {cluster_id: float(probs[idx]) for idx, cluster_id in enumerate(sorted(raw_scores))}
    return best_cluster, normalized, tokens, raw_scores, score_map


@st.cache_data(show_spinner=False)
def compute_visual_data():
    df = load_cluster_data()
    vectors = compute_doc_vectors()
    pca2 = PCA(n_components=2, random_state=42)
    points = pca2.fit_transform(vectors)
    plot_df = pd.DataFrame({
        "PC1": points[:, 0],
        "PC2": points[:, 1],
        "cluster": df["cluster"].astype(str),
        "label": df["cluster"].map(lambda c: f"Cluster {c}"),
    })

    pca100 = PCA(n_components=min(100, vectors.shape[1]), random_state=42)
    X_pca = pca100.fit_transform(vectors)
    X_norm = normalize(X_pca)
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_norm)
        rows.append({
            "K": k,
            "Inertia": km.inertia_,
            "Silhouette": silhouette_score(X_norm, labels),
            "Calinski-Harabasz": calinski_harabasz_score(X_norm, labels),
            "Davies-Bouldin": davies_bouldin_score(X_norm, labels),
        })
    return plot_df, pd.DataFrame(rows)


def top_bigrams_by_cluster(df, cluster_id, n=5):
    words = " ".join(df.loc[df["cluster"] == cluster_id, "lemmas"].dropna().astype(str)).split()
    return Counter(zip(words, words[1:])).most_common(n)


def page_input():
    st.markdown("# Pemetaan Teks ke Klaster Gejala Kesehatan Mental")
    st.markdown('<div class="small-muted">Masukkan teks/curhatan, lalu sistem akan menunjukkan klaster yang paling sesuai dengan hasil penelitian.</div>', unsafe_allow_html=True)

    default_text = "Akhir-akhir ini saya sering merasa cemas tanpa alasan yang jelas. Jantung saya berdebar-debar, napas terasa sesak, dan badan jadi lemas. Saya sulit tidur dan sering kepikiran hal-hal buruk terus-menerus."
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        text = st.text_area("Masukan Teks", value=default_text, height=150)
        run = st.button("Lihat Klaster", type="primary", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    if run or text.strip():
        cluster_id, normalized, tokens, raw_scores, scores = predict_cluster(text)
        info = CLUSTER_INFO[cluster_id]
        cols = st.columns([1, 5])
        with cols[0]:
            st.markdown(
                f'<div class="card" style="text-align:center;background:{info["soft"]};"><div style="font-size:3rem;color:{info["color"]};font-weight:900;">{cluster_id}</div><b>Cluster</b></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="small-muted">Teks ini dipetakan ke</div><div class="result-title" style="color:{info["color"]};">Cluster {cluster_id} <span class="success-chip">Paling Sesuai</span></div>', unsafe_allow_html=True)
            st.markdown(f"### {info['title']}")
            st.markdown(f'<div class="small-muted">{info["description"]}</div><hr/>', unsafe_allow_html=True)
            st.markdown("**Kata Kunci Terkait**")
            st.markdown(" ".join([f'<span class="pill">{kw}</span>' for kw in info["keywords"]]), unsafe_allow_html=True)
            with st.expander("Lihat teks setelah normalisasi"):
                st.write(normalized if normalized else "Tidak ada token yang cocok setelah preprocessing.")
            st.markdown('</div>', unsafe_allow_html=True)

        score_df = pd.DataFrame({
            "Klaster": [f"Cluster {i}" for i in sorted(scores)],
            "Skor Kesesuaian": [scores[i] for i in sorted(scores)],
        })
        fig = px.bar(score_df, x="Klaster", y="Skor Kesesuaian", color="Klaster", text_auto=".2f")
        fig.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=25, b=20), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="card"><h3>Daftar Klaster</h3>', unsafe_allow_html=True)
    for cid, info in CLUSTER_INFO.items():
        st.markdown(
            f'<div class="cluster-row"><span class="badge" style="background:{info["color"]};">{cid}</span><b>Cluster {cid} - {info["title"]}</b><span class="small-muted">{info["description"]}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def page_results():
    st.markdown("# Hasil Klastering Penelitian")
    st.markdown('<div class="small-muted">Visualisasi utama untuk memahami struktur klaster hasil penelitian.</div>', unsafe_allow_html=True)
    df = load_cluster_data()
    plot_df, metrics_df = compute_visual_data()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>Knee Locator / Elbow</h3>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_df["K"], y=metrics_df["Inertia"], mode="lines+markers", line=dict(color="#2563eb", width=3)))
        fig.add_vline(x=5, line_dash="dash", line_color="#2563eb")
        fig.add_annotation(x=5, y=float(metrics_df.loc[metrics_df["K"] == 5, "Inertia"].iloc[0]), text="Elbow di K=5", showarrow=True, arrowhead=2)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=15, b=10), xaxis_title="K", yaxis_title="Inertia / Distortion")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="small-muted">Elbow method menunjukkan titik siku pada K=5 sebagai kandidat awal jumlah klaster.</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>PCA Scatter Plot</h3>', unsafe_allow_html=True)
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="label", color_discrete_map={"Cluster 0": "#2563eb", "Cluster 1": "#16a34a", "Cluster 2": "#ef4444"}, opacity=0.72)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=15, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="small-muted">Visualisasi PCA 2D memperlihatkan pemisahan data berdasarkan tiga klaster.</div></div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="card"><h3>Metrik Evaluasi</h3>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-card"><div class="metric-label">Silhouette Score</div><div class="metric-value">{VALIDATION_METRICS["silhouette"]:.4f}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">Calinski-Harabasz</div><div class="metric-value" style="color:#159947;">{VALIDATION_METRICS["calinski"]:,.2f}</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-label">Davies-Bouldin</div><div class="metric-value" style="color:#dc2626;">{VALIDATION_METRICS["davies"]:.4f}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted" style="margin-top:1rem;">K=3 dipilih karena paling representatif secara interpretasi dan validasi psikolog.</div></div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card"><h3>Ringkasan Hasil Klaster</h3>', unsafe_allow_html=True)
        counts = df["cluster"].value_counts().sort_index()
        for cid, info in CLUSTER_INFO.items():
            st.markdown(
                f'<div class="cluster-row" style="grid-template-columns:52px 1.5fr 1fr;background:{info["soft"]};border-radius:8px;border-top:0;padding:.65rem .8rem;margin-bottom:.55rem;"><span class="badge" style="background:{info["color"]};">{cid}</span><b>Cluster {cid}: {info["title"]}</b><span class="small-muted">n = {counts.get(cid, 0):,}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Top Bigram per Klaster</h3>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, cid in enumerate(CLUSTER_INFO):
        with cols[idx]:
            info = CLUSTER_INFO[cid]
            st.markdown(f'**Cluster {cid} - {info["title"]}**')
            for (w1, w2), freq in top_bigrams_by_cluster(df, cid, 5):
                st.markdown(f'<span class="pill">{w1} {w2} ({freq})</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def page_validation():
    st.markdown("# Validasi Psikolog dan Alasan Pemilihan Klaster")
    st.markdown('<div class="small-muted">Hasil klastering ditinjau untuk memastikan bahwa setiap klaster bermakna secara psikologis dan sesuai dengan konteks gejala.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Validator Ahli</h3>', unsafe_allow_html=True)
    st.markdown("## Ni Gusti Ketut Diana Setiawati, M.Psi., Psikolog")
    st.markdown('<div class="small-muted">Psikolog Klinis / Praktik Mandiri Preema Psikologi. Validasi dilakukan pada 2 Juni 2026 untuk menilai kesesuaian hasil pengelompokan dengan gejala klinis dan kategori DSM-5.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Temuan Validasi</h3>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, cid in enumerate(CLUSTER_INFO):
        info = CLUSTER_INFO[cid]
        with cols[idx]:
            st.markdown(f'<div class="info-box" style="border-left:4px solid {info["color"]};"><span class="badge" style="background:{info["color"]};">{cid}</span><h3 style="color:{info["color"]};">Cluster {cid} - {info["title"]}</h3>' + " ".join([f'<span class="pill">{kw}</span>' for kw in info["keywords"][:4]]) + '</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.25, 1])
    with col1:
        st.markdown('<div class="card"><h3>Alasan Memilih K = 3</h3>', unsafe_allow_html=True)
        reasons = [
            "K=2 memiliki metrik baik, tetapi klasternya terlalu luas.",
            "K=3 lebih mudah diinterpretasikan secara psikologis.",
            "K=3 didukung visualisasi PCA dan pemisahan gejala yang jelas.",
            "K=3 selaras dengan validasi psikolog.",
            "K=5 dari elbow method kurang stabil secara interpretatif.",
        ]
        for i, reason in enumerate(reasons, 1):
            st.markdown(f'<div class="cluster-row" style="grid-template-columns:42px 1fr;"><span class="badge" style="background:#2563eb;width:28px;height:28px;">{i}</span><span>{reason}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-box" style="margin-top:1.4rem;"><h3>Berdasarkan evaluasi kuantitatif, visualisasi PCA, dan validasi psikolog, K=3 dipilih sebagai jumlah klaster paling representatif untuk penelitian ini.</h3></div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Dokumen Validasi</h3>', unsafe_allow_html=True)
    for file in sorted(VALIDASI_DIR.glob("*.docx")):
        with open(file, "rb") as fh:
            st.download_button(f"Unduh {file.name}", fh, file_name=file.name)
    st.markdown('</div>', unsafe_allow_html=True)


def page_about():
    st.markdown("# Tentang Penelitian")
    st.markdown('<div class="small-muted">Informasi singkat mengenai Tugas Akhir dan aplikasi pemetaan klaster gejala kesehatan mental.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Identitas Tugas Akhir</h3>', unsafe_allow_html=True)
    st.markdown(f"### {RESEARCH_INFO['title']}")
    rows = [
        ("Penulis", RESEARCH_INFO["author"]),
        ("NIM", RESEARCH_INFO["student_id"]),
        ("Program Studi", RESEARCH_INFO["program"]),
        ("Fakultas", RESEARCH_INFO["faculty"]),
        ("Universitas", RESEARCH_INFO["university"]),
        ("Tahun", RESEARCH_INFO["year"]),
    ]
    for label, value in rows:
        st.markdown(f'<div class="cluster-row" style="grid-template-columns:180px 1fr;"><b>{label}</b><span>{value}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown('<div class="card"><h3>Ringkasan Penelitian</h3>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Penelitian ini mengelompokkan teks curahan hati pengguna media sosial ke dalam klaster gejala gangguan kesehatan mental. Data teks diproses melalui normalisasi, tokenisasi, stopword removal, dan lemmatization, kemudian direpresentasikan menggunakan Word2Vec. Hasil representasi teks dikelompokkan dengan algoritma K-Means untuk menemukan pola gejala yang muncul pada data.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>Tujuan Aplikasi</h3>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Aplikasi ini menampilkan hasil penelitian secara interaktif: pengguna dapat memasukkan teks, melihat klaster yang paling sesuai, mengeksplorasi visualisasi hasil klastering, serta melihat ringkasan validasi psikolog.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Hasil Utama</h3>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (cid, info) in enumerate(CLUSTER_INFO.items()):
        with cols[idx]:
            st.markdown(f'<div class="info-box" style="border-left:4px solid {info["color"]};"><span class="badge" style="background:{info["color"]};">{cid}</span><h3 style="color:{info["color"]};">Cluster {cid}</h3><b>{info["title"]}</b><br/><span class="small-muted">{info["short"]}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


page = render_sidebar()
if page == "Input Teks":
    page_input()
elif page == "Hasil Klastering":
    page_results()
elif page == "Validasi Psikolog":
    page_validation()
else:
    page_about()
