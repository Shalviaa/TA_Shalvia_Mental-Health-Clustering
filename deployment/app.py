from pathlib import Path
from collections import Counter
import html as html_lib
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

# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KAMUS_DIR = DATA_DIR / "kamus"
VALIDASI_DIR = DATA_DIR / "validasi"
LOGO_PATH = DATA_DIR / "Telkom University logo.png"

CLUSTER_PATH = DATA_DIR / "hasil_clustering_w2v_tuned_pca100.csv"
TUNING_PATH = DATA_DIR / "bigrams_w2v_tuning_results_pca100.csv"
VECTOR_PATH = DATA_DIR / "w2v_vectors_pca100.npz"

# =========================
# CONTENT CONFIG
# Minimal version: cluster detail hanya ditampilkan di Input setelah klik tombol dan di Hasil Penelitian.
# =========================
CLUSTER_INFO = {
    0: {
        "title": "Anxiety & Physical Symptoms",
        "short": "Kecemasan dan keluhan fisik",
        "description": "Klaster ini merepresentasikan teks dengan pola kecemasan yang disertai keluhan fisik, seperti napas sesak, jantung berdebar, badan lemas, dan pikiran negatif.",
        "keywords": ["cemas lebih", "sesak nafas", "cemas takut", "badan lemas", "jantung debar"],
        "recommendation": "Catat kapan keluhan muncul, situasi pemicunya, dan intensitasnya. Jika keluhan sering terjadi atau mengganggu aktivitas, pertimbangkan untuk berkonsultasi dengan psikolog atau psikiater.",
        "color": "#159947",
        "soft": "#e8f8ee",
    },
    1: {
        "title": "Emotional Distress & Social Conflict",
        "short": "Distres emosional dan konflik sosial",
        "description": "Klaster ini menggambarkan teks dengan tekanan emosional, keluh kesah, konflik interpersonal, sudut pandang, dan ekspresi emosi negatif.",
        "keywords": ["putus asa", "keluh kesah", "caci maki", "sudut pandang", "sesak nafas"],
        "recommendation": "Coba kenali sumber tekanan, beri jeda sebelum mengambil keputusan, dan ceritakan kondisi kepada orang yang dipercaya. Jika tekanan berlangsung lama, konsultasi profesional dapat dipertimbangkan.",
        "color": "#2563eb",
        "soft": "#eaf1ff",
    },
    2: {
        "title": "Hopelessness & Crisis",
        "short": "Keputusasaan dan krisis",
        "description": "Klaster ini berisi pola teks yang berkaitan dengan keputusasaan, kehilangan arah, keinginan pergi, dan indikasi krisis psikologis yang lebih kuat.",
        "keywords": ["putus asa", "sehat mental", "hilang arah", "keluh kesah", "ingin pergi"],
        "recommendation": "Kondisi ini perlu diperhatikan lebih serius. Segera cari dukungan dari orang terdekat atau tenaga profesional, terutama jika muncul pikiran untuk menyakiti diri sendiri.",
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

DISCLAIMER = (
    "Hasil pemetaan ini bukan diagnosis klinis dan tidak dapat menggantikan pemeriksaan oleh psikolog atau psikiater. "
    "Aplikasi hanya menunjukkan kemiripan pola teks dengan klaster yang terbentuk dalam penelitian."
)

# =========================
# STREAMLIT CONFIG & STYLE
# =========================
st.set_page_config(
    page_title="Mental Health Clustering",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #f4f7fb;
    --card: #ffffff;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #d9e2ef;
    --primary: #3157d5;
    --primary-2: #7c3aed;
    --success: #159947;
    --warning-bg: #fff7ed;
    --warning-line: #fed7aa;
    --warning-text: #9a3412;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(124, 58, 237, .12), transparent 30rem),
        radial-gradient(circle at top right, rgba(37, 99, 235, .10), transparent 28rem),
        linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
    color: var(--ink);
}

.block-container {
    max-width: 1180px;
    padding-top: 2.0rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, .92);
    border-right: 1px solid var(--line);
    box-shadow: 8px 0 30px rgba(15, 23, 42, .04);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: var(--ink);
    letter-spacing: -.035em;
}

h1 {
    font-size: 2.65rem !important;
    line-height: 1.05 !important;
    font-weight: 900 !important;
    margin-bottom: .5rem !important;
}

h2 {
    font-size: 1.55rem !important;
    font-weight: 850 !important;
}

h3 {
    font-size: 1.12rem !important;
    font-weight: 800 !important;
}

button[kind="primary"] {
    border-radius: 999px !important;
    font-weight: 800 !important;
    padding: .55rem 1.15rem !important;
}

textarea, input {
    border-radius: 16px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 24px !important;
    border: 1px solid rgba(148, 163, 184, .28) !important;
    box-shadow: 0 16px 36px rgba(15, 23, 42, .06) !important;
    background: rgba(255, 255, 255, .84) !important;
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 2rem;
    background: linear-gradient(135deg, #14213d 0%, #3157d5 58%, #7c3aed 100%);
    color: white;
    box-shadow: 0 24px 60px rgba(49, 87, 213, .28);
    margin-bottom: 1rem;
}
.hero:after {
    content: "";
    position: absolute;
    width: 22rem;
    height: 22rem;
    border-radius: 999px;
    right: -7rem;
    top: -9rem;
    background: rgba(255,255,255,.13);
}
.hero h1 {
    color: white !important;
    max-width: 820px;
}
.hero p {
    color: rgba(255,255,255,.82);
    font-size: 1.04rem;
    max-width: 760px;
    margin-bottom: 0;
}
.tag {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    border-radius: 999px;
    padding: .38rem .72rem;
    background: rgba(255,255,255,.15);
    color: rgba(255,255,255,.94);
    font-weight: 800;
    font-size: .82rem;
    margin-bottom: .8rem;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .8rem;
    margin: .9rem 0 1.1rem;
}
.stat-card {
    border-radius: 22px;
    padding: 1.05rem 1.1rem;
    background: rgba(255,255,255,.90);
    border: 1px solid rgba(148, 163, 184, .24);
    box-shadow: 0 12px 28px rgba(15, 23, 42, .06);
}
.stat-label {
    font-size: .82rem;
    color: var(--muted);
    font-weight: 700;
    margin-bottom: .2rem;
}
.stat-value {
    font-size: 1.35rem;
    font-weight: 900;
    color: var(--ink);
}

.card-html {
    border-radius: 24px;
    padding: 1.35rem;
    background: rgba(255,255,255,.88);
    border: 1px solid rgba(148, 163, 184, .26);
    box-shadow: 0 16px 36px rgba(15, 23, 42, .06);
    margin-bottom: 1rem;
}
.card-title {
    display: flex;
    align-items: center;
    gap: .55rem;
    font-size: 1.05rem;
    font-weight: 900;
    color: var(--ink);
    margin-bottom: .45rem;
}
.muted {
    color: var(--muted);
    font-size: .94rem;
    line-height: 1.65;
}

.disclaimer {
    border-radius: 20px;
    padding: 1rem 1.1rem;
    background: var(--warning-bg);
    border: 1px solid var(--warning-line);
    color: var(--warning-text);
    font-weight: 650;
    line-height: 1.55;
    margin: .8rem 0 1.1rem;
}

.badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 999px;
    color: white;
    font-weight: 900;
}
.badge-sm {
    width: 26px;
    height: 26px;
    font-size: .76rem;
}
.pill {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: .42rem .72rem;
    margin: .22rem .24rem .22rem 0;
    background: #edf4ff;
    color: #1d4ed8;
    font-size: .83rem;
    font-weight: 800;
}
.cluster-card {
    border-radius: 22px;
    padding: 1.2rem;
    min-height: 200px;
    border: 1px solid rgba(148, 163, 184, .35);
    box-shadow: 0 14px 28px rgba(15, 23, 42, .05);
    background: rgba(255,255,255,.88);
}
.result-card {
    border-radius: 28px;
    padding: 1.45rem;
    background: white;
    border: 1px solid rgba(148, 163, 184, .28);
    box-shadow: 0 18px 44px rgba(15, 23, 42, .08);
}
.result-kicker {
    color: var(--muted);
    font-weight: 800;
    font-size: .86rem;
    text-transform: uppercase;
    letter-spacing: .06em;
}
.result-title {
    font-size: 2.05rem;
    font-weight: 950;
    letter-spacing: -.04em;
    margin: .25rem 0 .35rem;
}
.success-chip {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: .36rem .72rem;
    background: #dcfce7;
    color: #166534;
    font-size: .8rem;
    font-weight: 900;
    margin-left: .4rem;
}
.score-row {
    display: grid;
    grid-template-columns: 120px 1fr 56px;
    gap: .75rem;
    align-items: center;
    padding: .45rem 0;
}
.score-label {
    font-weight: 850;
    color: var(--ink);
    font-size: .9rem;
}
.score-value {
    text-align: right;
    color: var(--muted);
    font-weight: 800;
}
.table-row {
    display: grid;
    grid-template-columns: 170px 1fr;
    gap: 1rem;
    border-bottom: 1px solid #e2e8f0;
    padding: .78rem 0;
}
.table-row:last-child { border-bottom: 0; }
.sidebar-logo {
    width: 54px;
    height: 54px;
    border-radius: 18px;
    background: linear-gradient(135deg, #3157d5, #7c3aed);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.55rem;
    font-weight: 950;
    box-shadow: 0 14px 30px rgba(49,87,213,.28);
    margin-bottom: .9rem;
}
.sidebar-title {
    font-weight: 950;
    line-height: 1.05;
    font-size: 1.35rem;
    color: var(--ink);
    margin-bottom: .35rem;
}
.sidebar-subtitle {
    color: var(--muted);
    font-size: .85rem;
    line-height: 1.45;
    margin-bottom: 1.2rem;
}
.sidebar-box {
    border-radius: 18px;
    padding: .9rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #475569;
    font-size: .86rem;
    line-height: 1.55;
    margin-top: 1rem;
}

@media (max-width: 900px) {
    .stat-grid { grid-template-columns: 1fr; }
    .score-row { grid-template-columns: 1fr; gap: .3rem; }
    .table-row { grid-template-columns: 1fr; }
    h1 { font-size: 2.1rem !important; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# SMALL RENDER HELPERS
# =========================
def html(content: str):
    st.markdown(content, unsafe_allow_html=True)


def render_disclaimer():
    html(f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>')


def render_hero(title: str, subtitle: str, tag: str = "Tugas Akhir · K-Means Clustering"):
    html(
        f"""
        <div class="hero">
            <div class="tag">🧠 {tag}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """
    )


def render_sidebar():
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=150)
    else:
        st.sidebar.markdown('<div class="sidebar-logo">MH</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="sidebar-title">Mental Health<br/>Clustering</div>
        <div class="sidebar-subtitle">Demo interaktif hasil penelitian clustering teks kesehatan mental.</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Menu",
        ["Input Teks", "Hasil Penelitian", "Tentang"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-box">
            <b>{RESEARCH_INFO['author']}</b><br/>
            {RESEARCH_INFO['student_id']}<br/>
            {RESEARCH_INFO['program']}<br/>
            {RESEARCH_INFO['university']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


# =========================
# DATA LOADING
# =========================
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


# =========================
# PREPROCESSING & PREDICTION
# =========================
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
    return np.vstack([document_vector(tokens, vocab, word_vectors) for tokens in df["tokens_lemma"]])


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


def raw_examples_by_cluster(df, cluster_id, n=15):
    source = df.loc[df["cluster"] == cluster_id, "post_text"].dropna().astype(str)
    examples = []
    seen = set()
    for value in source:
        clean = " ".join(value.split())
        if len(clean) < 20 or clean in seen:
            continue
        seen.add(clean)
        examples.append(html_lib.escape(clean))
        if len(examples) >= n:
            break
    return examples


def style_plotly(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="#0f172a"),
        margin=dict(l=18, r=18, t=20, b=20),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.22)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.22)", zeroline=False)
    return fig


# =========================
# PAGES
# =========================
def page_input():
    df = load_cluster_data()
    render_hero(
        "Pemetaan Teks ke Klaster Gejala Kesehatan Mental",
        "Masukkan teks curahan hati. Sistem akan memetakan teks ke klaster yang paling mirip berdasarkan model Word2Vec, PCA, dan K-Means pada hasil penelitian.",
    )
    render_disclaimer()

    total_data = len(df)
    html(
        f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-label">Data Final</div><div class="stat-value">{total_data:,} teks</div></div>
            <div class="stat-card"><div class="stat-label">Jumlah Klaster</div><div class="stat-value">3 klaster</div></div>
            <div class="stat-card"><div class="stat-label">Metode</div><div class="stat-value">Word2Vec · PCA · K-Means</div></div>
        </div>
        """
    )

    with st.container(border=True):
        st.subheader("Coba Masukkan Teks")
        text = st.text_area(
            "Teks pengguna",
            value="",
            placeholder="Contoh: Saya sering merasa cemas, sulit tidur, napas terasa sesak, dan badan terasa lemas.",
            height=150,
            label_visibility="collapsed",
        )
        col_btn, col_hint = st.columns([1, 3])
        with col_btn:
            run = st.button("Lihat Klaster", type="primary", use_container_width=True)
        with col_hint:
            st.caption("Teks akan diproses melalui cleaning, normalisasi, stopword removal, dan representasi vektor.")

    if not run:
        html(
            """
            <div class="card-html">
                <div class="card-title">💡 Cara membaca hasil</div>
                <div class="muted">
                    Tulis teks, klik tombol <b>Lihat Klaster</b>, lalu aplikasi akan menampilkan satu klaster yang paling sesuai. Hasil ini hanya untuk demonstrasi penelitian, bukan diagnosis klinis.
                </div>
            </div>
            """
        )
        return

    if not text.strip():
        st.warning("Masukkan teks terlebih dahulu sebelum melihat klaster.")
        return

    with st.spinner("Memproses teks dan menghitung kemiripan klaster..."):
        cluster_id, normalized, tokens, raw_scores, scores = predict_cluster(text)

    info = CLUSTER_INFO[cluster_id]
    html(
        f"""
        <div class="result-card">
            <div class="result-kicker">Hasil Pemetaan</div>
            <div class="result-title" style="color:{info['color']};">
                Cluster {cluster_id} <span class="success-chip">Paling Sesuai</span>
            </div>
            <h3>{info['title']}</h3>
            <div class="muted">{info['description']}</div>
            <div style="margin-top:1rem;">
                {''.join([f'<span class="pill">{kw}</span>' for kw in info['keywords']])}
            </div>
        </div>
        """
    )

    col_left, col_right = st.columns([1.05, 1])
    with col_left:
        with st.container(border=True):
            st.subheader("Skor Kesesuaian")
            for cid in sorted(scores):
                pct = max(0, min(1, scores[cid]))
                st.markdown(f"**Cluster {cid}** · {pct:.0%}")
                st.progress(pct)

            with st.expander("Lihat teks setelah normalisasi"):
                st.write(normalized if normalized else "Tidak ada token yang cocok setelah preprocessing.")

    with col_right:
        html(
            f"""
            <div class="card-html" style="border-left:5px solid {info['color']};">
                <div class="card-title">📝 Rekomendasi Umum</div>
                <div class="muted">{info['recommendation']}</div>
                <div style="height:.85rem;"></div>
                <div class="muted"><b>Catatan:</b> rekomendasi ini bersifat umum, bukan saran medis dan bukan diagnosis.</div>
            </div>
            """
        )



def render_cluster_overview():
    html('<div class="card-html"><div class="card-title">🧩 Ringkasan Tiga Klaster</div><div class="muted">Setiap klaster merupakan hasil interpretasi dari pola teks yang terbentuk pada penelitian.</div></div>')
    cols = st.columns(3)
    for idx, (cid, info) in enumerate(CLUSTER_INFO.items()):
        with cols[idx]:
            html(
                f"""
                <div class="cluster-card" style="border-top:5px solid {info['color']}; background:{info['soft']};">
                    <span class="badge" style="background:{info['color']};">{cid}</span>
                    <h3 style="color:{info['color']}; margin-top:.85rem;">Cluster {cid}</h3>
                    <b>{info['title']}</b>
                    <div class="muted" style="margin-top:.45rem;">{info['short']}</div>
                    <div style="margin-top:.7rem;">
                        {''.join([f'<span class="pill">{kw}</span>' for kw in info['keywords'][:3]])}
                    </div>
                </div>
                """
            )


def page_results():
    render_hero(
        "Dashboard Hasil Klastering",
        "Halaman ini merangkum visualisasi elbow, sebaran PCA, metrik evaluasi, ukuran klaster, bigram dominan, dan validasi psikolog.",
        "Dashboard Penelitian",
    )

    df = load_cluster_data()
    plot_df, metrics_df = compute_visual_data()
    counts = df["cluster"].value_counts().sort_index()

    html(
        f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-label">Silhouette K=3</div><div class="stat-value">{VALIDATION_METRICS['silhouette']:.4f}</div></div>
            <div class="stat-card"><div class="stat-label">Calinski-Harabasz</div><div class="stat-value">{VALIDATION_METRICS['calinski']:,.2f}</div></div>
            <div class="stat-card"><div class="stat-label">Davies-Bouldin</div><div class="stat-value">{VALIDATION_METRICS['davies']:.4f}</div></div>
        </div>
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Elbow / Knee Locator")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics_df["K"],
                y=metrics_df["Inertia"],
                mode="lines+markers",
                line=dict(color="#3157d5", width=4),
                marker=dict(size=8),
            ))
            fig.add_vline(x=5, line_dash="dash", line_color="#7c3aed")
            fig.add_annotation(
                x=5,
                y=float(metrics_df.loc[metrics_df["K"] == 5, "Inertia"].iloc[0]),
                text="Elbow K=5",
                showarrow=True,
                arrowhead=2,
                bgcolor="white",
                bordercolor="#d9e2ef",
            )
            fig.update_layout(xaxis_title="Jumlah Klaster (K)", yaxis_title="Inertia")
            st.plotly_chart(style_plotly(fig, 360), use_container_width=True)
            st.caption("Elbow method menunjukkan K=5 sebagai kandidat awal, tetapi interpretasi akhir tetap mempertimbangkan validasi psikolog dan metrik lain.")

    with col2:
        with st.container(border=True):
            st.subheader("PCA Scatter Plot")
            fig = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="label",
                color_discrete_map={"Cluster 0": "#159947", "Cluster 1": "#2563eb", "Cluster 2": "#7c3aed"},
                opacity=0.72,
            )
            st.plotly_chart(style_plotly(fig, 360), use_container_width=True)
            st.caption("Visualisasi PCA 2D digunakan untuk melihat pola sebaran data berdasarkan tiga klaster.")

    col3, col4 = st.columns([1, 1])
    with col3:
        with st.container(border=True):
            st.subheader("Ukuran Klaster")
            for cid, info in CLUSTER_INFO.items():
                html(
                    f"""
                    <div style="display:grid;grid-template-columns:44px 1fr 86px;gap:.7rem;align-items:center;background:{info['soft']};border-radius:16px;padding:.75rem .9rem;margin:.45rem 0;">
                        <span class="badge badge-sm" style="background:{info['color']};">{cid}</span>
                        <div><b>Cluster {cid}</b><br/><span class="muted">{info['short']}</span></div>
                        <b style="color:{info['color']};text-align:right;">n = {counts.get(cid, 0):,}</b>
                    </div>
                    """
                )

    with col4:
        with st.container(border=True):
            st.subheader("Alasan K=3")
            reasons = [
                "K=2 memiliki metrik kuat, tetapi tema terlalu luas.",
                "K=3 lebih mudah diinterpretasikan secara psikologis.",
                "Pola PCA masih menunjukkan pemisahan yang cukup jelas.",
                "Validasi psikolog mendukung tiga tema utama.",
            ]
            for i, reason in enumerate(reasons, 1):
                html(
                    f"""
                    <div style="display:grid;grid-template-columns:34px 1fr;gap:.75rem;align-items:start;padding:.55rem 0;border-bottom:1px solid #e2e8f0;">
                        <span class="badge badge-sm" style="background:#3157d5;">{i}</span>
                        <span>{reason}</span>
                    </div>
                    """
                )

    st.subheader("Validasi Psikolog")
    col_val1, col_val2 = st.columns([1, 1.35])
    with col_val1:
        html(
            """
            <div class="card-html" style="margin-bottom:0;">
                <div class="card-title">Validator Ahli</div>
                <h3 style="margin-bottom:.35rem;">Ni Gusti Ketut Diana Setiawati, M.Psi., Psikolog</h3>
                <div class="muted">
                    Psikolog Klinis / Praktik Mandiri Preema Psikologi. Validasi dilakukan untuk menilai apakah tema klaster bermakna secara psikologis dan sesuai dengan konteks gejala.
                </div>
            </div>
            """
        )
    with col_val2:
        html(
            """
            <div class="card-html" style="margin-bottom:0;">
                <div class="card-title">Kesimpulan Validasi</div>
                <div class="muted">
                    K=3 dipilih karena memberikan pembagian yang lebih interpretatif dibanding K=2 yang terlalu luas, serta lebih stabil secara makna dibanding K=5 dari elbow method. Validasi psikolog digunakan sebagai penguat bahwa klaster yang terbentuk tidak hanya terbaca secara statistik, tetapi juga dapat dijelaskan dari sudut pandang psikologis.
                </div>
            </div>
            """
        )

    files = sorted(VALIDASI_DIR.glob("*.docx"))
    if files:
        with st.container(border=True):
            st.subheader("Dokumen Validasi")
            for file in files:
                with open(file, "rb") as fh:
                    st.download_button(f"Unduh {file.name}", fh, file_name=file.name)

    with st.container(border=True):
        st.subheader("Top Bigram per Klaster")
        cols = st.columns(3)
        for idx, cid in enumerate(CLUSTER_INFO):
            info = CLUSTER_INFO[cid]
            with cols[idx]:
                st.markdown(f"**Cluster {cid} - {info['title']}**")
                for (w1, w2), freq in top_bigrams_by_cluster(df, cid, 5):
                    html(f'<span class="pill">{w1} {w2} - {freq}</span>')

    st.subheader("Contoh Teks dari Dataset per Klaster")
    st.caption("Contoh berikut diambil dari kolom post_text pada dataset hasil clustering. Masing-masing klaster menampilkan minimal 15 teks.")
    for cid, info in CLUSTER_INFO.items():
        with st.expander(f"Cluster {cid} - {info['title']}", expanded=(cid == 0)):
            examples = raw_examples_by_cluster(df, cid, 15)
            for idx, example in enumerate(examples, 1):
                html(f'<div class="table-row" style="grid-template-columns:44px 1fr;"><b>{idx}</b><span>{example}</span></div>')


def page_about():
    render_hero(
        "Tentang Penelitian",
        "Informasi singkat mengenai Tugas Akhir, metode yang digunakan, dan tujuan aplikasi sebagai media demonstrasi hasil clustering.",
        "Profil Penelitian",
    )

    html(
        f"""
        <div class="card-html">
            <div class="card-title">🎓 Identitas Tugas Akhir</div>
            <h3>{RESEARCH_INFO['title']}</h3>
            <div class="table-row"><b>Penulis</b><span>{RESEARCH_INFO['author']}</span></div>
            <div class="table-row"><b>NIM</b><span>{RESEARCH_INFO['student_id']}</span></div>
            <div class="table-row"><b>Program Studi</b><span>{RESEARCH_INFO['program']}</span></div>
            <div class="table-row"><b>Fakultas</b><span>{RESEARCH_INFO['faculty']}</span></div>
            <div class="table-row"><b>Universitas</b><span>{RESEARCH_INFO['university']}</span></div>
            <div class="table-row"><b>Tahun</b><span>{RESEARCH_INFO['year']}</span></div>
        </div>
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        html(
            """
            <div class="card-html">
                <div class="card-title">📌 Ringkasan Penelitian</div>
                <div class="muted">
                    Penelitian ini mengelompokkan teks curahan hati pengguna media sosial ke dalam klaster gejala kesehatan mental. Data teks diproses melalui cleaning, normalisasi, tokenisasi, stopword removal, dan lemmatization. Representasi teks dibuat menggunakan Word2Vec, lalu direduksi dengan PCA dan dikelompokkan menggunakan K-Means.
                </div>
            </div>
            """
        )
    with col2:
        html(
            """
            <div class="card-html">
                <div class="card-title">🎯 Tujuan Aplikasi</div>
                <div class="muted">
                    Aplikasi ini digunakan sebagai media demonstrasi hasil penelitian. Pengguna dapat memasukkan teks, melihat klaster yang paling sesuai, mengeksplorasi visualisasi hasil klastering, dan membaca ringkasan validasi psikolog.
                </div>
            </div>
            """
        )

    html(
        """
        <div class="card-html">
            <div class="card-title">⚠️ Batasan Aplikasi</div>
            <div class="muted">
                Aplikasi ini bukan alat diagnosis. Hasil yang ditampilkan hanya menunjukkan kemiripan teks dengan pola klaster pada data penelitian. Interpretasi klinis tetap memerlukan evaluasi psikolog atau psikiater.
            </div>
        </div>
        """
    )


# =========================
# APP ROUTER
# =========================
page = render_sidebar()

if page == "Input Teks":
    page_input()
elif page == "Hasil Penelitian":
    page_results()
else:
    page_about()
