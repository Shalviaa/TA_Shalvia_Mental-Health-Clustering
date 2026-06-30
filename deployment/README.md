# Mental Health Clustering Streamlit Deployment

Aplikasi Streamlit untuk pemetaan teks ke klaster gejala kesehatan mental berdasarkan hasil penelitian Word2Vec + K-Means.

## File utama

- `app.py` - aplikasi Streamlit
- `data/w2v_vectors_pca100.npz` - vektor Word2Vec hasil ekspor dari model penelitian
- `data/hasil_clustering_w2v_tuned_pca100.csv` - hasil klastering K=3
- `data/bigrams_w2v_tuning_results_pca100.csv` - hasil tuning Word2Vec
- `data/validasi/*.docx` - dokumen validasi psikolog dan kata kunci DSM-5

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Streamlit Community Cloud

1. Push folder ini ke repository GitHub.
2. Di Streamlit Cloud pilih repository `Shalviaa/TA_Shalvia_Mental-Health-Clustering`.
3. Set main file path ke:

```text
deployment/app.py
```

4. Deploy.

Catatan: file `.pkl` asli tetap berada di folder `output`, sedangkan deployment memakai format `.npz` agar tidak perlu menginstall `gensim` di Streamlit Cloud.
