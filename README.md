# TA Shalvia Mental Health Clustering

Repository ini berisi aplikasi deployment Streamlit untuk Tugas Akhir:

**Pengelompokan Gejala Gangguan Kesehatan Mental Berdasarkan Curahan Hati Pengguna di Media Sosial Menggunakan Algoritma K-Means Clustering**

**Penulis:** Shalvia Retno Salsabil  
**NIM:** 1202220027  
**Program Studi:** S1 Sistem Informasi  
**Fakultas:** Fakultas Rekayasa Industri  
**Universitas:** Telkom University  
**Tahun:** 2026

## Tentang Penelitian

Penelitian ini bertujuan mengelompokkan teks curahan hati pengguna media sosial ke dalam beberapa klaster gejala kesehatan mental. Tahapan utama yang digunakan meliputi preprocessing teks, pembentukan representasi teks menggunakan Word2Vec, reduksi dimensi menggunakan PCA, dan pengelompokan menggunakan algoritma K-Means Clustering.

Aplikasi Streamlit pada folder `deployment` digunakan sebagai media demonstrasi hasil penelitian. Pengguna dapat memasukkan teks, melihat klaster yang paling sesuai, melihat visualisasi hasil clustering, serta membaca ringkasan validasi dan contoh teks dari dataset.

## Struktur Utama

```text
deployment/
  app.py
  requirements.txt
  data/
    hasil_clustering_w2v_tuned_pca100.csv
    bigrams_w2v_tuning_results_pca100.csv
    w2v_vectors_pca100.npz
    Telkom University logo.png
    kamus/
    validasi/
```

## Data dan Model Deployment

File utama yang digunakan aplikasi:

- `hasil_clustering_w2v_tuned_pca100.csv`: data hasil clustering yang memuat `post_text`, `lemmas`, dan label `cluster`.
- `bigrams_w2v_tuning_results_pca100.csv`: hasil tuning parameter Word2Vec.
- `w2v_vectors_pca100.npz`: hasil ekspor vektor Word2Vec dari model penelitian. Format ini dipakai agar deployment Streamlit tidak perlu menginstall `gensim`.
- `validasi/*.docx`: dokumen validasi ahli dan validasi kata kunci.

## Sumber dan Penyusunan Kamus

Kamus digunakan pada tahap preprocessing untuk membantu normalisasi teks, mengurangi noise, dan menjaga konsistensi token sebelum proses modeling.

### 1. Kamus Alay

File: `kamusalay .csv`

Kamus alay digunakan untuk mengubah bentuk kata tidak baku atau kata gaul menjadi bentuk yang lebih baku. Kamus ini diperoleh dari sumber publik yang tersedia di GitHub, kemudian digunakan sebagai referensi awal normalisasi teks. Setelah itu, isi kamus tetap disesuaikan dengan kebutuhan data penelitian agar tidak mengubah makna konteks curahan hati.

### 2. Kamus Slang Indonesia

File: `slang_indo.csv`

Kamus slang Indonesia digunakan untuk menangani kata informal yang sering muncul pada teks media sosial. Sumber awal kamus berasal dari kamus publik/GitHub. Validasi dilakukan dengan melihat kesesuaian pasangan kata slang dan bentuk normalnya terhadap konteks bahasa Indonesia sehari-hari, khususnya pada data curahan hati.

### 3. Custom Stopword

File: `custom_stopword.csv`

Custom stopword dibuat dari gabungan:

- daftar stopword bahasa Indonesia dari repository `stopwords-iso/stopwords-id`: https://github.com/stopwords-iso/stopwords-id/blob/master/stopwords-id.txt
- tambahan stopword yang disusun dengan bantuan AI untuk menangkap kata umum yang kurang informatif pada konteks dataset
- kurasi manual peneliti agar kata yang penting untuk konteks gejala kesehatan mental tidak ikut terhapus

Validasi custom stopword dilakukan dengan memeriksa apakah kata yang dihapus benar-benar bersifat umum dan tidak membawa makna gejala. Kata yang masih berhubungan dengan kondisi psikologis, emosi, keluhan fisik, atau indikasi krisis tidak dimasukkan sebagai stopword.

### 4. Tambahan Typo

File: `tambahan_typo.csv`

Kamus tambahan typo dibuat berdasarkan validasi manual peneliti terhadap data. Kata-kata yang sering salah ketik dan belum tercakup dalam kamus alay atau slang ditambahkan ke file ini. Penyusunan dilakukan melalui pengecekan langsung pada token hasil preprocessing, sehingga koreksi typo tetap sesuai dengan bentuk kata yang muncul pada dataset.

### 5. Keyword Mental Health

File: `keywords_mental_health.csv`

Keyword mental health disusun berdasarkan rujukan gejala dari buku DSM-5. Keyword ini digunakan sebagai referensi konseptual untuk membantu proses filtering dan interpretasi istilah yang berkaitan dengan kesehatan mental. Keyword tidak digunakan sebagai diagnosis, melainkan sebagai daftar kata bantu untuk mengidentifikasi teks yang relevan dengan konteks penelitian.

## Validasi Kamus dan Keyword

Validasi kamus dilakukan secara bertahap:

1. **Validasi sumber**: kamus publik seperti kamus alay, slang, dan stopword digunakan sebagai referensi awal karena sudah umum dipakai untuk pemrosesan teks bahasa Indonesia.
2. **Validasi konteks dataset**: setiap kamus disesuaikan dengan pola bahasa pada data curahan hati, sehingga kata penting tidak hilang akibat normalisasi atau stopword removal.
3. **Validasi manual peneliti**: tambahan typo dan sebagian stopword diperiksa langsung berdasarkan pengamatan pada dataset.
4. **Validasi konseptual**: keyword mental health mengacu pada DSM-5 sebagai rujukan gejala, kemudian digunakan secara hati-hati untuk mendukung filtering dan interpretasi.
5. **Validasi ahli**: hasil akhir klaster ditinjau melalui dokumen validasi psikolog yang tersedia pada folder `deployment/data/validasi`.

## Catatan Etis

Aplikasi ini bukan alat diagnosis klinis. Hasil klaster hanya menunjukkan kemiripan pola teks terhadap data penelitian. Interpretasi kondisi psikologis tetap memerlukan konsultasi dengan psikolog, psikiater, atau tenaga kesehatan profesional.

## Menjalankan Aplikasi Lokal

```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Streamlit

Pada Streamlit Community Cloud, gunakan:

```text
Main file path: deployment/app.py
```
