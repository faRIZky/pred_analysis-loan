# Predictive Analysis of Loan Approval Classification

## Domain Background

Dalam industri keuangan, permintaan pinjaman terus meningkat setiap tahunnya. Namun, proses evaluasi kelayakan peminjam yang masih konvensional sering kali tidak efisien dan rentan terhadap kesalahan penilaian. Keputusan yang tidak akurat dapat menyebabkan kerugian finansial atau kehilangan calon peminjam potensial.

Untuk mengatasi tantangan tersebut, pendekatan berbasis machine learning dapat digunakan untuk membangun sistem prediktif. Model ini mampu mengevaluasi permohonan pinjaman secara otomatis dan tepat sasaran berdasarkan data historis yang tersedia.

Menurut penelitian Awan et al. (2020), algoritma seperti Random Forest dan Decision Tree menunjukkan performa tinggi dalam kasus klasifikasi kelayakan pinjaman, terutama saat digunakan dengan metrik akurasi sebagai tolok ukur utama.

## Business Understanding

### Problem Statements
- Bagaimana memanfaatkan data terstruktur untuk memprediksi apakah permohonan pinjaman disetujui?
- Algoritma machine learning apa yang paling efektif untuk memodelkan data persetujuan pinjaman?

### Goals
- Mengembangkan model klasifikasi untuk memprediksi persetujuan pinjaman berdasarkan fitur finansial.
- Menentukan model terbaik di antara Random Forest, Naive Bayes, Decision Tree, dan KNN menggunakan metrik evaluasi.

### Solution Statement
- Membangun model klasifikasi menggunakan Random Forest, Naive Bayes, Decision Tree, dan K-Nearest Neighbors (KNN).
- Melakukan preprocessing data terstruktur (encoding dan normalisasi).
- Evaluasi model menggunakan akurasi, precision, recall, dan F1-score. Pemilihan model terbaik berdasarkan akurasi, sesuai referensi studi terdahulu.

## Data Understanding

Dataset yang digunakan diambil dari Kaggle dengan judul "Loan-Solutions Elite Dataset", tersedia pada tautan berikut:

https://www.kaggle.com/datasets/ayessa/loan-solutions-elite

### Informasi Umum
- Jumlah data: 24,000 baris
- Target variabel: `Approval` (Approved / Rejected)
- Distribusi kelas tidak seimbang:
  - Rejected: 20,067
  - Approved: 3,933

### Fitur pada Data
Beberapa fitur utama pada dataset:
- Income
- Credit_Score
- Loan_Amount
- Loan_Purpose
- Employment_Status
- Marital_Status
- Education_Level
- dan beberapa lainnya

### Exploratory Data Analysis
Beberapa visualisasi dilakukan untuk memahami pola dalam data:

- <eda distribusi income>
- <eda loan purpose vs approval>
- <eda employment status vs approval>
- <correlation heatmap>

## Data Preparation

### 1. Encoding
Fitur kategorikal seperti `Loan_Purpose`, `Employment_Status`, `Marital_Status`, dan `Education_Level` diubah ke bentuk numerik menggunakan LabelEncoder agar dapat diproses oleh model machine learning.

### 2. Normalisasi
Fitur numerik seperti `Income`, `Loan_Amount`, dan `Credit_Score` dinormalisasi menggunakan StandardScaler agar memiliki skala yang seragam.

### 3. Split Data
Data dibagi menjadi data latih dan data uji dengan rasio 80:20 untuk menguji kemampuan generalisasi model.

## Modeling

Empat algoritma klasifikasi digunakan untuk menyelesaikan permasalahan prediksi persetujuan pinjaman:

### Decision Tree
- Mudah dipahami dan divisualisasikan.
- Cepat diproses namun rentan overfitting.

### Random Forest
- Ensemble learning yang menggabungkan banyak decision tree.
- Lebih stabil dan akurat dibanding satu pohon tunggal.

### Naive Bayes
- Cocok untuk data dengan asumsi distribusi tertentu.
- Cepat dan sederhana, namun performanya dapat turun jika data tidak memenuhi asumsi.

### K-Nearest Neighbors
- Non-parametrik dan berbasis kedekatan jarak antar data.
- Performa sangat dipengaruhi oleh skala dan jumlah data.

Semua model dilatih menggunakan data latih, dan hasil prediksi diuji pada data uji.

## Evaluation

Untuk mengevaluasi kinerja model klasifikasi, digunakan metrik accuracy, precision, recall, dan F1-score. Meskipun data tidak seimbang (20,067 Rejected vs 3,933 Approved), metrik accuracy tetap dijadikan dasar pemilihan model terbaik karena mengacu pada referensi penelitian sebelumnya.

Berikut ringkasan hasil evaluasi keempat model:

- **Decision Tree** menunjukkan akurasi mendekati sempurna (100%), dengan performa yang baik untuk kedua kelas.
- **Random Forest** tampil sebagai model dengan akurasi 100% dan nilai precision, recall, serta F1-score yang merata di semua kelas.
- **Naive Bayes** memiliki akurasi yang jauh lebih rendah (72%), terutama karena kesulitan dalam mengenali kelas Approved secara akurat.
- **KNN** menunjukkan akurasi tinggi (99%) dengan performa yang stabil dan seimbang antar kelas.

Dua model yang mencetak akurasi sempurna adalah Random Forest dan Decision Tree. Namun, Random Forest dipilih sebagai model terbaik karena kemampuannya dalam mengurangi overfitting melalui teknik ensemble dan menghasilkan evaluasi yang lebih stabil secara umum.

### Hubungan dengan Business Understanding

Evaluasi model menunjukkan bahwa proyek ini berhasil menjawab semua problem statement dan mencapai tujuan yang telah ditetapkan:

- Model berhasil memanfaatkan data terstruktur untuk memprediksi apakah pengajuan pinjaman akan disetujui atau ditolak. Keakuratan tinggi dari Random Forest menunjukkan bahwa data finansial dapat digunakan secara efektif untuk prediksi ini.
- Model terbaik telah teridentifikasi berdasarkan metrik evaluasi, menjawab pertanyaan mengenai algoritma machine learning yang paling efektif dalam konteks ini.
- Semua solusi yang dirancang—dari preprocessing, pemilihan algoritma, hingga evaluasi berbasis metrik—telah diimplementasikan dan terbukti efektif.

Dengan demikian, hasil proyek ini berdampak langsung pada efisiensi dan akurasi proses persetujuan pinjaman, sesuai dengan konteks awal business understanding. Model yang dibangun berpotensi diintegrasikan ke sistem otomasi untuk membantu lembaga keuangan dalam mengurangi risiko dan mempercepat pengambilan keputusan berbasis data.

## Future Development

Walaupun model yang dibangun telah menunjukkan performa yang sangat tinggi, proyek ini masih memiliki ruang pengembangan lebih lanjut. Di antaranya adalah:

- Menerapkan teknik penanganan data tidak seimbang seperti SMOTE untuk mengetahui dampaknya terhadap metrik selain akurasi.
- Mencoba pendekatan hyperparameter tuning untuk mengoptimalkan performa model lebih lanjut.
- Menyertakan fitur tambahan atau eksternal (misalnya riwayat transaksi, catatan kredit pihak ketiga) untuk meningkatkan kompleksitas dan kekuatan prediksi model.

Proyek ini telah memberikan solusi prediktif yang kuat, namun eksplorasi lanjutan akan membantu dalam menghadirkan model yang lebih robust dan siap digunakan di lingkungan produksi nyata.
