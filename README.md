  # Predictive Analysis of Loan Approval Classification

## Domain Background

Seiring dengan meningkatnya kebutuhan masyarakat terhadap pinjaman keuangan, lembaga keuangan menghadapi tantangan besar dalam menyaring dan mengevaluasi aplikasi pinjaman secara efektif. Proses peninjauan manual terhadap aplikasi pinjaman sering kali membutuhkan waktu yang lama dan berpotensi menimbulkan bias, sehingga dibutuhkan solusi berbasis teknologi untuk meningkatkan efisiensi dan akurasi proses evaluasi.

Proyek ini bertujuan untuk membangun model prediktif menggunakan machine learning guna menentukan apakah suatu aplikasi pinjaman layak disetujui atau tidak. Dengan mengandalkan kombinasi data terstruktur seperti pendapatan, rasio utang terhadap pendapatan, dan skor kredit, serta data tidak terstruktur berupa narasi dari pemohon, model ini diharapkan dapat membantu lembaga keuangan mengambil keputusan secara cepat dan tepat.

Proyek ini juga merujuk pada penelitian terdahulu yang dilakukan oleh Viswanatha et al. (2023) yang mengimplementasikan beberapa algoritma machine learning untuk prediksi persetujuan pinjaman dengan hasil akurasi tertinggi sebesar 83,73% menggunakan algoritma Naive Bayes. Penelitian ini menggunakan fitur-fitur serupa seperti pendapatan dan status pekerjaan, sehingga menjadi referensi yang relevan dalam pengembangan proyek ini.

**Referensi:**

**Prediction of loan approval in banks using machine learning approach**. Viswanatha, V., Ramachandra, A. C., Vishwas, K. N., & Adithya, G. (2023), *International Journal of Engineering and Management Research, 13(4), 7-19*.

**Batasan Masalah:** 

Dalam proyek ini, hanya data terstruktur yang digunakan untuk membangun model prediksi persetujuan pinjaman. Fokus model sepenuhnya pada variabel numerik dan kategorikal seperti pendapatan, skor kredit, jumlah pinjaman, rasio utang terhadap pendapatan, dan status pekerjaan.

## Business Understanding

Permintaan pinjaman semakin meningkat, namun lembaga keuangan menghadapi tantangan dalam mengevaluasi kelayakan peminjam secara efisien dan akurat. Keputusan yang salah dapat menyebabkan kerugian finansial atau kehilangan peluang bisnis.

Machine learning dapat digunakan untuk membangun sistem prediktif berbasis data historis guna membantu proses persetujuan pinjaman secara otomatis dan tepat sasaran.

#### **Problem Statements**
- Bagaimana memanfaatkan data terstruktur untuk memprediksi apakah permohonan pinjaman disetujui?

- Algoritma machine learning apa yang paling efektif untuk memodelkan data persetujuan pinjaman?

#### **Goals**
- Mengembangkan model klasifikasi untuk memprediksi persetujuan pinjaman berdasarkan fitur finansial.

- Menentukan model terbaik di antara Random Forest, Naive Bayes, Decision Tree, dan KNN menggunakan metrik evaluasi.

#### **Solution Statement**
- Membangun model klasifikasi menggunakan:

  - Random Forest

  - Naive Bayes

  - Decision Tree

  - K-Nearest Neighbors (KNN)

- Melakukan preprocessing data terstruktur (encoding dan normalisasi).

- Evaluasi model menggunakan akurasi, precision, recall, dan F1-score. Pemilihan model terbaik menggunakan akurasi--merujuk pada penelitian yang menjadi refrensi proyek ini. 

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset yang men-*support* ML Loan-Solutions Elite project, sebuah model yang dirancang untuk memprediksi persetujuan pinjaman berdasarkan data keuangan pemohon. Dataset ini terdiri dari data terstruktur dan tersedia dalam format CSV. Dataset dapat diakses melalui Kaggle: https://www.kaggle.com/datasets/abhishekmishra08/loan-approval-datasets.

Dataset tersedia dalam format CSV dan memiliki 7 kolom:

- Text (string): Deskripsi alasan pengajuan pinjaman. (Tidak digunakan dalam proyek ini)

- Income (float): Total pendapatan pemohon.

- Credit_Score (integer): Skor kredit pemohon.

- Loan_Amount (float): Jumlah pinjaman yang diajukan.

- DTI_Ratio (float): Debt-to-Income Ratio, yaitu rasio antara utang dan pendapatan.

- Employment_Status (categorical): Status pekerjaan pemohon (misalnya employed, unemployed).

- Approval (categorical - target): Status pengajuan pinjaman (Approved atau Rejected).

Catatan:
Fitur Text tidak digunakan dalam proyek ini untuk menjaga fokus pada pemodelan berbasis data numerik dan kategorikal.

### Exploratory Data Analysis
Beberapa visualisasi dilakukan untuk memahami pola dalam data:

- preview dataset
<p align='center'>
      <img src ="(https://github.com/faRIZky/pred_analysis-loan/blob/main/images/preview%20dataset.png?raw=true)" alt="preview dataset"> 
</p>
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
