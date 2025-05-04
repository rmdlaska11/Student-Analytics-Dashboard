# Proyek Akhir: Monitoring dan Prediksi Dropout Mahasiswa - Jaya Jaya Institut

## Business Understanding

**Jaya Jaya Institut** adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan memiliki reputasi yang baik dalam mencetak lulusan berkualitas. Namun, dalam beberapa tahun terakhir, institusi ini menghadapi permasalahan serius terkait tingginya angka *dropout* mahasiswa. Hal ini dapat mempengaruhi citra institusi dan efektivitas sistem pendidikan yang diterapkan.

Untuk mengatasi hal ini, Jaya Jaya Institut ingin menerapkan pendekatan berbasis data untuk:
- Memantau performa dan karakteristik mahasiswa secara menyeluruh melalui dashboard.
- Mengembangkan sistem prediktif untuk mendeteksi risiko dropout mahasiswa sejak dini.

### Permasalahan Bisnis

- Tingginya angka *dropout* mahasiswa
- Kebutuhan akan monitoring performa mahasiswa yang lebih efisien dan visual
- Kurangnya alat bantu analisis untuk mendeteksi potensi dropout secara proaktif
- Kesulitan dalam segmentasi dan analisis karakteristik mahasiswa berdasarkan atribut seperti umur, gender, beasiswa, jam kuliah, dll.

### Cakupan Proyek

- Eksplorasi dan pembersihan dataset performa mahasiswa
- Pembuatan business dashboard untuk visualisasi performa dan distribusi mahasiswa
- Pengembangan model machine learning untuk prediksi risiko dropout
- Pembuatan prototype sistem untuk prediksi dan monitoring
- Penyusunan rekomendasi berbasis data untuk membantu pihak institusi

### Persiapan

**Sumber data:**
- Dataset performa mahasiswa (students' performance) dari Jaya Jaya Institut ([tautan](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv))

**Setup environment:**
```
# Instal virtual environment dengan Conda
conda create --name student_analytics python=3.10

# Aktifkan environment
conda activate student_analytics

# Install dependencies dari requirements.txt
pip install -r requirements.txt

# Jalankan aplikasi Streamlit
streamlit run app.py
```

**Setup Dashboard Metabase:**

```
# 1. Download Docker Image untuk Metabase
docker pull metabase/metabase:v0.46.4

# 2. Jalankan Metabase di Container Baru
docker run -p 3000:3000 --name metabase metabase/metabase

# 3. Akses Metabase dari Browser
http://localhost:3000

# 4. Login dengan akun berikut:
Username: rahmadlaska11@gmail.com  
Password: rahmad12
```

## Business Dashboard

### 1. Tujuan dan Fungsi Dashboard  
**Student Performance Monitoring Dashboard** dirancang untuk memberikan pandangan menyeluruh terkait kondisi akademik dan demografis mahasiswa. Melalui visualisasi data yang interaktif dan real-time, dashboard ini membantu pemangku kepentingan untuk:
- Mengidentifikasi pola keberhasilan dan risiko akademik
- Memantau distribusi siswa berdasarkan faktor sosial dan ekonomi
- Mengevaluasi efektivitas program studi
- Merancang intervensi yang ditargetkan untuk peningkatan performa akademik

### 2. Fitur Filter untuk Analisis Dinamis  
Dashboard ini mendukung eksplorasi data yang fleksibel melalui filter berikut:
- **Program Studi**
- **Kewarganegaraan**
- **Status Pernikahan**
- **Usia Saat Pendaftaran**
- **Status Akademik (Enrolled, Dropout, Graduate)**

### 3. Metrik Utama yang Dimonitor

Beberapa indikator kunci yang ditampilkan di bagian atas dashboard:
- **Total Mahasiswa**: 4.424
- **Jumlah Program Studi**: 17
- **Negara Asal Mahasiswa**: 21
- **Tingkat Kelulusan (Graduation Rate)**: 49,93%
- **Tingkat Dropout**: 32,12%
- **Rata-rata Usia Saat Pendaftaran**: 23,27 tahun

### 4. Analisis Visual: Faktor Kunci Performa Mahasiswa
- **Distribusi Mahasiswa Berdasarkan Program Studi**  
  - **Nursing** memiliki pendaftar terbanyak (**766 mahasiswa**), diikuti **Management** (380) dan **Social Service** (355).
  - Informasi ini berguna untuk mengalokasikan sumber daya akademik secara efisien.

- **Status Beasiswa dan Keterlibatan Akademik**  
  - Mayoritas mahasiswa **tidak menerima beasiswa**.
  - Institusi dapat mempertimbangkan penyesuaian bantuan keuangan untuk meningkatkan partisipasi akademik.

- **Kelas Siang vs Malam**  
  - **89% mahasiswa** mengikuti kelas **siang**, dan hanya **11%** di kelas **malam**.
  - Ini memberi gambaran tentang preferensi waktu belajar mahasiswa dan kapasitas jadwal pengajaran.

- **Gender dan Status Akademik**  
  - Mahasiswa perempuan mendominasi kategori **lulus** dan **aktif**, sementara **dropout** cukup seimbang antara laki-laki dan perempuan.
  - Menandakan perbedaan performa akademik berdasarkan gender.

- **Usia Saat Pendaftaran**    
  - Mayoritas mendaftar di usia **15-23 tahun (3.047 mahasiswa)**.
  - Tren ini menunjukkan bahwa institusi menarik sebagian besar siswa dari kelompok usia muda.

- **Status Pernikahan dan Fokus Belajar**    
  - Sebagian besar mahasiswa berstatus **lajang (3.919)**.
  - Kelompok mahasiswa menikah mungkin memerlukan sistem pembelajaran yang lebih fleksibel.

### 5. Kesimpulan  
Dashboard ini menyajikan gambaran menyeluruh yang sangat berguna dalam mengevaluasi dan meningkatkan kualitas pendidikan. Beberapa faktor utama yang harus selalu dipantau antara lain adalah status akademik mahasiswa, distribusi gender, status sosial (seperti pernikahan dan beasiswa), serta usia saat mendaftar. Dengan memanfaatkan informasi ini secara berkala, institusi dapat lebih tanggap dalam merancang kebijakan untuk mengurangi tingkat dropout, meningkatkan tingkat kelulusan, serta menciptakan pengalaman belajar yang lebih adil dan inklusif.

## Menjalankan Sistem Machine Learning
### Deskripsi Singkat
Aplikasi ini memanfaatkan model machine learning (menggunakan **XGBoost**) untuk memprediksi status akhir mahasiswa (**Dropout**, **Enrolled**, atau **Graduate**) berdasarkan data akademik dan demografis. Dibangun menggunakan **Python** dan **Streamlit**.

🔗 **Akses aplikasi**:  
[Link Streamlit](https://rmd-student-analytics.streamlit.app/)


### Cara Menjalankan Aplikasi

#### 1. Input Data Mahasiswa
Terdapat dua cara input:

#### A. Input Manual (satu per satu)
- Pilih tab **Input Manual**.
- Isi setiap kolom sesuai kondisi mahasiswa: status pernikahan, mode pendaftaran, nilai masuk, status orang tua, nilai semester, dan lainnya.
- Klik tombol **Prediksi Status Mahasiswa**.
- Hasil prediksi akan muncul di bawah tabel **Riwayat Prediksi Sebelumnya**, dan bisa diunduh sebagai CSV.

#### B. Upload CSV (untuk banyak mahasiswa)
- Pilih tab **Upload CSV**.
- Klik **Browse files** atau **drag-and-drop** file CSV.
- Pastikan format file sesuai dengan template contoh (bisa diunduh pada menu **Contoh Data**).
- Setelah file terunggah, klik tombol **Prediksi dari File Upload**.
- Hasil prediksi akan tampil dan bisa diunduh.

### 🧩 Fitur Tambahan
- Tersedia tombol **Unduh Sample CSV** sebagai acuan format input.
- Hasil prediksi disimpan sementara dalam riwayat dan dapat diunduh kembali.
- Tampilan antarmuka dibuat ramah pengguna dan mendukung upload hingga **200MB** data.

## Conclusion

Proyek ini dirancang untuk membantu Jaya Jaya Institut menurunkan angka dropout melalui pendekatan berbasis data. Dengan menggabungkan dashboard analitik, model machine learning, dan aplikasi Streamlit, institusi kini memiliki alat yang efektif untuk mendeteksi mahasiswa berisiko dan merancang intervensi lebih dini.

Hasil analisis menunjukkan bahwa faktor akademik, seperti jumlah mata kuliah yang disetujui pada awal studi dan status pembayaran, merupakan penentu utama kelulusan. Program studi seperti Management (kelas malam) memiliki tingkat dropout lebih tinggi, sedangkan Nursing dan Social Service mencatat tingkat kelulusan terbaik. Usia juga berperan, dengan mahasiswa dropout rata-rata berusia 26 tahun saat masuk, lebih tua dibandingkan lulusan. Meskipun beasiswa bukan faktor penentu utama, dukungan finansial tetap penting untuk kelompok rentan.

Dengan sistem ini, Jaya Jaya Institut dapat menyusun strategi intervensi yang lebih proaktif, mulai dari pendampingan akademik di awal studi, pemantauan masalah pembayaran, hingga perluasan dukungan finansial. Pendekatan ini diharapkan mampu menekan angka dropout, meningkatkan tingkat kelulusan, dan memperkuat reputasi institusi.

## Rekomendasi Action Items

Berikut beberapa rekomendasi yang dapat dilakukan oleh Jaya Jaya Institut:

- **Perkuat dukungan akademik di awal studi:**  
  Sediakan program pendampingan (mentoring), bimbingan belajar, atau kelas remedial khusus untuk mahasiswa baru agar mereka dapat menyelesaikan mata kuliah semester 1 dan 2 dengan baik.

- **Pantau dan bantu masalah pembayaran mahasiswa:**  
  Bangun sistem peringatan dini untuk mahasiswa yang menunggak biaya kuliah dan sediakan opsi pembayaran yang fleksibel atau bantuan finansial.

- **Analisis khusus pada jurusan dropout tinggi (misalnya Management):**  
  Lakukan evaluasi terhadap kurikulum, beban studi, atau kualitas pengajaran di jurusan yang mencatat tingkat dropout tinggi.

- **Perluas dukungan psikososial:**  
  Meskipun faktor sosial tidak dominan, sediakan layanan konseling untuk membantu mahasiswa yang mengalami masalah pribadi, khususnya mahasiswa yang lebih tua dan berisiko dropout.
