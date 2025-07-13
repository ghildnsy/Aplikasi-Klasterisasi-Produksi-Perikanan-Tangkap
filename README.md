# 📊 Aplikasi Web Analisis Klasterisasi Produksi Perikanan (K-Medoids)

Aplikasi ini melakukan analisis dan klasterisasi terhadap dataset hasil _Produksi Perikanan Tangkap di Provinsi Jawa Barat_ menggunakan algoritma _K-Medoids_ dan visualisasi berbasis web dengan **Flask**.

## 🐟 Fitur

- Membaca dataset Excel produksi perikanan.
- Melakukan klasterisasi menggunakan K-Medoids dengan 3 klaster (Rendah, Sedang, Tinggi).
- Menampilkan data asli dan hasil klaster dalam tabel HTML.
- Visualisasi 2D hasil PCA dari data produksi.
- Menyediakan tombol untuk mengunduh hasil clustering dalam format Excel.

---

## 🛠️ Cara Instalasi Secara Lokal

## Buka terminal pada vscode

### 1. Masuk ke direktori project:

    cd SPK_FLASK/src

### 2. Buat dan Jalankan Virtual Environment (Optional)

#### Buat Virtual Environment:

    -m venv venv

#### Aktifkan Virtual Environment:

##### Windows:

    venv\Scripts\activate

##### Macos

    source venv/bin/activate

### 3. Install Dependensi Sesuai requirements.txt

    pip install -r requirements.txt

### 4. Jalankan aplikasi

     main.py

### 5. Aplikasi Berjalan

    http://127.0.0.1:5000/
    *tergantung terminal masing-masing
