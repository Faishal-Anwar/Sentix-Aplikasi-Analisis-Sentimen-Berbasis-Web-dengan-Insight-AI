# Sentix: Aplikasi Analisis Sentimen Berbasis Web dengan Insight AI
by Faishal Anwar Hasyim

Sebuah aplikasi web intuitif untuk menganalisis sentimen dari data teks (seperti ulasan atau tweet) secara otomatis, mengubah data mentah menjadi visualisasi dan insight yang mudah dipahami.

![image](https://github.com/user-attachments/assets/2dc07f7d-1fe4-4e42-a563-ac1f32aaaad2)


---

## **Tentang Proyek Ini**

Analisis sentimen adalah proses yang krusial untuk memahami opini publik, namun seringkali memerlukan keahlian teknis dalam pemrograman dan machine learning. Prosesnya yang rumit, mulai dari pembersihan data hingga pembuatan model di Jupyter Notebook, menjadi penghalang bagi banyak orang.

**Sentix** dibangun untuk menjembatani jurang tersebut. Proyek ini menyediakan antarmuka yang ramah pengguna di mana siapa pun dapat mengunggah dataset mereka (dalam format CSV atau Excel) dan mendapatkan laporan analisis sentimen yang komprehensif hanya dengan beberapa klik. Aplikasi ini tidak hanya memberikan klasifikasi sentimen (Positif, Negatif, Netral), tetapi juga menyajikan visualisasi data interaktif dan insight yang dihasilkan oleh AI generatif untuk pemahaman yang lebih mendalam.

Proyek ini sepenuhnya siap untuk di-deploy menggunakan **Docker**, memastikan proses setup yang konsisten, portabel, dan skalabel.

## **Fitur Utama**

* 📤 **Upload Data Fleksibel:** Mendukung file `.csv` dan `.xlsx` sebagai sumber data.
* 🤖 **Analisis Sentimen Otomatis:** Menggunakan model dari Hugging Face Transformers untuk mengklasifikasikan teks ke dalam kategori Positif, Negatif, dan Netral.
* 📊 **Visualisasi Interaktif:** Menghasilkan Pie Chart dan Bar Chart distribusi sentimen menggunakan Plotly.
* ☁️ **Word Cloud Dinamis:** Menampilkan kata-kata kunci yang paling sering muncul untuk setiap kategori sentimen.
* 💡 **Insight Berbasis AI:** Menggunakan Google Gemini untuk memberikan ringkasan dan tema utama dari setiap sentimen, membantu pengguna memahami "mengapa" di balik data.
* 🐳 **Siap Docker:** Dilengkapi dengan `Dockerfile` yang dioptimalkan untuk deployment yang mudah dan efisien.

## **Tumpukan Teknologi (Tech Stack)**

* **Backend:** Python, FastAPI, Uvicorn
* **AI & Machine Learning:** Transformers (Hugging Face), PyTorch, NLTK, Sastrawi
* **Generative AI:** Google Generative AI (Gemini)
* **Analisis & Visualisasi:** Pandas, Plotly, WordCloud
* **Frontend:** HTML5, Bootstrap 5, Jinja2 Templates
* **Deployment:** Docker

## **Struktur Proyek**

Struktur direktori yang direkomendasikan untuk menjalankan proyek ini dengan `Dockerfile` yang disediakan:

```
SENTIMENT_APP/
├── Dockerfile
├── main.py
├── requirements.txt
├── .env
├── static/
│   └── (File CSS atau gambar statis lainnya)
└── templates/
    ├── index.html
    └── results.html
```

## **Panduan Menjalankan Aplikasi**

Ada dua cara untuk menjalankan aplikasi ini: menggunakan Docker (direkomendasikan) atau menjalankan secara lokal.

### **1. Menjalankan dengan Docker (Direkomendasikan)**

Ini adalah cara termudah dan paling andal untuk menjalankan aplikasi.

**Prasyarat:**
* Docker sudah terinstall di sistem Anda.

**Langkah-langkah:**

1.  **Clone Repositori (Jika ada di Git):**
    ```bash
    git clone [URL_REPOSITORY_ANDA]
    cd SENTIMENT_APP
    ```

2.  **Buat file `.env`:**
    Buat sebuah file bernama `.env` di dalam folder `SENTIMENT_APP` dan tambahkan kunci API Anda.
    ```
    GOOGLE_API_KEY="MASUKKAN_KUNCI_API_ANDA_DI_SINI"
    ```

3.  **Bangun (Build) Image Docker:**
    Buka terminal di dalam folder `SENTIMENT_APP` dan jalankan:
    ```bash
    docker build -t sentix .
    ```

4.  **Jalankan Kontainer Docker:**
    Jalankan kontainer dengan meneruskan file `.env` untuk variabel lingkungan.
    ```bash
    docker run -d -p 8000:8000 --env-file .env sentix
    ```
    * `-d`: Menjalankan di latar belakang (detached).
    * `-p 8000:8000`: Memetakan port 8000 dari komputer Anda ke port 8000 di dalam kontainer.
    * `--env-file .env`: Cara aman untuk memuat variabel lingkungan (seperti API Key) dari file.

5.  **Akses Aplikasi:**
    Buka browser Anda dan kunjungi **`http://localhost:8000`**.

### **2. Menjalankan Secara Lokal (Tanpa Docker)**

**Prasyarat:**
* Python 3.8+ terinstall.
* `pip` package manager.

**Langkah-langkah:**

1.  **Clone Repositori dan Masuk ke Folder:**
    ```bash
    git clone [URL_REPOSITORY_ANDA]
    cd SENTIMENT_APP
    ```

2.  **Buat dan Aktifkan Virtual Environment (Sangat Direkomendasikan):**
    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Semua Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Buat file `.env`:**
    Sama seperti pada metode Docker, buat file `.env` dan isi dengan kunci API Anda.
    ```
    GOOGLE_API_KEY="MASUKKAN_KUNCI_API_ANDA_DI_SINI"
    ```

5.  **Jalankan Server Aplikasi:**
    ```bash
    uvicorn main:app --reload
    ```
    * `--reload`: Server akan otomatis restart jika Anda membuat perubahan pada kode.

6.  **Akses Aplikasi:**
    Buka browser Anda dan kunjungi **`http://127.0.0.1:8000`**.
