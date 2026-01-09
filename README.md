# twitter-forecasting-lightgbm-
Time series forecasting of Twitter/X topic, word, and hashtag frequencies using LightGBM with Optuna-based hyperparameter optimization to support data-driven social media marketing strategies.

# Twitter Data Analysis Pipeline

Pipeline lengkap untuk crawling, preprocessing, dan forecasting data Twitter/X menggunakan Python dan machine learning.

## 📋 Daftar File

1. **twitter_crawler.py** - Script untuk crawling data dari Twitter/X
2. **merge_data.py** - Script untuk menggabungkan multiple file CSV/XLSX
3. **data_preprocessing.py** - Script untuk preprocessing dan cleaning data
4. **forecasting.py** - Script untuk time series forecasting dengan LightGBM + Optuna

## 🔧 Requirements

### Software Requirements
- Python 3.8+
- Node.js 18+ (untuk twitter_crawler.py)
- NPM (biasanya terinstall bersama Node.js)

### Python Libraries

Install semua dependencies dengan perintah:

```bash
pip install pandas numpy plotly langdetect emoji transformers sacremoses sentencepiece bertopic sentence-transformers LazyProphet optuna optuna-integration[lightgbm] lightgbm scikit-learn matplotlib seaborn openpyxl
```

Atau buat file `requirements.txt`:

```txt
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.14.0
langdetect>=1.0.9
emoji>=2.2.0
transformers>=4.30.0
sacremoses>=0.0.53
sentencepiece>=0.1.99
bertopic>=0.15.0
sentence-transformers>=2.2.0
LazyProphet>=0.3.0
optuna>=3.2.0
optuna-integration[lightgbm]>=3.2.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
```

Lalu install dengan:
```bash
pip install -r requirements.txt
```

## 📖 Cara Penggunaan

### 1. Twitter Crawler (`twitter_crawler.py`)

Script untuk crawling data dari Twitter/X menggunakan tweet-harvest.

#### Persiapan:
1. **Install Node.js dan NPM**
   - Download dari https://nodejs.org/
   - Verifikasi instalasi: `node --version` dan `npm --version`

2. **Dapatkan Twitter Auth Token**
   - Login ke Twitter/X di browser
   - Buka Developer Tools (tekan F12)
   - Buka tab "Network"
   - Reload halaman
   - Cari request yang mengandung cookies
   - Temukan dan copy nilai `auth_token`

3. **Edit Script**
   - Buka `twitter_crawler.py`
   - Ganti `TWITTER_AUTH_TOKEN` dengan token Anda
   - Sesuaikan `CRAWL_CONFIGS` sesuai kebutuhan (akun, keyword, tanggal)

#### Menjalankan:
```bash
python twitter_crawler.py
```

Pilih mode:
- **Mode 1 (Batch)**: Crawl multiple accounts sekaligus
- **Mode 2 (Custom)**: Crawl dengan keyword custom

#### Konfigurasi Crawling:

Edit bagian `CRAWL_CONFIGS` di script:

```python
CRAWL_CONFIGS = [
    {
        'name': 'alfamart',  # Nama untuk identifikasi
        'filename': 'data_alfamart.csv',  # Output filename
        'keyword': 'from:alfamart promo OR sale since:2024-11-27 until:2025-11-27 lang:id',  # Search query
        'limit': 1000  # Maximum tweets
    },
    # Tambahkan konfigurasi lain di sini
]
```

#### Format Search Keyword:
- `from:username` - Tweet dari akun tertentu
- `promo OR sale OR diskon` - Kata kunci dengan operator OR
- `since:YYYY-MM-DD` - Tanggal mulai
- `until:YYYY-MM-DD` - Tanggal akhir
- `lang:id` - Bahasa (id = Indonesia)

#### Output:
- File CSV untuk setiap akun yang di-crawl
- Format: `data_[nama_akun].csv`

---

### 2. Merge Data (`merge_data.py`)

Script untuk menggabungkan multiple file CSV/XLSX menjadi satu file.

#### Persiapan:
1. Buat folder bernama `data_new` di direktori yang sama dengan script
2. Pindahkan semua file CSV/XLSX hasil crawling ke folder tersebut

Atau edit path di script:
```python
folder_path = './data_new'  # Ganti sesuai lokasi folder Anda
```

#### Menjalankan:
```bash
python merge_data.py
```

#### Output:
- File: `data(fix).xlsx`
- Berisi gabungan semua file CSV/XLSX dari folder input

---

### 3. Data Preprocessing (`data_preprocessing.py`)

Script untuk membersihkan dan memproses data Twitter.

#### Persiapan:
Edit path input file di script:
```python
INPUT_FILE = 'data(fix).xlsx'  # File hasil dari merge_data.py
```

#### Proses yang Dilakukan:
1. ✅ Load data dari Excel
2. ✅ Pilih kolom yang relevan
3. ✅ Rename kolom untuk konsistensi
4. ✅ Ekstrak fitur (mentions, hashtags, panjang tweet, jumlah kata)
5. ✅ Clean text (hapus URL, karakter khusus)
6. ✅ Normalisasi emoji
7. ✅ Deteksi bahasa
8. ✅ Filter data bahasa Indonesia
9. ✅ Export hasil

#### Menjalankan:
```bash
python data_preprocessing.py
```

#### Output:
- File: `data_preprocessed.xlsx`
- Berisi data yang sudah dibersihkan dan siap untuk analisis

#### Kolom Output:
- `id` - ID tweet
- `tanggal` - Timestamp
- `tweet` - Teks tweet original
- `tweet_clean` - Teks yang sudah dibersihkan
- `tweet_preprocessed` - Teks final untuk analisis
- `mentions_count` - Jumlah mention (@)
- `hashtags_count` - Jumlah hashtag (#)
- `hashtags_list` - List semua hashtag
- `tweet_length` - Panjang karakter tweet
- `word_count` - Jumlah kata
- `detected_lang` - Bahasa yang terdeteksi
- Engagement metrics: `like`, `reply`, `retweet`, `quote`
- User info: `username`, `followers`, `following`, `verified_status`

---

### 4. Forecasting (`forecasting.py`)

Script untuk time series forecasting menggunakan machine learning.

#### Persiapan:
Edit path dan konfigurasi di script:
```python
INPUT_FILE = 'data bersih.xlsx'  # Atau 'data_preprocessed.xlsx'
N_TRIALS = 50  # Jumlah trial Optuna (lebih banyak = lebih akurat tapi lebih lama)
TEST_SIZE_RATIO = 0.2  # 20% data untuk testing
```

#### Proses yang Dilakukan:
1. ✅ Topic modeling dengan BERTopic (IndoBERTweet)
2. ✅ Agregasi data harian (topik, kata, hashtag)
3. ✅ Feature engineering (lag features, rolling statistics)
4. ✅ Hyperparameter tuning dengan Optuna
5. ✅ Forecasting dengan LightGBM
6. ✅ Evaluasi (RMSE, MAE, RMSSE)
7. ✅ Visualisasi hasil

#### Menjalankan:
```bash
python forecasting.py
```

**⚠️ Perhatian**: 
- Script ini membutuhkan waktu lama (30 menit - 2 jam tergantung ukuran data)
- Membutuhkan RAM yang cukup besar (minimal 8GB)
- Proses download model IndoBERTweet (~500MB) di run pertama

#### Output:
1. **Excel File**: `forecast_results_lightgbm_optuna.xlsx`
   - Sheet "Topics": Forecast untuk topik
   - Sheet "Words": Forecast untuk kata
   - Sheet "Hashtags": Forecast untuk hashtag
   - Berisi: item name, RMSE, MAE, RMSSE, actual values, predicted values

2. **Visualisasi**:
   - `forecast_topics_lgbm.png` - Plot forecast topik
   - `forecast_words_lgbm.png` - Plot forecast kata
   - `forecast_hashtags_lgbm.png` - Plot forecast hashtag

#### Metrik Evaluasi:
- **RMSE** (Root Mean Squared Error): Error rata-rata prediksi
- **MAE** (Mean Absolute Error): Rata-rata absolut error
- **RMSSE** (Root Mean Squared Scaled Error): Error yang dinormalisasi

Semakin kecil nilai metrik, semakin baik modelnya.

---

## 🔄 Workflow Lengkap

### Pipeline Standar:

```
1. twitter_crawler.py
   ↓ (menghasilkan: data_*.csv)
   
2. merge_data.py
   ↓ (menghasilkan: data(fix).xlsx)
   
3. data_preprocessing.py
   ↓ (menghasilkan: data_preprocessed.xlsx)
   
4. forecasting.py
   ↓ (menghasilkan: forecast results + visualisasi)
```

### Contoh Eksekusi:

```bash
# Step 1: Crawl data
python twitter_crawler.py
# Pilih mode 1, ikuti instruksi
# Output: data_alfamart.csv, data_shopee.csv, dll

# Step 2: Merge semua file
python merge_data.py
# Output: data(fix).xlsx

# Step 3: Preprocessing
python data_preprocessing.py
# Output: data_preprocessed.xlsx

# Step 4: Forecasting
python forecasting.py
# Output: forecast_results_lightgbm_optuna.xlsx + visualisasi
```

---

## ⚙️ Kustomisasi

### Mengubah Periode Crawling:

Edit keyword di `twitter_crawler.py`:
```python
'keyword': 'from:alfamart promo since:2024-01-01 until:2024-12-31 lang:id'
#                                      ^^^^^^^^^^       ^^^^^^^^^^
#                                      tanggal mulai    tanggal akhir
```

### Mengubah Jumlah Tweet:

```python
'limit': 5000  # Ubah sesuai kebutuhan
```

### Mengubah Stopwords:

Edit `INDONESIAN_STOPWORDS` di `forecasting.py` untuk menambah/mengurangi kata yang diabaikan.

### Mengubah Jumlah Forecast:

Di `forecasting.py`, ubah:
```python
for topic in sorted(topics)[:10]:  # Ubah 10 menjadi N
```

---

## ❗ Troubleshooting

### Problem: ModuleNotFoundError
**Solusi**: Install library yang kurang
```bash
pip install [nama-library]
```

### Problem: Node.js not found (twitter_crawler.py)
**Solusi**: 
- Install Node.js dari https://nodejs.org/
- Restart terminal setelah install
- Cek dengan: `node --version`

### Problem: Auth token tidak valid
**Solusi**:
- Pastikan Anda login ke Twitter di browser
- Dapatkan token baru dari browser cookies
- Token biasanya berubah setelah logout/login

### Problem: Memory error saat forecasting
**Solusi**:
- Kurangi `N_TRIALS` menjadi 20 atau 30
- Kurangi jumlah item yang di-forecast
- Close aplikasi lain untuk free up RAM

### Problem: File tidak ditemukan
**Solusi**:
- Pastikan path file benar
- Pastikan file ada di direktori yang sama atau sesuaikan path di script
- Gunakan absolute path jika perlu: `/full/path/to/file.xlsx`

### Problem: Model download gagal (forecasting.py)
**Solusi**:
- Pastikan koneksi internet stabil
- Model IndoBERTweet akan didownload otomatis di run pertama (~500MB)
- Jika gagal, coba jalankan ulang script

---

## 📊 Expected Results

### Preprocessing Output:
```
[1] Loading data...
✅ Data berhasil dimuat: 3549 baris, 18 kolom

[2] Memilih kolom yang relevan...
✅ 12 kolom dipilih

[3] Rename kolom...
✅ Kolom berhasil direname

[4] Ekstrak fitur dari tweet...
✅ Fitur berhasil diekstrak

[5] Membersihkan teks...
✅ Teks berhasil dibersihkan

[6] Mendeteksi bahasa...
📊 Distribusi bahasa:
   id: 2845 (80.2%)
   en: 456 (12.9%)
   ...

[7] Filter data bahasa Indonesia...
✅ Data terfilter: 2845 baris (dari 3549 baris)

[9] Menyimpan hasil ke data_preprocessed.xlsx...
✅ File berhasil disimpan!
```

### Forecasting Output:
```
[3] FORECASTING FREKUENSI TOPIK
  → Forecasting: Topic_1: shopee, promo, gratis
    Total data points: 488
    Train size: 380, Test size: 94
    Running Optuna optimization (50 trials)...
    ✓ Best RMSE: 0.5432

...

FORECASTING SELESAI!
Output files:
  1. forecast_results_lightgbm_optuna.xlsx
  2. forecast_topics_lgbm.png
  3. forecast_words_lgbm.png
  4. forecast_hashtags_lgbm.png
```

---

## 📝 Tips & Best Practices

1. **Crawling**:
   - Jangan crawl terlalu banyak tweet sekaligus (max 5000 per akun)
   - Beri jeda antar crawling untuk menghindari rate limit
   - Backup auth token Anda di tempat aman

2. **Preprocessing**:
   - Periksa hasil deteksi bahasa, sesuaikan threshold jika perlu
   - Simpan hasil intermediate untuk debugging
   - Check missing values sebelum lanjut ke forecasting

3. **Forecasting**:
   - Start dengan N_TRIALS=20 untuk testing cepat
   - Gunakan N_TRIALS=50-100 untuk hasil final
   - Monitor penggunaan RAM saat running
   - Forecasting lebih akurat dengan data >= 6 bulan

4. **General**:
   - Selalu backup data original
   - Dokumentasikan perubahan konfigurasi
   - Gunakan version control (Git) untuk track changes

---

## 📚 Referensi

- **tweet-harvest**: https://github.com/helmisatria/tweet-harvest
- **BERTopic**: https://maartengr.github.io/BERTopic/
- **IndoBERTweet**: https://huggingface.co/indolem/indobertweet-base-uncased
- **LightGBM**: https://lightgbm.readthedocs.io/
- **Optuna**: https://optuna.readthedocs.io/


---

## 📄 License

Gunakan dengan bijak dan hormati Terms of Service Twitter/X.

---

## 🆘 Support

Jika mengalami masalah:
1. Baca troubleshooting section di atas
2. Check apakah semua requirements sudah terinstall
3. Pastikan file path sudah benar
4. Periksa format data input sesuai expected format

**Selamat menganalisis data Twitter! 🚀**
