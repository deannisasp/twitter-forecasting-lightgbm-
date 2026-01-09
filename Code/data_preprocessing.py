#!/usr/bin/env python3
"""
data_preprocessing.py
Script untuk preprocessing data Twitter/X

Proses yang dilakukan:
1. Load data dari Excel
2. Data cleaning dan filtering
3. Text preprocessing (lowercasing, remove symbols, stemming, dll)
4. Language detection dan translation
5. Feature extraction
6. Export hasil

Author: Converted from Jupyter Notebook
Usage: python data_preprocessing.py

Requirements:
- pandas
- numpy
- plotly
- langdetect
- emoji
- transformers
- sacremoses
- sentencepiece
"""

import pandas as pd
import numpy as np
import plotly.express as px
import re
from langdetect import detect, DetectorFactory, detect_langs, LangDetectException
from langdetect.lang_detect_exception import LangDetectException
import emoji
from transformers import MarianMTModel, MarianTokenizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set seed untuk konsistensi hasil deteksi bahasa
DetectorFactory.seed = 0

# =============================================================================
# KONFIGURASI
# =============================================================================

# Path input file - sesuaikan dengan lokasi file Anda
INPUT_FILE = 'data(fix).xlsx'

# Path output file
OUTPUT_FILE = 'data_preprocessed.xlsx'

# =============================================================================
# FUNGSI-FUNGSI HELPER
# =============================================================================

def count_mentions_and_hashtags(text):
    """Menghitung jumlah mention (@) dan hashtag (#)"""
    mentions = len(re.findall(r'@\w+', text))
    hashtags = len(re.findall(r'#\w+', text))
    return mentions, hashtags

def extract_hashtags(text):
    """Ekstrak semua hashtag dari teks"""
    hashtags = re.findall(r'#\w+', text)
    return hashtags

def clean_text(text):
    """Membersihkan teks dari URL dan karakter khusus"""
    # Menghapus URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Menghapus karakter non-alfabet (kecuali spasi dan angka)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def normalize_emoji(text):
    """Menggantikan emoji dengan label khusus"""
    # Mengonversi emoji menjadi nama kata (misalnya ":grinning_face:")
    text = emoji.demojize(text)
    # Menambahkan label [emoji_<label>] untuk jenis emoji tertentu
    text = re.sub(r':(\w+_face|_\w+):', r'[emoji_\1]', text)
    return text

def detect_language_enhanced(text, lang_column=None):
    """Deteksi bahasa dengan enhanced accuracy"""
    if not text or len(str(text).strip()) < 3:
        return 'unknown'
    
    try:
        # Jika sudah ada kolom lang dari Twitter, gunakan itu sebagai hint
        if lang_column and lang_column in ['id', 'in']:
            detected = detect(str(text))
            if detected in ['id', 'in']:
                return 'id'
        
        # Deteksi bahasa menggunakan langdetect
        detected = detect(str(text))
        return detected
    except LangDetectException:
        return 'unknown'

def translate_text(text, source_lang, target_lang='id'):
    """Translate text menggunakan MarianMT model"""
    try:
        if source_lang == target_lang or source_lang == 'unknown':
            return text
        
        # Load model untuk translasi
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Tokenize dan translate
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**tokens)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def preprocess_text(text):
    """Preprocessing teks lengkap"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_keywords(text, top_n=10):
    """Ekstrak kata-kata yang sering muncul"""
    words = text.lower().split()
    # Filter kata-kata pendek dan stopwords sederhana
    words = [w for w in words if len(w) > 3 and w not in ['yang', 'untuk', 'dengan', 'dari', 'pada']]
    word_freq = Counter(words)
    return word_freq.most_common(top_n)

# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def main():
    """Fungsi utama untuk menjalankan preprocessing"""
    
    print("="*80)
    print("DATA PREPROCESSING SCRIPT")
    print("="*80)
    
    # 1. Load data
    print("\n[1] Loading data...")
    try:
        data = pd.read_excel(INPUT_FILE)
        print(f"✅ Data berhasil dimuat: {len(data)} baris, {len(data.columns)} kolom")
    except FileNotFoundError:
        print(f"❌ File {INPUT_FILE} tidak ditemukan!")
        print("💡 Pastikan file ada di direktori yang sama dengan script ini")
        return
    except Exception as e:
        print(f"❌ Error saat membaca file: {e}")
        return
    
    # 2. Pilih kolom yang diperlukan
    print("\n[2] Memilih kolom yang relevan...")
    required_cols = ['id_str', 'tanggal', 'full_text', 'lang', 'favorite_count', 
                     'quote_count', 'reply_count', 'retweet_count', 'username', 
                     'followers', 'following', 'verified_status']
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"⚠️  Kolom yang tidak ditemukan: {missing_cols}")
        print("📋 Kolom yang tersedia:", list(data.columns))
        # Gunakan kolom yang ada saja
        required_cols = [col for col in required_cols if col in data.columns]
    
    df = data[required_cols].copy()
    print(f"✅ {len(required_cols)} kolom dipilih")
    
    # 3. Rename kolom untuk konsistensi
    print("\n[3] Rename kolom...")
    rename_dict = {
        'id_str': 'id',
        'full_text': 'tweet',
        'favorite_count': 'like',
        'quote_count': 'quote',
        'reply_count': 'reply',
        'retweet_count': 'retweet'
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
    print("✅ Kolom berhasil direname")
    
    # 4. Ekstrak fitur dari teks
    print("\n[4] Ekstrak fitur dari tweet...")
    if 'tweet' in df.columns:
        # Hitung mentions dan hashtags
        df[['mentions_count', 'hashtags_count']] = df['tweet'].apply(
            lambda x: pd.Series(count_mentions_and_hashtags(str(x)))
        )
        
        # Ekstrak list hashtags
        df['hashtags_list'] = df['tweet'].apply(lambda x: extract_hashtags(str(x)))
        
        # Hitung panjang tweet
        df['tweet_length'] = df['tweet'].apply(lambda x: len(str(x)))
        
        # Hitung jumlah kata
        df['word_count'] = df['tweet'].apply(lambda x: len(str(x).split()))
        
        print(f"✅ Fitur berhasil diekstrak")
    
    # 5. Clean text
    print("\n[5] Membersihkan teks...")
    if 'tweet' in df.columns:
        df['tweet_clean'] = df['tweet'].apply(lambda x: clean_text(str(x)))
        df['tweet_normalized'] = df['tweet_clean'].apply(lambda x: normalize_emoji(str(x)))
        df['tweet_preprocessed'] = df['tweet_normalized'].apply(preprocess_text)
        print("✅ Teks berhasil dibersihkan")
    
    # 6. Deteksi bahasa
    print("\n[6] Mendeteksi bahasa...")
    if 'tweet_clean' in df.columns:
        lang_col = 'lang' if 'lang' in df.columns else None
        df['detected_lang'] = df.apply(
            lambda row: detect_language_enhanced(row['tweet_clean'], row.get(lang_col)), 
            axis=1
        )
        
        lang_dist = df['detected_lang'].value_counts()
        print("📊 Distribusi bahasa:")
        for lang, count in lang_dist.head(10).items():
            print(f"   {lang}: {count} ({count/len(df)*100:.1f}%)")
    
    # 7. Filter data bahasa Indonesia
    print("\n[7] Filter data bahasa Indonesia...")
    if 'detected_lang' in df.columns:
        df_filtered = df[df['detected_lang'].isin(['id', 'in'])].copy()
        print(f"✅ Data terfilter: {len(df_filtered)} baris (dari {len(df)} baris)")
        print(f"   Persentase: {len(df_filtered)/len(df)*100:.1f}%")
    else:
        df_filtered = df.copy()
    
    # 8. Statistik deskriptif
    print("\n[8] Statistik data:")
    if all(col in df_filtered.columns for col in ['like', 'reply', 'retweet']):
        print(f"   Rata-rata like: {df_filtered['like'].mean():.2f}")
        print(f"   Rata-rata reply: {df_filtered['reply'].mean():.2f}")
        print(f"   Rata-rata retweet: {df_filtered['retweet'].mean():.2f}")
        
        if 'followers' in df_filtered.columns:
            print(f"   Rata-rata followers: {df_filtered['followers'].mean():.2f}")
    
    # 9. Save hasil
    print(f"\n[9] Menyimpan hasil ke {OUTPUT_FILE}...")
    try:
        df_filtered.to_excel(OUTPUT_FILE, index=False)
        print(f"✅ File berhasil disimpan!")
        print(f"📁 Lokasi: {OUTPUT_FILE}")
        print(f"📊 Total baris: {len(df_filtered)}")
        print(f"📊 Total kolom: {len(df_filtered.columns)}")
    except Exception as e:
        print(f"❌ Error saat menyimpan file: {e}")
    
    # 10. Summary
    print("\n" + "="*80)
    print("PREPROCESSING SELESAI!")
    print("="*80)
    print(f"\n📋 Kolom yang tersedia di output:")
    for i, col in enumerate(df_filtered.columns, 1):
        print(f"   {i}. {col}")
    
    print(f"\n💡 Tip: Gunakan file '{OUTPUT_FILE}' untuk analisis lebih lanjut")
    
    return df_filtered

if __name__ == "__main__":
    df_result = main()
