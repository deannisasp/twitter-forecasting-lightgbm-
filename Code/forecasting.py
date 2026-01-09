#!/usr/bin/env python3
"""
forecasting.py
Script untuk time series forecasting data Twitter menggunakan LightGBM + Optuna

Proses yang dilakukan:
1. Topic modeling dengan BERTopic
2. Aggregasi data harian untuk topik, kata, dan hashtag
3. Hyperparameter tuning dengan Optuna
4. Forecasting dengan LightGBM
5. Evaluasi dan visualisasi hasil

Author: Converted from Jupyter Notebook
Usage: python forecasting.py

Requirements:
- pandas
- numpy
- bertopic
- sentence-transformers
- LazyProphet
- optuna
- lightgbm
- scikit-learn
- matplotlib
- seaborn
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from LazyProphet.LazyProphet import LazyProphet
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import re
from collections import Counter

warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI
# =============================================================================

# Path input file
INPUT_FILE = 'data bersih.xlsx'  # Ganti dengan nama file Anda

# Path output files
OUTPUT_EXCEL = 'forecast_results_lightgbm_optuna.xlsx'
OUTPUT_PLOT_TOPICS = 'forecast_topics_lgbm.png'
OUTPUT_PLOT_WORDS = 'forecast_words_lgbm.png'
OUTPUT_PLOT_HASHTAGS = 'forecast_hashtags_lgbm.png'

# Hyperparameter Optuna
N_TRIALS = 50  # Jumlah trial untuk tuning
TEST_SIZE_RATIO = 0.2  # Proporsi data test

# =============================================================================
# STOPWORDS BAHASA INDONESIA
# =============================================================================

INDONESIAN_STOPWORDS = [
    # Kata sambung
    'dan', 'atau', 'tetapi', 'namun', 'sedangkan', 'padahal', 'serta',
    'maupun', 'melainkan', 'bahwa', 'karena', 'sebab', 'jika', 'kalau', 'ini',
    # Kata depan
    'di', 'ke', 'dari', 'pada', 'untuk', 'dengan', 'dalam', 'oleh',
    'terhadap', 'atas', 'antara', 'kepada', 'bagi', 'tentang',
    # Kata ganti
    'saya', 'aku', 'kamu', 'anda', 'dia', 'ia', 'mereka', 'kami', 'kita',
    'nya', 'ku', 'mu', 'kak',
    # Kata kerja bantu
    'adalah', 'ialah', 'merupakan', 'yaitu', 'yakni', 'akan', 'telah',
    'sudah', 'sedang', 'dapat', 'bisa', 'boleh', 'harus', 'perlu',
    # Kata keterangan
    'sangat', 'lebih', 'paling', 'terlalu', 'cukup', 'agak', 'sedikit',
    'banyak', 'semua', 'setiap', 'seluruh', 'beberapa', 'berbagai',
    # Kata tanya
    'apa', 'siapa', 'kapan', 'dimana', 'kemana', 'darimana', 'mengapa',
    'kenapa', 'bagaimana', 'berapa',
    # Kata penunjuk
    'ini', 'itu', 'tersebut', 'begitu', 'begini',
    # Partikel
    'lah', 'kah', 'pun', 'per', 'tah',
    # Kata umum lainnya
    'yang', 'juga', 'ada', 'tidak', 'ya', 'bukan', 'belum', 'masih',
    'hanya', 'saja', 'lagi', 'pula', 'jadi', 'maka', 'bila', 'walau',
    'meski', 'walaupun', 'meskipun', 'hingga', 'sampai', 'supaya',
    # Twitter specific
    'rt', 'via', 'follow', 'followers', 'following', 'retweet', 'reply'
]

# =============================================================================
# FUNGSI HELPER
# =============================================================================

def clean_tweets(df, column='final_tweet'):
    """Membersihkan teks tweet"""
    keywords = r'\b(promo|diskon|sale|flash\s*sale|amp)\b'
    
    df[column] = (
        df[column]
        .astype(str)
        .str.lower()
        .str.replace(keywords, '', regex=True)
        .str.replace(r'\b\w{1,2}\b', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return df

def get_topic_name(topic_id, topic_model):
    """Generate nama topik berdasarkan top keywords"""
    if topic_id == -1:
        return "Outlier"
    
    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        top_words = [word for word, _ in topic_words[:3]]
        return f"Topic_{topic_id}: {', '.join(top_words)}"
    else:
        return f"Topic_{topic_id}"

def create_time_features(df):
    """Membuat fitur temporal untuk modeling"""
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Lag features
    for lag in [1, 7, 14]:
        df[f'lag_{lag}'] = df['count'].shift(lag)
    
    # Rolling features
    for window in [7, 14]:
        df[f'rolling_mean_{window}'] = df['count'].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['count'].shift(1).rolling(window=window).std()
    
    return df.fillna(0)

def calculate_rmsse(y_true, y_pred, y_train):
    """Calculate Root Mean Squared Scaled Error"""
    n = len(y_train)
    denominator = np.sum(np.square(np.diff(y_train))) / (n - 1)
    numerator = np.mean(np.square(y_true - y_pred))
    
    if denominator == 0:
        return np.inf
    
    return np.sqrt(numerator / denominator)

# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def perform_topic_modeling(df):
    """Melakukan topic modeling dengan BERTopic"""
    print("\n" + "="*80)
    print("TOPIC MODELING DENGAN BERTOPIC")
    print("="*80)
    
    tweets = df['final_tweet'].tolist()
    print(f"\n[1/5] Loading IndoBERTweet model...")
    embedding_model = SentenceTransformer("indolem/indobertweet-base-uncased")
    print("✓ Model loaded")
    
    print(f"\n[2/5] Creating embeddings...")
    embeddings = embedding_model.encode(tweets, show_progress_bar=True)
    print(f"✓ Embeddings created: {embeddings.shape}")
    
    print(f"\n[3/5] Setting up BERTopic...")
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=INDONESIAN_STOPWORDS,
        min_df=2,
        max_df=0.95
    )
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=5,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True
    )
    print("✓ BERTopic configured")
    
    print(f"\n[4/5] Fitting model...")
    topics, probs = topic_model.fit_transform(tweets, embeddings)
    print(f"✓ Model fitted")
    print(f"✓ Total topics found: {len(set(topics)) - 1}")
    
    print(f"\n[5/5] Adding results to dataframe...")
    topic_names = [get_topic_name(topic_id, topic_model) for topic_id in topics]
    df['topic_id'] = topics
    df['topic_name'] = topic_names
    
    topic_probabilities = []
    for i, topic_id in enumerate(topics):
        if topic_id == -1:
            topic_probabilities.append(0.0)
        else:
            topic_probabilities.append(probs[i][topic_id])
    
    df['topic_probability'] = topic_probabilities
    print("✓ Topic modeling completed")
    
    return df, topic_model

def aggregate_daily_data(df):
    """Aggregasi data per hari untuk topik, kata, dan hashtag"""
    print("\n" + "="*80)
    print("AGREGASI DATA HARIAN")
    print("="*80)
    
    # Pastikan kolom tanggal dalam format datetime
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    
    # 1. Agregasi topik per hari
    print("\n[1/3] Aggregating topics...")
    topic_daily = (
        df[df['topic_id'] != -1]
        .groupby([pd.Grouper(key='tanggal', freq='D'), 'topic_name'])
        .size()
        .reset_index(name='count')
    )
    print(f"✓ Topic aggregation done: {len(topic_daily)} records")
    
    # 2. Agregasi kata per hari
    print("\n[2/3] Aggregating words...")
    all_words = []
    for idx, row in df.iterrows():
        date = row['tanggal']
        words = str(row['final_tweet']).split()
        for word in words:
            if len(word) > 3:  # Filter kata pendek
                all_words.append({'tanggal': date, 'word': word})
    
    words_df = pd.DataFrame(all_words)
    word_daily = (
        words_df.groupby([pd.Grouper(key='tanggal', freq='D'), 'word'])
        .size()
        .reset_index(name='count')
    )
    print(f"✓ Word aggregation done: {len(word_daily)} records")
    
    # 3. Agregasi hashtag per hari
    print("\n[3/3] Aggregating hashtags...")
    all_hashtags = []
    for idx, row in df.iterrows():
        date = row['tanggal']
        hashtags = re.findall(r'##\w+', str(row.get('hashtags_list', '')))
        for hashtag in hashtags:
            all_hashtags.append({'tanggal': date, 'hashtag': hashtag})
    
    hashtag_daily = pd.DataFrame(all_hashtags)
    if not hashtag_daily.empty:
        hashtag_daily = (
            hashtag_daily.groupby([pd.Grouper(key='tanggal', freq='D'), 'hashtag'])
            .size()
            .reset_index(name='count')
        )
    print(f"✓ Hashtag aggregation done: {len(hashtag_daily)} records")
    
    return topic_daily, word_daily, hashtag_daily

def forecast_with_lightgbm_optuna(ts_data, item_name, n_trials=50):
    """Forecasting dengan LightGBM + Optuna hyperparameter tuning"""
    
    if len(ts_data) < 20:
        print(f"    ⚠ Warning: Insufficient data ({len(ts_data)} points), skipping...")
        return None
    
    # Prepare data
    ts_data = ts_data.set_index('tanggal')
    ts_data = create_time_features(ts_data)
    
    # Split train/test
    test_size = max(int(len(ts_data) * TEST_SIZE_RATIO), 1)
    train = ts_data[:-test_size]
    test = ts_data[-test_size:]
    
    if len(train) < 10:
        print(f"    ⚠ Warning: Insufficient training data ({len(train)} points), skipping...")
        return None
    
    print(f"    Total data points: {len(ts_data)}")
    print(f"    Train size: {len(train)}, Test size: {len(test)}")
    
    X_train = train.drop(['count'], axis=1)
    y_train = train['count']
    X_test = test.drop(['count'], axis=1)
    y_test = test['count']
    
    # Optuna optimization
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }
        
        model = lgb.LGBMRegressor(**params, n_estimators=100, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                 callbacks=[lgb.early_stopping(10, verbose=False)])
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse
    
    print(f"    Running Optuna optimization ({n_trials} trials)...")
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1
    })
    
    final_model = lgb.LGBMRegressor(**best_params, n_estimators=100, random_state=42)
    final_model.fit(X_train, y_train)
    
    # Predict
    y_pred_test = final_model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    rmsse = calculate_rmsse(y_test, y_pred_test, y_train)
    
    print(f"    ✓ Best RMSE: {rmse:.4f}")
    
    return {
        'item': item_name,
        'rmse': rmse,
        'mae': mae,
        'rmsse': rmsse,
        'test_actual': y_test.tolist(),
        'test_pred': y_pred_test.tolist(),
        'test_dates': test.index.tolist()
    }

def visualize_forecasts(results_df, output_file, title):
    """Visualisasi hasil forecasting"""
    top_items = results_df.nsmallest(5, 'rmse')
    
    fig, axes = plt.subplots(len(top_items), 1, figsize=(12, 4*len(top_items)))
    
    if len(top_items) == 1:
        axes = [axes]
    
    for idx, (_, row) in enumerate(top_items.iterrows()):
        ax = axes[idx]
        dates = pd.to_datetime(row['test_dates'])
        ax.plot(dates, row['test_actual'], label='Actual', marker='o')
        ax.plot(dates, row['test_pred'], label='Predicted', marker='s')
        ax.set_title(f"{row['item']} (RMSE: {row['rmse']:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")

def main():
    """Fungsi utama"""
    print("="*80)
    print("TIME SERIES FORECASTING DENGAN LIGHTGBM + OPTUNA")
    print("="*80)
    
    # 1. Load data
    print(f"\n[1] LOADING DATA")
    print("="*80)
    try:
        df = pd.read_excel(INPUT_FILE)
        print(f"✅ Data loaded: {len(df)} rows")
    except FileNotFoundError:
        print(f"❌ File {INPUT_FILE} not found!")
        return
    
    # 2. Clean data
    print(f"\n[2] CLEANING DATA")
    print("="*80)
    df = clean_tweets(df)
    print("✅ Data cleaned")
    
    # 3. Topic modeling
    df, topic_model = perform_topic_modeling(df)
    
    # 4. Aggregate data
    topic_daily, word_daily, hashtag_daily = aggregate_daily_data(df)
    
    # 5. Forecasting Topics
    print("\n" + "="*80)
    print("[3] FORECASTING FREKUENSI TOPIK")
    print("="*80)
    
    topics = topic_daily['topic_name'].unique()
    topic_results = []
    
    for topic in sorted(topics)[:10]:  # Top 10 topics
        print(f"\n  → Forecasting: {topic}")
        ts = topic_daily[topic_daily['topic_name'] == topic][['tanggal', 'count']]
        result = forecast_with_lightgbm_optuna(ts, topic, n_trials=N_TRIALS)
        if result:
            topic_results.append(result)
    
    # 6. Forecasting Words
    print("\n" + "="*80)
    print("[4] FORECASTING FREKUENSI KATA")
    print("="*80)
    
    top_words = word_daily.groupby('word')['count'].sum().nlargest(10).index
    word_results = []
    
    for word in top_words:
        print(f"\n  → Forecasting: {word}")
        ts = word_daily[word_daily['word'] == word][['tanggal', 'count']]
        result = forecast_with_lightgbm_optuna(ts, word, n_trials=N_TRIALS)
        if result:
            word_results.append(result)
    
    # 7. Forecasting Hashtags
    print("\n" + "="*80)
    print("[5] FORECASTING FREKUENSI HASHTAG")
    print("="*80)
    
    if not hashtag_daily.empty:
        top_hashtags = hashtag_daily.groupby('hashtag')['count'].sum().nlargest(10).index
        hashtag_results = []
        
        for hashtag in top_hashtags:
            print(f"\n  → Forecasting: {hashtag}")
            ts = hashtag_daily[hashtag_daily['hashtag'] == hashtag][['tanggal', 'count']]
            result = forecast_with_lightgbm_optuna(ts, hashtag, n_trials=N_TRIALS)
            if result:
                hashtag_results.append(result)
    else:
        hashtag_results = []
    
    # 8. Save results
    print("\n" + "="*80)
    print("[6] EXPORT HASIL")
    print("="*80)
    
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        if topic_results:
            pd.DataFrame(topic_results).to_excel(writer, sheet_name='Topics', index=False)
        if word_results:
            pd.DataFrame(word_results).to_excel(writer, sheet_name='Words', index=False)
        if hashtag_results:
            pd.DataFrame(hashtag_results).to_excel(writer, sheet_name='Hashtags', index=False)
    
    print(f"✓ Results saved to: {OUTPUT_EXCEL}")
    
    # 9. Visualize
    print("\n" + "="*80)
    print("[7] VISUALISASI")
    print("="*80)
    
    if topic_results:
        visualize_forecasts(pd.DataFrame(topic_results), OUTPUT_PLOT_TOPICS, "Topic Forecasts")
    if word_results:
        visualize_forecasts(pd.DataFrame(word_results), OUTPUT_PLOT_WORDS, "Word Forecasts")
    if hashtag_results:
        visualize_forecasts(pd.DataFrame(hashtag_results), OUTPUT_PLOT_HASHTAGS, "Hashtag Forecasts")
    
    print("\n" + "="*80)
    print("FORECASTING SELESAI!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  1. {OUTPUT_EXCEL}")
    print(f"  2. {OUTPUT_PLOT_TOPICS}")
    print(f"  3. {OUTPUT_PLOT_WORDS}")
    print(f"  4. {OUTPUT_PLOT_HASHTAGS}")

if __name__ == "__main__":
    main()
