#!/usr/bin/env python3
"""
twitter_crawler.py
Script untuk crawling data Twitter menggunakan tweet-harvest

Note: Script ini memerlukan Node.js dan NPM untuk menjalankan tweet-harvest.
Untuk versi Python ini, kita menggunakan subprocess untuk menjalankan tool tersebut.

Author: Converted from Jupyter Notebook
Usage: python twitter_crawler.py

Requirements:
- Node.js dan NPM harus terinstall
- Twitter auth token (dapatkan dari browser developer tools)

Cara mendapatkan Twitter Auth Token:
1. Login ke Twitter/X di browser
2. Buka Developer Tools (F12)
3. Buka tab Network
4. Reload halaman
5. Cari request yang mengandung 'auth_token' di cookies
6. Copy nilai auth_token tersebut
"""

import subprocess
import os
import sys

# =============================================================================
# KONFIGURASI
# =============================================================================

# IMPORTANT: Ganti dengan auth token Anda sendiri!
# Auth token bisa didapatkan dari browser cookies saat login ke Twitter
TWITTER_AUTH_TOKEN = '33557dd01c7df6d88d9ccac806cd67fcd05cb799'

# =============================================================================
# DAFTAR AKUN DAN KONFIGURASI CRAWLING
# =============================================================================

# Daftar akun yang akan di-crawl dengan konfigurasi masing-masing
CRAWL_CONFIGS = [
    {
        'name': 'alfamart',
        'filename': 'data_alfamart.csv',
        'keyword': 'from:alfamart promo OR sale OR diskon OR flash sale since:2024-11-27 until:2025-11-27 lang:id',
        'limit': 1000
    },
    {
        'name': 'indomaret',
        'filename': 'data_indomaret.csv',
        'keyword': 'from:indomaret promo OR sale OR diskon OR flash sale since:2024-11-27 until:2025-11-27 lang:id',
        'limit': 1000
    },
    {
        'name': 'shopee',
        'filename': 'data_shopee.csv',
        'keyword': 'from:shopeeid promo OR sale OR diskon OR flash sale since:2024-11-27 until:2025-11-27 lang:id',
        'limit': 1000
    },
    {
        'name': 'tokopedia',
        'filename': 'data_tokopedia.csv',
        'keyword': 'from:tokopedia promo OR sale OR diskon OR flash sale since:2024-11-27 until:2025-11-27 lang:id',
        'limit': 1000
    },
    {
        'name': 'lazada',
        'filename': 'data_lazada.csv',
        'keyword': 'from:lazadaid promo OR sale OR diskon OR flash sale since:2024-11-27 until:2025-11-27 lang:id',
        'limit': 1000
    },
    # Tambahkan akun lain di sini sesuai kebutuhan
]

# =============================================================================
# FUNGSI HELPER
# =============================================================================

def check_nodejs_installed():
    """Cek apakah Node.js sudah terinstall"""
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        version = result.stdout.strip()
        print(f"✅ Node.js terdeteksi: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js tidak terdeteksi!")
        print("💡 Install Node.js terlebih dahulu dari https://nodejs.org/")
        return False

def check_npm_installed():
    """Cek apakah NPM sudah terinstall"""
    try:
        result = subprocess.run(['npm', '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        version = result.stdout.strip()
        print(f"✅ NPM terdeteksi: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ NPM tidak terdeteksi!")
        print("💡 NPM biasanya terinstall bersama Node.js")
        return False

def crawl_tweets(config, auth_token):
    """
    Crawl tweets menggunakan tweet-harvest tool
    
    Args:
        config: Dictionary berisi konfigurasi crawling
        auth_token: Twitter authentication token
    """
    filename = config['filename']
    keyword = config['keyword']
    limit = config['limit']
    name = config['name']
    
    print(f"\n{'='*80}")
    print(f"Crawling: {name}")
    print(f"{'='*80}")
    print(f"Keyword: {keyword}")
    print(f"Limit: {limit} tweets")
    print(f"Output: {filename}")
    print()
    
    # Construct command untuk menjalankan tweet-harvest via npx
    cmd = [
        'npx',
        '-y',
        'tweet-harvest@2.6.1',
        '-o', filename,
        '-s', keyword,
        '--tab', 'LATEST',
        '-l', str(limit),
        '--token', auth_token
    ]
    
    try:
        # Run command
        print("⏳ Memulai crawling...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Crawling selesai!")
        
        # Cek apakah file berhasil dibuat
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"📁 File tersimpan: {filename} ({file_size:,} bytes)")
        else:
            print(f"⚠️  File tidak ditemukan: {filename}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error saat crawling:")
        print(f"   Return code: {e.returncode}")
        if e.stderr:
            print(f"   Error message: {e.stderr}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def main():
    """Fungsi utama untuk menjalankan crawling"""
    
    print("="*80)
    print("TWITTER DATA CRAWLER")
    print("Menggunakan tweet-harvest via NPX")
    print("="*80)
    
    # 1. Cek requirement
    print("\n[1] Checking requirements...")
    print("-" * 80)
    
    if not check_nodejs_installed():
        return
    
    if not check_npm_installed():
        return
    
    # 2. Validasi auth token
    print("\n[2] Validating auth token...")
    print("-" * 80)
    
    if not TWITTER_AUTH_TOKEN or TWITTER_AUTH_TOKEN == '33557dd01c7df6d88d9ccac806cd67fcd05cb799':
        print("⚠️  PERINGATAN: Anda masih menggunakan auth token default!")
        print("💡 Silakan ganti TWITTER_AUTH_TOKEN dengan token Anda sendiri")
        print()
        response = input("Lanjutkan dengan token default? (y/n): ")
        if response.lower() != 'y':
            print("Dibatalkan.")
            return
    else:
        print(f"✅ Auth token: {TWITTER_AUTH_TOKEN[:20]}...")
    
    # 3. Konfirmasi crawling
    print(f"\n[3] Crawling configuration...")
    print("-" * 80)
    print(f"Total akun yang akan di-crawl: {len(CRAWL_CONFIGS)}")
    for i, config in enumerate(CRAWL_CONFIGS, 1):
        print(f"  {i}. {config['name']} (limit: {config['limit']} tweets)")
    
    print()
    response = input("Mulai crawling? (y/n): ")
    if response.lower() != 'y':
        print("Dibatalkan.")
        return
    
    # 4. Mulai crawling
    print(f"\n[4] Starting crawl process...")
    print("-" * 80)
    
    success_count = 0
    failed_count = 0
    
    for i, config in enumerate(CRAWL_CONFIGS, 1):
        print(f"\nProgress: [{i}/{len(CRAWL_CONFIGS)}]")
        
        success = crawl_tweets(config, TWITTER_AUTH_TOKEN)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        # Jeda antar crawling untuk menghindari rate limit
        if i < len(CRAWL_CONFIGS):
            print(f"\n⏸  Menunggu 5 detik sebelum crawling berikutnya...")
            import time
            time.sleep(5)
    
    # 5. Summary
    print("\n" + "="*80)
    print("CRAWLING SELESAI!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Total akun: {len(CRAWL_CONFIGS)}")
    print(f"   Berhasil: {success_count}")
    print(f"   Gagal: {failed_count}")
    
    if success_count > 0:
        print(f"\n✅ File hasil crawling:")
        for config in CRAWL_CONFIGS:
            filename = config['filename']
            if os.path.exists(filename):
                print(f"   - {filename}")
    
    print(f"\n💡 Tip: Gunakan merge_data.py untuk menggabungkan semua file CSV")

def custom_crawl():
    """Mode interaktif untuk crawling custom"""
    print("\n" + "="*80)
    print("CUSTOM CRAWLING MODE")
    print("="*80)
    
    print("\nMasukkan parameter crawling:")
    
    # Input dari user
    name = input("Nama akun/keyword: ")
    filename = input("Nama file output (contoh: data_custom.csv): ")
    
    print("\nBuat search keyword (format Twitter advanced search):")
    print("Contoh: from:username promo OR sale since:2024-01-01 until:2024-12-31 lang:id")
    keyword = input("Search keyword: ")
    
    limit = input("Limit tweets (default: 1000): ") or "1000"
    
    # Konfirmasi
    print("\nKonfigurasi crawling:")
    print(f"  Nama: {name}")
    print(f"  Output: {filename}")
    print(f"  Keyword: {keyword}")
    print(f"  Limit: {limit}")
    
    response = input("\nLanjutkan? (y/n): ")
    if response.lower() != 'y':
        print("Dibatalkan.")
        return
    
    # Crawl
    config = {
        'name': name,
        'filename': filename,
        'keyword': keyword,
        'limit': int(limit)
    }
    
    crawl_tweets(config, TWITTER_AUTH_TOKEN)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                        TWITTER DATA CRAWLER                                ║
║                    Using tweet-harvest NPX package                         ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("Mode:")
    print("  1. Batch crawling (multiple accounts)")
    print("  2. Custom crawling (single account/keyword)")
    print("  3. Exit")
    
    choice = input("\nPilih mode (1/2/3): ")
    
    if choice == "1":
        main()
    elif choice == "2":
        if not check_nodejs_installed() or not check_npm_installed():
            print("❌ Requirements tidak terpenuhi")
        else:
            custom_crawl()
    else:
        print("Keluar.")
