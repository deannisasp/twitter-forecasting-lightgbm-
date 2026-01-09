"""
merge_data.py
Script untuk menggabungkan multiple file CSV/XLSX menjadi satu file

Author: Converted from Jupyter Notebook
Usage: python merge_data.py
"""

import os
import pandas as pd

def main():
    """
    Fungsi utama untuk menggabungkan file CSV dan XLSX
    """
    # Path folder yang berisi file-file yang ingin digabungkan
    # Sesuaikan path ini dengan lokasi folder data Anda
    folder_path = './data_new'  # Ganti dengan path folder Anda
    
    # Cek apakah folder ada
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} tidak ditemukan!")
        print(f"💡 Silakan buat folder atau ubah path di dalam script")
        return
    
    # Ambil semua file csv dan xlsx
    files = [
        file for file in os.listdir(folder_path)
        if file.endswith('.csv') or file.endswith('.xlsx')
    ]
    
    if not files:
        print(f"❌ Tidak ada file CSV atau XLSX di folder {folder_path}")
        return
    
    print(f"📁 Menemukan {len(files)} file:")
    for file in files:
        print(f"   - {file}")
    
    # List untuk menampung DataFrame
    df_list = []
    
    # Baca setiap file sesuai ekstensi
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        try:
            if file.endswith('.csv'):
                temp_df = pd.read_csv(file_path)
                print(f"✅ Berhasil membaca {file} ({len(temp_df)} baris)")
            elif file.endswith('.xlsx'):
                temp_df = pd.read_excel(file_path)
                print(f"✅ Berhasil membaca {file} ({len(temp_df)} baris)")
            
            df_list.append(temp_df)
            
        except Exception as e:
            print(f"❌ Gagal membaca {file}: {e}")
    
    if not df_list:
        print("❌ Tidak ada data yang berhasil dibaca!")
        return
    
    # Gabungkan semua DataFrame
    df = pd.concat(df_list, ignore_index=True)
    
    print(f"\n✅ Berhasil menggabungkan {len(df_list)} file")
    print(f"📊 Total baris: {len(df)}")
    print(f"📊 Total kolom: {len(df.columns)}")
    print(f"\nKolom yang tersedia: {list(df.columns)}")
    
    # Menyimpan DataFrame ke file Excel
    output_file = 'data(fix).xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n💾 File berhasil disimpan sebagai: {output_file}")
    
    # Tampilkan preview data
    print(f"\n📋 Preview 5 baris pertama:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = main()
