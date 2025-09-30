# %% [markdown]
# ## Cell 1: Setup Proyek dan Impor Library

# %%
# ==============================================================================
# SEL 1: SETUP DAN IMPOR LIBRARY
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from PIL import Image
from IPython.display import display

# Mengatur style visualisasi agar lebih menarik
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
print("✅ Library berhasil diimpor.")

# %% [markdown]
# ## Cell 2: Memuat Dataset Mentah

# %%
# ==============================================================================
# SEL 2: MEMUAT DATASET MENTAH
# ==============================================================================
# Tentukan path utama ke folder dataset
BASE_PATH = "../dataset/"

# Definisikan path untuk setiap file CSV dan folder gambar
train_truth_path = os.path.join(BASE_PATH, "MILK10k_Training_GroundTruth.csv")
test_meta_path = os.path.join(BASE_PATH, "MILK10k_Test_Metadata.csv")
train_img_path = os.path.join(BASE_PATH, "MILK10k_Training_Input")
test_img_path = os.path.join(BASE_PATH, "MILK10k_Test_Input")

# Memuat data CSV
try:
    df_train_truth = pd.read_csv(train_truth_path)
    df_test_meta = pd.read_csv(test_meta_path)
    print("✅ Data mentah berhasil dimuat.")

    # PENTING: Memastikan tipe data kolom diagnosis adalah angka (float)
    # untuk menghindari error perhitungan di sel berikutnya.
    diagnosis_columns = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
    df_train_truth[diagnosis_columns] = df_train_truth[diagnosis_columns].astype(float)
    print("✅ Tipe data kolom diagnosis telah dipastikan numerik.")
    
except FileNotFoundError as e:
    print(f"❌ ERROR: File tidak ditemukan. Pastikan file CSV ada di folder 'dataset'.")
    print(f"   Detail: {e}")

# %% [markdown]
# ## Cell 3: Analisis dan Verifikasi Data Awal

# %%
# ==============================================================================
# SEL 3: ANALISIS DAN VERIFIKASI DATA AWAL
# ==============================================================================
print("--- Info Data Ground Truth Training ---")
display(df_train_truth.head())
print("\n--- Info Data Metadata Test ---")
display(df_test_meta.head())

# Verifikasi Rasio Pembagian Dataset
num_train_lesions = df_train_truth['lesion_id'].nunique()
num_test_lesions = df_test_meta['lesion_id'].nunique()
total_lesions = num_train_lesions + num_test_lesions
train_ratio = (num_train_lesions / total_lesions) * 100
test_ratio = (num_test_lesions / total_lesions) * 100

print("\n--- Analisis Rasio Pembagian Dataset ---")
print(f"Total Lesi Unik      : {total_lesions}")
print(f"Jumlah Lesi Training   : {num_train_lesions} ({train_ratio:.2f}%)")
print(f"Jumlah Lesi Testing    : {num_test_lesions} ({test_ratio:.2f}%)")

# %% [markdown]
# ## Cell 4: Transformasi Data (Menggabungkan Path Gambar)

# %%
# ==============================================================================
# SEL 4: TRANSFORMASI DATA (MENGGABUNGKAN PATH GAMBAR)
# ==============================================================================

# --- 4.1 Transformasi Data Training ---
print("Memproses data training...")
# Membuat daftar semua path gambar di folder training
all_train_images = glob.glob(os.path.join(train_img_path, '*', '*.jpg'))
path_data = []
for path in all_train_images:
    parts = path.replace('\\', '/').split('/')
    lesion_id, filename = parts[-2], parts[-1]
    path_data.append({'lesion_id': lesion_id, 'image_path': path})

df_train_paths = pd.DataFrame(path_data)

# Mengelompokkan berdasarkan lesion_id dan membuat list berisi 2 path gambar
df_train_pivot = df_train_paths.groupby('lesion_id')['image_path'].apply(list).reset_index()

# Menggabungkan dengan tabel ground truth
df_train_processed = pd.merge(df_train_truth, df_train_pivot, on='lesion_id')

# Memisahkan list path menjadi dua kolom terpisah
# Ini adalah langkah kunci untuk menangani 2 gambar per lesi
try:
    df_train_processed[['image_path_1', 'image_path_2']] = pd.DataFrame(df_train_processed['image_path'].tolist(), index=df_train_processed.index)
    df_train_processed = df_train_processed.drop(columns=['image_path'])
    print("✅ Transformasi data training selesai.")
except ValueError:
    print("⚠️ Peringatan: Beberapa lesi training tidak memiliki pasangan gambar yang lengkap.")


# --- 4.2 Transformasi Data Test ---
print("\nMemproses data test...")
df_test_meta['image_path'] = df_test_meta.apply(lambda r: os.path.join(test_img_path, r['lesion_id'], r['isic_id'] + '.jpg'), axis=1)
df_test_clinical = df_test_meta[df_test_meta['image_type'] == 'clinical: close-up'].rename(columns={'image_path': 'clinical_path', 'isic_id': 'clinical_isic_id'})
df_test_dermoscopic = df_test_meta[df_test_meta['image_type'] == 'dermoscopic'].rename(columns={'image_path': 'dermoscopic_path', 'isic_id': 'dermoscopic_isic_id'})

# Menggabungkan kembali menjadi satu baris per lesi
df_test_processed = pd.merge(
    df_test_clinical.drop(columns=['image_type']),
    df_test_dermoscopic[['lesion_id', 'dermoscopic_path', 'dermoscopic_isic_id']],
    on='lesion_id'
)
print("✅ Transformasi data test selesai.")

# --- Tampilkan Hasil Transformasi ---
print("\n--- Contoh Tabel Training SETELAH Transformasi ---")
display(df_train_processed.head())

print("\n--- Contoh Tabel Test SETELAH Transformasi ---")
display(df_test_processed.head())

# %% [markdown]
# ## Cell 5: Visualisasi Distribusi dan Perbandingan

# %%
# ==============================================================================
# SEL 5: VISUALISASI DISTRIBUSI DAN PERBANDINGAN
# ==============================================================================

# === PERBAIKAN: Definisikan variabel yang dibutuhkan di awal sel ===
# Daftar kolom diagnosis
diagnosis_columns = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']

# Pemetaan dari singkatan ke nama lengkap untuk label plot yang lebih jelas
diagnosis_mapping = {
    'AKIEC': 'Actinic Keratosis / IEC',
    'BCC': 'Basal Cell Carcinoma',
    'BEN_OTH': 'Other Benign Proliferations',
    'BKL': 'Benign Keratinocytic Lesion',
    'DF': 'Dermatofibroma',
    'INF': 'Inflammatory & Infectious',
    'MAL_OTH': 'Other Malignant Proliferations',
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus',
    'SCCKA': 'Squamous Cell Carcinoma / KA',
    'VASC': 'Vascular Lesions'
}
# ======================================================================

# --- 5.1 Visualisasi Distribusi Diagnosis (Gambar 2) ---
# Kode ini sekarang akan berjalan karena 'diagnosis_mapping' sudah ada
diagnosis_counts_mapped = df_train_truth[diagnosis_columns].sum().rename(index=diagnosis_mapping).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=diagnosis_counts_mapped.values, y=diagnosis_counts_mapped.index, palette="viridis", orient='h')
plt.title('Gambar 2: Distribusi Kelas Diagnosis pada Data Training', fontsize=20, pad=20)
plt.xlabel('Jumlah Lesi', fontsize=15); plt.ylabel('Kategori Diagnosis', fontsize=15)
plt.show()

# --- 5.2 Visualisasi Perbandingan Gambar (Gambar 3) ---
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized)
        normalized_array = img_array / 255.0
        return img, normalized_array
    except (FileNotFoundError, AttributeError):
        return None, None

# Ambil satu contoh gambar dari DataFrame yang sudah diproses
# Pastikan df_train_processed sudah ada dari sel sebelumnya
if 'df_train_processed' in globals() and not df_train_processed.empty:
    example_path = df_train_processed.iloc[5]['image_path_1'] 
    original_image, processed_image = preprocess_image(example_path)

    if original_image and (processed_image is not None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_image)
        axes[0].set_title(f"SEBELUM\nUkuran: {original_image.size}\nPiksel: 0-255", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(processed_image)
        axes[1].set_title(f"SESUDAH\nUkuran: {processed_image.shape[:2]}\nPiksel: {processed_image.min():.1f}-{processed_image.max():.1f}", fontsize=14)
        axes[1].axis('off')
        
        plt.suptitle("Gambar 3: Perbandingan Citra Sebelum & Sesudah Pra-pemrosesan", fontsize=20, y=1.02)
        plt.show()
else:
    print("⚠️ DataFrame 'df_train_processed' belum dibuat. Jalankan sel sebelumnya terlebih dahulu.")

# %% [markdown]
# ## Cell 6: Menyimpan Hasil Akhir
# 

# %%
# ==============================================================================
# SEL 6: MENYIMPAN HASIL AKHIR (KE FOLDER TERPISAH)
# ==============================================================================
# Pada versi ini, kita akan menyimpan file yang sudah diproses ke folder terpisah
# bernama 'processed_data' agar tidak tercampur dengan data mentah di 'dataset'.

# === PERBAIKAN: Tentukan folder output baru ===
# Path '../processed_data/' berarti "naik satu level dari folder 'notebook',
# lalu masuk ke folder 'processed_data'".
OUTPUT_PATH = "../dataset/processed_data/"

# Perintah ini akan membuat folder 'processed_data' secara otomatis jika belum ada.
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"Folder output '{os.path.abspath(OUTPUT_PATH)}' telah disiapkan.")
# ============================================

# Tentukan path lengkap untuk file output di dalam folder baru
train_output_path = os.path.join(OUTPUT_PATH, 'train_processed.csv')
test_output_path = os.path.join(OUTPUT_PATH, 'test_processed.csv')

# Menyimpan DataFrame ke lokasi baru
df_train_processed.to_csv(train_output_path, index=False)
df_test_processed.to_csv(test_output_path, index=False)

print(f"\n✅ DataFrame yang telah diproses berhasil disimpan di folder baru:")
print(f"   - {train_output_path}")
print(f"   - {test_output_path}")


