# backend/app/algorithm/fitness.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score # Atau metrik jarak lain
from .chromosome_setup import FEATURE_ORDER, FEATURE_DETAILS 
import random

# --- Helper Functions ---

def get_target_profile(target_genus_specie, dataset_df, 
                       numerical_feature_names, categorical_feature_names, label_col='Genus_&_Specie'):
    """
    Menghasilkan profil fitur target (rata-rata untuk numerik, modus untuk kategorikal)
    untuk Genus_&_Specie tertentu dari dataset.

    Args:
        target_genus_specie (str): Nama Genus_&_Specie target.
        dataset_df (pd.DataFrame): DataFrame Evolution_DataSets.csv yang sudah dimuat.
        numerical_feature_names (list): Daftar nama fitur numerik.
        categorical_feature_names (list): Daftar nama fitur kategorikal.
        label_col (str): Nama kolom label.

    Returns:
        dict: Dictionary berisi profil fitur target, atau None jika target tidak ditemukan.
              Format: {'NamaFitur1': nilai_target1, 'NamaFitur2': nilai_target2, ...}
    """
    target_samples = dataset_df[dataset_df[label_col] == target_genus_specie]
    if target_samples.empty:
        print(f"Peringatan: Tidak ada sampel ditemukan untuk target '{target_genus_specie}' dalam dataset.")
        return None

    profile = {}
    for feature in FEATURE_ORDER: # Menggunakan FEATURE_ORDER untuk konsistensi
        if feature in numerical_feature_names:
            # Menggunakan median agar lebih robust terhadap outlier daripada mean
            profile[feature] = target_samples[feature].median() 
        elif feature in categorical_feature_names:
            # Menggunakan modus (nilai paling sering muncul)
            try:
                profile[feature] = target_samples[feature].mode()[0] 
            except KeyError: # Jika kolom tidak ada di sampel target (seharusnya tidak terjadi jika fitur ada di FEATURE_ORDER)
                profile[feature] = None 
            except IndexError: # Jika mode kosong (misal semua NaN atau tidak ada data)
                 profile[feature] = None # Atau nilai default lain
        # Fitur yang tidak numerik atau kategorikal (seharusnya tidak ada jika FEATURE_DETAILS benar)
        # tidak akan dimasukkan ke profil, atau bisa diberi nilai default.
    
    # Membersihkan profil dari nilai None jika ada fitur kategorikal yang tidak memiliki modus
    profile_cleaned = {k: v for k, v in profile.items() if v is not None}
    if len(profile_cleaned) < len(FEATURE_ORDER):
        print(f"Peringatan: Beberapa fitur tidak dapat dihitung profilnya untuk '{target_genus_specie}'.")

    return profile_cleaned


def calculate_feature_similarity(chromosome_dict, target_profile_dict, user_input_dict=None):
    """
    Menghitung kemiripan antara individu GA (chromosome_dict) dengan 
    profil target dan (opsional) input pengguna.

    Args:
        chromosome_dict (dict): Individu GA dalam format {'NamaFitur': nilai, ...}.
        target_profile_dict (dict): Profil fitur target Genus_&_Specie.
        user_input_dict (dict, optional): Input parameter dari pengguna.

    Returns:
        float: Skor kemiripan gabungan (nilai lebih tinggi lebih baik).
               Atau tuple (similarity_to_target, similarity_to_user_input)
    """
    if not target_profile_dict: # Jika profil target tidak bisa dibuat
        return 0.0

    # Normalisasi/Scaling diperlukan sebelum menghitung jarak untuk fitur numerik
    # Kita akan melakukan perbandingan fitur per fitur
    
    num_matching_features = 0
    total_similarity_score_target = 0.0
    
    # --- Skala untuk normalisasi fitur numerik (berdasarkan rentang di FEATURE_DETAILS) ---
    # Ini bisa di-precompute
    scalers = {}
    for feature_name, details in FEATURE_DETAILS.items():
        if details['type'] == 'numerical':
            scalers[feature_name] = MinMaxScaler(feature_range=(0, 1))
            # Fit scaler pada rentang yang mungkin (min, max dari FEATURE_DETAILS)
            # Ini hanya contoh sederhana, idealnya scaling disesuaikan dengan distribusi data
            # Untuk MinMax, cukup gunakan rentang yang diketahui.
            # scalers[feature_name].fit(np.array(details['range']).reshape(-1, 1))


    # 1. Kemiripan dengan Profil Target
    common_features_target = 0
    for feature_name in FEATURE_ORDER:
        chromo_val = chromosome_dict.get(feature_name)
        target_val = target_profile_dict.get(feature_name)

        if chromo_val is None or target_val is None:
            continue # Lewati jika salah satu nilai tidak ada
        
        common_features_target += 1
        details = FEATURE_DETAILS[feature_name]

        if details['type'] == 'numerical':
            # Normalisasi nilai sebelum perbandingan (menggunakan rentang dari FEATURE_DETAILS)
            min_val, max_val = details['range']
            if max_val == min_val: # Hindari pembagian dengan nol jika rentang 0
                norm_chromo_val = 0.5 if chromo_val == min_val else 0 # Atau logika lain
                norm_target_val = 0.5 if target_val == min_val else 0
            else:
                norm_chromo_val = (chromo_val - min_val) / (max_val - min_val)
                norm_target_val = (target_val - min_val) / (max_val - min_val)
            
            # Jarak absolut ternormalisasi, kemudian dibalik jadi similarity (1 - distance)
            similarity = 1 - abs(norm_chromo_val - norm_target_val)
            total_similarity_score_target += similarity
        
        elif details['type'] == 'categorical':
            if chromo_val == target_val:
                total_similarity_score_target += 1 # Skor penuh jika sama
            else:
                total_similarity_score_target += 0 # Skor nol jika beda (bisa dibuat lebih halus)

    avg_similarity_target = total_similarity_score_target / common_features_target if common_features_target > 0 else 0.0

    # 2. Kemiripan dengan Input Pengguna (Opsional)
    avg_similarity_user = 0.0
    if user_input_dict:
        total_similarity_score_user = 0.0
        common_features_user = 0
        for feature_name in FEATURE_ORDER:
            chromo_val = chromosome_dict.get(feature_name)
            user_val = user_input_dict.get(feature_name)

            if chromo_val is None or user_val is None: # Jika pengguna tidak mengisi semua
                continue
            
            common_features_user +=1
            details = FEATURE_DETAILS[feature_name]

            if details['type'] == 'numerical':
                min_val, max_val = details['range']
                if max_val == min_val:
                    norm_chromo_val = 0.5 if chromo_val == min_val else 0
                    norm_user_val = 0.5 if user_val == min_val else 0
                else:
                    norm_chromo_val = (chromo_val - min_val) / (max_val - min_val)
                    # Nilai pengguna mungkin di luar rentang FEATURE_DETAILS, perlu di-clamp atau di-handle
                    clamped_user_val = max(min_val, min(float(user_val), max_val))
                    norm_user_val = (clamped_user_val - min_val) / (max_val - min_val)

                similarity = 1 - abs(norm_chromo_val - norm_user_val)
                total_similarity_score_user += similarity
            
            elif details['type'] == 'categorical':
                if chromo_val == user_val: # Asumsi user_input sudah divalidasi nilainya
                    total_similarity_score_user += 1
                else:
                    total_similarity_score_user += 0
        
        avg_similarity_user = total_similarity_score_user / common_features_user if common_features_user > 0 else 0.0

    # Kombinasi Skor Fitness (Contoh: bobot)
    # Sesuaikan bobot ini berdasarkan seberapa penting kesesuaian dengan target vs. input pengguna
    weight_target = 0.7 
    weight_user = 0.3 

    if user_input_dict and common_features_user > 0 :
        final_fitness = (weight_target * avg_similarity_target) + (weight_user * avg_similarity_user)
    else: # Jika tidak ada input pengguna atau tidak ada fitur yang bisa dibandingkan
        final_fitness = avg_similarity_target
        
    return final_fitness


# --- Fungsi Fitness Utama untuk GA ---
# Fungsi ini akan dipanggil oleh ga_core.py

# Global cache untuk profil target agar tidak dihitung ulang setiap evaluasi fitness
TARGET_PROFILES_CACHE = {}

def calculate_combined_fitness(chromosome_list, # Ini adalah list nilai dari GA
                               target_genus_specie, # String, misal "Homo sapiens"
                               dataset_df, # DataFrame Evolution_DataSets.csv
                               user_input_dict, # Dict input dari pengguna (sudah diproses oleh chromosome_setup.user_input_to_chromosome)
                               numerical_cols_original, # Daftar nama kolom numerik asli
                               categorical_cols_original, # Daftar nama kolom kategorikal asli
                               label_col_in_dataset='Genus_&_Specie',
                               fitness_weights=None): # Bobot untuk berbagai komponen fitness
    """
    Fungsi fitness utama yang dipanggil oleh GA.
    Menggabungkan berbagai aspek untuk menilai seberapa "baik" sebuah kromosom.
    """
    
    # 1. Konversi chromosome_list (dari GA) ke dictionary agar mudah diakses
    chromosome_dict = {feature: chromosome_list[i] for i, feature in enumerate(FEATURE_ORDER)}

    # 2. Dapatkan Profil Fitur Target dari dataset (atau dari cache)
    if target_genus_specie not in TARGET_PROFILES_CACHE:
        profile = get_target_profile(
            target_genus_specie, dataset_df,
            numerical_cols_original, categorical_cols_original, label_col_in_dataset
        )
        if profile is None: # Target tidak ditemukan atau profil tidak bisa dibuat
            return 0.0 # Fitness sangat rendah
        TARGET_PROFILES_CACHE[target_genus_specie] = profile
    
    target_profile = TARGET_PROFILES_CACHE[target_genus_specie]

    # 3. Hitung Skor Kemiripan
    # user_input_dict di sini adalah parameter yang diberikan pengguna di awal,
    # yang mungkin sudah dikonversi ke format yang sama dengan kromosom oleh chromosome_setup.
    fitness_score = calculate_feature_similarity(chromosome_dict, target_profile, user_input_dict)
    
    # Di sini Anda bisa menambahkan komponen fitness lain jika diperlukan:
    # - Model klasifikasi (probabilitas individu GA diklasifikasikan sebagai target_genus_specie)
    # - Penalti untuk nilai fitur yang tidak realistis (jika mutasi menghasilkan sesuatu di luar domain)
    # - Reward untuk "jalur evolusi" yang masuk akal (lebih lanjut)

    return fitness_score

# --- Contoh Penggunaan (untuk testing) ---
if __name__ == '__main__':
    # Buat DataFrame dummy untuk dataset dan input pengguna
    # Ini harus diganti dengan pemuatan Evolution_DataSets.csv yang sebenarnya
    
    # A. Setup fitur (seperti di chromosome_setup.py)
    # FEATURE_ORDER dan FEATURE_DETAILS diasumsikan sudah didefinisikan
    
    # B. Buat dummy dataset_df (Evolution_DataSets.csv)
    # Anda perlu memuat dataset yang asli di sini.
    # Kolom-kolomnya harus sesuai dengan FEATURE_ORDER + kolom label
    data_dict_dummy = {
        'Genus_&_Specie': ['Homo habilis', 'Homo habilis', 'Homo erectus', 'Homo erectus', 'Homo sapiens', 'Homo sapiens'],
        'Time': [2.0, 1.8, 1.5, 1.0, 0.1, 0.05],
        'Location': ['Africa', 'Africa', 'Asia', 'Africa', 'Africa', 'Europe'],
        'Zone': ['Savanna', 'Savanna', 'Temperate Forest', 'Savanna', 'Grassland', 'Temperate Forest'],
        'Cranial_Capacity': [650, 680, 900, 1000, 1350, 1400],
        'Height': [130, 135, 160, 165, 170, 175],
        'Jaw_Shape': ['Parabolic', 'Parabolic', 'Parabolic', 'Robust Parabolic', 'Gracile Parabolic', 'Gracile Parabolic']
        # ... tambahkan fitur lain dari FEATURE_ORDER ...
    }
    # Untuk testing, pastikan semua fitur di FEATURE_ORDER ada di data_dict_dummy
    for f_name in FEATURE_ORDER:
        if f_name not in data_dict_dummy:
            details = FEATURE_DETAILS[f_name]
            if details['type'] == 'numerical':
                data_dict_dummy[f_name] = [random.uniform(details['range'][0], details['range'][1]) for _ in range(6)]
            else:
                data_dict_dummy[f_name] = [random.choice(details['categories']) for _ in range(6)]

    dummy_evolution_df = pd.DataFrame(data_dict_dummy)
    print("--- Dummy Evolution Dataset ---")
    print(dummy_evolution_df.head())

    # Identifikasi kolom numerik dan kategorikal dari FEATURE_ORDER dan FEATURE_DETAILS
    test_numerical_cols = [f for f in FEATURE_ORDER if FEATURE_DETAILS[f]['type'] == 'numerical']
    test_categorical_cols = [f for f in FEATURE_ORDER if FEATURE_DETAILS[f]['type'] == 'categorical']

    # C. Buat dummy chromosome_list (output dari GA) dan user_input_dict
    # Ini harusnya berasal dari chromosome_setup.initialize_chromosome() atau user_input_to_chromosome()
    from chromosome_setup import initialize_chromosome, user_input_to_chromosome 
    
    test_chromosome_list = initialize_chromosome() # Individu GA yang akan dievaluasi
    
    sample_user_input_params = { # Input dari pengguna
        'Time': 0.5, 'Location': 'Europe', 'Cranial_Capacity': 1300, 'Height': 170,
        # ... tambahkan fitur lain atau biarkan kosong untuk diisi acak oleh user_input_to_chromosome
    }
    # Konversi input pengguna ke format dictionary yang lengkap (jika ada fitur yang tidak diisi pengguna)
    # Fungsi user_input_to_chromosome dari chromosome_setup.py mengembalikan list, kita butuh dictnya
    # Mari kita buat versi dict dari input pengguna yang sudah diproses (diisi jika ada yang kosong)
    processed_user_input_list = user_input_to_chromosome(sample_user_input_params)
    processed_user_input_dict = {feature: processed_user_input_list[i] for i, feature in enumerate(FEATURE_ORDER)}


    print("\n--- Testing calculate_combined_fitness ---")
    target_species_to_test = 'Homo sapiens'
    
    fitness = calculate_combined_fitness(
        chromosome_list=test_chromosome_list,
        target_genus_specie=target_species_to_test,
        dataset_df=dummy_evolution_df,
        user_input_dict=processed_user_input_dict, # Ini adalah parameter awal dari pengguna
        numerical_cols_original=test_numerical_cols,
        categorical_cols_original=test_categorical_cols,
        label_col_in_dataset='Genus_&_Specie'
    )
    print(f"\nFitness untuk kromosom random vs target '{target_species_to_test}' (dengan input pengguna): {fitness:.4f}")

    # Tes jika target tidak ada
    target_species_to_test_nonexist = 'Alienus Minimus'
    fitness_nonexist = calculate_combined_fitness(
        chromosome_list=test_chromosome_list,
        target_genus_specie=target_species_to_test_nonexist,
        dataset_df=dummy_evolution_df,
        user_input_dict=processed_user_input_dict,
        numerical_cols_original=test_numerical_cols,
        categorical_cols_original=test_categorical_cols
    )
    print(f"Fitness untuk target tidak ada '{target_species_to_test_nonexist}': {fitness_nonexist:.4f}")
    
    # Kosongkan cache untuk pengujian berikutnya jika perlu
    TARGET_PROFILES_CACHE.clear()