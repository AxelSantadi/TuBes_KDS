# backend/app/genetic_algorithm/fitness.py

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
# Asumsi encoders.py ada di ../preprocessing/
# Anda mungkin perlu menyesuaikan path impor berdasarkan struktur akhir proyek Anda
import sys
sys.path.append('../') # Menambahkan parent directory (app) agar bisa impor preprocessing
from preprocessing_data.encoders import preprocess_data # Mengimpor fungsi dari encoders.py

def calculate_fitness(chromosome, original_df, all_original_feature_names, 
                      numerical_cols_original, categorical_cols_original, 
                      label_col, svm_kernel='rbf', svm_c=1.0, k_folds=5, 
                      penalty_factor=0.01): # (opsional penalti fitur banyak)
    """
    Menghitung fitness sebuah kromosom untuk feature selection.
    Fitness adalah akurasi klasifikasi dari model SVM yang dilatih
    menggunakan fitur-fitur yang dipilih oleh kromosom, dengan opsi penalti
    untuk jumlah fitur yang banyak.

    Args:
        chromosome (list atau np.array): Array biner, 1 berarti fitur dipilih, 0 tidak.
        original_df (pd.DataFrame): DataFrame asli sebelum pra-pemrosesan.
        all_original_feature_names (list): Daftar semua nama fitur asli sesuai urutan di kromosom.
        numerical_cols_original (list): Daftar nama kolom numerik asli.
        categorical_cols_original (list): Daftar nama kolom kategorikal asli.
        label_col (str): Nama kolom label.
        svm_kernel (str): Kernel untuk SVM.
        svm_c (float): Parameter C untuk SVM.
        k_folds (int): Jumlah fold untuk cross-validation.
        penalty_factor (float): Faktor penalti untuk jumlah fitur.

    Returns:
        float: Nilai fitness.
    """
    selected_feature_indices = [i for i, bit in enumerate(chromosome) if bit == 1]

    if not selected_feature_indices:
        return 0.0  # Tidak ada fitur terpilih, fitness sangat rendah

    selected_feature_names = [all_original_feature_names[i] for i in selected_feature_indices]

    # Buat DataFrame baru hanya dengan fitur terpilih dan kolom label
    df_subset = original_df[selected_feature_names + [label_col]].copy()

    # Identifikasi ulang kolom numerik dan kategorikal dari subset fitur terpilih
    current_numerical_cols = [col for col in selected_feature_names if col in numerical_cols_original]
    current_categorical_cols = [col for col in selected_feature_names if col in categorical_cols_original]

    # Jika tidak ada fitur numerik atau kategorikal yang valid di subset (misal semua fitur numerik tidak terpilih)
    # fungsi preprocess_data mungkin error. Perlu penanganan.
    if not current_numerical_cols and not current_categorical_cols:
         # Ini bisa terjadi jika kromosom hanya memilih fitur yang tipenya tidak terdaftar, atau tidak memilih apa pun
        return 0.01 # Fitness rendah jika tidak ada fitur valid untuk diproses


    try:
        # Pra-pemrosesan hanya pada data subset
        X_processed, y_original_labels, _, _ = preprocess_data(
            df_subset,
            current_numerical_cols,
            current_categorical_cols,
            label_col
        )
    except ValueError as e:
        # Contoh error: jika current_numerical_cols atau current_categorical_cols kosong
        # padahal ColumnTransformer mengharapkannya.
        # print(f"Error saat pra-pemrosesan untuk kromosom {chromosome}: {e}")
        return 0.02 # Fitness rendah jika ada error saat preprocessing

    if X_processed.shape[1] == 0: # Tidak ada fitur yang berhasil diproses
        return 0.03

    # Encode label jika berupa string
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_original_labels)

    # Model Klasifikasi (SVM)
    model = SVC(kernel=svm_kernel, C=svm_c, random_state=42) # (menggunakan SVM untuk fitness)

    # Evaluasi dengan Cross-Validation
    # Menggunakan StratifiedKFold untuk menjaga proporsi kelas pada setiap fold, penting untuk klasifikasi
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    try:
        scores = cross_val_score(model, X_processed, y_encoded, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)
    except ValueError as e:
        # Bisa terjadi jika y_encoded hanya memiliki satu kelas setelah filtering/subsetting,
        # atau X_processed kosong.
        # print(f"Error saat cross-validation untuk kromosom {chromosome}: {e}")
        return 0.04 # Fitness rendah

    # Penalti untuk jumlah fitur (opsional, sesuai dokumen desain Anda)
    num_selected_features = sum(chromosome)
    total_features = len(chromosome)

    # Penalti dihitung berdasarkan rasio jumlah fitur terpilih terhadap total fitur
    # Semakin banyak fitur, semakin besar penalti.
    # Pastikan penalty_factor disesuaikan agar tidak mendominasi akurasi.
    fitness = accuracy - (penalty_factor * (num_selected_features / total_features))

    # Pastikan fitness tidak negatif
    return max(0, fitness)