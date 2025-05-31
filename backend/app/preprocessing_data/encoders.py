# backend/app/preprocessing/encoders.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def preprocess_data(df, numerical_cols, categorical_cols, label_col):
    """
    Melakukan pra-pemrosesan pada DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame input.
        numerical_cols (list): Daftar nama kolom numerik.
        categorical_cols (list): Daftar nama kolom kategorikal.
        label_col (str): Nama kolom label/target.
        
    Returns:
        X_processed (np.ndarray): Matriks fitur yang sudah diproses.
        y (np.ndarray): Array label.
        preprocessor (ColumnTransformer): Objek preprocessor yang sudah di-fit.
                                          Ini bisa disimpan untuk memproses data baru
                                          dengan cara yang sama.
        feature_names_out (list): Daftar nama fitur setelah encoding.
    """
    
    X = df.drop(columns=[label_col])
    y = df[label_col].values # Asumsi label tidak perlu encoding khusus (misal sudah numerik atau akan dihandle terpisah)

    # Membuat pipeline untuk fitur numerik
    numerical_transformer = StandardScaler()

    # Membuat pipeline untuk fitur kategorikal
    # handle_unknown='ignore' akan mengabaikan kategori baru saat transform (jika ada di data tes)
    # sparse_output=False agar hasilnya berupa dense array, bisa diubah jika perlu
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Menggabungkan transformer menggunakan ColumnTransformer
    # remainder='passthrough' akan membiarkan kolom lain yang tidak disebut (jika ada)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' 
    )

    # Fit dan transform data
    X_processed = preprocessor.fit_transform(X)
    
    # Mendapatkan nama fitur setelah one-hot encoding
    try:
        # Untuk scikit-learn versi baru
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback untuk versi lama (mungkin perlu penyesuaian lebih lanjut)
        ohe_feature_names = []
        if categorical_cols:
            ohe_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
        
        feature_names_out = numerical_cols + ohe_feature_names
        # Tambahkan nama kolom dari 'remainder' jika ada
        # Ini perlu penyesuaian jika 'remainder' menghasilkan banyak kolom
        num_processed_cols = X_processed.shape[1]
        current_feature_count = len(feature_names_out)
        if num_processed_cols > current_feature_count:
             # Cari kolom mana saja yang masuk ke remainder
            processed_cols_set = set(numerical_cols) | set(categorical_cols)
            remainder_cols = [col for col in X.columns if col not in processed_cols_set]
            feature_names_out.extend(remainder_cols)


    return X_processed, y, preprocessor, feature_names_out

if __name__ == '__main__':
    # Contoh penggunaan (asumsi file CSV ada di path yang benar)
    # Anda perlu menyesuaikan path ke file CSV Anda
    try:
        data_df = pd.read_csv('../../data/raw/Evolution_DataSets.csv')
    except FileNotFoundError:
        print("Pastikan file 'Evolution_DataSets.csv' ada di direktori 'data/raw/'")
        data_df = None

    if data_df is not None:
        print("Dataset berhasil dimuat.")
        print("Beberapa baris pertama dataset:")
        print(data_df.head())
        print("\nInfo dataset:")
        data_df.info()

        # Identifikasi kolom numerik dan kategorikal berdasarkan deskripsi proyek Anda
        # Ini adalah contoh, Anda HARUS menyesuaikannya dengan dataset Anda
        # Time (jutaan tahun), Cranial_Capacity, Height -> Numerik
        # Location, Zone, Jaw_Shape, Incisor_Size, Current_Country, Habitat -> Kategorikal
        # Label: Genus_&_Specie
        
        # Hapus baris dengan nilai NaN jika ada, atau lakukan imputasi
        # Dokumen desain menyatakan "Data sudah bersih sehingga tidak ada missing values lagi"
        # Namun, selalu baik untuk memeriksa:
        print(f"\nJumlah missing values sebelum cleaning: \n{data_df.isnull().sum()}")
        data_df_cleaned = data_df.dropna() # Atau metode imputasi lain jika diperlukan
        print(f"Jumlah baris setelah menghapus NaN (jika ada): {len(data_df_cleaned)}")


        # Kolom numerik dan kategorikal (SESUAIKAN DENGAN NAMA KOLOM DI CSV ANDA)
        # Berdasarkan dokumen desain Anda, atributnya ada 27
        # Anda perlu daftar lengkapnya. Berikut contoh parsial:
        # Perhatikan bahwa 'Genus_&_Specie' adalah label.
        
        # Contoh (PERLU DISESUAIKAN DENGAN CSV ANDA):
        # Pastikan nama kolom ini sama persis dengan yang ada di file CSV.
        potential_numerical_cols = ['Time (Million Years)', 'Cranial Capacity (cc)', 'Height (feet)', 
                                     'Incisor Size (teeth)', 'Canine Size (teeth)', 'Premolar Size (teeth)', 
                                     'Molar Size (teeth)', 'Body Mass (kg)']
        potential_categorical_cols = ['Location', 'Zone', 'Habitat', 'Current Country', 'Jaw Shape', 
                                       'Foramen Magnum Position', 'Teeth Shape', 'Brain Shape', 'Face Shape', 
                                       'Bipedalism', 'Tool Use', 'Diet', 'Sexual Dimorphism', 'Group Size', 
                                       'Lifespan (years)', 'Gestation (months)'] 
                                       # 'Evolutionary Stage' dan 'Species Classification' bisa jadi target atau fitur tambahan.
                                       # Asumsi 'Genus_&_Specie' adalah target utama.

        label_column = 'Genus_&_Specie' # Sesuai dokumen Anda

        # Filter kolom yang benar-benar ada di DataFrame
        actual_numerical_cols = [col for col in potential_numerical_cols if col in data_df_cleaned.columns]
        actual_categorical_cols = [col for col in potential_categorical_cols if col in data_df_cleaned.columns]

        if not actual_numerical_cols and not actual_categorical_cols:
            print("Tidak ada kolom numerik atau kategorikal yang teridentifikasi. Periksa nama kolom.")
        elif label_column not in data_df_cleaned.columns:
            print(f"Kolom label '{label_column}' tidak ditemukan dalam dataset.")
        else:
            print(f"\nKolom numerik teridentifikasi: {actual_numerical_cols}")
            print(f"Kolom kategorikal teridentifikasi: {actual_categorical_cols}")
            print(f"Kolom label: {label_column}")

            X_processed, y_labels, preprocessor_obj, feature_names = preprocess_data(
                data_df_cleaned, 
                actual_numerical_cols, 
                actual_categorical_cols, 
                label_column
            )

            print("\nDimensi data fitur setelah pra-pemrosesan:", X_processed.shape)
            print("Contoh data fitur setelah pra-pemrosesan (5 baris pertama):")
            print(X_processed[:5])
            print("\nLabel (5 label pertama):")
            print(y_labels[:5])
            print("\nNama fitur setelah encoding:")
            print(feature_names)
            print(f"Jumlah fitur setelah encoding: {len(feature_names)}")
            
            # Verifikasi jumlah fitur (seharusnya sekitar 27 sebelum encoding jika semua asli adalah fitur)
            # Jumlah fitur setelah one-hot encoding akan lebih banyak dari 27.
            # Dokumen Anda menyebutkan "27 atribut morfologi/kategorikal" sebagai input untuk feature selection.
            # Ini berarti input ke GA adalah representasi biner dari pemilihan fitur-fitur ASLI ini,
            # bukan fitur yang sudah di-one-hot-encode.
            # Fungsi fitness di GA akan melakukan encoding ini secara internal untuk setiap subset fitur yang dievaluasi.