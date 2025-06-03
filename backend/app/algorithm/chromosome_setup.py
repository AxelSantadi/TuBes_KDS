# backend/app/algorithm/chromosome_setup.py

import random
import numpy as np

# 1. Definisi Fitur dan Urutannya dalam Kromosom
# Urutan ini PENTING dan harus konsisten.
# Ini adalah fitur-fitur yang nilainya akan dioptimasi/dievolusikan oleh GA.
# Daftar ini harus sesuai dengan input yang akan diberikan pengguna di frontend.
FEATURE_ORDER = [
    'Current_Country',
    'Habitat',
    'Canine_Size', # Perhatikan penulisan 'Canine Size' (dengan spasi)
    'Arms',
    'Diet'
]

# 2. Rentang Nilai dan Kategori Valid untuk Setiap Fitur
# Ini penting untuk inisialisasi acak dan operator mutasi.
# Anda PERLU MENGISI ini dengan nilai yang masuk akal berdasarkan data atau pengetahuan domain.

# 'Current_Country', 'Habitat', 'Canine Size', 'Arms', 'Diet'
FEATURE_DETAILS = {
    'Time': {'type': 'numerical', 'range': (0.1, 8.0)},  # Juta tahun yang lalu
    'Location': {'type': 'categorical', 'categories': ['Africa', 'Asia', 'Europe']},
    'Zone': {'type': 'categorical', 'categories': ['oriental', 'south', 'central', 'west']},
    'Current_Country': {'type': 'categorical', 'categories': ['Ethiopia', 'Kenya', 'South Africa', 'Indonesia', 'Georgia', 'Spain', 'Republic of Chad', 'Germany']}, # Contoh, perlu diperluas
    'Habitat': {'type': 'categorical', 'categories': ['savannah', 'mixed', 'forest', 'forest-gallery', 'cold forest', 'jungle', 'forest-savanna', 'peninsular']},
    'Cranial_Capacity': {'type': 'numerical', 'range': (100, 1500)},
    'Height': {'type': 'numerical', 'range': (80, 200)},
    'Incisor_Size': {'type': 'categorical', 'categories': ['small', 'big', 'megadony', 'very small', 'medium large']}, # Skala relatif
    'Jaw_Shape': {'type': 'categorical', 'categories': ['U shape', 'V shape', 'modern', 'conical']},
    'Torus_Supraorbital': {'type': 'categorical', 'categories': ['very protruding', 'little protruding', 'ultra protruding', 'less protruding', 'flat']},
    'Prognathism': {'type': 'categorical', 'categories': ['medium-high', 'high', 'reduced', 'very high', 'medium', 'absent']},
    'Foramen_Magnum_Position': {'type': 'categorical', 'categories': ['posterior', 'modern', 'anterior', 'semi-anterior']},
    'Canine_Size': {'type': 'categorical', 'categories': ['small', 'big']},
    'Canines_Shape': {'type': 'categorical', 'categories': ['incisiform', 'conicalls']},
    'Tooth_Enamel': {'type': 'categorical', 'categories': ['thick', 'medium-thin', 'very thick', 'medium-thick', 'thin', 'thick-medium', 'very thin']},
    'Tecno': {'type': 'categorical', 'categories': ['no', 'yes', 'likely']},
    'Tecno_type': {'type': 'categorical', 'categories': ['no', 'mode 1', 'mode 2', 'mode 3', 'mode 4', 'primitive']},
    'biped': {'type': 'categorical', 'categories': ['modern', 'yes', 'low probability', 'high probability']},
    'Arms': {'type': 'categorical', 'categories': ['climbing', 'manipulate', 'manipulate with precision']},
    'Foots': {'type': 'categorical', 'categories': ['walk', 'climbing']},
    'Diet': {'type': 'categorical', 'categories': ['omnivore', 'dry fruits', 'hard fruits', 'carnivorous', 'soft fruits']},
    'Sexual_Dimorphism': {'type': 'categorical', 'categories': ['reduced', 'medium-high', 'High']}, # Berdasarkan perbedaan ukuran tubuh/canine
    'Hip': {'type': 'categorical', 'categories': ['wide', 'slim', 'modern', 'very modern']},
    'Vertical_Front': {'type': 'categorical', 'categories': ['no', 'modern', 'yes']}, # Dahi
    'Anatomy': {'type': 'categorical', 'categories': ['old', 'mixed', 'modern', 'very modern']}, # Struktur tubuh umum
    'Migrated': {'type': 'categorical', 'categories': ['no', 'yes']},
    'Skeleton': {'type': 'categorical', 'categories': ['light', 'robust', 'refined']} # Terkait dengan robustisitas
}
# Pastikan semua fitur di FEATURE_ORDER ada di FEATURE_DETAILS
assert all(feature in FEATURE_DETAILS for feature in FEATURE_ORDER), \
    "Tidak semua fitur di FEATURE_ORDER baru (5 fitur) ada di FEATURE_DETAILS. Periksa nama fitur 'Canine Size'."

NUM_FEATURES = len(FEATURE_ORDER)

def initialize_chromosome():
    """
    Menginisialisasi satu kromosom dengan nilai acak yang valid untuk setiap fitur.
    Kromosom adalah list di mana setiap elemen sesuai dengan fitur di FEATURE_ORDER.
    """
    chromosome = []
    for feature_name in FEATURE_ORDER:
        details = FEATURE_DETAILS[feature_name]
        if details['type'] == 'numerical':
            value = random.uniform(details['range'][0], details['range'][1])
            chromosome.append(value)
        elif details['type'] == 'categorical':
            value = random.choice(details['categories'])
            chromosome.append(value)
    return chromosome

def user_input_to_chromosome(user_input_dict):
    """
    Mengonversi dictionary input dari pengguna menjadi format kromosom (list).
    Input pengguna diharapkan berupa dictionary {'NamaFitur': nilai, ...}.
    Nilai numerik akan diambil apa adanya (setelah divalidasi).
    Nilai kategorikal akan divalidasi terhadap kategori yang ada.
    """
    chromosome = []
    missing_features = []
    invalid_values = {}

    for feature_name in FEATURE_ORDER:
        details = FEATURE_DETAILS[feature_name]
        user_value = user_input_dict.get(feature_name)

        if user_value is None:
            missing_features.append(feature_name)
            # Opsi: isi dengan nilai acak jika pengguna tidak menyediakan? Atau error?
            # Untuk saat ini, kita tandai sebagai hilang dan bisa di-handle nanti.
            # Atau, kita bisa langsung generate acak jika memang GA akan "mengisi" kekosongan
            if details['type'] == 'numerical':
                chromosome.append(random.uniform(details['range'][0], details['range'][1]))
            else: # categorical
                chromosome.append(random.choice(details['categories']))
            continue

        if details['type'] == 'numerical':
            try:
                val = float(user_value)
                if not (details['range'][0] <= val <= details['range'][1]):
                    invalid_values[feature_name] = f"Nilai '{val}' di luar rentang {details['range']}"
                    # Opsi: clamp ke rentang atau error
                    val = max(details['range'][0], min(val, details['range'][1]))
                chromosome.append(val)
            except ValueError:
                invalid_values[feature_name] = f"Nilai '{user_value}' bukan angka yang valid."
                chromosome.append(random.uniform(details['range'][0], details['range'][1])) # Fallback
        
        elif details['type'] == 'categorical':
            if user_value not in details['categories']:
                invalid_values[feature_name] = f"Kategori '{user_value}' tidak valid. Pilihan: {details['categories']}"
                chromosome.append(random.choice(details['categories'])) # Fallback
            else:
                chromosome.append(user_value)
    
    if missing_features:
        print(f"Peringatan: Fitur berikut tidak ada di input pengguna dan diisi acak: {missing_features}")
    if invalid_values:
        for feature, msg in invalid_values.items():
            print(f"Peringatan untuk fitur '{feature}': {msg}. Menggunakan nilai fallback/clamp.")
            
    return chromosome

# --- Contoh Penggunaan (bisa dihapus atau dikomentari di file produksi) ---
if __name__ == '__main__':
    print(f"Total fitur dalam kromosom: {NUM_FEATURES}")
    
    print("\n--- Contoh Kromosom Acak ---")
    random_chromo = initialize_chromosome()
    for i, feature_name in enumerate(FEATURE_ORDER):
        print(f"{feature_name}: {random_chromo[i]}")

    print("\n--- Contoh Konversi Input Pengguna ---")
    # Pengguna hanya mengisi beberapa, sisanya akan diisi acak dengan peringatan
    # sample_user_input = {
    #     'Time': 2.5,                              # Valid
    #     'Location': 'Africa',                     # Valid
    #     'Cranial_Capacity': 700,                  # Valid
    #     'Habitat': 'Savanna',                     # Valid
    #     'Jaw_Shape': 'Parabolic',                 # Valid
    #     'Height': 150,                            # Valid
    #     'biped': 'Obligate',                      # Valid
    #     'Diet': 'Omnivorous (mixed)',             # Valid
    #     'Tecno_type': 'Mode 1 (Oldowan)',         # Valid
    #     'NonExistentFeature': 'test',             # Akan diabaikan
    #     'Torus_Supraorbital': 'Super Pronounced'  # Tidak valid, akan difallback
    # }
    
    # user_chromo = user_input_to_chromosome(sample_user_input)
    # print("\nKromosom dari Input Pengguna (dengan fallback untuk yang hilang/tidak valid):")
    # for i, feature_name in enumerate(FEATURE_ORDER):
    #     print(f"{feature_name}: {user_chromo[i]}")

    # Contoh input pengguna yang lebih lengkap
    complete_user_input = {
        'Time': 3.0,
        'Location': 'Asia',
        'Zone': 'south',
        'Current_Country': 'Indonesia',
        'Habitat': 'jungle',
        'Cranial_Capacity': 1200,
        'Height': 175,
        'Incisor_Size': 'medium large',
        'Jaw_Shape': 'U shape',
        'Torus_Supraorbital': 'little protruding',
        'Prognathism': 'medium',
        'Foramen_Magnum_Position': 'posterior',
        'Canine_Size': 'big',
        'Canines_Shape': 'conicalls',
        'Tooth_Enamel': 'medium-thin',
        'Tecno': 'yes',
        'Tecno_type': 'mode 2',
        'biped': 'modern',
        'Arms': 'manipulate with precision',
        'Foots': 'walk',
        'Diet': 'dry fruits',
        'Sexual_Dimorphism': 'medium-high',
        'Hip': 'wide',
        'Vertical_Front': 'yes',
        'Anatomy': 'modern',
        'Migrated': 'no',
        'Skeleton': 'light'
    }
    print("\nKromosom dari Input Pengguna Lengkap:")
    full_user_chromo = user_input_to_chromosome(complete_user_input)
    for i, feature_name in enumerate(FEATURE_ORDER):
        print(f"{feature_name}: {full_user_chromo[i]}")