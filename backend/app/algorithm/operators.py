# backend/app/algorithm/operators.py

import random
import numpy as np
# Asumsi chromosome_setup.py ada di modul yang sama (algorithm)
from .chromosome_setup import FEATURE_ORDER, FEATURE_DETAILS, NUM_FEATURES, initialize_chromosome

# --- 1. Seleksi ---
def tournament_selection(population, fitness_scores, k=3):
    """
    Melakukan seleksi turnamen.
    Memilih individu terbaik dari k individu yang dipilih secara acak.
    """
    selected_parents = []
    population_size = len(population)
    
    for _ in range(population_size): # Kita butuh sejumlah parent yang sama dengan ukuran populasi
        tournament_indices = random.sample(range(population_size), k)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_index_in_tournament = np.argmax(tournament_fitness)
        winner_original_index = tournament_indices[winner_index_in_tournament]
        selected_parents.append(population[winner_original_index]) # Menyimpan seluruh kromosom (list)
            
    return selected_parents

# --- 2. Crossover ---
def uniform_crossover(parent1, parent2, crossover_probability):
    """
    Melakukan uniform crossover.
    Untuk setiap gen (fitur), pilih secara acak dari parent1 atau parent2.
    Ini cocok untuk kromosom di mana urutan gen tidak sepenting kombinasi nilai.
    """
    child1 = [None] * NUM_FEATURES
    child2 = [None] * NUM_FEATURES

    if random.random() < crossover_probability:
        for i in range(NUM_FEATURES):
            if random.random() < 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        return child1, child2
    else:
        # Jika tidak ada crossover, anak adalah salinan parent
        return list(parent1), list(parent2)

def arithmetic_crossover_numerical_only(parent1, parent2, feature_index, alpha=0.5):
    """
    Melakukan arithmetic crossover untuk satu fitur numerik.
    p1' = alpha * p1 + (1-alpha) * p2
    p2' = (1-alpha) * p1 + alpha * p2
    """
    val1 = parent1[feature_index]
    val2 = parent2[feature_index]
    
    child_val1 = alpha * val1 + (1 - alpha) * val2
    child_val2 = (1 - alpha) * val1 + alpha * val2
    
    # Pastikan nilai tetap dalam rentang yang valid (clamping)
    details = FEATURE_DETAILS[FEATURE_ORDER[feature_index]]
    min_val, max_val = details['range']
    
    child_val1 = max(min_val, min(child_val1, max_val))
    child_val2 = max(min_val, min(child_val2, max_val))
    
    return child_val1, child_val2

def combined_crossover(parent1, parent2, crossover_probability, numerical_crossover_alpha=0.5):
    """
    Kombinasi crossover: uniform untuk semua fitur,
    dan untuk fitur numerik, bisa juga dipertimbangkan arithmetic crossover (opsional).
    Untuk kesederhanaan awal, kita gunakan uniform crossover untuk semua.
    """
    # Untuk saat ini, kita gunakan uniform crossover untuk semua tipe fitur.
    # Ini lebih sederhana untuk diimplementasikan dan seringkali bekerja dengan baik.
    return uniform_crossover(parent1, parent2, crossover_probability)

    # --- Alternatif Crossover yang Lebih Kompleks (Contoh, tidak diimplementasikan penuh di sini) ---
    # child1 = list(parent1) # Salin
    # child2 = list(parent2) # Salin
    # if random.random() < crossover_probability:
    #     for i in range(NUM_FEATURES):
    #         feature_name = FEATURE_ORDER[i]
    #         details = FEATURE_DETAILS[feature_name]
    #         
    #         if details['type'] == 'numerical':
    #             # Bisa pilih antara uniform atau arithmetic untuk numerik
    #             if random.random() < 0.5: # 50% chance for arithmetic
    #                 val1, val2 = arithmetic_crossover_numerical_only(parent1, parent2, i, numerical_crossover_alpha)
    #                 child1[i] = val1
    #                 child2[i] = val2
    #             else: # Uniform
    #                 if random.random() < 0.5:
    #                     child1[i] = parent1[i]
    #                     child2[i] = parent2[i]
    #                 else:
    #                     child1[i] = parent2[i]
    #                     child2[i] = parent1[i]
    #         elif details['type'] == 'categorical':
    #             # Uniform crossover untuk kategorikal
    #             if random.random() < 0.5:
    #                 child1[i] = parent1[i]
    #                 child2[i] = parent2[i]
    #             else:
    #                 child1[i] = parent2[i]
    #                 child2[i] = parent1[i]
    # return child1, child2


# --- 3. Mutasi ---
def random_reset_mutation(chromosome, mutation_probability):
    """
    Melakukan mutasi dengan mereset nilai gen ke nilai acak baru yang valid.
    """
    mutated_chromosome = list(chromosome) # Salin kromosom
    for i in range(NUM_FEATURES):
        if random.random() < mutation_probability:
            feature_name = FEATURE_ORDER[i]
            details = FEATURE_DETAILS[feature_name]
            
            if details['type'] == 'numerical':
                mutated_chromosome[i] = random.uniform(details['range'][0], details['range'][1])
            elif details['type'] == 'categorical':
                mutated_chromosome[i] = random.choice(details['categories'])
    return mutated_chromosome

def creep_mutation_numerical_only(value, feature_details, creep_magnitude_ratio=0.1):
    """
    Melakukan creep mutation pada satu fitur numerik.
    Menambahkan nilai acak kecil (positif atau negatif) ke nilai saat ini.
    """
    min_val, max_val = feature_details['range']
    current_range = max_val - min_val
    
    if current_range == 0: # Jika rentang adalah 0, tidak ada creep
        return value

    # Besarnya creep adalah persentase dari rentang total fitur
    creep_value = (random.random() - 0.5) * 2 * creep_magnitude_ratio * current_range # antara -max_creep dan +max_creep
    
    mutated_value = value + creep_value
    
    # Pastikan nilai tetap dalam rentang (clamping)
    mutated_value = max(min_val, min(mutated_value, max_val))
    return mutated_value

def combined_mutation(chromosome, mutation_probability, numerical_creep_prob=0.5, creep_magnitude_ratio=0.1):
    """
    Kombinasi mutasi: 
    - Untuk fitur numerik: bisa random reset atau creep mutation.
    - Untuk fitur kategorikal: random reset (pilih kategori acak baru).
    """
    mutated_chromosome = list(chromosome)
    for i in range(NUM_FEATURES):
        if random.random() < mutation_probability: # Apakah gen ini akan dimutasi?
            feature_name = FEATURE_ORDER[i]
            details = FEATURE_DETAILS[feature_name]
            
            if details['type'] == 'numerical':
                if random.random() < numerical_creep_prob: # Peluang untuk creep mutation
                    mutated_chromosome[i] = creep_mutation_numerical_only(mutated_chromosome[i], details, creep_magnitude_ratio)
                else: # Random reset untuk numerik
                    mutated_chromosome[i] = random.uniform(details['range'][0], details['range'][1])
            
            elif details['type'] == 'categorical':
                # Random reset untuk kategorikal (pilih kategori acak lain)
                # Untuk memastikan nilai *berubah* jika memungkinkan:
                current_value = mutated_chromosome[i]
                possible_new_values = [cat for cat in details['categories'] if cat != current_value]
                if possible_new_values:
                    mutated_chromosome[i] = random.choice(possible_new_values)
                else: # Jika hanya ada satu kategori, tidak bisa berubah
                    mutated_chromosome[i] = current_value 
                    
    return mutated_chromosome


# --- Contoh Penggunaan (untuk testing) ---
if __name__ == '__main__':
    # Inisialisasi beberapa kromosom dummy (menggunakan struktur dari chromosome_setup)
    # Pastikan FEATURE_ORDER, FEATURE_DETAILS, NUM_FEATURES sudah diimpor/didefinisikan
    
    parent_a = initialize_chromosome()
    parent_b = initialize_chromosome()
    parent_c = initialize_chromosome()
    parent_d = initialize_chromosome()

    dummy_population = [parent_a, parent_b, parent_c, parent_d]
    dummy_fitness_scores = [0.7, 0.9, 0.6, 0.85] # Contoh fitness

    print("--- Testing Seleksi Turnamen ---")
    selected = tournament_selection(dummy_population, dummy_fitness_scores, k=2)
    print(f"Jumlah parent terpilih: {len(selected)}")
    # Anda bisa print untuk melihat apakah parent dengan fitness lebih tinggi lebih sering terpilih
    # for i, p in enumerate(selected):
    #     print(f"Parent terpilih {i}: Fitness (kurang lebih) = ...")

    print("\n--- Testing Crossover (Uniform) ---")
    child_x1, child_x2 = uniform_crossover(parent_a, parent_b, crossover_probability=0.9)
    # Print beberapa gen untuk melihat perbedaannya
    print("Parent A (awal):", [f"{val:.2f}" if isinstance(val, float) else val for val in parent_a[:5]])
    print("Parent B (awal):", [f"{val:.2f}" if isinstance(val, float) else val for val in parent_b[:5]])
    print("Child X1 (awal):", [f"{val:.2f}" if isinstance(val, float) else val for val in child_x1[:5]])
    print("Child X2 (awal):", [f"{val:.2f}" if isinstance(val, float) else val for val in child_x2[:5]])

    print("\n--- Testing Mutasi (Combined) ---")
    original_chromo = initialize_chromosome()
    print("Kromosom Asli (awal):", [f"{val:.2f}" if isinstance(val, float) else val for val in original_chromo[:7]])
    
    mutated_chromo = combined_mutation(original_chromo, mutation_probability=0.1, numerical_creep_prob=0.7) # 10% per gen, 70% creep jika numerik
    print("Kromosom Termutasi (awal):", [f"{val:.2f}" if isinstance(val, float) else val for val in mutated_chromo[:7]])
    
    changes = 0
    for i in range(NUM_FEATURES):
        if original_chromo[i] != mutated_chromo[i]:
            changes +=1
            # print(f"Perubahan di fitur {FEATURE_ORDER[i]}: {original_chromo[i]} -> {mutated_chromo[i]}")
    print(f"Total gen yang termutasi (diharapkan sekitar {NUM_FEATURES * 0.1}): {changes}")