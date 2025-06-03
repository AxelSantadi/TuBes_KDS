# backend/app/genetic_algorithm/ga_core.py

import random
import numpy as np
from .fitness import calculate_combined_fitness
from .operators import tournament_selection, uniform_crossover, combined_mutation
# from .operators import tournament_selection, combined_crossover, combined_mutation

class GeneticAlgorithmFeatureSelection:
    def __init__(self, original_df, label_col, all_original_feature_names,
                 numerical_cols_original, categorical_cols_original,
                 target_genus_specie_for_ga: str,
                 initial_user_params_for_ga: dict, # Ini adalah dict {nama_fitur: nilai}
                 population_size=50, num_generations=20,
                 crossover_prob=0.8, mutation_prob=0.01,
                 num_features=27,
                 fitness_params: dict = None): # (27 atribut)

        self.original_df = original_df
        self.label_col = label_col
        # Pastikan all_original_feature_names adalah daftar nama fitur yang benar, sesuai NUM_FEATURES
        self.all_original_feature_names = all_original_feature_names[:num_features]
        self.numerical_cols_original = [col for col in numerical_cols_original if col in self.all_original_feature_names]
        self.categorical_cols_original = [col for col in categorical_cols_original if col in self.all_original_feature_names]
        self.evolution_log_tuples = []         # Untuk (generation, fitness, best_chromosome_list)
        self.convergence_log = []

        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.num_features = num_features

        # SIMPAN PARAMETER BARU SEBAGAI ATRIBUT INSTANCE:
        self.target_genus_specie = target_genus_specie_for_ga
        self.user_input_dict_for_fitness = initial_user_params_for_ga # Ini dict input awal pengguna

        self.fitness_params = fitness_params if fitness_params else {}

        self.population = []
        self.fitness_scores = []
        self.best_chromosome_overall = None
        self.best_fitness_overall = -float('inf') # Inisialisasi dengan nilai sangat kecil
        self.evolution_log_tuples = [] # Untuk (generation, fitness, best_chromosome_list)

    def _initialize_population(self):
        """Inisialisasi populasi awal dengan kromosom biner."""
        self.population = []
        for _ in range(self.population_size):
            # Kromosom adalah vektor biner panjang num_features
            chromosome = [random.randint(0, 1) for _ in range(self.num_features)]
            # Pastikan minimal ada satu fitur terpilih untuk menghindari masalah di awal
            if sum(chromosome) == 0 and self.num_features > 0:
                chromosome[random.randint(0, self.num_features - 1)] = 1
            self.population.append(chromosome)

    def _evaluate_population(self):
        """Mengevaluasi fitness setiap individu dalam populasi."""
        self.fitness_scores = []
        for chromo_list in self.population:
            fitness = calculate_combined_fitness( # Pastikan nama fungsi ini benar
                chromosome_list=chromo_list,
                target_genus_specie=self.target_genus_specie,
                dataset_df=self.original_df,
                user_input_dict=self.user_input_dict_for_fitness,
                numerical_cols_original=self.numerical_cols_original,
                categorical_cols_original=self.categorical_cols_original,
                label_col_in_dataset=self.label_col
                # Anda bisa meneruskan self.fitness_params jika calculate_combined_fitness menerimanya
            )
            self.fitness_scores.append(fitness)

    def run(self):
        """Menjalankan algoritma genetik."""
        print("Memulai Algoritma Genetik untuk Seleksi Fitur...")
        self._initialize_population()

        for gen in range(self.num_generations):
            self._evaluate_population()

            current_best_fitness_in_gen = np.max(self.fitness_scores)
            current_best_chromo_in_gen = self.population[np.argmax(self.fitness_scores)]

            self.convergence_log.append((gen + 1, current_best_fitness_in_gen, list(current_best_chromo_in_gen)))

            if current_best_fitness_in_gen > self.best_fitness_overall:
                self.best_fitness_overall = current_best_fitness_in_gen
                self.best_chromosome_overall = list(current_best_chromo_in_gen) # Simpan sebagai list

            print(f"Generasi {gen + 1}/{self.num_generations} - Fitness Terbaik: {self.best_fitness_overall:.4f} (Akurasi di gen ini: {current_best_fitness_in_gen:.4f})")

            # Seleksi
            selected_parents = tournament_selection(self.population, self.fitness_scores)

            # Crossover dan Mutasi untuk membuat populasi baru
            next_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_parents[i]
                # Pastikan ada parent kedua jika ukuran populasi ganjil
                parent2 = selected_parents[i+1] if (i+1) < self.population_size else selected_parents[0] 

                child1, child2 = uniform_crossover(parent1, parent2, self.crossover_prob) # Menggunakan uniform_crossover

                # Menggunakan combined_mutation
                next_population.append(combined_mutation(child1, self.mutation_prob, numerical_creep_prob=0.5, creep_magnitude_ratio=0.1)) 
                if len(next_population) < self.population_size:
                    next_population.append(combined_mutation(child2, self.mutation_prob, numerical_creep_prob=0.5, creep_magnitude_ratio=0.1))

            self.population = next_population

        # Evaluasi terakhir untuk populasi final jika diperlukan, atau langsung ambil yang terbaik selama ini
        print("\nAlgoritma Genetik Selesai.")
        print(f"Kromosom terbaik ditemukan: {self.best_chromosome_overall}")
        print(f"Fitness terbaik: {self.best_fitness_overall:.4f}")

        selected_feature_names_final = [self.all_original_feature_names[i] for i, bit in enumerate(self.best_chromosome_overall) if bit == 1]
        print(f"Fitur terpilih: {selected_feature_names_final}")

        return self.best_chromosome_overall, self.best_fitness_overall, None, self.convergence_log