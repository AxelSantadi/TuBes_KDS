# backend/app/genetic_algorithm/ga_core.py

import random
import numpy as np
from .fitness import calculate_fitness # Tanda '.' menunjukkan impor dari modul dalam package yang sama
from .operators import tournament_selection, single_point_crossover, bit_flip_mutation

class GeneticAlgorithmFeatureSelection:
    def __init__(self, original_df, label_col, all_original_feature_names,
                 numerical_cols_original, categorical_cols_original,
                 population_size=50, num_generations=20, # (P=50, Gmax=20 untuk debug)
                 crossover_prob=0.8, mutation_prob=0.01, # (pc=0.8, pm=0.01 contoh)
                 num_features=27, fitness_params=None): # (27 atribut)

        self.original_df = original_df
        self.label_col = label_col
        self.all_original_feature_names = all_original_feature_names[:num_features] # Pastikan hanya menggunakan N fitur
        self.numerical_cols_original = [col for col in numerical_cols_original if col in self.all_original_feature_names]
        self.categorical_cols_original = [col for col in categorical_cols_original if col in self.all_original_feature_names]

        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.num_features = num_features # Jumlah total fitur yang bisa dipilih (panjang kromosom)

        self.population = []
        self.fitness_scores = []
        self.best_chromosome_overall = None
        self.best_fitness_overall = -1.0
        self.convergence_log = [] # Untuk menyimpan fitness terbaik per generasi

        self.fitness_params = fitness_params if fitness_params else {}

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
        for chromo in self.population:
            fitness = calculate_fitness(
                chromo,
                self.original_df,
                self.all_original_feature_names,
                self.numerical_cols_original,
                self.categorical_cols_original,
                self.label_col,
                **self.fitness_params 
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

            self.convergence_log.append(current_best_fitness_in_gen)

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

                child1, child2 = single_point_crossover(parent1, parent2, self.crossover_prob)

                next_population.append(bit_flip_mutation(child1, self.mutation_prob))
                if len(next_population) < self.population_size:
                    next_population.append(bit_flip_mutation(child2, self.mutation_prob))

            self.population = next_population

        # Evaluasi terakhir untuk populasi final jika diperlukan, atau langsung ambil yang terbaik selama ini
        print("\nAlgoritma Genetik Selesai.")
        print(f"Kromosom terbaik ditemukan: {self.best_chromosome_overall}")
        print(f"Fitness terbaik: {self.best_fitness_overall:.4f}")

        selected_feature_names_final = [self.all_original_feature_names[i] for i, bit in enumerate(self.best_chromosome_overall) if bit == 1]
        print(f"Fitur terpilih: {selected_feature_names_final}")

        return self.best_chromosome_overall, self.best_fitness_overall, selected_feature_names_final, self.convergence_log