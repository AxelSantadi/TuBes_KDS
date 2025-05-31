# backend/app/genetic_algorithm/operators.py

import random
import numpy as np

def tournament_selection(population, fitness_scores, k=3): # (Seleksi tournamen)
    """Melakukan seleksi turnamen."""
    selected_parents = []
    population_size = len(population)

    for _ in range(population_size): # Kita butuh sejumlah parent yang sama dengan ukuran populasi untuk generasi berikutnya
        tournament_indices = random.sample(range(population_size), k)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        winner_index_in_tournament = np.argmax(tournament_fitness)
        winner_original_index = tournament_indices[winner_index_in_tournament]
        selected_parents.append(population[winner_original_index])

    return selected_parents

def single_point_crossover(parent1, parent2, crossover_probability): # (crossover single-point)
    """Melakukan single-point crossover."""
    child1, child2 = list(parent1), list(parent2) # Salin parent
    if random.random() < crossover_probability:
        if len(parent1) > 1 : # Crossover hanya jika panjang kromosom > 1
            point = random.randint(1, len(parent1) - 1)
            child1 = list(parent1[:point]) + list(parent2[point:])
            child2 = list(parent2[:point]) + list(parent1[point:])
    return child1, child2

def bit_flip_mutation(chromosome, mutation_probability): # (mutasi flip bit)
    """Melakukan bit-flip mutation."""
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_probability:
            mutated_chromosome[i] = 1 - mutated_chromosome[i] # Flip bit
    return mutated_chromosome