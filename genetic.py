'''
Genetic Algorithm Module 
'''

'''
Applied genetic algorithm in diagonal magic cube problem
'''

# Import Relevant Libraries
import random
import numpy as np
from tensor import *
import copy

class GeneticAlgo:
    def __init__(self, cube, goal_cube, population_size, generation_rate):
        self.cube = cube
        self.goal_cube = goal_cube
        self.population_size = population_size
        self.generation_rate = generation_rate
        self.population = []
        mutation_rate = 0.1
        goal_fitness = goal_cube.objective_function()
        self.history = []

        # Initial Population
        print("Generating initial population...")
        self.population.append(self.cube)
        
        while len(self.population) < self.population_size:
            # print(f"Current population size: {len(self.population)}")
            initial_cube = copy.deepcopy(self.cube)
            new_cube = initial_cube.randomize_value()
            # print("Generated new cube")
            if all(not existing_cube.same_tensor(new_cube) for existing_cube in self.population):
                self.population.append(new_cube)
                # print("Added new cube to population")
            else:
                print("Cube already exists in population")
        print("Initial Population Generated with size: ", len(self.population))
        self.sort_population_by_fitness(self.population)
        self.history.append(self.population[0])

       
        # Evolution
        for i in range(self.generation_rate):
            new_population = []

            # Selection
            self.population = self.population[:2]
            parent1 = self.population[0]
            parent2 = self.population[1]
            # print("Parent 1 Fitness: ", parent1.objective_function())
            # print("Parent 2 Fitness: ", parent2.objective_function())

            while len(new_population) <= self.population_size:
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                if random.random() < mutation_rate:
                    self.mutate(child1)
                if random.random() < mutation_rate:
                    self.mutate(child2)

                # New Generation
                if all(not existing_cube.same_tensor(child1) for existing_cube in new_population):
                    new_population.append(child1)
                if len(new_population) < self.population_size:
                    if all(not existing_cube.same_tensor(child2) for existing_cube in new_population):
                        new_population.append(child2)
                
            self.sort_population_by_fitness(new_population)
            best_solution = new_population[0]
            self.history.append(best_solution)
            if best_solution.objective_function() == goal_fitness:
                break
            else:
                # Update Population
                self.population = new_population

            print("Generation: ", i+1, "Fitness: ", best_solution.objective_function())

        # Final Solution
        print("Best solution found:")
        best_solution.plot_cube()
        print("Fitness:", best_solution.objective_function())

    def sort_population_by_fitness(self, population):
        population.sort(key=lambda tensor: tensor.objective_function(), reverse=False)

    def crossover(self, parent1, parent2):
        child1 = Tensor(parent1.r, parent1.c, parent1.h)
        child2 = Tensor(parent2.r, parent2.c, parent2.h)
        child1.array = np.copy(parent1.array)
        child2.array = np.copy(parent2.array)

        row = random.randint(0, parent1.r - 1)
        col = random.randint(0, parent1.c - 1)
        height = random.randint(0, parent1.h - 1)

        child1.array[row, col, height], child2.array[row, col, height] = parent2.array[row, col, height], parent1.array[row, col, height]

        return child1, child2
    
    def mutate(self, cube):
        row = random.randint(0, cube.r - 1)
        col = random.randint(0, cube.c - 1)
        height = random.randint(0, cube.h - 1)
        cube.array[row, col, height] = random.randint(0, 125)

    def hist_plot(self):
       fitness_values = [cube.objective_function() for cube in self.history]
       plt.figure(figsize=(10, 5))
       plt.plot(fitness_values, label='Best Fitness Over Generations')
       plt.xlabel('Generation')
       plt.ylabel('Fitness Value')
       plt.title('Fitness Evolution Over Generations')
       plt.legend()
       plt.grid()
       plt.show()