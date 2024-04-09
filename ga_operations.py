import numpy as np


def roulette_wheel_selection(population, num_selections):
    fitness = 1 / (1 + population[:, -1])  # Assuming the last column is the cost
    total_fitness = np.sum(fitness)
    probs = fitness / total_fitness
    selected_indices = np.random.choice(range(len(population)), size=num_selections, p=probs)
    return population[selected_indices]


def single_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1)-1)  # Exclude cost column
    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:-1]), axis=None)
    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:-1]), axis=None)
    return offspring1, offspring2


def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)-1):  # Exclude cost column
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.rand()  # Assuming continuous values; adjust as needed
    return individual
