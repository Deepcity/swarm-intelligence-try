import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
import copy


n_top = 20
n_perturb = 8
p_shift = .1
p_swop = .2
eps = .1


def deepcopy(arr):
    return copy.deepcopy(arr)


class Evolutionary:
    def __init__(self, task):
        self.task = task
    
    def generate_genome(self, length):
        genome = np.arange(length)
        np.random.shuffle(genome)
        return genome
    
    def generate_population(self, size, genome_length):
        population = [self.generate_genome(genome_length) for _ in range(size)]
        return population
    
    def partial_map_crossover(self, parent1, parent2):
        n = len(parent1)
        point = random.randint(0, n-1)
        child1 = list(parent1[0:point])
        for j in parent2:
            if (j in child1) == False:
                child1.append(j)
        child2 = list(parent2[0:point])
        for j in parent1:
            if (j in child2) == False:
                child2.append(j)
        return child1, child2
    
    
    def run_evolution(self, population_size, generation_limit=5000, fitness_limit=1e99, crossover='single', verbose=True):
        ## ... define population ...
        population = self.generate_population(population_size,self.task.num_cities)
        
        best_fitness_seen = -1e9
        for i in tqdm(range(generation_limit)):
            population = sorted(
                population, key=lambda genome: self.task.fitness(genome), reverse=True
            )
            fitness = self.task.fitness(population[0])
            
            if verbose and (fitness > best_fitness_seen):
                best_fitness_seen = fitness
                self.task.visualize(population[0], save_id=i)
            if fitness >= fitness_limit:
                break
            
            ## ... elitism; keep best individuals and variants of them ...
            next_generation = population[:n_top]   # keep the n_top fittest individuals

            for _ in range(n_perturb):
            # select a candidate from population[:n_top]
                candidate = random.choice(population[:n_top])
                if np.random.random() < p_shift:
                    candidate = self.shift_to_end(candidate)
                if np.random.random() < p_swop:
                    candidate = self.swop(candidate)
                next_generation += [candidate]

            n_keep = n_top + n_perturb
            
            for j in range((population_size - n_keep)//2):
                parents = self.selection(
                    population, self.task.fitness, 
                    method='tournament'
                )
                if random.random() < 0.9:
                    offspring_a, offspring_b = self.partial_map_crossover(parents[0], parents[1])
                else:
                    offspring_a, offspring_b = parents[0], parents[1]
                if random.random() < 0.9:
                    offspring_a = self.swop(offspring_a)
                    offspring_b = self.swop(offspring_b)
                next_generation += [offspring_a, offspring_b]
            population = next_generation
        
        best_genome = population[0]
        return best_genome

    
    def selection(self, population, fitness_func, method='tournament'):
        if method == 'tournament':
            k = min(5, int(0.02*len(population)))
            sub_population1 = random.choices(
                population=population, k=k
            )
            sub_population2 = random.choices(
                population=population, k=k
            )
            return (
                sorted(sub_population1, key=fitness_func, reverse=True)[0], 
                sorted(sub_population2, key=fitness_func, reverse=True)[0]
            )        
        else: # roulette wheel
            min_fitness = min([fitness_func(gene) for gene in population])
            selected = random.choices(
                population=population,
                weights=[fitness_func(gene)+eps-min_fitness for gene in population],
                k=2
            )
            return tuple(selected)
    
    
    def shift_to_end(self, genome, num=1):
        new_genome = deepcopy(genome)
        for _ in range(num):
            a = random.sample(range(len(genome)), k=1)[0]
            ref = deepcopy(new_genome[a])
            if random.random() < 0.5:
                new_genome[1:a+1] = new_genome[:a]
                new_genome[0] = ref   # bring to first
            else:
                new_genome[a:-1] = new_genome[a+1:]
                new_genome[-1] = ref   # bring to last
        return new_genome

    
    def swop(self, genome, num=1):
        new_genome = deepcopy(genome)
        for _ in range(num):
            a, b = random.sample(range(len(genome)), k=2)
            new_genome[a], new_genome[b] = genome[b], genome[a]
        return new_genome


class Salesman:
    def __init__(self, num_cities, x_lim, y_lim, read_from_txt=None):
        if read_from_txt:
            self.city_locations = []
            f = open(read_from_txt)
            for i, line in enumerate(f.readlines()):
                if i==num_cities:
                    break
                node_val = line.split()
                self.city_locations.append(
                    (float(node_val[-2]), float(node_val[-1]))
                )
            self.num_cities = len(self.city_locations)
            self.x_lim = np.max(np.array(self.city_locations)[:,0])
            self.y_lim = np.max(np.array(self.city_locations)[:,1])
        
        else:   # generate randomly
            self.num_cities = num_cities
            self.x_lim = x_lim
            self.y_lim = y_lim
            x_loc = np.random.uniform(0, x_lim, size=num_cities)
            y_loc = np.random.uniform(0, y_lim, size=num_cities)
            self.city_locations = [
                (x,y) for x,y in zip(x_loc,y_loc)
            ]
        self.distances = self.calculate_distances()
    
    
    def calculate_distances(self):
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.sqrt((self.city_locations[i][0] - self.city_locations[j][0]) ** 2 + (self.city_locations[i][1] - self.city_locations[j][1]) ** 2)
                distances[i][j] = distances[j][i] = dist
        return distances

    
    def fitness(self, solution):
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distances[solution[i]][solution[i+1]]
        total_distance += self.distances[solution[self.num_cities - 1]][solution[0]]
        fitness = -total_distance
        return fitness
    

    def visualize(self, solution, save_id=None):
        n = len(solution)
        assert n == len(self.city_locations), 'The solution must correspond to all cities'
        for i, (x,y) in enumerate(self.city_locations):
            plt.plot(x, y, "ro")
            plt.annotate(i, (x, y))
        
        ordered_cities = [self.city_locations[idx] for idx in solution]
        x_coord = [x for (x,y) in  ordered_cities]
        y_coord = [y for (x,y) in  ordered_cities]
        distance = -self.fitness(solution)
        
        plt.plot(x_coord, y_coord, "gray")
        plt.title("Connected cities (%.1f) according to solution" % distance)
        if save_id is not None:
            filename = "results/plot_%03d.png" % save_id
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            

salesman = Salesman(
    num_cities=10, x_lim=100, y_lim=100, read_from_txt='city_locations.txt'
)
evo = Evolutionary(salesman)
best_genome = evo.run_evolution(
    population_size=200, generation_limit=1000, crossover='pmx', verbose=True
)
salesman.visualize(best_genome)
