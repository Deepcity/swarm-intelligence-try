import numpy as np
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt


n_top = 30
n_perturb = 30
p_shift = .1
p_swop = .2
eps = .1
p_up = .2
p_down = .2

class Evolutionary:
	def __init__(self,task):
		self.task = task

	def generate_genome(self,point_nums):
		genome = [[i,random.randint(0,point_nums[i]-1)] for i in range(0,len(point_nums))]
		return genome
	
	def generate_population(self, size, point_nums):
		population = [self.generate_genome(point_nums) for _ in range(size)]
		return population
	
	def partial_map_crossover(self, parent1, parent2):
		n = len(parent1)
		point = random.randint(0, n-1)
		child1 = list(parent1[0:point])
		for j in parent2:
			if any(j[0] == t[0] for t in child1) == False:
				child1.append(j)

		child2 = list(parent2[0:point])
		for j in parent1:
			if any(j[0] == t[0] for t in child2) == False:
				child2.append(j)

		return child1, child2
	
	def run_evolution(self, population_size, generation_times,fitness_limit=1e99, crossover='pmx', verbose=True):
		population = self.generate_population(population_size, self.task.point_nums)
		
		best_fitness_seen = -1e9
		for i in range(generation_times):
			population = sorted(
				population, key=lambda genome: self.task.fitness(genome), reverse=True
			)
			fitness = self.task.fitness(population[0])

			if verbose and (fitness > best_fitness_seen):
				best_fitness_seen = fitness
			if fitness >= fitness_limit:
				break

				# self.task.visualize(population[0],save_id = i)
			next_generation = population[:n_top] 

			# print(self.task.point_nums)
			for _ in range(n_perturb):
				candidate = random.choice(population[:n_top])
				# print(candidate)
				# print([len(t) for t in candidate])
				if np.random.random() < p_shift:
					candidate = self.shift_to_end(candidate)
				if np.random.random() < p_swop:
					candidate = self.swop(candidate)
					next_generation += [candidate]
				if np.random.random() < p_up:
					index = random.choice(range(len(candidate)))
					if candidate[index][1] < self.task.point_nums[candidate[index][0]] - 1:
						candidate[index][1] += 1
				if np.random.random() < p_down:
					index = random.choice(range(len(candidate)))
					if candidate[index][1] > 0:
						candidate[index][1] -= 1

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
		new_genome = copy.deepcopy(genome)
		for _ in range(num):
			a = random.sample(range(len(genome)), k=1)[0]
			ref = copy.deepcopy(new_genome[a])
			if random.random() < 0.5:
				new_genome[1:a+1] = new_genome[:a]
				new_genome[0] = ref   # bring to first
			else:
				new_genome[a:-1] = new_genome[a+1:]
				new_genome[-1] = ref   # bring to last
		return new_genome
	
	def swop(self, genome, num=1):
		new_genome = copy.deepcopy(genome)
		for _ in range(num):
			a, b = random.sample(range(len(genome)), k=2)
			new_genome[a], new_genome[b] = genome[b], genome[a]
		return new_genome
	

class Solution:
	def __init__(self, points):
		self.points = points
		self.point_nums = []
		for i in range(len(points)):
			self.point_nums.append(len(points[i]))
		self.x_lim = np.max([np.max(np.array(t)[:,0]) for t in points])
		self.y_lim = np.max([np.max(np.array(t)[:,0]) for t in points])

	def calculate_distance(self, pointa, pointb):
		return np.sqrt((pointa[0] - pointb[0]) ** 2 + (pointa[1] - pointb[1]) ** 2)

	def fitness(self, solution):
		total_distance = 0
		# print(solution)
		for i in range(1,len(solution)):
			total_distance += self.calculate_distance(self.points[solution[i-1][0]][solution[i-1][1]],
																						self.points[solution[i][0]][solution[i][1]])
		total_distance += self.calculate_distance(self.points[solution[len(solution)-1][0]][solution[len(solution)-1][1]],
																						self.points[solution[0][0]][solution[0][1]])
		fitness = -total_distance
		return fitness
	
	def visualize(self, solution, save_id = None):
		n = len(solution)
		assert n == len(self.point_nums), 'The solution must correspond to all cities'
		for profile in self.points:
			for i, (x,y) in enumerate(profile):
				plt.plot(x, y, "o")
				plt.annotate(i, (x, y))
        
		ordered_points= [self.points[idx][pidx] for (idx,pidx) in solution]
		ordered_points.append(self.points[solution[0][0]][solution[0][1]])
		x_coord = [x for (x,y) in  ordered_points]
		y_coord = [y for (x,y) in  ordered_points]
		distance = -self.fitness(solution)
        
		plt.plot(x_coord, y_coord, "gray")
		plt.title("Connected point (%.1f) according to solution" % distance)
		if save_id is not None:
			filename = "../results/plot_%03d.png" % save_id
			plt.savefig(filename, bbox_inches='tight')
			plt.close()
		else:
			plt.show()
  

points = [eval(group) for group in input().split('@')]
# print([len(t) for t in points])
printer = Solution(points)

evo = Evolutionary(printer)
best_genome = evo.run_evolution(
	population_size=100, generation_times=5000,crossover='pmx',verbose=True
)
# printer.visualize(best_genome)
print([points[idx][pidx] for (idx,pidx) in best_genome])