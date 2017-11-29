import numpy as np
import time
import copy
import gym

# Implementar Roulette Wheel Selection
# Implementar melhor _get_best_indv


class Individual(object):
    """Represents an Individual
       
       Attributes:
           chromosome: Array size [200][4] representing it's actions
           fitness: Fitness value"""

    def __init__(self):
        self._chromosome = []
        self._fitness = 0

    def getChromosome(self):
        return self._chromosome

    def getGene(self, i):
        return self._chromosome[i]
    
    def getFitness(self):
        return self._fitness

    def setGenes(self, gene, i):
        self._chromosome[i] = gene

    def setActions(self):
        self._chromosome = self._generate_actions()
   
    def setFitness(self, fitness):
        self._fitness = fitness

    def _generate_actions(self):
        return np.random.rand(200, 4) * 2 - 1




class Population(object):
    """Represents a Population of Individuals

       Attributes:
           size: size of the population
           population: A list of Individual-type Objects"""

    def __init__(self, size):
        self._size = size
        self._population = []
        self._init_population()
    
    def _init_population(self): 
        for i in range(self._size):
            self._population.append(Individual())
            self._population[i].setActions()

    def getPopulation(self):
        return self._population

    def get_best_indv(self):
        index = np.zeros(self._size, dtype=np.int_)
        index[::-1] = np.asarray(self.get_pop_fitness(), dtype=np.int_).argsort()
        return self._population[index[0]]

    def get_pop_fitness(self):
        return [self._population[_].getFitness() for _ in range(self._size)]

    def get_max_fitness(self):
        best_indv = self.get_best_indv()
        return best_indv.getFitness()

    def get_avg_fitness(self):
        x = []
        [x.append(self._population[_].getFitness()) for _ in range(self._size)]
        return np.average(x)

    def modify_population(self, elite, offs, i):
        self._population[:i] = elite
        self._population[i:] = offs

class Environment(object):
    """Represents the environment that the population will interact.
       Attributes:
           env: generates the environment by OpenAI Gym
           generation: generation counter for the GA
           pop_size: size of the population
           pop: Population-type Object
           score: maximum score to achieve
           start and end: time tracking"""

    def __init__(self, pop_size, score):
        self._env = gym.make('CartPole-v0')
        self._generation = 0
        self._pop_size = pop_size
        self._pop = Population(self._pop_size)
        self._score = score
        self._start = time.time()
        self._end = 0

    def start(self):
        while self._pop.get_max_fitness() < self._score:
            self._generation += 1
            [self._evaluate_fitness(self._pop.getPopulation()[i]) for i in range(self._pop_size)]
            best_indv = self._pop.get_best_indv()
            print('Generation %d  -  Max Score = %.4f' %(self._generation,
                                               best_indv.getFitness()))
            if best_indv.getFitness() >= self._score: break
            elite = self._elitism()
            offsprings = self._crossover()
            [self._mutation(offspring) for offspring in offsprings]
            self._pop.modify_population(elite, offsprings, 10)
        self._end = time.time()
        print('Best Score = %.4f, Generation = %d, Time Taken = %4.4f' %(
          best_indv.getFitness(), self._generation, (self._end - self._start)))
        return best_indv

    def _evaluate_fitness(self, indv, render=False):
        total_reward = 0
        obs = self._env.reset()
        for i in range(self._score):
            if render: self._env.render()
            action = 0 if np.matmul(indv.getChromosome()[i], obs) < 0 else 1
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            if done: break
        indv.setFitness(total_reward)

    def _elitism(self):
        index = np.zeros(self._pop_size, dtype=np.int_)
        index[::-1] = np.asarray(self._pop.get_pop_fitness(), dtype=np.int_).argsort()
        elite = []
        [elite.append(self._pop.getPopulation()[_]) for _ in index]
        return elite[:10]

    def _crossover(self):
        index = self._selection()
        offsprings = []
        for _ in range(45):
            r_1, r_2 = np.random.choice(index, 2)
            p = self._pc_calc(r_1, r_2)
            offs_1 = copy.copy(self._pop.getPopulation()[r_1])
            offs_2 = copy.copy(self._pop.getPopulation()[r_2])
            if np.random.uniform() <= p:
                for i in range(self._score):
                    if np.random.uniform() <= 0.5:
                        aux = offs_1.getGene(i)
                        offs_2.setGenes(offs_1.getGene(i), i)
                        offs_1.setGenes(aux, i)
            offsprings.append(offs_1)
            offsprings.append(offs_2)
        return offsprings

    def _selection(self):
        prob = np.array(self._pop.get_pop_fitness()) / np.sum(self._pop.get_pop_fitness())
        index = []
        while len(index) < 45:
            index.append(np.random.choice(range(self._pop_size), p=prob))
        return index

    def _pc_calc(self, r_1, r_2):
        f = np.maximum(self._pop.getPopulation()[r_1].getFitness(),
                        self._pop.getPopulation()[r_2].getFitness())
        if f<= self._pop.get_avg_fitness():
            return 1.0 * (self._pop.get_max_fitness() - f) / (self._pop.
                           get_max_fitness() - self._pop.get_avg_fitness())
        else:
            return 1.0

    def _mutation(self, indv):
        p = 0.5
        new_indv = copy.copy(indv)
        for i in range(self._score):
           x = np.random.choice(4) * 2 - 1
           if np.random.uniform() <= p: new_indv.setGenes(x, i)
        return new_indv

    def render(self, indv):
        self._evaluate_fitness(indv, True)
            


if __name__ == '__main__':
    alg = Environment(100, 200)
    best = alg.start()
    alg.render(best)
