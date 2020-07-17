

"""
Author: Sriranjani Sridharan
file: TSP_R00182510.py
"""

import random
from Individual_R00182510 import *
import sys

myStudentNum = 182510 # Replace 12345 with your student number
random.seed(myStudentNum)

class TSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations,_initial_soln,_selection,_crosssover_type,_mutation_type):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}
        self.initial_soln   = _initial_soln
        self.selection      = _selection
        self.crossover_type = _crosssover_type
        self.mutation_type  = _mutation_type

        self.readInstance()

        if self.initial_soln == 'Rand':
            self.initPopulation()
        elif self.initial_soln == 'Heuristic':
            self.heuristicPopulation()
        else:
            print('Incorrect Input - Enter a valid Initial Solution')


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ",self.best.getFitness())

    def heuristicPopulation(self):
        """
        Creating heuristic solution based individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            genes_old = individual.genes
            ind = random.randint(0, self.genSize-1)

            genes_new = [genes_old[ind]]
            del genes_old[ind]

            current_gene = genes_new[0]
            while(len(genes_old) > 0):
                Agene = genes_old[0]
                Acost = individual.euclideanDistance(current_gene,Agene)
                Aindex = 0
                for i in range(1,len(genes_old)):
                    gene = genes_old[i]
                    cost = individual.euclideanDistance(current_gene,gene)
                    if Acost > cost:
                        Acost = cost
                        Agene = gene
                        Aindex = i
                current_gene = Agene
                genes_new.append(current_gene)
                del genes_old[Aindex]

            individual.setGene(genes_new)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print("Best initial sol: ", self.best.getFitness())

    def updateBest(self, candidate):
        """
        print the updated best results as iterations process
        """
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def stochasticUniversalSampling(self):
        """
        Stochastic universal sampling Selection Implementation
        """

        fitness = []
        indices = []
        fitness_range = [0]
        item = 0

        select_num = self.popSize

        for ind_i in self.matingPool:
            fitness.append(ind_i.getFitness())

        fitness = [1/x for x in fitness]
        fitness_sum = sum(fitness)
        fitness_transformed = [x / fitness_sum for x in fitness]
        fitness_transformed_sum = sum(fitness_transformed)


        for i in range(len(fitness_transformed)):
            item += fitness_transformed[i]
            fitness_range.append(item)

        distance = fitness_transformed_sum / select_num
        pointer = random.uniform(0, distance)
        current_point = pointer

        for i in range(select_num):
            for j in range(0, len(fitness_range)):
                if current_point <= fitness_range[j]:
                    indices.append(j-1)
                    break
            current_point += pointer

        indA = self.matingPool[random.choice(indices)]
        indB = self.matingPool[random.choice(indices)]

        return [indA, indB]

    def uniformCrossover(self, indA, indB):
        """
        Uniform Crossover Implementation
        """
        childA = []
        childB = []
        j = 0
        k = 0

        size = random.randint(0, self.genSize-1)
        indices = (random.sample(range(0, self.genSize-1), size))
        indices.sort()

        tmpA = [indA.genes[i] for i in indices]
        tmpB = [indB.genes[i] for i in indices]

        auxA = [i for i in indB.genes if i not in tmpA]
        auxB = [i for i in indA.genes if i not in tmpB]

        for i in range(0, self.genSize):
            if i in indices:
                childA.append(tmpA[j])
                childB.append(tmpB[j])
                j += 1
            else:
                childA.append(auxA[k])
                childB.append(auxB[k])
                k += 1
        return childA,childB

    def PMXmap(self,num,tmp1,tmp2):
        """
        Mapping of randomly selected genes in PMX Crossover Implementation
        """
        for i in range(0, len(tmp1)):
            if num == tmp1[i]:
                temp = tmp2[i]
                if temp in tmp1:
                    return self.PMXmap(temp, tmp1, tmp2)
                else:
                    return tmp2[i]


    def pmxCrossover(self, indA, indB):
        """
        PMX Crossover Implementation
        """
        childA = []
        childB = []
        j = 0

        randA = random.randint(0, self.genSize-1)
        randB = random.randint(0, self.genSize-1)

        indexA = min(randA, randB)
        indexB = max(randA, randB)

        tmpA = [indB.genes[i] for i in range(0, self.genSize) if (i >= indexA and i <= indexB)]
        tmpB = [indA.genes[i] for i in range(0, self.genSize) if (i >= indexA and i <= indexB)]

        if indexA == 0 and indexB == self.genSize-1:
            childA = tmpA
        else:
            for i in range(0, self.genSize):
                if i >= indexA and i <= indexB:
                    childA.append(tmpA[j])
                    j += 1
                else:
                    if indA.genes[i] in tmpA:
                        mappedA = self.PMXmap(indA.genes[i],tmpA,tmpB)
                        childA.append(mappedA)
                    else:
                        childA.append(indA.genes[i])

        j = 0
        if indexA == 0 and indexB == self.genSize-1:
            childB = tmpB
        else:
            for i in range(0, self.genSize):
                if i >= indexA and i <= indexB:
                    childB.append(tmpB[j])
                    j += 1
                else:
                    if indB.genes[i] in tmpB:
                        mappedB = self.PMXmap(indB.genes[i],tmpB,tmpA)
                        childB.append(mappedB)
                    else:
                        childB.append(indB.genes[i])

        return childA, childB

    def reciprocalExchangeMutation(self, ind):
        """
        Reciprocal Exchange Mutation implementation
        Mutate an individual with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def inversionMutation(self, ind):
        """
        Inversion Mutation implementation
        Mutate an individual with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        tmp = []
        res = []
        j = 0

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        indA = min(indexA, indexB)
        indB = max(indexA, indexB)

        [tmp.append(ind.genes[i]) for i in range(self.genSize) if (i >= indA and i <= indB)]
        tmp = tmp[::-1]

        for i in range(0, self.genSize):
            if i >= indA and i <= indB:
                res.append(tmp[j])
                j += 1
            else:
                res.append(ind.genes[i])
        ind.setGene(res)

        ind.computeFitness()
        self.updateBest(ind)

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swapping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append( ind_i.copy() )

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        for i in range(0, int(len(self.population)/2),2):
            """
            Depending on the experiment the most suitable algorithms are used for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """

            #Selection of Parents
            if self.selection == 'Rand':
                parentA, parentB = self.randomSelection()
            elif self.selection == 'SUS':
                parentA, parentB = self.stochasticUniversalSampling()
            else:
                print('Incorrect Input - Enter a valid selection method')

            #Crossover
            if self.crossover_type == 'uniform':
                childA_l,childB_l = self.uniformCrossover(parentA, parentB)
            elif self.crossover_type == 'PMX':
                childA_l,childB_l = self.pmxCrossover(parentA,parentB)
            else:
                print('Incorrect Input - Enter a valid Crossover type')

            #Create new Individual-> child
            childA = Individual(self.genSize, self.data)
            childA.setGene(childA_l)
            childB = Individual(self.genSize, self.data)
            childB.setGene(childB_l)

            #Mutation
            if self.mutation_type == 'Inv':
                self.inversionMutation(childA)
                self.inversionMutation(childB)
            elif self.mutation_type == 'RecExc':
                self.reciprocalExchangeMutation(childA)
                self.reciprocalExchangeMutation(childB)
            else:
                print('Incorrect Input - Enter a valid Mutation type')

            childA.computeFitness()
            childB.computeFitness()
            self.population[i] = childA
            self.population[i+1] = childB

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        print ("Total iterations: ",self.iteration)
        print ("Best Solution: ", self.best.getFitness())

if len(sys.argv) < 5:
    print ("Error - Incorrect input")
    print ("Expecting python TSP_R00182510.py [instance] [population size] [mutation rate] [Iterations] ")
    sys.exit(0)

#Refer report to run required experiments for the below assignments
problem_file = sys.argv[1]
pop_size = int(sys.argv[2])
Mutation_rate = float(sys.argv[3])
Iteration = int(sys.argv[4])

print('Problem File : ',problem_file)
print('Population size : ',pop_size)
print('Mutation rate : ',Mutation_rate)
print('Iterations : ',Iteration)
print('SUS - num parents selected : popsize')

# To run Configurations 1 or 2 -> uncomment the code between the tags <BASIC>
#                               -> comment out the code between tags <EXTENSIVE>

# To run Configurations 3 to 8 (any) -> uncomment the code between tags <EXTENSIVE>
#                                    -> comment out the code between tags <BASIC>

#<BASIC>
print("\nConfiguration 1 - Random population, Random Selection, Uniform Crossover, Inversion mutation")
for i in range(5):
    ga1 = TSP(sys.argv[1], pop_size, Mutation_rate, Iteration, 'Rand', 'Rand', 'uniform', 'Inv')
    ga1.search()
    print("\n")

print("\nConfiguration 2 - Random population, Random Selection, PMX Crossover, Reciprocal Exchange mutation")
for i in range(5):
    ga2 = TSP(sys.argv[1], pop_size, Mutation_rate, Iteration, 'Rand', 'Rand', 'PMX', 'RecExc')
    ga2.search()
    print("\n")
#<BASIC>


# #<EXTENSIVE>
# print("\nConfiguration 3 - Random population, Stochastic Universal sampling, uniform crossover, Reciprocal Exchange mutation")
# for i in range(5):
#     ga3 = TSP(sys.argv[1], pop_size, Mutation_rate, Iteration, 'Rand', 'SUS', 'uniform', 'RecExc')
#     ga3.search()
#     print("\n")
#
# print("\nConfiguration 4 - Random population, Stochastic Universal sampling, PMX Crossover, Reciprocal Exchange mutation")
# for i in range(5):
#     ga4 = TSP(sys.argv[1], pop_size, Mutation_rate, Iteration, 'Rand', 'SUS', 'PMX', 'RecExc')
#     ga4.search()
#     print("\n")
#
# print("\nConfiguration 5 - Random population, Stochastic Universal sampling, PMX Crossover, Inversion mutation")
# for i in range(5):
#     ga5 = TSP(sys.argv[1], pop_size, Mutation_rate, Iteration, 'Rand', 'SUS', 'PMX', 'Inv')
#     ga5.search()
#     print("\n")
#
# print("\nConfiguration 6 - Random population, Stochastic Universal sampling, Uniform crossover, Inversion mutation")
# for i in range(5):
#     ga6 = TSP(sys.argv[1], pop_size, Mutation_rate, Iteration, 'Rand', 'SUS', 'uniform', 'Inv')
#     ga6.search()
#     print("\n")
#
# print("\nConfiguration 7 - Heuristic population, Stochastic Universal sampling, PMX crossover, Reciprocal Exchange")
# for i in range(5):
#     ga7 = TSP(sys.argv[1], pop_size, Mutation_rate, Iteration, 'Heuristic', 'SUS', 'PMX', 'RecExc')
#     ga7.search()
#     print("\n")
#
# print("\nConfiguration 8 -  Heuristic population, Stochastic Universal sampling, Uniform crossover, Inversion mutation")
# for i in range(5):
#     ga8 = TSP(sys.argv[1], pop_size, Mutation_rate, Iteration, 'Heuristic', 'SUS', 'uniform', 'Inv')
#     ga8.search()
#     print("\n")
# #<EXTENSIVE>