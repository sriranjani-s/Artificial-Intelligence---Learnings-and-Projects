
"""
Author: Alejandro Arbelaez (Alejandro.Arbelaez@cit.ie)
Monkey example
file: Example.py
"""
import random
#import os
from Individual import *

random.seed(12345)

class GA:
    def __init__(self, _mutation, _totalPopulation, _maxIterations, _target):
        """
        Parameters and general variables
        """
        self.mutationRate    = _mutation
        self.totalPopulation = _totalPopulation
        self.maxIterations   = _maxIterations
        self.target = _target

        self.population = []
        self.matingPool = []
        self.best = None
        self.iteration = 0
        self.printPopulation = False
        self.genSize = len(self.target)

        ##Init population
        for i in range(0, self.totalPopulation):
            self.population.append(Individual(self.genSize))
        self.best = self.population[0]


    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Selection
        2. Crossover
        3. Mutation

        Not including survival functionality
        """
        for gene_i in self.population:
            gene_i.computeFitness(self.target)

        ##Naive mating pool
        matingPool = []
        for gene_i in self.population:
            elementsInPool = int(gene_i.getFitness() * 100)
            #print gene_i.getPhrase()
            #print "elements: ",elementsInPool,gene_i.getFitness()
            for i in range(0, elementsInPool):
                matingPool.append(gene_i)

        ##New generation --> Replacing current population with a new one
#        for gene_i in range(0, len(self.population)):
        for gene_i in range(0, int(len(self.population)/2)):

            ##Random selection
            indexPartnerA = random.randint(0, len(matingPool)-1)
            indexPartnerB = random.randint(0, len(matingPool)-1)

            partnerA = matingPool[indexPartnerA]
            partnerB = matingPool[indexPartnerB]

            ##Crossover
 #           child = self.crossover(partnerA, partnerB)
            child1, child2 = self.crossover(partnerA, partnerB)

            ##Mutation
#            self.mutate(child)
            self.mutate(child1) 
            self.mutate(child2)


#            child.computeFitness(self.target)
            child1.computeFitness(self.target)
            child2.computeFitness(self.target)

#            self.population[gene_i] = child
            self.population[gene_i*2] = child1
            self.population[gene_i*2 +1] = child2

            if child1.getFitness() > self.best.getFitness():
                self.best = child1

                print ("Best so far =============")
                print ("Iteration: "+str(self.iteration))
                print ("Fitness: "+str(self.best.getPhrase()))
                print ("Cost: "+str(self.best.getFitness()))
                print ("=========================")
            elif child2.getFitness() > self.best.getFitness():
                self.best = child2

                print ("Best so far =============")
                print ("Iteration: "+str(self.iteration))
                print ("Fitness: "+str(self.best.getPhrase()))
                print ("Cost: "+str(self.best.getFitness()))
                print ("=========================")

    #Crossover
    def crossover(self, ind1, ind2):
        """
        Executes a two point crossover and returns a new individual
        :param ind1: The first parent (or individual)
        :param ind2: The second parent (or individual)
        :returns: A new individual
        """
#        child = Individual(self.genSize)
        child1 = Individual(self.genSize)
        child2 = Individual(self.genSize)

        #Two random points
        pt1 = random.randint(0, self.genSize)
        pt2 = random.randint(0, self.genSize)
        
        # Sort the random points
        if pt1>pt2:
            temp=pt2
            pt2=pt1
            pt1=temp

        for i in range(0, self.genSize):
#            if(i > midPoint):
            if(pt1 < i < pt2):
#                child.genes[i] = ind1.genes[i]
                child1.genes[i] = ind1.genes[i]
                child2.genes[i] = ind2.genes[i]
            else:
#                child.genes[i] = ind2.genes[i]
                child1.genes[i] = ind2.genes[i]
                child2.genes[i] = ind1.genes[i]
#        return child
        return child1, child2

    def mutate(self, ind):
        """
        Mutate and individual by replacing genes with certain probability (i.e., mutation rate)
        uniform random number between 32 and 128 --> ASCII codification
        :param ind: An individual
        :return: A new individual
        """
        for i in range(0, self.genSize):
            if(random.random() < self.mutationRate):
                ind.genes[i] = chr(random.randint(32, 128))

    def search(self):
        """
        General search template.
        Iterates until reaching a solution or after a given number of iterations
        """
        self.iteration = 0
        while self.iteration < self.maxIterations and self.best.getFitness() < 1:
            self.GAStep()
            self.iteration +=1
        print ("i: ",self.iteration)


#ga = GA(0.01, 1000, 5000, "to be or not to be")
ga = GA(0.01, 1000, 5000, "Diarmuid O'Greachain is ainm dom")
ga.search()