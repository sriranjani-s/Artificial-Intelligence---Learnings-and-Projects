
"""
Author: Alejandro Arbelaez (Alejandro.Arbelaez@cit.ie)
Monkey example
file: Example.py
"""
import random
import os
from Individual import *

class GA:
    def __init__(self, _mutation, _totalPopulation, _maxIterations, _target):
        """
        Parameters and general variables
        """
        self.mutationRate    = _mutation
        self.totalPopulation = _totalPopulation
        self.maxIterations   = _maxIterations
        self.target = "To be or not to be"

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
        for gene_i in range(0, len(self.population)):

            ##Random selection
            indexPartnerA = random.randint(0, len(matingPool)-1)
            indexPartnerB = random.randint(0, len(matingPool)-1)

            partnerA = matingPool[indexPartnerA]
            partnerB = matingPool[indexPartnerB]

            ##Crossover
            child = self.crossover(partnerA, partnerB)

            ##Mutation
            self.mutate(child)

            child.computeFitness(self.target)

            self.population[gene_i] = child

            if child.getFitness() > self.best.getFitness():
                self.best = child

                print ("Best so far =============")
                print ("Iteration: "+str(self.iteration))
                print ("Fitness: "+str(self.best.getPhrase()))
                print ("Cost: "+str(self.best.getFitness()))
                print ("=========================")

    #Crossover
    def crossover(self, ind1, ind2):
        """
        Executes a one point crossover and returns a new individual
        :param ind1: The first parent (or individual)
        :param ind2: The second parent (or individual)
        :returns: A new individual
        """
        child = Individual(self.genSize)

        #Random mid-point
        midPoint = random.randint(0, self.genSize)

        for i in range(0, self.genSize):
            if(i > midPoint):
                child.genes[i] = ind1.genes[i]
            else:
                child.genes[i] = ind2.genes[i]
        return child

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


ga = GA(0.01, 100, 5000, "to be or not to be")
ga.search()