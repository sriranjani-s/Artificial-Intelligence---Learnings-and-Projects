
"""
Author: Alejandro Arbelaez (Alejandro.Arbelaez@cit.ie)
Monkey example
file: Individual.py
"""


import random

class Individual:
    def __init__(self, num):
        """
        Parameters and general variables
        """
        self.fitness = -1
        self.genes = []
        self.genSize = num
        for i in range(0, num):
            self.genes.append( chr(random.randint(32, 128)) )

    def getPhrase(self):
        """
        String representation of the individual
        """
        return "".join(str(x) for x in self.genes)

    def getFitness(self):
        return self.fitness

    def computeFitness(self, target):
        """
        Calculating the fitness of the current individual
        % of correct symbols
        """
        score = 0.0
        for  i in range(0, len(self.genes)):
            if self.genes[i] == target[i]:
                score+=1
        self.fitness = score/len(target)
