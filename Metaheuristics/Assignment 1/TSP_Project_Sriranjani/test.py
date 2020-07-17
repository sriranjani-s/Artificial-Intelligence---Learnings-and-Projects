#PMX Crossover
from random import randint
import random

indA = ['5','3','1','2','4']
indB = ['3','2','4','1','5']
mapdictA = {}
mapdictB = {}
childA = []
childB = []
k = 0

randA = random.randint(0, len(indA)-1)
randB = random.randint(0, len(indB)-1)

#indexA = min(randA,randB)
#indexB = max(randA,randB)
indexA = 2
indexB = 4
print(indexA,indexB)

#tmpA = [indB[i] for i in range(0,len(indB)) if(i >= indexA and i <= indexB)]
#tmpB = [indA[i] for i in range(0,len(indA)) if(i >= indexA and i <= indexB)]
tmpA = ['4','1','5']
tmpB = ['1','2','4']
print(tmpA,tmpB)

j = 0
for i in range(0,len(tmpA)):
    if tmpB[i] in tmpA:
        temp = tmpB[i]
        while j in range(0,len(tmpA)):
            if tmpA[j] == temp:
                if tmpB[j] not in tmpA:
                    mapdictA[tmpA[i]] = tmpB[j]
                    j = 0
                    break
                else:
                    temp = tmpB[j]
                    j=0
            else:
                j += 1
    else:
        mapdictA[tmpA[i]] = tmpB[i]
print(mapdictA)

j = 0
for i in range(0,len(tmpB)):
    if tmpA[i] in tmpB:
        temp = tmpA[i]
        while j in range(0,len(tmpB)):
            if tmpB[j] == temp:
                if tmpA[j] not in tmpB:
                    mapdictB[tmpB[i]] = tmpA[j]
                    j = 0
                    break
                else:
                    temp = tmpA[j]
                    j=0
            else:
                j += 1
    else:
        mapdictB[tmpB[i]] = tmpA[i]
print(mapdictB)

j = 0
for i in range(0,len(indA)):
    if i >= indexA and i <= indexB:
        childA.append(tmpA[j])
        childB.append(tmpB[j])
        j += 1
    else:
        if indA[i] in tmpA:
            childA.append(mapdictA[indA[i]])
        else:
            childA.append(indA[i])
        if indB[i] in tmpB:
            childB.append(mapdictB[indB[i]])
        else:
            childB.append(indB[i])
print(childA,childB)