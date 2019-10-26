import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import math
import operator


def euclidean_dist(v1, v2):
    dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    for i in range(len(trainingSet)):
        dist = euclidean_dist(trainingSet[i], testInstance)
        distances.append([i, dist])
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])  #! prestar atencao na criacao dos neighbors
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]  #! definir adequadamente a response
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if x is predictions[x]:  #! como definir o correto para a acuracia
            correct += 1
    return (correct / float(len(testSet))) * 100.0


trainingDf = pd.read_csv("../ML_Dataset/ml-latest-small/training_movies.csv")
testDf = pd.read_csv("../ML_Dataset/ml-latest-small/test_movies.csv")
trainingSet = []
testSet = []

for i in tqdm(range(1, trainingDf.iloc[:, 0].values.size)):
    trainingSet.append(trainingDf.iloc[i, 1:].values)
for i in tqdm(range(1, testDf.iloc[:, 0].values.size)):
    testSet.append(testDf.iloc[i, 1:].values)


k = 5
predictions = []
for i in tqdm(range(len(testSet))):
    neighbors = getNeighbors(trainingSet, testSet[i], k)
    response = getResponse(neighbors)
    predictions.append(response)
accuracy = getAccuracy(testSet, predictions)
print("Accuracy: " + repr(accuracy) + "%")
