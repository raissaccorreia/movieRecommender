import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import math

# * 0 or 1 relation movie-gender each line a movie
movie_vectors = pd.read_csv("../ML_Dataset/ml-latest-small/movie_profiles.csv")


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    for i in tqdm(range(1, movie_vectors.iloc[:, 0].values.size)):
        if random.random() < split:
            trainingSet.append(movie_vectors.iloc[i, 1:].values)
        else:
            testSet.append(movie_vectors.iloc[i, 1:].values)
    return trainingSet, testSet


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
        neighbors.append(distances[x][1])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


trainingSet, testSet = loadDataset("movie_vectors", 0.70)
k = 5
predictions = []
for i in tqdm(range(len(testSet))):
    neighbors = getNeighbors(trainingSet, testSet[i], k)
    print(neighbors)
    response = getResponse(neighbors)
    predictions.append(response)
accuracy = getAccuracy(testSet, predictions)
print("Accuracy: " + repr(accuracy) + "%")
