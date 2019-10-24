import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import math

# * 0 or 1 relation movie-gender each line a movie
movie_vectors = pd.read_csv("../ML_Dataset/ml-latest-small/movie_profiles.csv")

genre_dict = {
    "Action": 0,
    "Adventure": 1,
    "Animation": 2,
    "Children": 3,
    "Comedy": 4,
    "Crime": 5,
    "Documentary": 6,
    "Drama": 7,
    "Fantasy": 8,
    "Film-Noir": 9,
    "Horror": 10,
    "Musical": 11,
    "Mystery": 12,
    "Romance": 13,
    "Sci-Fi": 14,
    "Thriller": 15,
    "War": 16,
    "Western": 17,
    "IMAX": 18,
    "(no genres listed)": 19,
}


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    trainingSet_line = []
    testSet_line = []
    for i in tqdm(range(movie_vectors.iloc[:, 0].values.size)):
        for j in range(len(genre_dict)):
            if random.random() < split:
                trainingSet_line.append(movie_vectors.iloc[i, j])
            else:
                testSet_line.append(movie_vectors.iloc[i, j])
        trainingSet.append(trainingSet_line)
        testSet.append(testSet_line)
        trainingSet_line = []
        testSet_line = []
    return np.array(trainingSet), np.array(testSet)


def euclidean_dist(v1, v2):
    dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    for i in tqdm(range(len(trainingSet))):
        this_trainSet = np.array(trainingSet[i])
        this_testInstance = np.array(testInstance)
        dist = euclidean_dist(this_trainSet, this_testInstance)
        distances.append([i, dist])
    distances.sort()
    print(distances)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


trainingSet, testSet = loadDataset("movie_vectors", 0.70)
trainingSet = np.delete(trainingSet, 0, axis=1)
testSet = np.delete(testSet, 0, axis=1)
k = 3
neighbors = getNeighbors(trainingSet, testSet[1], k)
