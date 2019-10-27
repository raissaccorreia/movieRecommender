import math
import numpy as np
import sklearn

# devendo similaridade entre feature para o soft cosine simil


def cos_simil(X, Y):
    return sklearn.metrics.pairwise.cosine_similarity(X, Y, dense_output=True)


def euclidean_dist(v1, v2):
    dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


def mse(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()
