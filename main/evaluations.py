import math
import numpy as np
import sklearn

def rmse(y_true, y_pred):
    return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())

def mae(y_true, y_pred, sample_weight):
    return sklearn.metrics.mean_absolute_error(y_true, y_pred, sample_weight, multioutput=’uniform_average’)

def f1(hit_set_size, test_set_size, N):
    recall = hit_set_size / test_set_size
    precision = hit_set_size / N
    return 2 * recall * precision / (recall + precision)

def ranking_dist(topN):
     return sklearn.metrics.pairwise.cosine_similarity(topN, Y=None, dense_output=True)

#def long_tail_plot():
#plotar os filmes e seu numero de ratings
#plotar os topn e seu numero de ratings