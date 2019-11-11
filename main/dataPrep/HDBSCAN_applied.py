import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan

movies = pd.read_csv("../../ML_Dataset/ml-latest-small/transformed_movies.csv")
users = pd.read_csv("../../ML_Dataset/ml-latest-small/transformed_users.csv")

clusterer_movies = hdbscan.HDBSCAN(
    algorithm="best",
    alpha=1.0,
    approx_min_span_tree=True,
    gen_min_span_tree=False,
    leaf_size=40,
    metric="euclidean",
    min_cluster_size=5,
    min_samples=5,
    cluster_selection_method="leaf",
    p=None,
)

print(clusterer_movies.fit(movies))

np.savetxt("fit_HDBSCAN_movies.txt", clusterer_movies.fit_predict(movies))
print(
    "Outline Rate in Movies:",
    np.count_nonzero(clusterer_movies.fit_predict(movies) == -1) / 2000,
)

clusterer_users = hdbscan.HDBSCAN(
    algorithm="best",
    alpha=1.0,
    approx_min_span_tree=True,
    gen_min_span_tree=False,
    leaf_size=40,
    metric="correlation",
    min_cluster_size=2,
    min_samples=2,
    cluster_selection_method="leaf",
    p=None,
)

print(clusterer_users.fit(users))
np.savetxt("fit_HDBSCAN_users.txt", clusterer_users.fit_predict(users))

print(
    "Outline Rate in Users:",
    np.count_nonzero(clusterer_users.fit_predict(users) == -1) / 589,
)
