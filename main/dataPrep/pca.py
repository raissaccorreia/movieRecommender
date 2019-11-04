# SOURCES:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://scikit-learn.org/stable/modules/decomposition.html#pca

# plot
# https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py


# PCA in movie_profiles and user_profiles
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
}

# * userId,movieId,rating,timestamp
user_prof = pd.read_csv("../../ML_Dataset/ml-latest-small/user_profile.csv")
# * it was taken off the NAN lines with dropna() and the column 19 from user_profile.py
user_prof = np.array(user_prof.iloc[:, 1:])

# * movieId,title,genres
movie_prof = pd.read_csv("../../ML_Dataset/ml-latest-small/movie_profiles.csv")
# * it was taken off the column 19 from user_profile.py
movie_prof = np.array(movie_prof.iloc[:, 1:])

pca = PCA(
    n_components="mle",
    copy=True,
    whiten=False,
    svd_solver="auto",
    tol=0.0,
    iterated_power="auto",
    random_state=None,
)
print("MOVIE PROFILES INFO\n")
print("PCA FIT: ", pca.fit(movie_prof))
print("PCA FIT Tranform", pca.fit_transform(movie_prof))

print("Expained Variance Ratio: ", pca.explained_variance_ratio_)
relevance_features = pca.explained_variance_ratio_
index = np.arange(len(relevance_features))
plt.bar(index, relevance_features, log=True)
plt.xlabel("Genre Features", fontsize=5)
plt.ylabel("Explained Variance Ratio", fontsize=5)
plt.xticks(index, genre_dict.keys(), fontsize=5, rotation=30)
plt.title("Explained Variance Ratio for Movie Profiles")
plt.show()

print("Singular Values: ", pca.singular_values_)
print("Get Covariance: ", pca.get_covariance())
print("Get Params: ", pca.get_params(deep=True))
print("Get Precision: ", pca.get_precision())
print("Score: ", pca.score(movie_prof, y=None))
print("N Components:", pca.n_components)
print("N Features:", pca.n_features_)

print("\n")

print("USER PROFILES INFO\n")
print("PCA FIT: ", pca.fit(user_prof))
print("PCA FIT Tranform", pca.fit_transform(user_prof))

print("Expained Variance Ratio: ", pca.explained_variance_ratio_)
relevance_features = pca.explained_variance_ratio_
index = np.arange(len(relevance_features))
plt.bar(index, relevance_features, log=True)
plt.xlabel("Genre Features", fontsize=5)
plt.ylabel("Explained Variance Ratio", fontsize=5)
plt.xticks(index, genre_dict.keys(), fontsize=5, rotation=30)
plt.title("Explained Variance Ratio for User Profiles")
plt.show()

print("Singular Values: ", pca.singular_values_)
print("Get Covariance: ", pca.get_covariance())
print("Get Params: ", pca.get_params(deep=True))
print("Get Precision: ", pca.get_precision())
print("Score: ", pca.score(user_prof, y=None))
print("N Components:", pca.n_components)
print("N Features:", pca.n_features_)
