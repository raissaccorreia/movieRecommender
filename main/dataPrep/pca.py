# SOURCES:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://scikit-learn.org/stable/modules/decomposition.html#pca

# plot
# https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py
# Explicacao detalhada
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

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
print("Original shape:   ", movie_prof.shape)
print("PCA FIT Tranform", pca.fit_transform(movie_prof))
transformed_movies = pd.DataFrame(data=pca.fit_transform(movie_prof))
transformed_movies.to_csv(
    index=False, path_or_buf="../../ML_Dataset/ml-latest-small/transformed_movies.csv"
)

print("Expained Variance Ratio: ", pca.explained_variance_ratio_)
print(
    pd.DataFrame(pca.components_, columns=genre_dict.keys(), index=list(range(0, 18)))
)
relevance_features = pca.explained_variance_ratio_.cumsum()
index = np.arange(len(relevance_features))
plt.plot(index, relevance_features, linewidth=2.0)
plt.xlabel("Genre Features", fontsize=5)
plt.ylabel("Explained Variance Ratio", fontsize=5)
plt.xticks(index, list(range(0, 18)), fontsize=5, rotation=30)
plt.title("Explained Variance Ratio for Movie Profiles")
plt.show()

print("Singular Values: ", pca.singular_values_)
print("Get Covariance: ", pca.get_covariance())
print("Get Params: ", pca.get_params(deep=True))
print("Get Precision: ", pca.get_precision())
print("Score: ", pca.score(movie_prof, y=None))
print("N Components:", pca.n_components)
print("N Features:", pca.n_features_)
print("Transformed Shape:", movie_prof.shape)

print("\n")

print("USER PROFILES INFO\n")
print("PCA FIT: ", pca.fit(user_prof))
print("Original shape:   ", user_prof.shape)
print("PCA FIT Tranform", pca.fit_transform(user_prof))

transformed_users = pd.DataFrame(data=pca.fit_transform(user_prof))
transformed_users.to_csv(
    index=False, path_or_buf="../../ML_Dataset/ml-latest-small/transformed_users.csv"
)

print("Expained Variance Ratio: ", pca.explained_variance_ratio_)
print(
    pd.DataFrame(pca.components_, columns=genre_dict.keys(), index=list(range(0, 18)))
)
relevance_features = pca.explained_variance_ratio_.cumsum()
index = np.arange(len(relevance_features))
plt.plot(index, relevance_features, linewidth=2.0)
plt.xlabel("Genre Features", fontsize=5)
plt.ylabel("Explained Variance Ratio", fontsize=5)
plt.xticks(index, list(range(0, 18)), fontsize=5, rotation=30)
plt.title("Explained Variance Ratio for User Profiles")
plt.show()

print("Singular Values: ", pca.singular_values_)
print("Get Covariance: ", pca.get_covariance())
print("Get Params: ", pca.get_params(deep=True))
print("Get Precision: ", pca.get_precision())
print("Score: ", pca.score(user_prof, y=None))
print("N Components:", pca.n_components)
print("N Features:", pca.n_features_)
print("Transformed Shape:", movie_prof.shape)

# https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn

# https://pypi.org/project/hdbscan/

