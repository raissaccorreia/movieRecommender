# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

# PCA in movie_profiles and user_profiles
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# * userId,movieId,rating,timestamp
user_prof = pd.read_csv("../../ML_Dataset/ml-latest-small/user_profile.csv")
user_prof = np.array(user_prof)
# * movieId,title,genres
movie_prof = pd.read_csv("../../ML_Dataset/ml-latest-small/movie_profiles.csv")
movie_prof = np.array(movie_prof)

pca = PCA(
    n_components="mle",
    copy=True,
    whiten=False,
    svd_solver="auto",
    tol=0.0,
    iterated_power="auto",
    random_state=None,
)

# https://scikit-learn.org/stable/modules/decomposition.html#pca

