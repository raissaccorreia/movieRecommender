import numpy as np
import pandas as pd

user_prof = pd.read_csv("../../ML_Dataset/ml-latest-small/user_profile.csv")
movie_prof = pd.read_csv("../../ML_Dataset/ml-latest-small/movie_profiles.csv")

# user_prof.drop(labels=["Unnamed: 0"], axis=1, inplace=True)
user_prof.to_csv(
    index=False, path_or_buf="../../ML_Dataset/ml-latest-small/user_profile.csv"
)

# movie_prof.drop(labels=["Unnamed: 0"], axis=1, inplace=True)
movie_prof.to_csv(
    index=False, path_or_buf="../../ML_Dataset/ml-latest-small/movie_profiles.csv"
)

for col in user_prof.columns:
    print(col)

for col in movie_prof.columns:
    print(col)
