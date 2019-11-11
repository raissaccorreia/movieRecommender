import pandas as pd
import numpy as np
from tqdm import tqdm
import random

movies = pd.read_csv("../../ML_Dataset/ml-latest-small/transformed_movies.csv")
users = pd.read_csv("../../ML_Dataset/ml-latest-small/transformed_users.csv")

movies_test = movies.sample(frac=0.3)
movies_train = movies.sample(frac=0.7)
users_test = users.sample(frac=0.3)
users_train = users.sample(frac=0.7)

print("Train Shape and Test Shape from Movies: ", movies_train.shape, movies_test.shape)
print("Train Shape and Test Shape from Users: ", users_train.shape, users_test.shape)

movies_test.to_csv(
    index=False, path_or_buf="../../ML_Dataset/ml-latest-small/movies_test.csv"
)
movies_train.to_csv(
    index=False, path_or_buf="../../ML_Dataset/ml-latest-small/movies_train.csv"
)
users_test.to_csv(
    index=False, path_or_buf="../../ML_Dataset/ml-latest-small/users_test.csv"
)
users_train.to_csv(
    index=False, path_or_buf="../../ML_Dataset/ml-latest-small/users_train.csv"
)
