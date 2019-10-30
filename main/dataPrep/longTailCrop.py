import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import operator

# * userId,movieId,rating,timestamp
ratings = pd.read_csv("../../ML_Dataset/ml-latest-small/ratings.csv")

users_ratings = {}
movies_ratings = {}

# creating distribution of ratings among users and movies dictionary
for i in tqdm(range(ratings.iloc[:, 0].values.size)):
    user = ratings.iloc[i, 0]
    movie = ratings.iloc[i, 1]

    if user not in users_ratings.keys():
        users_ratings[user] = 1
    else:
        users_ratings[user] += 1

    if movie not in movies_ratings.keys():
        movies_ratings[movie] = 1
    else:
        movies_ratings[movie] += 1

users_ratings = sorted(users_ratings.items(), key=operator.itemgetter(1), reverse=True)
movies_ratings = sorted(
    movies_ratings.items(), key=operator.itemgetter(1), reverse=True
)

df_user = pd.DataFrame(data=users_ratings, columns=["users", "ratings"])
df_movie = pd.DataFrame(data=movies_ratings, columns=["movies", "ratings"])
# display scatter plot data
df_user.plot(
    y="ratings",
    marker="o",
    markerfacecolor="blue",
    markersize=1,
    color="blue",
    linewidth=1,
    label="users",
)
df_movie.plot(
    y="ratings",
    marker="o",
    markerfacecolor="red",
    markersize=1,
    color="red",
    linewidth=1,
    label="movies",
)
# plt.show()

movies = pd.read_csv("../../ML_Dataset/ml-latest-small/movies.csv")

# * user with at least 128 ratings
df_user = df_user.truncate(before=0, after=200, axis=0)
# * movie with at least 11 ratings
df_movie = df_movie.truncate(before=0, after=2000, axis=0)

new_movies = pd.merge(
    df_movie,
    movies,
    how="left",
    on=None,
    left_on="movies",
    right_on="movieId",
    left_index=False,
    right_index=False,
    sort=True,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
)
new_movies = new_movies.drop(
    labels="movies",
    axis=1,
    index=None,
    columns=None,
    level=None,
    inplace=False,
    errors="raise",
)

new_movies.to_csv("../../ML_Dataset/ml-latest-small/relevant_movies.csv")

new_users = []
for i in range(df_user.iloc[:, 0].values.size):
    new_users.append(df_user.iloc[i, 0])

ratings.drop(
    labels="timestamp",
    axis=1,
    index=None,
    columns=None,
    level=None,
    inplace=True,
    errors="raise",
)

for i in tqdm(range(88580)):
    user = ratings.iloc[i, 0]
    if user not in new_users:
        ratings.drop(
            labels=None,
            axis=0,
            index=i,
            columns=None,
            level=None,
            inplace=True,
            errors="raise",
        )

ratings.to_csv("../../ML_Dataset/ml-latest-small/relevant_ratings.csv")

# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.truncate.html
# * https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
# * https://datatofish.com/plot-dataframe-pandas/
# * https://python-graph-gallery.com/124-spaghetti-plot/
