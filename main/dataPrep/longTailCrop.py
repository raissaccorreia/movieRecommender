import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import operator

# * https://datatofish.com/plot-dataframe-pandas/
# * https://python-graph-gallery.com/124-spaghetti-plot/

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

# fazer a lista de filmes e usuarios que vao ficar
# via merge/join criar novo ratings.csv e movies.csv

# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.truncate.html
# * https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
print(df_movie)
print(df_user)
movies = pd.read_csv("../../ML_Dataset/ml-latest-small/movies.csv")

