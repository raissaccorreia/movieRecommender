import pandas as pd
import numpy as np
from tqdm import tqdm

# * userId,movieId,rating,timestamp
ratings = pd.read_csv("../../ML_Dataset/ml-latest-small/relevant_ratings.csv")
# * movieId,title,genres
movie_genres = pd.read_csv("../../ML_Dataset/ml-latest-small/relevant_movies.csv")

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
    "(no genres listed)": 19,
}

num_movies = 2000  # * due to long tail crop
genre_movie_matrix = np.zeros(shape=(num_movies, len(genre_dict)), dtype=int)

for i in tqdm(range(num_movies)):
    genre_list = movie_genres.iloc[i, 4]  # * genres list action|adventure
    genre_array = genre_list.split("|")
    for j in range(len(genre_array)):
        index_y = genre_dict[genre_array[j]]
        genre_movie_matrix[movie_genres.iloc[i, 0], index_y] = 1

genre_movie_df = pd.DataFrame(genre_movie_matrix)
movie_ids = pd.DataFrame(movie_genres.iloc[:, 2])
genre_movie_df.to_csv("../../ML_Dataset/ml-latest-small/movie_profiles.csv")

movie_id_list = movie_ids["movieId"].tolist()
movies_dict = {movie_id_list[i]: i for i in range(0, len(movie_id_list))}

user_profiles = pd.DataFrame(columns=genre_dict, dtype=float)
num_ratings_user = 0
first_user = False
last_user = False

num_ratings = 89721

for i in tqdm(range(num_ratings)):
    # * actual,previous, next, first and last to make the decisions
    if i == 0:
        first_user = True
    if i == (num_ratings) - 1:
        last_user = True
    if first_user:
        user_id = ratings.iloc[i, 1]
        next_user = ratings.iloc[i + 1, 1]
        relation_gender_user = np.zeros(len(genre_dict), dtype=float)
        num_ratings_user = 0
    elif last_user:
        prev_user = ratings.iloc[i - 1, 1]
        user_id = ratings.iloc[i, 1]
        next_user = user_id + 1
    else:
        prev_user = ratings.iloc[i - 1, 1]
        user_id = ratings.iloc[i, 1]
        next_user = ratings.iloc[i + 1, 1]
        if user_id > prev_user:
            relation_gender_user = np.zeros(len(genre_dict), dtype=float)
            num_ratings_user = 0
    # the same user
    movie_id = ratings.iloc[i, 2]
    try:
        movie_index = movies_dict[movie_id]
        relation_gender_user += genre_movie_df.loc[movie_index] * ratings.iloc[i, 3]
        num_ratings_user += 1
    except:
        pass
    # if its the last movie of this user
    if user_id < next_user:
        relation_gender_user = relation_gender_user * (1 / (num_ratings_user))
        user_profiles = pd.concat(
            [relation_gender_user, user_profiles], axis=1, ignore_index=True
        )
    # to end the loop
    first_user = False
    last_user = False

user_profiles = user_profiles.transpose()
user_profiles.to_csv("../../ML_Dataset/ml-latest-small/user_profile.csv")
