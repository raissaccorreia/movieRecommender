import pandas as pd
import random
from tqdm import tqdm

# * 0 or 1 relation movie-gender each line a movie
movie_vectors = pd.read_csv("../ML_Dataset/ml-latest-small/movie_profiles.csv")

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


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    for i in tqdm(range(1, movie_vectors.iloc[:, 0].values.size)):
        if random.random() < split:
            trainingSet.append(movie_vectors.iloc[i, 1:].values)
        else:
            testSet.append(movie_vectors.iloc[i, 1:].values)
    return trainingSet, testSet


trainingSet, testSet = loadDataset(movie_vectors, 0.7)
trainData = pd.DataFrame(data=trainingSet, columns=genre_dict, dtype=int)
testData = pd.DataFrame(data=testSet, columns=genre_dict, dtype=int)
trainData.to_csv("../ML_Dataset/ml-latest-small/training_movies.csv")
testData.to_csv("../ML_Dataset/ml-latest-small/test_movies.csv")
