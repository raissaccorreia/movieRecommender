import pandas as pd
import csv

ratings = pd.read_csv("../ML_Dataset/ml-latest-small/ratings.csv")
ratings = ratings.loc[:, ["userId", "movieId", "rating"]]

# Select column by index position using iloc[]
rating_list = ratings.iloc[:, 2]
print("Ratings list size: ", rating_list.values.size)

# Ratings Normalization from 1-5 to 0.1-1
new_rating_list = pd.array
new_rating_list.values = rating_list.values / 5
print("Column Contents : ", new_rating_list.values)

