import pandas as pd

# get ratings original file
ratings = pd.read_csv("../../ML_Dataset/ml-latest/ratings.csv")

# getting rating column
rating_list = ratings.iloc[:, 2]

# Ratings Normalization from 1-5 to 0.1-1
new_rating_list = pd.array([])
new_rating_list.values = rating_list.values / 5

# join dataframes
df = pd.DataFrame(ratings.loc[:, ["userId", "movieId"]])
df2 = pd.DataFrame(
    data=new_rating_list.values,
    index=range(new_rating_list.values.size),
    columns=["rating"],
)
df = df.join(df2)
df.to_csv("../../ML_Dataset/ml-latest/ratings_norm.csv")
