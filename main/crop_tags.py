import pandas as pd

# get scores original file
scores = pd.read_csv("../ML_Dataset/ml-latest/genome-scores.csv")

# eliminating too tiny scores
high_relevance = scores.drop(
    scores[scores["relevance"] < 0.05].index, axis=0, inplace=False
)

high_relevance.to_csv("../ML_Dataset/ml-latest/cropped_tags.csv")
