import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV

reader = Reader(
    line_format="user item rating timestamp", sep=",", rating_scale=(1, 5), skip_lines=1
)

data = Dataset.load_from_file(
    file_path="../../ML_Dataset/ml-latest-small/ratings.csv", reader=reader
)

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
gs = GridSearchCV(
    SVD, param_grid, measures=["rmse", "mae"], cv=3, return_train_measures=True
)
gs.fit(data)

# predict, test,fit?
# best RMSE score

print("Best Score\n", gs.best_score)
# combination of parameters that gave the best RMSE score
print("Best Params\n", gs.best_params)
print("Best Estimators\n", gs.best_estimator)
print("Best Index\n", gs.best_index)
print("Results Dicts: \n")
results_df = pd.DataFrame.from_dict(gs.cv_results)
print(results_df)

# define a cross-validation iterator
# kf = KFold(n_splits=3)

# We'll use the famous SVD algorithm.
# algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
# for trainset, testset in kf.split(data):
#    predictions = algo.fit(trainset).test(testset)
#    accuracy.rmse(predictions)

# Run 5-fold cross-validation and print results
# cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

# GET TOP-N OF ALL USERS, GET PRECISION AND RECALL(FAQ)


# CONTENT BASED LINKS:
# https://github.com/nikitaa30/Content-based-Recommender-System/blob/master/recommender_system.py
# https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243
# https://towardsdatascience.com/how-we-built-a-content-based-filtering-recommender-system-for-music-with-python-c6c3b1020332
# LIMITS TO SURPRISE ARXIV
# https://arxiv.org/pdf/1807.03905.pdf
