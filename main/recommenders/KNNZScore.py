import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from surprise import KNNWithZScore
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV


def get_top_n(predictions, n=10):
    # * First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # * Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    # * First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # * Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # * Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # * Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # * Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # * Precision at K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # * Recall at K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


# * using reader to be able to deal with the imported CSV
reader = Reader(
    line_format="user item rating timestamp", sep=",", rating_scale=(1, 5), skip_lines=1
)
# * loading the csv
data = Dataset.load_from_file(
    file_path="../../ML_Dataset/ml-latest-small/ratings.csv", reader=reader
)
# * dividing in train and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# * define a cross-validation iterator
kf = KFold(n_splits=5)

# * Choosing KNN with Z-Score as algorithm
algo = KNNWithZScore()

# * Train the algorithm on the trainset, and predict ratings for the testset
for trainset, testset in kf.split(data):
    predictions = algo.fit(trainset).test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    accuracy.mse(predictions)
    accuracy.fcp(predictions)
    print("Precision: ", sum(prec for prec in precisions.values()) / len(precisions))
    print("Recall: ", sum(rec for rec in recalls.values()) / len(recalls))

df = pd.DataFrame(predictions, columns=["uid", "iid", "rui", "est", "details"])
df["err"] = abs(df.est - df.rui)
df.to_csv("predictions_KNNZScore.csv")
