import pandas as pd
import numpy as np

# get relevance and tag label
relevance = pd.read_csv("../ML_Dataset/ml-latest/cropped_tags.csv")
tag_labels = pd.read_csv("../ML_Dataset/ml-latest/genome-tags.csv")

#create labels list
labels = []

#for all tag numbers crpped list for all movies
#find its associated label and append to the labels list
for i in range(relevance.iloc[:, 2].values.size):
    tag = relevance.iloc[i, 2]
    label = tag_labels.iloc[tag - 1, 1]
    labels.append(label)

#create a new dataframe with this new column
associated_label = pd.DataFrame(
    data=({"label": labels}),
    index=range(relevance.iloc[:, 1].values.size),
    columns=["label"],
)

#join and put all in the new genome_labels.csv
relevance = relevance.join(associated_label)
relevance.to_csv("../ML_Dataset/ml-latest/genome_labels.csv")
