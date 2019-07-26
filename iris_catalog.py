from __future__ import absolute_import, division, print_function, unicode_literals
import os
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

train_dataset_url = (
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)
train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url), origin=train_dataset_url
)
# column order in CSV file
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

feature_names = column_names[:-1]
label_name = column_names[-1]

class_names = ["Iris setosa", "Iris versicolor", "Iris virginica"]

batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1,
)

features, labels = next(iter(train_dataset))

plt.scatter(
    features["petal_length"].numpy(),
    features["sepal_length"].numpy(),
    c=labels.numpy(),
    cmap="viridis",
)

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()

# ate aqui tudo correu bem

