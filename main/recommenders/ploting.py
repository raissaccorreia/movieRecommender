import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# RMSE Plot

# line 1 SVD
x1 = [1, 2, 3, 4, 5]
y1 = [0.8792, 0.8729, 0.8692, 0.8688, 0.8805]
plt.plot(x1, y1, label="SVD")

# line 2 NMF
x2 = [1, 2, 3, 4, 5]
y2 = [0.9252, 0.9244, 0.9281, 0.9124, 0.9227]
plt.plot(x2, y2, label="NMF")

# line 3 KNN Basic
x3 = [1, 2, 3, 4, 5]
y3 = [0.9448, 0.9474, 0.9427, 0.9618, 0.9406]
plt.plot(x3, y3, label="KNN Basic")

# line 4 KNN Means
x4 = [1, 2, 3, 4, 5]
y4 = [0.8909, 0.8974, 0.8991, 0.9005, 0.8931]
plt.plot(x4, y4, label="KNN Means")

# line 5 KNN Baseline
x5 = [1, 2, 3, 4, 5]
y5 = [0.8851, 0.8684, 0.8625, 0.8820, 0.8719]
plt.plot(x5, y5, label="KNN Baseline")

# line 6 KNN Z-Score
x6 = [1, 2, 3, 4, 5]
y6 = [0.9013, 0.8983, 0.8882, 0.8922, 0.9000]
plt.plot(x6, y6, label="KNN Z-Score")

# line 7 Normal Predictor
# x7 = [1, 2, 3, 4, 5]
# y7 = [1.4243, 1.4217, 1.4182, 1.4131, 1.4334]
# plt.plot(x7, y7, label="Normal Predictor")

# line 8 Baseline
x8 = [1, 2, 3, 4, 5]
y8 = [0.8731, 0.8712, 0.8713, 0.8718, 0.8751]
plt.plot(x8, y8, label="Baseline")

# line 9 SVD++
x9 = [1, 2, 3]
y9 = [0.8571, 0.8669, 0.8598]
plt.plot(x9, y9, label="SVD++")

plt.xlabel("Number of Cross-Validations")
# Set the y axis label of the current axis.
plt.ylabel("RMSE")
# Set a title of the current axes.
plt.title("RMSE per Algorithm per Cross Validations")
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()

# MAE Plot

# line 1 SVD
x1 = [1, 2, 3, 4, 5]
y1 = [0.6736, 0.6689, 0.6670, 0.6685, 0.6774]
plt.plot(x1, y1, label="SVD")

# line 2 NMF
x2 = [1, 2, 3, 4, 5]
y2 = [0.7066, 0.7064, 0.7104, 0.7021, 0.7100]
plt.plot(x2, y2, label="NMF")

# line 3 KNN Basic
x3 = [1, 2, 3, 4, 5]
y3 = [0.7216, 0.7283, 0.7255, 0.7351, 0.7215]
plt.plot(x3, y3, label="KNN Basic")

# line 4 KNN Means
x4 = [1, 2, 3, 4, 5]
y4 = [0.6801, 0.6838, 0.6858, 0.6898, 0.6832]
plt.plot(x4, y4, label="KNN Means")

# line 5 KNN Baseline
x5 = [1, 2, 3, 4, 5]
y5 = [0.6735, 0.6626, 0.6602, 0.6731, 0.6691]
plt.plot(x5, y5, label="KNN Baseline")

# line 6 KNN Z-Score
x6 = [1, 2, 3, 4, 5]
y6 = [0.6827, 0.6819, 0.6734, 0.6771, 0.6832]
plt.plot(x6, y6, label="KNN Z-Score")

# line 8 Baseline
x8 = [1, 2, 3, 4, 5]
y8 = [0.6720, 0.6697, 0.6707, 0.6767, 0.6750]
plt.plot(x8, y8, label="Baseline")

# line 9 SVD++
x9 = [1, 2, 3]
y9 = [0.6582, 0.6612, 0.6594]
plt.plot(x9, y9, label="SVD++")

plt.xlabel("Number of Cross-Validations")
# Set the y axis label of the current axis.
plt.ylabel("MAE")
# Set a title of the current axes.
plt.title("MAE per Algorithm per Cross Validations")
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()

# F1 Plot

# line 1 SVD
x1 = [1, 2, 3, 4, 5]
p1 = [
    0.8481147540983618,
    0.8530925013683645,
    0.8686371100164216,
    0.8447181171319115,
    0.8387061403508786,
]
r1 = [
    0.24170115988125696,
    0.2536395287362205,
    0.25231624761779725,
    0.25223331323371767,
    0.25701956088972955,
]
y1 = [None] * 5
for i in range(5):
    y1[i] = (2 * r1[i] * p1[i]) / (r1[i] + p1[i])
plt.plot(x1, y1, label="SVD")

# line 2 NMF
x2 = [1, 2, 3, 4, 5]
p2 = [
    0.810708401976937,
    0.7867486338797828,
    0.8165573770491813,
    0.8098795840175164,
    0.7978379857690218,
]
r2 = [
    0.28999761565053706,
    0.2677791375565105,
    0.27345356864666714,
    0.273782666200474,
    0.259213327836551,
]
y2 = [None] * 5
for i in range(5):
    y2[i] = (2 * r2[i] * p2[i]) / (r2[i] + p2[i])
plt.plot(x2, y2, label="NMF")

# line 3 KNN Basic
x3 = [1, 2, 3, 4, 5]
p3 = [
    0.7832786885245915,
    0.7806239737274229,
    0.7742076502732251,
    0.7914887794198151,
    0.7875478927203078,
]
r3 = [
    0.3035644229350757,
    0.2847442887126069,
    0.28445027012666463,
    0.28234175051664545,
    0.29581133810484195,
]
y3 = [None] * 5
for i in range(5):
    y3[i] = (2 * r3[i] * p3[i]) / (r3[i] + p3[i])
plt.plot(x3, y3, label="KNN Basic")

# line 4 KNN Means
x4 = [1, 2, 3, 4, 5]
p4 = [
    0.8432129173508502,
    0.8306284153005477,
    0.8495081967213131,
    0.8336612021857938,
    0.8392896174863399,
]
r4 = [
    0.28450853226175904,
    0.2861859503636732,
    0.27835581972304946,
    0.2710868176204517,
    0.26352107818415105,
]
y4 = [None] * 5
for i in range(5):
    y4[i] = (2 * r4[i] * p4[i]) / (r4[i] + p4[i])
plt.plot(x4, y4, label="KNN Basic")

# line 5 KNN Baseline
x5 = [1, 2, 3, 4, 5]
p5 = [
    0.8185792349726788,
    0.836065573770493,
    0.8352764094143416,
    0.8139573070607566,
    0.8168035030104008,
]
r5 = [
    0.27647798690214187,
    0.2711692575462154,
    0.27209239779673466,
    0.26492154572910404,
    0.26346044213190506,
]
y5 = [None] * 5
for i in range(5):
    y5[i] = (2 * r5[i] * p5[i]) / (r5[i] + p5[i])
plt.plot(x5, y5, label="KNN Baseline")

# line 6 KNN Z-Score
x6 = [1, 2, 3, 4, 5]
p6 = [
    0.8236842105263169,
    0.8370005473453762,
    0.8260416666666681,
    0.837759562841532,
    0.8239583333333349,
]
r6 = [
    0.2928648424263283,
    0.28114731756025785,
    0.2771353994230258,
    0.2828973003051379,
    0.286656282968948,
]
y6 = [None] * 5
for i in range(5):
    y5[i] = (2 * r6[i] * p6[i]) / (r6[i] + p6[i])
plt.plot(x6, y6, label="KNN Z-Score")

# line 8 Baseline
x8 = [1, 2, 3, 4, 5]
p8 = [
    0.8768032786885256,
    0.8772950819672141,
    0.8659289617486351,
    0.8900109469074994,
    0.8587848932676531,
]
r8 = [
    0.22207904574004916,
    0.22973044367868478,
    0.22737102588216626,
    0.2351213382698622,
    0.2317364066060379,
]
y8 = [None] * 5
for i in range(5):
    y5[i] = (2 * r8[i] * p8[i]) / (r8[i] + p8[i])
plt.plot(x8, y8, label="Baseline")

# line 9 SVD++
x9 = [1, 2, 3]
p9 = [0.8642349726775966, 0.8547892720306527, 0.8722677595628425]
r9 = [0.25565750023700096, 0.25249711783351003, 0.25478409940848273]
y9 = [None] * 3
for i in range(3):
    y9[i] = (2 * r9[i] * p9[i]) / (r9[i] + p9[i])
plt.plot(x9, y9, label="SVD++")

plt.xlabel("Number of Cross-Validations")
# Set the y axis label of the current axis.
plt.ylabel("F1")
# Set a title of the current axes.
plt.title("F1 per Algorithm per Cross Validations")
# show a legend on the plot
plt.legend(loc=9)
# Display a figure.
plt.show()

svd_preds = pd.read_csv("../results/predictions_svd.csv")
nmf_preds = pd.read_csv("../results/predictions_NMF.csv")
knn_basic_preds = pd.read_csv("../results/predictions_KNNBasic.csv")
knn_means_preds = pd.read_csv("../results/predictions_KNNMeans.csv")
knn_baseline_preds = pd.read_csv("../results/predictions_KNNBaseline.csv")
knn_z_preds = pd.read_csv("../results/predictions_KNNZScore.csv")

# error distribution
x = np.arange(20167)

svd_err = svd_preds.iloc[:, 6].sort_values()
plt.plot(x, svd_err, label="SVD")

nmf_err = nmf_preds.iloc[:, 6].sort_values()
plt.plot(x, nmf_err, label="NMF")

knnBasic_err = knn_basic_preds.iloc[:, 6].sort_values()
plt.plot(x, knnBasic_err, label="KNN Basic")

knnMeans_err = knn_means_preds.iloc[:, 6].sort_values()
plt.plot(x, knnMeans_err, label="KNN Means")

knnBaseline_err = knn_baseline_preds.iloc[:, 6].sort_values()
plt.plot(x, knnBaseline_err, label="KNN Baseline")

knnZ_err = knn_z_preds.iloc[:, 6].sort_values()
plt.plot(x, knnZ_err, label="KNN Z-Score")

plt.xlabel("Number of Recomendation Sorted by Erro")
# Set the y axis label of the current axis.
plt.ylabel("Absolute Error")
# Set a title of the current axes.
plt.title("Absolute Error Distribution")
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()

# prediction distribution
# The _err variables will be reused

svd_err = svd_preds.iloc[:, 4].sort_values()
plt.plot(x, svd_err, label="SVD")

nmf_err = nmf_preds.iloc[:, 4].sort_values()
plt.plot(x, nmf_err, label="NMF")

knnBasic_err = knn_basic_preds.iloc[:, 4].sort_values()
plt.plot(x, knnBasic_err, label="KNN Basic")

knnMeans_err = knn_means_preds.iloc[:, 4].sort_values()
plt.plot(x, knnMeans_err, label="KNN Means")

knnBaseline_err = knn_baseline_preds.iloc[:, 4].sort_values()
plt.plot(x, knnBaseline_err, label="KNN Baseline")

knnZ_err = knn_z_preds.iloc[:, 4].sort_values()
plt.plot(x, knnZ_err, label="KNN Z-Score")

plt.xlabel("Number of Recomendation Sorted by Prediction")
# Set the y axis label of the current axis.
plt.ylabel("Prediction Value")
# Set a title of the current axes.
plt.title("Predicted Rating Distribution")
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
