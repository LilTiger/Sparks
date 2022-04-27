import matplotlib.pyplot as plt
import sys
import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE, MDS
import time

df = pd.read_excel("data/manifold_drawing/Bert.xlsx", engine="openpyxl")
# data sets that don't contain classification label
X = df.drop("TRUE VALUE", axis=1)
# Columns that contain classification label
labels = df["TRUE VALUE"]

tag = list(np.unique(labels))
variable = X.columns

# data need to be standardized before dimension reduction for dimensionless method
scaler = StandardScaler()
X_processed = pd.DataFrame(scaler.fit_transform(X))


# four common manifold learning algorithms
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
mds = MDS(n_components=2, random_state=42)
isomap = Isomap(n_components=2)
tsne = TSNE(n_components=2, random_state=42)

# model_manifold = [(221, lle, "LocallyLinearEmbedding, n_neighbors=10"),
#                   (222, mds, "Multidimensional Scaling"),
#                   (223, isomap, "Isomap"),
#                   (224, tsne, "t-Distributed Stochastic Neighbor Embedding")]

model_manifold = [(mds, "Multi-dimensional Scaling")]


def plot_2D_reduced(model, X, labels, tag, fig_name=True):
    """
    2D mapping of data points using dimensionality reduction method

    :param model: dimension reduction algorithm
    :param X: data sets only contain eigenvalues
    :param labels: data sets only contain label values
    :param tag: true category tag type
    :param fig_name: figure name
    """

    plt.figure(figsize=(11, 8))
    legend = []
    for pca, title in model:
        X_reduced = pca.fit_transform(X)
        # plt.title(title, fontsize=14)
        for i, label in enumerate(tag):
            plt.scatter(X_reduced[:, 0][labels == label], X_reduced[:, 1][labels == label])
            legend.append("{}".format(label))
        # if subplot == 221 or subplot == 223:
        #     plt.ylabel("$x_2$", fontsize=18, rotation=0)
        # if subplot == 223 or subplot == 224:
        #     plt.xlabel("$x_1$", fontsize=18)
        plt.grid(True)
        plt.legend(legend)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(fig_name, dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_2D_reduced(model_manifold, X_processed, labels, tag, fig_name="manifold_plot")
