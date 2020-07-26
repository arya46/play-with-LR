from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as cma
# from sklearn.linear_model import LogisticRegression
from utils.models.logreg_classifier import LogisticRegression
from utils.configs import *

def generate_random_data():
    X, Y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1)
    return ((X + DATA_SHIFT) * DATA_SCALE), Y

def get_decision_boundary(X, Y, model):

    cMap = cma.ListedColormap(["#6b76e8", "#c775d1"])

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .05  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
    Z = Z.reshape(xx.shape)

    plt.figure(1, figsize=(20, 20))
    plt.axis('off')

    plt.pcolormesh(xx, yy, Z, cmap=cMap)
    plt.scatter(X[:, 0], X[:, 1], c=Y, marker = "o", edgecolors='k', cmap=cMap)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.savefig("static/image/xyz.png", bbox_inches='tight')

    return "static/image/xyz.png"

def logRegClassifier(data_x, data_y, learning_rate, max_iter, C, STD, MEAN):

    clf = LogisticRegression(learning_rate=learning_rate, max_iter=max_iter, C=C)
    clf.fit(data_x, data_y)
    
    points_x = [((0 - MEAN[0]) * (1 / STD[0])), ((500 - MEAN[0]) * (1 / STD[0]))] #np.array(ax.get_xlim())
    
    line_bias = clf.get_params()['intercept']
    line_w = clf.get_params()['coef']

    points_y = [(line_w[0]*x + line_bias) / (-1*line_w[1]) for x in points_x]

    x_vals = [MEAN[0]+j for j in [STD[0]*i for i in points_x]]
    y_vals = [MEAN[1]+j for j in [STD[1]*i for i in points_y]]
    
    return {'points': list(zip(x_vals, y_vals))}

def standardize_data(X):
    STD = X.std(axis=0)
    MEAN = X.mean(axis = 0)
    scaled_X = (X - MEAN) * (1 / STD)
    return scaled_X, STD, MEAN