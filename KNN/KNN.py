#KNN 算法
# 稳了 2018.04.05
import numpy as np
from sklearn import *
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('d01_te.csv')
print(data.shape)
print(data)
print('================data.value=====================')
print(data.values)  #多了第一列0-959的序号
data = data.values[:,1:]
print(data)
print(data.shape)    #960*52
data = (data - data.min()) / (data.max() - data.min())
# data = data.values #把Pandas中的dataframe转成numpy中的array
#
X = data
# print(data)
#
y0 = np.zeros((160,1))
print(y0)
y1 = np.ones((800,1))
print(y1)
y = np.vstack((y0,y1))
print(y)
print(np.shape(y))###hstack()在行上合并 vstack()在列上合并
#
neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(X,y)

print(X[1,:])

print(y[1,:])
print('================Predict================')
print( neigh.predict([ X[800,:] ]) )



# n_neighbors = 15
# h = 0.02 # step size in the mesh
# # Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
# for weights in ['uniform', 'distance']:
#     # we create an instance of Neighbours Classifier and fit the data.
#     clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#     clf.fit(X, y)
#
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure()
#     plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
#     # Plot also the training points
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
#                 edgecolor='k', s=20)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title("3-Class classification (k = %i, weights = '%s')"
#               % (n_neighbors, weights))
#
# plt.show()


print('OK')
