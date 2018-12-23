#Importing libraries
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#finding the number of k using dendograms
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plot.show()

#Fitting hc algo with k = 5
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage= "ward")
y_hc = hc.fit_predict(X)

#Plotting the results
plot.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], c = 'red', s = 50, label = 'Carefull')
plot.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], c = 'blue', s = 50, label = 'Carefull')
plot.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], c = 'green', s = 50, label = 'Carefull')
plot.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], c = 'cyan', s = 50, label = 'Carefull')
plot.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], c = 'magenta', s = 50, label = 'Carefull')
plot.title('Clusters of clients')
plot.xlabel('Annual Income')
plot.ylabel('Spending Score')
plot.legend()
plot.show()