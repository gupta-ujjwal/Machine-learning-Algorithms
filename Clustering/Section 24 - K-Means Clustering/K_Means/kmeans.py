#Importing libraries
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#finding the number of k using elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plot.plot(range(1,11),wcss)
plot.show()

#Fitting kmeans algo with k = 5
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_means = kmeans.fit_predict(X)

#Plotting the results
plot.scatter(X[y_means == 0, 0], X[y_means == 0, 1], c = 'red', s = 50, label = 'Carefull')
plot.scatter(X[y_means == 1, 0], X[y_means == 1, 1], c = 'blue', s = 50, label = 'Carefull')
plot.scatter(X[y_means == 2, 0], X[y_means == 2, 1], c = 'green', s = 50, label = 'Carefull')
plot.scatter(X[y_means == 3, 0], X[y_means == 3, 1], c = 'cyan', s = 50, label = 'Carefull')
plot.scatter(X[y_means == 4, 0], X[y_means == 4, 1], c = 'magenta', s = 50, label = 'Carefull')
plot.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')
plot.title('Clusters of clients')
plot.xlabel('Annual Income')
plot.ylabel('Spending Score')
plot.legend()
plot.show()