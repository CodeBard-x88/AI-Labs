#Kmean Machine Learning Algorithm Using built in library
#Done by
#Name : Muhammad Tayyab
#Roll # : 21L-5340
#Section: F2

import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_excel("k mean dataset.xlsx")
print(df)

data = df.drop(columns=['datapoints'])

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("Centroids:")
print(centroids)
print("Labels:")
print(labels)