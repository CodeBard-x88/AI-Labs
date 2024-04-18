#Kmean Machine Learning Algorithm from Scratch
#Done by
#Name : Muhammad Tayyab
#Roll # : 21L-5340
#Section: F2


import pandas as pd
import math
df = pd.read_excel("k mean dataset.xlsx")
data = df.drop(columns=['datapoints'])
print(data)

total_clusters = 3
intial_centroid_datapoints = ['a1','b1','c1']   #adjust the numbers of initial centroids according to total clusters

clusters = [0 for i in range(len(data))]
distance = [0 for i in range(total_clusters)]

for i in range(len(data)):
    for j in range(total_clusters):
        temp = df.loc[df['datapoints'] == intial_centroid_datapoints[j]]
        x = temp['x']
        y = temp['y']
        distance[j] = float(math.sqrt(pow(data['x'].iloc[i] - float(x.iloc[0]), 2) + pow(data['y'].iloc[i] - float(y.iloc[0]), 2))) #Euclidean Function to calculate the distance
    
    clusters[i] = distance.index(min(distance))

print("Total Clusters: ", total_clusters)

print("Labels: ")
print (clusters)

total_elements_in_cluster = 0
x_sum = 0
y_sum= 0 
new_centroids = [[] for _ in range(total_clusters)]

#The following calculates the new_centroids after clustering

for j in range(total_clusters):
    total_elements_in_cluster = 0
    x_sum = 0
    y_sum = 0
    for k in range(len(data)):
        if(clusters[k] == j):
            total_elements_in_cluster += 1
            x_sum += data['x'].iloc[k]
            y_sum += data['y'].iloc[k]
    if total_elements_in_cluster != 0:
        new_centroids[j].append((x_sum/total_elements_in_cluster, y_sum/total_elements_in_cluster))

print("New Centroids: ")
print(new_centroids)