#KNN Machine Learning Algorithm from Scratch
#Done by
#Name : Muhammad Tayyab
#Roll # : 21L-5340
#Section: F2


import pandas as pd
import math
import heapq
from collections import Counter

def k_smallest_with_index(lst, k):
    heap = [(val, idx) for idx, val in enumerate(lst)]  
    heapq.heapify(heap)  
    smallest = []
    for _ in range(k):
        val, idx = heapq.heappop(heap)  
        smallest.append((val, idx))

    return smallest

def most_frequent(lst):
    counts = Counter(lst)
    most_common_element, count = counts.most_common(1)[0]
    return most_common_element

df =  pd.read_excel("KNN dataset.xlsx")
print("Initial Dataset")
print(df,"\n\n")

data_elements = len(df)
K = int(math.sqrt(data_elements))              #intializing K ,initially, with the squareroot of size of dataset
print("K = ",K)          
distances = [0.0 for _ in range(data_elements)]
test_data = (113,9586,3)    #matches, runs, wickets
print("Test Dataset: ", test_data)

#Calculating Distances

for i in range(data_elements):
    for j in range (len(test_data)):
        x1 = df.iloc[i,j]
        y1 = test_data[j]
        distances[i] += float(pow(float(x1-y1),2))
    distances[i] = float(math.sqrt(distances[i]))

df['distances'] = distances
print (df)

#Selecting k nearest classes based on distances
k_nearest_neighbours = k_smallest_with_index(distances,K)
print("Selected nearest neighbours are: ", k_nearest_neighbours)

elements_for_voting = []
for i in range(K):
    elements_for_voting.append(df['class'].iloc[k_nearest_neighbours[i][1]])

print("Selected Classes for voting are:", elements_for_voting)
print("Selected test dataset belongs to class:",most_frequent(elements_for_voting))