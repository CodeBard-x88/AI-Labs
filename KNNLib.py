#KNN Machine Learning Algorithm using Library
#Done by
#Name : Muhammad Tayyab
#Roll # : 21L-5340
#Section: F2

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_excel("KNN dataset.xlsx")

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Test Dataset: ")
print(X_test)
print("Training Dataset: ")
print(X_train)

k = 3  
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Predicted Values:",y_pred)