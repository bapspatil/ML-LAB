import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import sklearn.metrics as sm
l1 = [0,1,2]
def rename(s):
	l2 = []
	# Putting non-duplicate items from s into l2
	for i in s:
		if i not in l2:
			l2.append(i)
	# Getting the position of items that are common to l2 and s 
	# + Using that position's element in l1 to store it in s
	for i in range(len(s)):
		pos = l2.index(s[i])
		s[i] = l1[pos]
	return s

iris_dataset = load_iris()
X = iris_dataset["data"]
y = iris_dataset["target"]

# Using the elbow method to find the optimal number of clusters
# k-means part
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=40, cmap='viridis')
km = rename(y_kmeans)
print("Accuracy KM : ",sm.accuracy_score(y,km))

plt.show()

# EM part
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_kmeans = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=40, cmap='viridis')
em = rename(y_kmeans)
print("Accuracy EM : ",sm.accuracy_score(y,em))

plt.show()
