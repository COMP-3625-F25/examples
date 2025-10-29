from sklearn.datasets import make_classification, make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# create a toy dataset for clustering
X, y = make_blobs(n_samples=100, n_features=2, centers=6)
# X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=4, n_clusters_per_class=1, class_sep=2)

# instantiate k-means clustering algorithm
kmeans = KMeans(n_clusters=5)

# run the clustering
kmeans.fit(X)

# plot
fig, ax = plt.subplots(1, 2)
ax[0].scatter(X[:, 0], X[:, 1], c=y)
ax[0].set_title('true cluster memberships')
ax[1].scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap='jet')
ax[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='k')
ax[1].set_title('k-means clusters')
plt.show()