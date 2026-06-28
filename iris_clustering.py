from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title("Iris Flower Clustering using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig("cluster_plot.png")
plt.show()

# Save the trained model
joblib.dump(kmeans, "iris_kmeans.pkl")

# Compare predicted clusters with true labels
print("Confusion Matrix:")
print(confusion_matrix(y, clusters))

print("Project completed successfully!")
