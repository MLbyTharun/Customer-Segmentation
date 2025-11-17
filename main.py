# ==========================
# STEP 1: DATA LOADING + EDA
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# 1. Load Dataset
# --------------------------
df = pd.read_csv("Mall_Customers.csv")   # change to your filename

print("=== Dataset Loaded Successfully ===")
print(f"Shape: {df.shape}")

# --------------------------
# 2. Peek at the Data
# --------------------------
print("\n=== Head ===")
print(df.head())

print("\n=== Info ===")
print(df.info())

print("\n=== Summary Statistics ===")
print(df.describe().T)


# --------------------------
# 3. Check Missing Values
# --------------------------
print("\n=== Missing Values ===")
print(df.isnull().sum())

# --------------------------
# 4. Encode Gender (Male=1, Female=0)
# --------------------------

df['Genre'] = df['Genre'].map({'Male': 1, 'Female': 0})

# --------------------------
# 5. Correlation Heatmap
# --------------------------

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="viridis")
plt.title("Correlation Heatmap")
plt.show()

# ==========================
# STEP 2: FEATURE SELECTION + SCALING + CLUSTER SELECTION
# ==========================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# 1. Select Features for Clustering
# ----------------------------------------
# Option 1: Use the most common 3 features

X = df[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
print("\n=== Selected Features ===")
print(X.head())

# ----------------------------------------
# 2. Scaling Features
# ----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n=== Scaling Completed ===")

# ----------------------------------------
# 3. Elbow Method (Find Optimal K)
# ----------------------------------------

inertia_list = []
K_range = range(2, 14)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_list.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(K_range, inertia_list, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# ----------------------------------------
# 4. Silhouette Scores (Secondary Check)
# ----------------------------------------

silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

# Plot Silhouette Curve
plt.figure(figsize=(8,5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Score for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()

print("\n=== Silhouette Scores ===")
for k, score in zip(K_range, silhouette_scores):
    print(f"k={k}: score={score:.4f}")
    
# ==========================
# STEP 3: APPLYING K-MEANS + VISUALIZING CLUSTERS
# ==========================

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Choosing Optimal K ( from Step 2)
# ----------------------------------------
optimal_k =11  # based on Elbow/Silhouette result

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Adding cluster labels to original dataframe
df['Cluster'] = cluster_labels

print("\n=== Clustering Completed ===")
print(df[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']])

# ----------------------------------------
# 2. PCA for 2D Visualization
# ----------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=60)
plt.title("Customer Segments (2D PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()

# ----------------------------------------
# 3. PCA for 3D Visualization
# ----------------------------------------
from mpl_toolkits.mplot3d import Axes3D

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c=cluster_labels, s=60)

ax.set_title("Customer Segments (3D PCA Visualization)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")

plt.show()

# ----------------------------------------
# 4. Showing Cluster Centers (in scaled feature space)
# ----------------------------------------
print("\n=== Cluster Centers (Scaled Feature Space) ===")
print(kmeans.cluster_centers_)
# ==========================
# STEP 4: CLUSTER INTERPRETATION
# ==========================

# Combining PCA components + original cluster labels for profiling
df_profile = df.copy()

# Calculating cluster-wise averages
cluster_summary = df_profile.groupby("Cluster").mean()

print("\n=== Cluster Summary (Mean Values per Cluster) ===")
print(cluster_summary)

# Plotting each cluster distribution
numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.boxplot(x="Cluster", y=col, data=df_profile)
    plt.title(f"{col} Distribution Across Clusters")
    plt.show()

# Counting of customers per cluster
plt.figure(figsize=(6,4))
sns.countplot(x="Cluster", data=df_profile)
plt.title("Customers per Cluster")
plt.show()
# ==========================
# STEP 6: CLUSTER NAMING / INTERPRETATION
# ==========================

cluster_characteristics = df.groupby('Cluster').mean()[[
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]]

print("\n=== Cluster Characteristics (Mean Values) ===")
print(cluster_characteristics)

cluster_names = {}

for cluster in cluster_characteristics.index:
    age = cluster_characteristics.loc[cluster, 'Age']
    income = cluster_characteristics.loc[cluster, 'Annual Income (k$)']
    score = cluster_characteristics.loc[cluster, 'Spending Score (1-100)']

    if income > 70 and score > 70:
        cluster_names[cluster] = "Premium High Spenders"
    elif income > 70 and score < 40:
        cluster_names[cluster] = "Rich Low Spenders"
    elif income < 40 and score > 60:
        cluster_names[cluster] = "Budget High Spenders"
    elif income < 40 and score < 40:
        cluster_names[cluster] = "Low Income Low Spenders"
    else:
        cluster_names[cluster] = "Mid-Tier Average Customers"

df["Cluster_Name"] = df["Cluster"].map(cluster_names)

print("\n=== Cluster Names Assigned ===")
print(df[['Cluster', 'Cluster_Name']].head())

