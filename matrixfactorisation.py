## Matrix factorisation could be used as an alternative to biclustering for the cirriculum part of the project.
# SVD or ALS works as a recommender by reconstructing the interaction matrix in a lower latent space based on the interactions of puzzles by all the users.

#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
import implicit
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import SpectralBiclustering
from scipy.spatial.distance import pdist, squareform, euclidean

from biclusfunctions import loadprocess_data

# Define the project's folder as the root directory
ROOT_DIR = Path(__file__).parent

# Define the list of actual problem categories. Required to clean the `problems` table which has a mix of both category and tags
PROB_CATS = [
    "opening",
    "middle-game",
    "endgame",
    "capturing",
    "connect-and-cut",
    "ko-fight",
    "tesuji",
    "shapes",
    "life-and-death",
    "estimation",
    "positional-judgment",
    "joseki",
    "standards",
    "rules",
    "terms",
    "history",
    "value",
    "punishment"
]

data, users, prob_attempts, probs = loadprocess_data(ROOT_DIR, PROB_CATS)

# Adjust weights for first-time pass (1) and second-time pass (2)
def assign_weight(result):
    if result == 1:
        return 10  # First-time pass
    elif result == 2:
        return 5   # Second-time pass
    elif result == -1:
        return -10 # Failure
    else:
        return 0   # No interaction

data['result'] = data['result'].apply(assign_weight)

# Compute the relative rank between user and problem
data['rel_rank'] = data['prob_rank'] - data['user_rank']

data = data[data['user_rank'] >= 1]

# Sum rather than avergae
#data = data.groupby(['user_id', 'prob_id'])['result'].sum().reset_index()

# Create a pivot table for user_id, prob_id and results
# Fill value is set to 0 to represent a absence of edge/link between user and problem
pivot = data.pivot_table(values='result', index='user_id', columns='prob_id', fill_value=0)
#pivot = data.pivot_table(values='result', index='user_id', columns='cats', fill_value=0)
#pivot = data.pivot_table(values='total_time', index='user_id', columns='prob_id', fill_value=0, observed=False)
#pivot = data.pivot_table(values='total_time', index='user_id', columns='cats', fill_value=0, observed=False)
#pivot = data.pivot_table(values='rel_rank', index='user_id', columns='prob_id', fill_value=0)
print(pivot.head())

# Subsample the pivot table for matrix factorisation
subsample_pivot = pivot.sample(n=8000, random_state=42).sample(n=1000, axis=1, random_state=42)

print(subsample_pivot.head())

# Perform Singular Value Decomposition (SVD) for Matrix Factorisation
np_pivot = subsample_pivot.to_numpy()
svd = TruncatedSVD(n_components=200)  # latent features (components)
U = svd.fit_transform(np_pivot)    # User matrix (n_users x n_components)
Sigma = svd.singular_values_       # Singular values (n_components,)
Vt = svd.components_               # Problem matrix (n_components x n_problems)

# Reconstruct the original matrix approximation
reconstructed_matrix = np.dot(U, Vt)

# Visualise the original vs. reconstructed matrix
plt.matshow(np_pivot, cmap=plt.cm.Blues, aspect='auto')
plt.title('Original Interaction Matrix')
plt.show()
plt.matshow(reconstructed_matrix, cmap=plt.cm.Blues, aspect='auto')
plt.title('Reconstructed Matrix from SVD')
plt.show()

# Compute the error between the original and reconstructed matrices
error = np.linalg.norm(np_pivot - reconstructed_matrix, 'fro')
print(f"Reconstruction Error (Frobenius norm): {error}")

# Perform Alternating Least Squares (ALS) for Matrix Factorisation
pivot_sparse = csr_matrix(np_pivot)  # Convert to sparse matrix for memory efficiency

# Instantiate ALS model (using user and problem latent factors). captures the strengths of each player in a lower latent space
model = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.1, iterations=100)

# Fit the model
model.fit(pivot_sparse)

# Get the user and item factors (U and Vt)
user_factors = model.user_factors
item_factors = model.item_factors

# Reconstruct the matrix by dot multiplying user_factors and item_factors
reconstructed_matrix_als = np.dot(user_factors, item_factors.T)

# Visualise the results. ALS predicts missing values, so recommends puzzles that will continue to get good scores and recommends to players those puzzles that 
# players have excelled in but havent been attempted by themselves/limited interactions (strong fits). Can add a filter layer to disregard completed puzzles
plt.matshow(reconstructed_matrix_als, cmap=plt.cm.Blues, aspect='auto')
plt.title('Reconstructed Matrix from ALS (Strength Recommendations)')
plt.show()

# The inverse of the ALS. Takes the max value of the ALS matrix and subtracts the score of the cell to recommend puzzles that the player would struggle with
weakness_matrix = np.max(reconstructed_matrix_als) - reconstructed_matrix_als

plt.matshow(weakness_matrix, cmap=plt.cm.Reds, aspect='auto')
plt.title('Inverted ALS (Weakness Recommendations)')
plt.show()

# Strength embeddings directly from ALS normalised for clustering
strength_embeddings = StandardScaler().fit_transform(user_factors)

# Compute pairwise cosine similarity
distance_matrix = pairwise_distances(strength_embeddings, metric='cosine')

n_clusters = 5

# Fit K-means clustering based on the distance matrix
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
user_clusters = kmeans.fit_predict(distance_matrix)

# Visualise the clusters (PCA to 2D for clarity)
pca = PCA(n_components=2)
reduced_profiles = pca.fit_transform(strength_embeddings)

plt.scatter(reduced_profiles[:, 0], reduced_profiles[:, 1], c=user_clusters, cmap='viridis')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("User Clusters Based on Maximally Distances Strengths")
plt.show()

# Compute hierarchical clustering using Wards
linkage_matrix = linkage(strength_embeddings, method='ward')

# dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("User ID")
plt.ylabel("Distance")
plt.show()

# Cut the dendrogram to form clusters
clusters_hier = fcluster(linkage_matrix, t=20, criterion='maxclust')
print("Hierarchical clusters:", clusters_hier)

# I did the elbow method here and recommended 2-5 clusters for rows and 10 clustered for columns

# Trying out some biclustering as well
n_row_clusters = 5  # TODO refine this to be the number of user clusters i.e. playstyles?
n_col_clusters = 10  # TODO Refine this to be metacategorising the categories to be about openings, midgame etc.

# Perform spectral biclustering
bicluster_model = SpectralBiclustering(n_clusters=(n_row_clusters, n_col_clusters),
                                    method='log', svd_method='arpack', random_state=42)
bicluster_model.fit(np_pivot)

# Retrieve row (user) cluster labels
user_bicluster_labels = bicluster_model.row_labels_

# Rearrange the data to match bicluster orderings
reordered_rows = np.argsort(bicluster_model.row_labels_)
reordered_cols = np.argsort(bicluster_model.column_labels_)

# Ensure data is a NumPy array
data = np.array(np_pivot)
# Reorder rows and columns using NumPy indexing
reordered_data = data[reordered_rows][:, reordered_cols]

plt.figure(figsize=(8, 6))

# Plot the reordered matrix
plt.matshow(reordered_data, cmap=plt.cm.Blues, aspect='auto', fignum=False)

# Add labels and colorbar
plt.title("Biclustered User-Category Performance Matrix", pad=20)
plt.xlabel("Categories")
plt.ylabel("Users")
plt.colorbar(label="Performance Level")

plt.show()

column_labels = bicluster_model.column_labels_

# Convert the labels into a DataFrame for easy analysis
categories = subsample_pivot.columns 
column_labels_df = pd.DataFrame({'Category': categories, 'Cluster': column_labels})

# Group the categories by their cluster labels
clustered_categories = column_labels_df.groupby('Cluster')['Category'].apply(list)

# Display the categories grouped by clusters
print(clustered_categories)

# Compute centroid (mean performance vector) for each user bicluster
unique_clusters = np.unique(user_bicluster_labels)
centroids = []
for cluster in unique_clusters:
    # Find indices of users in this bicluster
    cluster_indices = np.where(user_bicluster_labels == cluster)[0]
    centroid = np.mean(np_pivot[cluster_indices, :], axis=0)
    centroids.append(centroid)
centroids = np.array(centroids)

# Compute pairwise distances between centroids
centroid_distances = squareform(pdist(centroids))
# To find two clusters that are maximally different, ignore the diagonal.
i, j = np.unravel_index(np.argmax(centroid_distances, axis=None), centroid_distances.shape)
print(f"Most opposing biclusters: Cluster {i} vs Cluster {j}")

# Extract the indices of users belonging to these two clusters.
cluster_i_indices = np.where(user_bicluster_labels == unique_clusters[i])[0]
cluster_j_indices = np.where(user_bicluster_labels == unique_clusters[j])[0]

overall_strength = np.linalg.norm(np_pivot, axis=1)

# Get strengths for each cluster and sort indices
sorted_i = cluster_i_indices[np.argsort(overall_strength[cluster_i_indices])]
sorted_j = cluster_j_indices[np.argsort(overall_strength[cluster_j_indices])]

# Pair users from the two clusters based on the smaller cluster
n_pairs = min(len(sorted_i), len(sorted_j))
pairings = list(zip(sorted_i[:n_pairs], sorted_j[:n_pairs]))

pairings_df = pd.DataFrame(pairings, columns=['User_from_Cluster_{}'.format(unique_clusters[i]),
                                            'User_from_Cluster_{}'.format(unique_clusters[j])])
print("User Pairings from Opposing Biclusters:")
print(pairings_df)




# TODO Pairings?
# Calculate overall strength for each player, TODO rethink this its ugly
print(strength_embeddings)
overall_strength = np.linalg.norm(strength_embeddings, axis=1)

# Create a DataFrame to track players and their strengths
players_df = pd.DataFrame({'Player': range(len(strength_embeddings)), 'Strength': overall_strength})

# Sort by strength for easier pairing
players_df.sort_values('Strength', inplace=True)

print(players_df.head)

### 1. Even-strength but different puzzle specialties
# Find two players with similar overall strength but different category strengths
best_diff_pair = None
best_diff_score = float('inf')

for i, player1 in players_df.iterrows():
    for j, player2 in players_df.iterrows():
        if i >= j:  # Skip duplicates and self-comparisons
            continue
        strength_diff = abs(player1['Strength'] - player2['Strength'])
        category_diff = euclidean(strength_embeddings[player1['Player']], strength_embeddings[player2['Player']])
        
        # Prioritise players with similar overall strength but different category spread
        if strength_diff < 0.1 and category_diff > best_diff_score:
            best_diff_score = category_diff
            best_diff_pair = (player1['Player'], player2['Player'])

### 2. Most evenly matched pair
closest_pair = None
closest_diff = float('inf')

for i, player1 in players_df.iterrows():
    for j, player2 in players_df.iterrows():
        if i >= j:
            continue
        strength_diff = abs(player1['Strength'] - player2['Strength'])
        
        if strength_diff < closest_diff:
            closest_diff = strength_diff
            closest_pair = (player1['Player'], player2['Player'])

### 3. Strong vs Weak Matchup
strong_player = players_df.iloc[-1]['Player']
weak_player = players_df.iloc[0]['Player']

# Output the matchups
print("Matchup 1 (Even strength, different specialties):", best_diff_pair)
print("Matchup 2 (Closest in strength):", closest_pair)
print("Matchup 3 (Strong vs Weak):", (strong_player, weak_player))
