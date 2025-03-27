## Place for checking the statistics of the data i.e. distributions in an effort to better understand.

#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

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

print(users.head())

# Drop the users that're below rank 1 (24000/32000)

users_drop = users.drop(labels='user_id', axis=1)

users_drop = users_drop[users_drop['user_rank'] >= 1]

dropped_count = len(users) - len(users_drop)
print(len(users_drop)) 
print(f"Number of users dropped: {dropped_count}")

variables = ['reading', 'streak_max', 'user_rank', 'streak', 'xp']

#scaler = MinMaxScaler()
#users_scaled = pd.DataFrame(scaler.fit_transform(users_drop), columns=users.columns)

# Chech the distributions of the variables

plt.figure(figsize=(15, 12))

for i, var in enumerate(variables, 1):
    plt.subplot(3, 2, i)
    mean = users_drop[var].mean()
    sd = users_drop[var].std()
    plt.text(-0.6, mean, f'Mean: {mean:.2f}', ha='center', color='black', fontsize=10)
    plt.text(-0.6, mean + sd, f'SD: {sd:.2f}', ha='center', color='black', fontsize=10)
    
    sns.boxplot(users_drop[var], color="royalblue")

    plt.title(f"Distribution of {var}")
    plt.tight_layout()

plt.show()

users_summary = users.describe()
print(users_summary)

prob_attempts_summary = prob_attempts.describe()
print(prob_attempts_summary)

probs_summary = probs.describe()
print(probs_summary)

X = 'user_id'

y = ['user_rank', 'xp', 'streak']

y, X = dmatrices('user_id ~ user_rank + xp + streak', data=users, return_type='dataframe')

model = sm.OLS(y, X)
results = model.fit()

print(results.summary())

coefficients = results.params
print(coefficients)

standard_errors = results.bse
print(standard_errors)

r_squared = results.rsquared
print("r_squared:", r_squared)

p_values = results.pvalues
print(p_values)

confidence_intervals = results.conf_int()
print(confidence_intervals)

# PCA clustering to analysing the most contributory variables

user_data = users.drop(columns=['user_id', 'opening', 'middle_game', 'endgame', 
                                'fighting', 'tesuji', 'life_and_death', 'analysis', 'knowledge', 'reading', 'reading_max', 'streak_max', 'streak'])

user_data = user_data[user_data['user_rank'] >= 1]

scaler = MinMaxScaler()
user_data_scaled = scaler.fit_transform(user_data)

pca = PCA(n_components=2)
user_pca = pca.fit_transform(user_data_scaled)

pca_df = pd.DataFrame(user_pca, columns=['PC1', 'PC2'])

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))

plt.figure(figsize=(10, 7))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

pca_full = PCA().fit(user_data_scaled)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.show()

loadings = pca.components_

loading_df = pd.DataFrame(
    loadings,
    columns=user_data.columns,
    index=[f'PC{i+1}' for i in range(loadings.shape[0])]
)

print(loading_df)

plt.figure(figsize=(10, 6))
sns.heatmap(loading_df, annot=True, cmap='coolwarm', center=0)
plt.title('PCA Loadings (Variable Contributions)')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data=users, x='user_id', y='user_rank', alpha=0.7)
plt.title('Rank vs User')
plt.xlabel('User')
plt.ylabel('Rank')
plt.grid(True)
plt.show()

# DBSCAN clustering to find outliers

dcscan_vars = users.drop(columns=['fighting', 'tesuji', 'life_and_death', 'analysis', 'knowledge', 'reading', 'reading_max', 'streak_max', 'streak'])

dcscan_vars = dcscan_vars[dcscan_vars['user_rank'] >= 1]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(dcscan_vars[['user_id', 'user_rank', 'opening', 'middle_game', 'endgame']])

# elbow method to find optimal epsilon, change k for tuning.
k = 4
nbrs = NearestNeighbors(n_neighbors=k).fit(scaled_data)
distances, indices = nbrs.kneighbors(scaled_data)

# Sort distances to find elbow
sorted_distances = np.sort(distances[:, k-1]) 
plt.figure(figsize=(8, 5))
plt.plot(sorted_distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}-th Nearest Neighbor Distance")
plt.title("Elbow Method for Choosing eps")
plt.show()

eps_values = [3, 4, 5, 6, 7]  # Adjust based on the k-distance plot
min_samples_values = [3, 5, 7, 10]

best_score = -1
best_params = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(scaled_data)
        
        # Ignore if all points are noise (-1 labels)
        if len(set(labels)) > 1:
            score = silhouette_score(scaled_data, labels)
            print(f"eps={eps}, min_samples={min_samples}, silhouette={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)

print(f"\nBest Parameters: eps={best_params[0]}, min_samples={best_params[1]}")

# Adjuts the epsilon and min_samples based on results from the silouette scores

dbscan = DBSCAN(eps=0.5, min_samples=5)
dcscan_vars['dbscan_cluster'] = dbscan.fit_predict(scaled_data)

plt.figure(figsize=(10, 6))

# Plot the noise clusters as white dots, plot them first so that they dont crowd the plot too mkuch.
sns.scatterplot(
    data=dcscan_vars[dcscan_vars['dbscan_cluster'] == -1], x='user_id', y='user_rank', color='white',edgecolor='black', linewidth=0.25,
    label='Noise',
    s=10)

sns.scatterplot(
    data=dcscan_vars[dcscan_vars['dbscan_cluster'] != -1], x='user_id', y='user_rank',hue='dbscan_cluster', palette='viridis', alpha=0.7,
    legend='full'
)

plt.title('User ID vs User Rank with DBSCAN Clusters')
plt.xlabel('User ID')
plt.ylabel('User Rank')
plt.legend(title='Cluster')
plt.show()

print(f"Number of unique clusters: {len(dcscan_vars['dbscan_cluster'].unique())}")
