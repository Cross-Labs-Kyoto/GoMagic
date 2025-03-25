## A space for biclustering the cleaned data. 
# Both spectral and 

#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralBiclustering
from sklearn.cluster import SpectralCoclustering
import matplotlib.colors as mcolors

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

# Compute the relative rank between user and problem
data['rel_rank'] = data['prob_rank'] - data['user_rank']

# Removed the users that are lower than rank 1 (~24000)
data = data[data['user_rank'] >= 1]

# Create a pivot table for user_id, prob_id and results
# Fill value is set to 0 to represent a absence of edge/link between user and problem
#pivot = data.pivot_table(values='result', index='user_id', columns='prob_id', fill_value=0)
pivot = data.pivot_table(values='result', index='user_id', columns='cats', fill_value=0)
#pivot = data.pivot_table(values='total_time', index='user_id', columns='prob_id', fill_value=0, observed=False)
#pivot = data.pivot_table(values='total_time', index='user_id', columns='cats', fill_value=0, observed=False)
#pivot = data.pivot_table(values='rel_rank', index='user_id', columns='prob_id', fill_value=0)
print(pivot)

# Subsample the pivot table
subsample_pivot = pivot.sample(n=2000, random_state=42).sample(n=20, axis=1, random_state=42)

if False:
    # Build an adjacency matrix
    np_pivot = pivot.to_numpy()
    rows, cols = np_pivot.shape
    top = np.hstack([np_pivot, np.zeros((rows, rows))])
    bot = np.hstack([np.zeros((cols, cols)), np_pivot.T])
    adj_mat = np.vstack([top, bot])

    # TMP
    deg_mat = np.diag(adj_mat.sum(axis=1))
    laplacian = deg_mat - adj_mat
    vals, vecs = np.linalg.eig(laplacian)
    vals = vals[np.argsort(vals)]
    #plt.matshow(deg_mat, cmap=plt.cm.Purples)
    plt.plot(vals)
    plt.show()
    exit()

# Fit biclustering
model = SpectralBiclustering(n_clusters=(5, 5), svd_method='arpack')
#model = SpectralCoclustering(n_clusters=5, svd_method='arpack')
#model.fit(pivot)
model.fit(subsample_pivot)

print(model.row_labels_)
with np.printoptions(threshold=np.inf, linewidth=160):
    print(model.column_labels_)

# Reorder rows and columns for display
reordered_row = pivot.iloc[np.argsort(model.row_labels_)]
re_pivot = reordered_row.iloc[:, np.argsort(model.column_labels_)]

plt.matshow(re_pivot, cmap=plt.cm.Blues, aspect='auto')

# Identify boundaries where row/column clusters change
row_boundaries = np.where(np.diff(np.sort(model.row_labels_)))[0]
col_boundaries = np.where(np.diff(np.sort(model.column_labels_)))[0]

# Draw the boundary lines between cells, offsetting by +0.5
for boundary in row_boundaries:
    plt.axhline(boundary + 0.5, color='red', linewidth=1)

for boundary in col_boundaries:
    plt.axvline(boundary + 0.5, color='red', linewidth=1)

plt.xlabel('Problem')
plt.ylabel('User')
plt.title('Biclustered User-Problem Interaction Matrix')
plt.show()
