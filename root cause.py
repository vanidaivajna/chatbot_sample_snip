import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load conversation data
df = pd.read_csv('conversation_data.csv')

# Vectorize conversation text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Cluster conversation text using K-Means
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

# Identify the cluster with the highest number of bot failures
bot_failure_cluster = kmeans.labels_[df['bot_failed']]
highest_failure_cluster = max(set(bot_failure_cluster), key=bot_failure_cluster.tolist().count)

# Print the most frequent terms in the highest failure cluster
terms = vectorizer.get_feature_names()
cluster_terms = X[kmeans.labels_ == highest_failure_cluster].sum(axis=0).tolist()[0]
cluster_terms_scores = list(zip(terms, cluster_terms))
cluster_terms_scores.sort(key=lambda x: x[1], reverse=True)
print(f"Most frequent terms in the highest failure cluster: {cluster_terms_scores[:10]}")


#---------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Load the data
data = pd.read_csv("data.csv")

# Vectorize the data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data["text"])

# Define the clustering model
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

# Fit the model to the data
model = model.fit(X.toarray())

# Visualize the dendrogram
fig = plt.figure(figsize=(10, 7))
dn = dendrogram(model, truncate_mode='level', p=5)
plt.show()

# Get the cluster labels
labels = model.labels_

# Add the cluster labels to the dataframe
data["cluster"] = labels

# Print the number of documents in each cluster
print(data["cluster"].value_counts())


#--------------------------------------------------------------------------------
