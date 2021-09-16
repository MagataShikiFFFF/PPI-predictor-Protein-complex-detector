import networkx as nx

G = nx.Graph()

#Build graph G by known PPI and predicted PPI
for i in range(len(ppi)):
	G.add_edge(ppi[i][0], ppi[i][1], weight=1)

from sklearn.cluster import SpectralClustering

X = nx.normalized_laplacian_matrix(G) 

# Set k clusters
k1 = k
clustering = SpectralClustering(n_clusters=k1,
         assign_labels="discretize",
         random_state=0).fit(X)
