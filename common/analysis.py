import sklearn
from matplotlib import pyplot as plt


def plot_elbow_method(embs):
    sum_of_squared_distances = []
    K = range(2, 50, 5)
    for k in K:
        km = sklearn.cluster.KMeans(n_clusters=k)
        km = km.fit(embs)
        sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
