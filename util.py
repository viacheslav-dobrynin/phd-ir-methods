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


def create_model_name(model, desc=""):
    return ("iVAE_s" + str(model.latent_dim) +
            "_elbo_" + str(model.elbo_loss_alpha).replace(".", "") +
            f"_{model.regularization_loss.__class__.__name__}_{str(model.regularization_loss.alpha).replace('.', '')}" +
            "_dist_" + str(model.distance_loss.alpha).replace(".", "") +
            "_n_clusts_" + str(model.embs_kmeans.n_clusters) +
            desc)
