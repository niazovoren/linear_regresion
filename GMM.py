import numpy as np
import scipy as sc
from sklearn import datasets
from sklearn.cluster import KMeans


class GMM:
    def __init__(self, data, number_of_clusters):
        self.X = data
        self.clusters = []
        self.n_clusters = number_of_clusters
        self.totals = None
        self.gamma_nk = []

    @staticmethod
    def gaussian(X, mu, cov):
        n = X.shape[1]  # number of columns
        diff = (X - mu).T
        return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(
            -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)

    def initialize_clusters(self):
        # We use the KMeans centroids to initialise the GMM
        kmeans = KMeans(self.n_clusters).fit(self.X)
        mu_k = kmeans.cluster_centers_

        for i in range(self.n_clusters):
            self.clusters.append({
                'pi_k': 1.0 / self.n_clusters,
                'mu_k': mu_k[i],
                'cov_k': np.cov(X.T)  # need to use mu_k
            })
        return self.clusters

    def expectation_step(self):  # calculating responsibility matrix
        N = self.X.shape[0]  # number of rows
        K = len(self.clusters)
        self.totals = np.zeros((N, 1), dtype=np.float64)
        gamma_nk = np.zeros((N, K), dtype=np.float64)

        for k, cluster in enumerate(self.clusters):
            pi_k = cluster['pi_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']

            gamma_nk[:, k] = (pi_k * self.gaussian(self.X, mu_k, cov_k)).ravel()

        self.totals = np.sum(gamma_nk, 1)
        gamma_nk /= np.expand_dims(self.totals, 1)
        return self.gamma_nk

    def maximization_step(self):
        N = float(self.X.shape[0])

        for k, cluster in enumerate(self.clusters):
            gamma_k = np.expand_dims(self.gamma_nk[:, k], 1)
            N_k = np.sum(gamma_k, axis=0)

            pi_k = N_k / N
            mu_k = np.sum(gamma_k * self.X, axis=0) / N_k
            cov_k = (gamma_k * (self.X - mu_k)).T @ (self.X - mu_k) / N_k

            cluster['pi_k'] = pi_k
            cluster['mu_k'] = mu_k
            cluster['cov_k'] = cov_k

    def get_likelihood(self):
        sample_likelihoods = np.log(self.totals)
        return np.sum(sample_likelihoods), sample_likelihoods

    def train_gmm(self, n_epochs):
        clusters = self.initialize_clusters()
        likelihoods = np.zeros((n_epochs,))
        scores = np.zeros((X.shape[0], self.n_clusters))
        history = []
        stop = 100
        for i in range(n_epochs):
            clusters_snapshot = []

            # This is just for our later use in the graphs
            for cluster in clusters:
                clusters_snapshot.append({
                    'mu_k': cluster['mu_k'].copy(),
                    'cov_k': cluster['cov_k'].copy()
                })

            history.append(clusters_snapshot)

            self.expectation_step()
            self.maximization_step()

            likelihood, sample_likelihoods = self.get_likelihood()
            likelihoods[i] = likelihood

            print('Epoch: ', i + 1, 'Likelihood: ', likelihood)

        scores = np.log(self.gamma_nk)

        return clusters, likelihoods, scores, sample_likelihoods, history


if __name__ == '__main__':
    data = datasets.load_iris()
    X = data.data
    gmm = GMM(X, 4)
    gmm.train_gmm(15)
