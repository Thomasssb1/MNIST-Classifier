from utils import *
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class CNNModel:
    def __init__(self, train_vectors, train_labels):
        self.train_vectors = train_vectors
        self.train_labels = train_labels

    def cosineDistance(self, dist):
        nearest = np.argmax(dist, axis=1)
        mdist = np.max(dist, axis=1)
        return self.train_labels[nearest], mdist

    def predict(self, test_vectors):
        print(test_vectors.shape, self.train_vectors.shape)
        print(self.train_labels.shape)
        x = np.dot(test_vectors, self.train_vectors.transpose())

        modtest = np.sqrt(np.sum(test_vectors * test_vectors, axis=1))
        modtrain = np.sqrt(np.sum(self.train_vectors * self.train_vectors, axis=1))

        modtest[modtest == 0] = 1e-10
        modtrain[modtrain == 0] = 1e-10

        dist = x / np.outer(modtest, modtrain.transpose())

        labels, mdist = self.cosineDistance(dist)

        return labels


class LDA:
    def __init__(self, images, labels):
        self.labelled_images = addLabels(images, labels)
        self.labels = labels

    def _computeMeanVectors(self):
        temp = np.zeros((10, self.labelled_images.shape[1] - 1))

        for label in range(0, 10):
            mean = np.mean(
                self.labelled_images[self.labelled_images[:, 0] == label, 1:], axis=0
            )
            temp[label, :] = mean
        return temp

    def _computeWithinClassScatterMatrices(self, mean_vectors):
        feature_count = self.labelled_images.shape[1] - 1
        temp = np.zeros((feature_count, feature_count))

        for label, mean in zip(range(0, 10), mean_vectors):
            class_scatter = np.zeros(temp.shape)
            for row in self.labelled_images[self.labelled_images[:, 0] == label, 1:]:
                row, mean = row.reshape(feature_count, 1), mean.reshape(
                    feature_count, 1
                )
                class_scatter += (row - mean).dot((row - mean).T)
            temp += class_scatter
        return temp

    def _computeBetweenClassScatterMatrices(self, mean_vectors):
        overall_mean = np.mean(self.labelled_images[:, 1:], axis=0)
        feature_count = self.labelled_images.shape[1] - 1
        temp = np.zeros((feature_count, feature_count))

        for label, mean in enumerate(mean_vectors):
            n = self.labelled_images[self.labelled_images[:, 0] == label, 1:].shape[0]
            mean = mean.reshape(feature_count, 1)
            overall_mean = overall_mean.reshape(feature_count, 1)
            temp += n * (mean - overall_mean).dot((mean - overall_mean).T)
        return temp

    def _computeTopEigenvectors(within_matrix, between_matrix, N=40):
        eigenvalues, eigenvectors = np.linalg.eig(
            np.linalg.pinv(within_matrix).dot(between_matrix)
        )

        eigenpairs = [
            (np.abs(eigenvalues[i]), eigenvectors[:, i])
            for i in range(len(eigenvalues))
        ]
        eigenpairs = sorted(eigenpairs, key=lambda k: k[0], reverse=True)

        return np.hstack(
            [eigenpairs[i][1].reshape(within_matrix.shape[0], 1) for i in range(N)]
        )

    def compute(self):
        mean_vectors = self._computeMeanVectors()
        within = self._computeWithinClassScatterMatrices(mean_vectors)
        between = self._computeBetweenClassScatterMatrices(mean_vectors)
        W = LDA._computeTopEigenvectors(within, between)
        lda_data = self.labelled_images[:, 1:].dot(W)
        return addLabels(lda_data, self.labels, datatype=complex)


class PCA:
    def __init__(self, images, features):
        self.images = images
        self.features = features
        self.v = None

    def calculatePrincipalComponents(self):
        cov_matrix = np.cov(self.images, rowvar=0)
        N = cov_matrix.shape[0]  # number of features
        if self.features > N:
            self.features = N
        # rowvar = 0, as each feature is a column
        e_value, e_vector = scipy.linalg.eigh(
            cov_matrix, subset_by_index=(N - self.features, N - 1)
        )
        self.v = np.fliplr(e_vector)  # descending order

    def projectToPrincipalComponents(self, images):
        if self.v is None:
            raise Exception(
                "Principal components need to have been computed during the training stage."
            )
        pca_train_data = np.dot((images - np.mean(self.images)), self.v)
        return pca_train_data

    def standardiseVector(self, vector):
        return np.array(
            (vector - np.mean(self.images)) / np.sqrt(np.var(self.images)), dtype=float
        )


def addLabels(images, labels, datatype=None):
    """"""
    new_size = (images.shape[0], images.shape[1] + 1)
    temp = np.zeros(new_size, dtype=datatype)
    temp[:, 1:] = images
    temp[:, 0] = labels
    return temp


def divergence(class1, class2):
    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)
    var1 = np.var(class1, axis=0)
    var2 = np.var(class2, axis=0)

    return 0.5 * (var1 / var2 + var2 / var1 - 2) + 0.5 * (mean1 - mean2) * (
        mean1 - mean2
    ) * (1.0 / var1 + 1.0 / var2)


def findBestFeatures(labelled_pca_images, N=9):
    if N > labelled_pca_images.shape[0]:
        N = labelled_pca_images.shape[0]

    d = 0
    features = []
    for char1 in np.arange(0, 10):
        char1_data = labelled_pca_images[labelled_pca_images[:, 0] == char1, 1:]
        for char2 in np.arange(char1 + 1, 10):
            char2_data = labelled_pca_images[labelled_pca_images[:, 0] == char2, 1:]
            d12 = divergence(char1_data, char2_data)
            d = d + d12
            sorted_indexes = np.argsort(-d)
            features = sorted_indexes[0:N]
    return features


def image_to_reduced_feature(images, labels=None, split="test"):
    """
    Finds the most suitable features, reducing to N-features using PCA/LDA
    By default the images are 28x28 pixels
    """

    pca_data = None
    if split != "train":
        pca_model: PCA = load_model("pca_model.pkl")
        standardised_images = pca_model.standardiseVector(images)
        pca_data = pca_model.projectToPrincipalComponents(standardised_images)
    else:
        pca_model = PCA(images, 20)
        standardised_images = pca_model.standardiseVector(images)
        pca_model.images = standardised_images
        pca_model.calculatePrincipalComponents()
        pca_data = pca_model.projectToPrincipalComponents(standardised_images)
        save_model(pca_model, "pca_model.pkl")

    labelled_pca = addLabels(pca_data, labels)
    best_features = findBestFeatures(labelled_pca, 9)

    return pca_data[:, best_features]


def training_model(train_features, train_labels):
    return CNNModel(train_features, train_labels)
