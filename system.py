from utils import *
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class KNNModel:
    def __init__(self, train_vectors, train_labels):
        self.train_vectors = train_vectors
        self.train_labels = train_labels

    def mode_numpy(arr):
        """Calculate the mode of an array using pure numpy."""
        unique, counts = np.unique(arr, return_counts=True)
        index = np.argmax(counts)
        return unique[index]

    def predict(self, test_vectors, k=3):
        x = np.dot(test_vectors, self.train_vectors.transpose())

        modtest = np.sqrt(np.sum(test_vectors * test_vectors, axis=1))
        modtrain = np.sqrt(np.sum(self.train_vectors * self.train_vectors, axis=1))

        dist = x / np.outer(modtest, modtrain.transpose())
        print(dist.shape)
        sorted_dist = np.argsort(dist, axis=1)[:, -k:]

        final_labels = []
        for i in range(len(sorted_dist)):
            mode_label = KNNModel.mode_numpy(self.train_labels[sorted_dist[i]])
            final_labels.append(mode_label)

        return np.array(final_labels)


class LDA:
    def __init__(self, images, labels):
        self.images = images
        self.images_mean = np.mean(images, axis=0)
        self.images_std = np.sqrt(np.var(images, axis=0))
        self.images_std[self.images_std == 0] = 1
        self.labels = labels
        self.W = None

    def _computeMeanVectors(self):
        unique_labels = np.unique(self.labels)

        temp = [
            np.mean(self.images[self.labels == label], axis=0)
            for label in unique_labels
        ]
        return temp

    def _computeWithinClassScatterMatrices(self, mean_vectors):
        feature_count = self.images.shape[1]
        unique_labels = np.unique(self.labels)
        temp = np.zeros((feature_count, feature_count))

        for label, mean in zip(unique_labels, mean_vectors):
            class_scatter = np.zeros(temp.shape)
            for row in self.images[self.labels == label]:
                row, mean = row.reshape(feature_count, 1), mean.reshape(
                    feature_count, 1
                )
                class_scatter += (row - mean).dot((row - mean).T)
            temp += class_scatter
        return temp

    def _computeBetweenClassScatterMatrices(self, mean_vectors):
        overall_mean = np.mean(self.images, axis=0)
        feature_count = self.images.shape[1]
        unique_labels = np.unique(self.labels)
        temp = np.zeros((feature_count, feature_count))

        for label, mean in zip(unique_labels, mean_vectors):
            n = self.images[self.labels == label, :].shape[0]
            mean = mean.reshape(feature_count, 1)
            overall_mean = overall_mean.reshape(feature_count, 1)
            temp += n * (mean - overall_mean).dot((mean - overall_mean).T)
        return temp

    def _computeTopEigenvectors(within_matrix, between_matrix, N=5):
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

    def standardiseVector(self, vector):
        return np.array(
            (vector - self.images_mean) / self.images_std,
            dtype=float,
        )

    def centreData(self, vector):
        return vector - self.images_mean

    def projectData(self, vector):
        if self.W is None:
            raise Exception(
                "Top eigenvectors for LDA need to be computed during the training stage."
            )
        return np.dot(vector, self.W)

    def compute(self):
        mean_vectors = self._computeMeanVectors()
        within = self._computeWithinClassScatterMatrices(mean_vectors)
        between = self._computeBetweenClassScatterMatrices(mean_vectors)
        self.W = LDA._computeTopEigenvectors(within, between)
        print(self.W.shape)


class PCA:
    def __init__(self, images, features):
        self.images = images
        self.images_mean = np.mean(images)
        self.images_std = np.sqrt(np.var(images))
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
        pca_train_data = np.dot((images - self.images_mean), self.v)
        return pca_train_data

    def standardiseVector(self, vector):
        # std[std == 0] = 1
        normal = np.array(
            (vector - self.images_mean) / self.images_std,
        )
        return normal


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


def evenlyChooseImages(images, labels, amount):
    unique_labels = np.unique(labels)
    per_letter = amount // len(unique_labels)

    new_images = None
    new_labels = None

    for i in range(len(unique_labels)):
        indices = np.where(labels == unique_labels[i])[0]
        random_indices = indices[:per_letter]

        if new_images is None:
            new_images = images[random_indices, :]
        else:
            new_images = np.vstack([new_images, images[random_indices, :]], dtype=float)

        if new_labels is None:
            new_labels = labels[random_indices]
        else:
            new_labels = np.hstack([new_labels, labels[random_indices]])

    return new_images, new_labels


def generateNoiseData(images, labels, amount=300, max_noise=5):
    new_images, new_labels = evenlyChooseImages(images, labels, amount)

    for i in range(new_images.shape[0]):
        noise = np.random.normal(0, max_noise, new_images.shape[1])
        new_images[i] += noise

    return np.vstack([images, new_images]), np.hstack([labels, new_labels])


def generateMaskedData(images, labels, amount=300, masked_pixels=180):
    if masked_pixels > images.shape[1]:
        masked_pixels = images.shape[1]

    new_images, new_labels = evenlyChooseImages(images, labels, amount)

    for i in range(new_images.shape[0]):
        random_masks = np.random.choice(
            new_images.shape[1], size=masked_pixels, replace=False
        )
        new_images[i, random_masks] = 0

    return np.vstack([images, new_images]), np.hstack([labels, new_labels])


def image_to_reduced_feature(images, labels=None, split="test"):
    """
    Finds the most suitable features, reducing to N-features using PCA/LDA
    By default the images are 28x28 pixels
    """

    pca_data = None
    if split != "train":
        pca_model: PCA = load_model("pca_model.pkl")
        standardised_images = pca_model.standardiseVector(images)
        pca_data = pca_model.projectToPrincipalComponents(images)
    else:
        images, labels = generateNoiseData(images, labels)
        # images, labels = generateMaskedData(images, labels, masked_pixels=100)

        pca_model = PCA(images, 460)
        standardised_images = pca_model.standardiseVector(images)
        pca_model.images = standardised_images
        pca_model.calculatePrincipalComponents()
        pca_data = pca_model.projectToPrincipalComponents(images)
        save_model(pca_model, "pca_model.pkl")

    labelled_pca = addLabels(pca_data, labels, datatype=complex)
    best_features = findBestFeatures(labelled_pca, 9)

    return pca_data[:, best_features]


def training_model(train_features, train_labels):
    old_train_features = train_features[:-300]
    _, labels = generateNoiseData(old_train_features, train_labels)
    # _, labels = generateMaskedData(old_train_features, labels)
    return KNNModel(train_features, labels)
