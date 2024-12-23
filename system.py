from utils import *
import numpy as np
import scipy.linalg
from typing import List


class KNNModel:
    """
    K-Nearest Neighbour classification model using cosine distance
    """

    def __init__(self, train_vectors: np.ndarray, train_labels: np.ndarray):
        self.train_vectors = train_vectors
        self.train_labels = train_labels

    def mode(self, vectors: np.ndarray) -> int:
        """Calculates the mode within a 1-d vector

        Args:
            vectors: 1-dimensional vector
        Returns:
            int: the mode of the vector
        """
        unique, count = np.unique(vectors, return_counts=True)
        mode_count = np.argmax(count)
        return unique[mode_count]

    def predict(self, test_vectors: np.ndarray, k: int = 3) -> np.ndarray:
        """Produces a 1-d vector of labels which were predicted based off the
        test_vectors used as input. Uses K-NN implementation to find the most
        closely related vectors.

        Args:
            test_vectors: input test image vector which should have been reduced
            k: number of neighbours to check for
        Returns:
            ndarray: predicted labels
        """
        x = np.dot(test_vectors, self.train_vectors.transpose())

        modtest = np.sqrt(np.sum(test_vectors * test_vectors, axis=1))
        modtrain = np.sqrt(np.sum(self.train_vectors * self.train_vectors, axis=1))

        dist = x / np.outer(modtest, modtrain.transpose())
        # sorts distances and keeps the k largest
        sorted_dist = np.argsort(dist, axis=1)[:, -k:]

        final_labels = []
        # determine the most popular neighbour and assign that as the label
        for i in range(len(sorted_dist)):
            mode_label = self.mode(self.train_labels[sorted_dist[i]])
            final_labels.append(mode_label)

        return np.array(final_labels)


class LDA:
    """
    LDA implementation using labels and images
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.images_mean = np.mean(images, axis=0)
        self.images_std = np.sqrt(np.var(images, axis=0))
        # prevent divide by 0 error
        self.images_std[self.images_std == 0] = 1
        self.labels = labels
        self.W = None

    def _computeMeanVectors(self) -> np.ndarray:
        """Compute the mean vectors for each class

        Returns:
            ndarray: mean vector for each class
        """
        unique_labels = np.unique(self.labels)

        return [
            np.mean(self.images[self.labels == label], axis=0)
            for label in unique_labels
        ]

    def _computeWithinClassScatterMatrices(
        self, mean_vectors: np.ndarray
    ) -> np.ndarray:
        """Compute the Within-Class Scatter Matrix

        Args:
            mean_vectors: calculated mean vectors for each class
        Returns:
            ndarray: within-class scatter matrix
        """
        feature_count = self.images.shape[1]
        unique_labels = np.unique(self.labels)
        temp = np.zeros((feature_count, feature_count))  # initialise matrix

        for label, mean in zip(unique_labels, mean_vectors):
            # create scatter matrix for each class
            class_scatter = np.zeros(temp.shape)
            for row in self.images[self.labels == label]:
                row, mean = row.reshape(feature_count, 1), mean.reshape(
                    feature_count, 1
                )
                class_scatter += (row - mean).dot((row - mean).T)
            # add the class scatter matrix to the within-class scatter matrix
            temp += class_scatter
        return temp

    def _computeBetweenClassScatterMatrices(
        self, mean_vectors: np.ndarray
    ) -> np.ndarray:
        """Compute the Between-Class Scatter Matrix

        Args:
            mean_vectors: calculated mean vectors for each class
        Returns:
            ndarray: between-class scatter matrix
        """
        overall_mean = np.mean(self.images, axis=0)
        feature_count = self.images.shape[1]
        unique_labels = np.unique(self.labels)
        temp = np.zeros((feature_count, feature_count))  # initialise

        for label, mean in zip(unique_labels, mean_vectors):
            # filter and then compare to the overall_mean
            n = self.images[self.labels == label, :].shape[0]
            mean = mean.reshape(feature_count, 1)
            overall_mean = overall_mean.reshape(feature_count, 1)
            temp += n * (mean - overall_mean).dot((mean - overall_mean).T)
        return temp

    def _computeTopEigenvectors(
        within_matrix: np.ndarray, between_matrix: np.ndarray, N: int = 5
    ) -> np.ndarray:
        """Compute the top N eigenvectors

        Args:
            within_matrix: within-class scatter matrix
            between_matrix: between-class scatter matrix
            N: number of eigenvectors to select
        Returns:
            ndarray: contains N eigenvectors horizontally sorted
        """
        eigenvalues, eigenvectors = np.linalg.eig(
            np.linalg.pinv(within_matrix).dot(between_matrix)
        )

        # sort eigenvalues and eigenvectors
        eigenpairs = [
            (np.abs(eigenvalues[i]), eigenvectors[:, i])
            for i in range(len(eigenvalues))
        ]
        eigenpairs = sorted(eigenpairs, key=lambda k: k[0], reverse=True)

        # select N eigenpairs from sorted ndarray
        return np.hstack(
            [eigenpairs[i][1].reshape(within_matrix.shape[0], 1) for i in range(N)]
        )

    def standardiseVector(self, vector: np.ndarray) -> np.ndarray:
        """Standardises the input vector

        Returns:
            ndarray: standardised vector
        """
        return np.array(
            (vector - self.images_mean) / self.images_std,
            dtype=float,
        )

    def centreData(self, vector: np.ndarray) -> np.ndarray:
        """Centres the data by removing the training mean"""
        return vector - self.images_mean

    def projectData(self, vector: np.ndarray) -> np.ndarray:
        """Project data onto the precomputed LDA components

        Args:
            vector: input vector
        Returns:
            ndarray: data projected onto LDA components
        """
        # requires W to have been calculated during training
        if self.W is None:
            raise Exception(
                "Top eigenvectors for LDA need to be computed during the training stage."
            )
        return np.dot(vector, self.W)

    def compute(self):
        """Computes the LDA components using default N=2 top eigenvectors"""
        mean_vectors = self._computeMeanVectors()
        within = self._computeWithinClassScatterMatrices(mean_vectors)
        between = self._computeBetweenClassScatterMatrices(mean_vectors)
        self.W = LDA._computeTopEigenvectors(within, between)


class PCA:
    """
    PCA implementation
    """

    def __init__(self, images: np.ndarray, features: int):
        """
        Args:
            images: input training images
            features: number of principal components to find
        """
        self.images = images
        # store pre-standardised mean and standard deviation
        self.images_mean = np.mean(images)
        self.images_std = np.sqrt(np.var(images))
        self.features = features
        self.v = None

    def calculatePrincipalComponents(self):
        """Calculates principal components and stores result in self.v"""
        cov_matrix = np.cov(self.images, rowvar=0)
        N = cov_matrix.shape[0]
        # ensure that the components can't exceed the total number of features
        if self.features > N:
            self.features = N

        _, e_vector = scipy.linalg.eigh(
            cov_matrix, subset_by_index=(N - self.features, N - 1)
        )
        self.v = np.fliplr(e_vector)  # descending order

    def projectToPrincipalComponents(self, images: np.ndarray) -> np.ndarray:
        # check to ensure that principal components were already computed
        if self.v is None:
            raise Exception(
                "Principal components need to have been computed during the training stage."
            )
        return np.dot((images - self.images_mean), self.v)

    def standardiseVector(self, vector: np.ndarray) -> np.ndarray:
        """Standardises the input vector by setting all mean = 0 and std = 1

        Returns:
            ndarray: standardised vector
        """
        return np.array(
            (vector - self.images_mean) / self.images_std,
        )


def divergence(class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """Calculates the vector of 1-D divergences from two classes

    Args:
        class1: data matrix for class 1
        class2: data matrix for class 2
    Returns:
        ndarray: a vector of 1-D divergence scores
    """
    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)
    var1 = np.var(class1, axis=0)
    var2 = np.var(class2, axis=0)

    return 0.5 * (var1 / var2 + var2 / var1 - 2) + 0.5 * (mean1 - mean2) * (
        mean1 - mean2
    ) * (1.0 / var1 + 1.0 / var2)


def findBestFeatures(images: np.ndarray, labels: np.ndarray, N: int = 9) -> List[int]:
    """Finds the best features given a feature vector.

    Args:
        images: input feature vector
        labels: input label vector
        N: number of features
    Returns:
        List[int]: list of the top N features
    """
    # prevent N features exceeding total features
    if N > images.shape[1]:
        N = images.shape[1]

    d = 0
    features = []
    for char1 in np.arange(0, 10):
        # calculate divergence between all classes
        char1_data = images[labels == char1, :]
        for char2 in np.arange(char1 + 1, 10):
            char2_data = images[labels == char2, :]
            d12 = divergence(char1_data, char2_data)
            d = d + d12
            # reduce to top N features
            sorted_indexes = np.argsort(-d)
            features = sorted_indexes[0:N]
    return features


def evenlyChooseImages(
    images: np.ndarray, labels: np.ndarray, amount: int
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly distribute the amount of images so each label
    has an even number of samples

    Args:
        images: input image feature vector
        labels: 1-d labels vector
        amount: total number of samples
    Returns:
        tuple of two elements:
            - ndarray: hosen images sampled from image parameter
            - ndarray: matching chosen labels sampled from label parameter
    """
    unique_labels = np.unique(labels)
    per_letter = amount // len(unique_labels)

    new_images = None
    new_labels = None

    # go through each label and select samples for each label
    for i in range(len(unique_labels)):
        indices = np.where(labels == unique_labels[i])[0]
        chosen_indices = indices[:per_letter]

        # store the chosen images & labels together with previously selected ones
        if new_images is None:
            new_images = images[chosen_indices, :]
        else:
            new_images = np.vstack([new_images, images[chosen_indices, :]], dtype=float)

        if new_labels is None:
            new_labels = labels[chosen_indices]
        else:
            new_labels = np.hstack([new_labels, labels[chosen_indices]])

    return new_images, new_labels


def generateNoiseData(
    images: np.ndarray, labels: np.ndarray, amount: int = 300, max_noise: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generates noise data to be used for noise injection

    Args:
        images: input image feature vector
        labels: input label vector
        amount: number of samples to generate
        max_noise: maximum noise scale to be used
    Returns:
        tuple of two elements:
            - ndarray: newly updated image vector with generated noise images added
            - ndarray: newly updated label vector with generated labels added
    """
    new_images, new_labels = evenlyChooseImages(images, labels, amount)

    # randomly add noise to each image in the sample
    for i in range(new_images.shape[0]):
        noise = np.random.normal(0, max_noise, new_images.shape[1])
        new_images[i] += noise

    return np.vstack([images, new_images]), np.hstack([labels, new_labels])


def generateMaskedData(
    images: np.ndarray, labels: np.ndarray, amount: int = 300, masked_pixels: int = 180
) -> tuple[np.ndarray, np.ndarray]:
    """Generates masked data to be used for mask injection

    Args:
        images: input image feature vector
        labels: input label vector
        amount: number of samples to generate
        masked_pixels: number of pixels to be masked
    Returns:
        tuple of two elements:
            - ndarray: newly updated image vector with generated noise images added
            - ndarray: newly updated label vector with generated labels added
    """
    # prevent masked pixels from being bigger than the image size 28x28
    if masked_pixels > images.shape[1]:
        masked_pixels = images.shape[1]

    new_images, new_labels = evenlyChooseImages(images, labels, amount)

    # add random mask for each image sample
    for i in range(new_images.shape[0]):
        random_masks = np.random.choice(
            new_images.shape[1], size=masked_pixels, replace=False
        )
        # set random pixels to black
        new_images[i, random_masks] = 0

    return np.vstack([images, new_images]), np.hstack([labels, new_labels])


def image_to_reduced_feature(
    images: np.ndarray, labels: np.ndarray | None = None, split: str = "test"
) -> np.ndarray:
    """Reduces the image feature vector by using PCA and standardisation.
    If ran with split="train", noise and mask data will be injected into the dataset.
    Args:
        images: input image feature vector
        labels: input feature labels
        split: type of training split to use
    Returns:
        ndarray: updated, reduced feature vector
    """

    pca_data = None
    if split != "train":
        # load the model to retain training mean for standardisation
        pca_model: PCA = load_model("pca_model.pkl")
        standardised_images = pca_model.standardiseVector(images)
        pca_data = pca_model.projectToPrincipalComponents(images)
    else:
        # inject noise and masked data into the dataset
        images, labels = generateNoiseData(images, labels)
        images, labels = generateMaskedData(images, labels, masked_pixels=100)

        # perform PCA
        pca_model = PCA(images, 100)
        standardised_images = pca_model.standardiseVector(images)
        pca_model.images = standardised_images
        pca_model.calculatePrincipalComponents()
        pca_data = pca_model.projectToPrincipalComponents(images)
        save_model(pca_model, "pca_model.pkl")

    return pca_data


def training_model(train_features: np.ndarray, train_labels: np.ndarray) -> KNNModel:
    """Create the training model for classification

    Args:
        train_features: reduced training feature vector
        train_labels: feature labels not including the injected labels
    """
    # update labels to match the injected data
    old_train_features = train_features[:-600]  # remove the injected feature vectors
    _, labels = generateNoiseData(old_train_features, train_labels)
    _, labels = generateMaskedData(old_train_features, labels)
    return KNNModel(train_features, labels)
