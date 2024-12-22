from utils import *
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def addLabels(images, labels, datatype=None):
    """"""
    new_size = (images.shape[0], images.shape[1] + 1)
    temp = np.zeros(new_size, dtype=datatype)
    temp[:, 1:] = images
    temp[:, 0] = labels
    return temp


def calculatePrincipalComponents(images, final_num_features=40):
    cov_matrix = np.cov(images, rowvar=0)
    N = cov_matrix.shape[0]  # number of features
    if final_num_features > N:
        final_num_features = N

    # rowvar = 0, as each feature is a column
    e_value, e_vector = scipy.linalg.eigh(
        cov_matrix, subset_by_index=(N - final_num_features, N - 1)
    )
    return np.fliplr(e_vector)  # descending order


def projectToPrincipalComponents(images, v):
    pca_train_data = np.dot((images - np.mean(images)), v)
    return pca_train_data


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


def computeMeanVectors(labelled_images):
    temp = np.zeros((10, labelled_images.shape[1] - 1))

    for label in range(0, 10):
        mean = np.mean(labelled_images[labelled_images[:, 0] == label, 1:], axis=0)
        temp[label, :] = mean
    return temp


def computeWithinClassScatterMatrices(labelled_images, mean_vectors):
    feature_count = labelled_images.shape[1] - 1
    temp = np.zeros((feature_count, feature_count))

    for label, mean in zip(range(0, 10), mean_vectors):
        class_scatter = np.zeros(temp.shape)
        for row in labelled_images[labelled_images[:, 0] == label, 1:]:
            row, mean = row.reshape(feature_count, 1), mean.reshape(feature_count, 1)
            class_scatter += (row - mean).dot((row - mean).T)
        temp += class_scatter
    return temp


def computeBetweenClassScatterMatrices(labelled_images, mean_vectors):
    overall_mean = np.mean(labelled_images[:, 1:], axis=0)
    feature_count = labelled_images.shape[1] - 1
    temp = np.zeros((feature_count, feature_count))

    for label, mean in enumerate(mean_vectors):
        n = labelled_images[labelled_images[:, 0] == label, 1:].shape[0]
        mean = mean.reshape(feature_count, 1)
        overall_mean = overall_mean.reshape(feature_count, 1)
        temp += n * (mean - overall_mean).dot((mean - overall_mean).T)
    return temp


def computeTopEigenvectors(within_matrix, between_matrix, N=40):
    eigenvalues, eigenvectors = np.linalg.eig(
        np.linalg.pinv(within_matrix).dot(between_matrix)
    )

    eigenpairs = [
        (np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))
    ]
    eigenpairs = sorted(eigenpairs, key=lambda k: k[0], reverse=True)

    return np.hstack(
        [eigenpairs[i][1].reshape(within_matrix.shape[0], 1) for i in range(N)]
    )


def image_to_reduced_feature(images, labels=None, split="train"):
    """
    Finds the most suitable features, reducing to N-features using PCA/LDA
    By default the images are 28x28 pixels
    """

    labelled_images = addLabels(images, labels)
    mean_vectors = computeMeanVectors(labelled_images)
    within = computeWithinClassScatterMatrices(labelled_images, mean_vectors)
    between = computeBetweenClassScatterMatrices(labelled_images, mean_vectors)

    print(within.shape)
    print(between.shape)

    W = computeTopEigenvectors(within, between, N=40)
    data_lda = labelled_images[:, 1:].dot(W)
    labelled_lda = addLabels(data_lda, labels, datatype=complex)

    best_features = findBestFeatures(labelled_lda)
    print(images[:, best_features].shape)

    return images[:, best_features]


def training_model(train_features, train_labels):
    return NullModel()
