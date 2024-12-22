from utils import *
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def addLabels(images, labels):
    """"""
    new_size = (images.shape[0], images.shape[1] + 1)
    temp = np.zeros(new_size)
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


def image_to_reduced_feature(images, labels=None, split="train"):
    """
    Finds the most suitable features, reducing to N-features using PCA/LDA
    By default the images are 28x28 pixels
    """

    v = calculatePrincipalComponents(images, 100)
    pca_train_data = projectToPrincipalComponents(images, v)
    # add labels to the first column
    labelled_images = addLabels(pca_train_data, labels)
    print(labelled_images.shape)
    best_features = findBestFeatures(labelled_images)
    print(best_features)

    print(images[:, best_features].shape)

    return images[:, best_features]


def training_model(train_features, train_labels):
    return NullModel()
