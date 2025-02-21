# MNIST-Classifier 
An implementation of a KNN classifier using PCA to classify numbers from the MNIST dataset with a __93.20%__ accuracy on noise data and __79.20%__ accuracy on masked data.

## Overview
The model uses PCA to reduce the feature space from 784 to 100 to increase accuracy. <br>
K = 15 was chosen for the weighted KNN using distance similarity in order to address noise and occluded data. <br>

The feature extraction method implemented uses PCA and standardisation. Before the image is processed using PCA, 
it is standardised ensuring that each sample feature has a mean of 0 and variance of 1 to prevent principal 
components becoming biased towards features with higher variances. In this step noise and 
masked data is injected into the training set to increase the robustness of the model against occluded and malformed input. 
Once the new data has been introduced, the new image feature vector is standardised and PCA is performed on it to 
calculate the 100 principal components - reducing the feature space from 784 to 100. PCA is performed to reduce the high-dimensional image data, which leads to 
more efficient computations. The hyperparameter value chosen was determined as it needed to have enough principal 
components to remain robust to noise and masking whilst still being able to classify clean data but reducing the 
feature set was important as computations can be more efficient and accurate. <br>

LDA is also implemented for this model but did not perform well. A reason for LDA 
performing poorly could possibly be due to class overlaps and how certain numbers are very similar, such as 8 and 0 
because it maximises the ratio of between-class variance and within-class variance whereas PCA maximises the 
variance of the whole set. 

The classifier method implemented is a K-Nearest Neighbour algorithm, where k=15. The hyperparameter k was chosen 
to be 15 based off the fact that the test data was to be performed on noise and masked data, so a higher number smooths 
out the decision boundary allowing for better generalisation. When k is low, the determined label can be influenced 
easily by a single misclassification. Another reason for k being 15 is because odd numbers can be used to break ties, 
where even numbers can result in ties which doesn’t lead to a majority vote. Ultimately, the value was tuned based off 
results when evaluating the model with different values of k – where k=15 was found to be most effective. <br>
K-NN was used over a condensed nearest neighbour as the training data is small and so reducing it further 
seemed redundant and came with the risk of data loss. The distance chosen for the K-NN was the cosine distance, which is 
scale invariant, and is used as it measures the orientation as opposed to scale. Despite optimising the K-NN with efficient 
parameters, it still has drawbacks such as computation time, which has been partially reduced due to the dimensionality 
reduction step, it can be further improved by adding weighting or editing the distance measure. However, compared with a 
standard nearest-neighbour approach, with no k, allowing for majority voting based on closest neighbours has increased the 
accuracy of the classifier.

## Accuracy
Accuracy on noise_test set: 93.20% <br>
Accuracy on mask_test set: 79.20% <br>
Accuracy can vary for each run due to the images selected for noise and mask injection.
