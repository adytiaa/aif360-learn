"""
This notebook demonstrates the use of adversarial debiasing algorithm to learn
a fair classifier.
Adversarial debiasing [1] is an in-processing technique that learns a
classifier to maximize prediction accuracy and simultaneously reduce
an adversary's ability to determine the protected attribute from the
predictions. This approach leads to a fair classifier as the predictions
cannot carry any group discrimination information that the adversary can
exploit. We will see how to use this algorithm for learning models with
and without fairness constraints and apply them on the Adult dataset.
"""

import sys

import tensorflow as tf
from IPython.display import Markdown, display
from sklearn.preprocessing import MaxAbsScaler

from aiflearn.algorithms.inprocessing.adversarial_debiasing import \
    AdversarialDebiasing
from aiflearn.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions \
    import (load_preproc_data_adult)
from aiflearn.metrics import BinaryLabelDatasetMetric, ClassificationMetric

sys.path.append("../")

# Load dataset and set options



# Get the dataset and split into train and test
dataset_orig = load_preproc_data_adult()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)


# print out some labels, names, etc.
# Training Dataset shape
print(dataset_orig_train.features.shape)
# Favorable and unfavorable labels
print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
display(Markdown("#### Protected attribute names"))
print(dataset_orig_train.protected_attribute_names)
# Privileged and unprivileged protected attribute values
print(dataset_orig_train.privileged_protected_attributes,
      dataset_orig_train.unprivileged_protected_attributes)
display(Markdown("#### Dataset feature names"))
print(dataset_orig_train.feature_names)

# #### Metric for original training data

# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(
    dataset_orig_train,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print(
    "Train set: Difference in mean outcomes between unprivileged and "
    "privileged groups = %f"
    % metric_orig_train.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(
    dataset_orig_test,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)
print(
    "Test set: Difference in mean outcomes between unprivileged and "
    "privileged groups = %f"
    % metric_orig_test.mean_difference())

min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(
    dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(
    dataset_orig_test.features)
metric_scaled_train = BinaryLabelDatasetMetric(
    dataset_orig_train,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)
# Scaled dataset - Verify that the scaling does not affect the group label
# statistics

print(
    "Train set: Difference in mean outcomes between unprivileged and privileged "
    "groups = %f"
    % metric_scaled_train.mean_difference())
metric_scaled_test = BinaryLabelDatasetMetric(
    dataset_orig_test,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)
print(
    "Test set: Difference in mean outcomes between unprivileged and privileged "
    "groups = %f"
    % metric_scaled_test.mean_difference())

# ### Learn plan classifier without debiasing

# In[6]:

# Load post-processing algorithm that equalizes the odds
# Learn parameters with debias set to False
sess = tf.Session()
plain_model = AdversarialDebiasing(
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    scope_name='plain_classifier',
    debias=False,
    sess=sess)


plain_model.fit(dataset_orig_train)

# Apply the plain model to test data
dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)


# Metrics for the dataset from plain model (without debiasing)
display(Markdown("#### Plain model - without debiasing - dataset metrics"))
metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(
    dataset_nodebiasing_train,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

print(
    "Train set: Difference in mean outcomes between unprivileged and privileged"
    "groups = %f"
    % metric_dataset_nodebiasing_train.mean_difference())

metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(
    dataset_nodebiasing_test,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

print(
    "Test set: Difference in mean outcomes between unprivileged and privileged "
    "groups = %f"
    % metric_dataset_nodebiasing_test.mean_difference())

display(
    Markdown("#### Plain model - without debiasing - classification metrics"))
classified_metric_nodebiasing_test = ClassificationMetric(
    dataset_orig_test,
    dataset_nodebiasing_test,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)
print("Test set: Classification accuracy = %f" %
      classified_metric_nodebiasing_test.accuracy())
TPR = classified_metric_nodebiasing_test.true_positive_rate()
TNR = classified_metric_nodebiasing_test.true_negative_rate()
bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
print("Test set: Balanced classification accuracy = %f" %
      bal_acc_nodebiasing_test)
print("Test set: Disparate impact = %f" %
      classified_metric_nodebiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" %
      classified_metric_nodebiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" %
      classified_metric_nodebiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" %
      classified_metric_nodebiasing_test.theil_index())

# Apply in-processing algorithm based on adversarial learning


sess.close()
tf.reset_default_graph()
sess = tf.Session()


# Learn parameters with debias set to True
debiased_model = AdversarialDebiasing(
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    scope_name='debiased_classifier',
    debias=True,
    sess=sess)


debiased_model.fit(dataset_orig_train)


# Apply the plain model to test data
dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
dataset_debiasing_test = debiased_model.predict(dataset_orig_test)


# Metrics for the dataset from plain model (without debiasing)
# Plain model - without debiasing - dataset metrics
print(
    "Train set: Difference in mean outcomes between unprivileged and privileged"
    " groups = %f"
    % metric_dataset_nodebiasing_train.mean_difference())
print(
    "Test set: Difference in mean outcomes between unprivileged and privileged"
    " groups = %f"
    % metric_dataset_nodebiasing_test.mean_difference())

# Metrics for the dataset from model with debiasing
# Model - with debiasing - dataset metrics
metric_dataset_debiasing_train = BinaryLabelDatasetMetric(
    dataset_debiasing_train,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

print(
    "Train set: Difference in mean outcomes between unprivileged and privileged"
    " groups = %f"
    % metric_dataset_debiasing_train.mean_difference())

metric_dataset_debiasing_test = BinaryLabelDatasetMetric(
    dataset_debiasing_test,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

print(
    "Test set: Difference in mean outcomes between unprivileged and privileged "
    "groups = %f"
    % metric_dataset_debiasing_test.mean_difference())

# Plain model - without debiasing - classification metrics
print("Test set: Classification accuracy = %f" %
      classified_metric_nodebiasing_test.accuracy())
TPR = classified_metric_nodebiasing_test.true_positive_rate()
TNR = classified_metric_nodebiasing_test.true_negative_rate()
bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
print("Test set: Balanced classification accuracy = %f" %
      bal_acc_nodebiasing_test)
print("Test set: Disparate impact = %f" %
      classified_metric_nodebiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" %
      classified_metric_nodebiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" %
      classified_metric_nodebiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" %
      classified_metric_nodebiasing_test.theil_index())

# Model - with debiasing - classification metrics
classified_metric_debiasing_test = ClassificationMetric(
    dataset_orig_test,
    dataset_debiasing_test,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)
print("Test set: Classification accuracy = %f" %
      classified_metric_debiasing_test.accuracy())
TPR = classified_metric_debiasing_test.true_positive_rate()
TNR = classified_metric_debiasing_test.true_negative_rate()
bal_acc_debiasing_test = 0.5 * (TPR + TNR)
print(
    "Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
print("Test set: Disparate impact = %f" %
      classified_metric_debiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" %
      classified_metric_debiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" %
      classified_metric_debiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" %
      classified_metric_debiasing_test.theil_index())

#
#     References:
#     [1] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating UnwantedBiases
#     with Adversarial Learning,"
#     AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2018.


