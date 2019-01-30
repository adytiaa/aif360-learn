#!/usr/bin/env python
# coding: utf-8


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.append("../")
from aiflearn.datasets import BinaryLabelDataset
from aiflearn.datasets import AdultDataset, GermanDataset, CompasDataset
from aiflearn.metrics import BinaryLabelDatasetMetric
from aiflearn.metrics import ClassificationMetric
from aiflearn.metrics.utils import compute_boolean_conditioning_vector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from aiflearn.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aiflearn.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier
from aiflearn.algorithms.inprocessing.celisMeta.utils import getStats
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




display(Markdown("### Meta-Algorithm for fair classification."))
display(Markdown("The fairness metrics to be optimized have to specified as \"input\". Currently we can handle the following fairness metrics."))
display(Markdown("Statistical Rate, False Positive Rate, True Positive Rate, False Negative Rate, True Negative Rate,"))
display(Markdown("Accuracy Rate, False Discovery Rate, False Omission Rate, Positive Predictive Rate, Negative Predictive Rate."))
display(Markdown("#### -----------------------------"))
display(Markdown("The example below considers the case of False Discovery Parity."))



dataset_orig = load_preproc_data_adult()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)




display(Markdown("#### Training Dataset shape"))
print(dataset_orig_train.features.shape)
display(Markdown("#### Favorable and unfavorable labels"))
print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
display(Markdown("#### Protected attribute names"))
print(dataset_orig_train.protected_attribute_names)
display(Markdown("#### Privileged and unprivileged protected attribute values"))
print(dataset_orig_train.privileged_protected_attributes, 
      dataset_orig_train.unprivileged_protected_attributes)
display(Markdown("#### Dataset feature names"))
print(dataset_orig_train.feature_names)




display(Markdown("#### Training Dataset shape"))
print(dataset_orig_train.features.shape)
display(Markdown("#### Favorable and unfavorable labels"))
print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
display(Markdown("#### Protected attribute names"))
print(dataset_orig_train.protected_attribute_names)
display(Markdown("#### Privileged and unprivileged protected attribute values"))
print(dataset_orig_train.privileged_protected_attributes, 
      dataset_orig_train.unprivileged_protected_attributes)
display(Markdown("#### Dataset feature names"))
print(dataset_orig_train.feature_names)




metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())




min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Scaled dataset - Verify that the scaling does not affect the group label statistics"))
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())


# Get classifier without fairness constraints
biased_model = MetaFairClassifier(tau=0, sensitive_attr="sex")
biased_model.fit(dataset_orig_train)



# Apply the unconstrained model to test data
dataset_bias_test = biased_model.predict(dataset_orig_test)

predictions = [1 if y == dataset_orig_train.favorable_label else -1 for y in list(dataset_bias_test.labels)]
y_test = np.array([1 if y == [dataset_orig_train.favorable_label] else -1 for y in dataset_orig_test.labels])
x_control_test = pd.DataFrame(data=dataset_orig_test.features, columns=dataset_orig_test.feature_names)["sex"]

acc, sr, unconstrainedFDR = getStats(y_test, predictions, x_control_test)
print(unconstrainedFDR)



# Learn debiased classifier
tau = 0.8
debiased_model = MetaFairClassifier(tau=tau, sensitive_attr="sex")
debiased_model.fit(dataset_orig_train)



# Apply the debiased model to test data
dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
dataset_debiasing_test = debiased_model.predict(dataset_orig_test)


# Metrics for the dataset from model with debiasing
display(Markdown("#### Model - with debiasing - dataset metrics"))
metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_train.mean_difference())

metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_test.mean_difference())



display(Markdown("#### Model - with debiasing - classification metrics"))
classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test, 
                                                 dataset_debiasing_test,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
TPR = classified_metric_debiasing_test.true_positive_rate()
TNR = classified_metric_debiasing_test.true_negative_rate()
bal_acc_debiasing_test = 0.5*(TPR+TNR)
print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())


### Testing 
predictions = list(dataset_debiasing_test.labels)
predictions = [1 if y == dataset_orig_train.favorable_label else -1 for y in predictions]
y_test = np.array([1 if y == [dataset_orig_train.favorable_label] else -1 for y in dataset_orig_test.labels])
x_control_test = pd.DataFrame(data=dataset_orig_test.features, columns=dataset_orig_test.feature_names)["sex"]

acc, sr, fdr = getStats(y_test, predictions, x_control_test)
print(fdr, unconstrainedFDR)
assert(fdr >= unconstrainedFDR)



biased_model = MetaFairClassifier(tau=0, sensitive_attr="race")
biased_model.fit(dataset_orig_train)

dataset_bias_test = biased_model.predict(dataset_orig_test)

predictions = [1 if y == dataset_orig_train.favorable_label else -1 for y in list(dataset_bias_test.labels)]
y_test = np.array([1 if y == [dataset_orig_train.favorable_label] else -1 for y in dataset_orig_test.labels])
x_control_test = pd.DataFrame(data=dataset_orig_test.features, columns=dataset_orig_test.feature_names)["race"]

acc, sr, unconstrainedFDR = getStats(y_test, predictions, x_control_test)




display(Markdown("#### Running the algorithm for different tau values"))

accuracies, false_discovery_rates, statistical_rates = [], [], []
s_attr = "race"
# Converting to form used by celisMeta.utils file
y_test = np.array([1 if y == [dataset_orig_train.favorable_label] else -1 for y in dataset_orig_test.labels])
x_control_test = pd.DataFrame(data=dataset_orig_test.features, columns=dataset_orig_test.feature_names)[s_attr]

all_tau = np.linspace(0.1, 0.9, 9)
for tau in all_tau:
    print("Tau: %.2f" % tau)
    debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=s_attr)
    debiased_model.fit(dataset_orig_train)
    
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    predictions = dataset_debiasing_test.labels
    predictions = [1 if y == dataset_orig_train.favorable_label else -1 for y in predictions]
    
    acc, sr, fdr = getStats(y_test, predictions, x_control_test)
    
    ## Testing
    assert (tau < unconstrainedFDR) or (fdr >= unconstrainedFDR)
    
    accuracies.append(acc)
    false_discovery_rates.append(fdr)
    statistical_rates.append(sr)
    


display(Markdown("### Plot of accuracy and output fairness vs input constraint (tau)"))

display(Markdown("#### Output fairness is represented by $\gamma_{fdr}$, which is the ratio of false discovery rate of different sensitive attribute values."))

fig, ax1 = plt.subplots(figsize=(13,7))
ax1.plot(all_tau, accuracies, color='r')
ax1.set_title('Accuracy and $\gamma_{fdr}$ vs Tau', fontsize=16, fontweight='bold')
ax1.set_xlabel('Input Tau', fontsize=16, fontweight='bold')
ax1.set_ylabel('Accuracy', color='r', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)

ax2 = ax1.twinx()
ax2.plot(all_tau, false_discovery_rates, color='b')
ax2.set_ylabel('$\gamma_{fdr}$', color='b', fontsize=16, fontweight='bold')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)




# # 
#     References:
#         Celis, L. E., Huang, L., Keswani, V., & Vishnoi, N. K. (2018). 
#         "Classification with Fairness Constraints: A Meta-Algorithm with Provable Guarantees.""

