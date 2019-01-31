"""

This notebook demonstrates the use of the learning fair representations
algorithm for bias mitigation.

Learning fair representations [1] is a pre-processing technique that finds a
latent representation which encodes the data well but obfuscates information
about protected attributes. We will see how to use this algorithm for learning
representations that encourage individual fairness and apply them on the
 Adult dataset.
"""



get_ipython().run_line_magic('matplotlib', 'inline')
# Load all necessary packages
import sys
sys.path.append("../")
from aiflearn.metrics import BinaryLabelDatasetMetric

from aiflearn.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
    import load_preproc_data_adult
from aiflearn.algorithms.preprocessing.lfr import LFR

from IPython.display import Markdown, display

# #### Load dataset and set options




# Get the dataset and split into train and test
dataset_orig = load_preproc_data_adult()
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)


# #### Clean up training data




# print out some labels, names, etc.
# Training Dataset shape
print(dataset_orig_train.features.shape)
# Favorable and unfavorable labels
print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
# Protected attribute names
print(dataset_orig_train.protected_attribute_names)
# Privileged and unprivileged protected attribute values
print(dataset_orig_train.privileged_protected_attributes, 
      dataset_orig_train.unprivileged_protected_attributes)
# Dataset feature names
print(dataset_orig_train.feature_names)


# #### Metric for original training data

# Metric for the original dataset
privileged_groups = [{'sex': 1.0}]
unprivileged_groups = [{'sex': 0.0}]
metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
# Original training dataset
print("Difference in mean outcomes between unprivileged and privileged groups = %f"
      % metric_orig_train.mean_difference())


# #### Train with and transform the original training data


# Input recontruction quality - Ax
# Fairness constraint - Az
# Output prediction error - Ay

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
    
TR = LFR(unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups)
TR = TR.fit(dataset_orig_train)


# In[6]:


# Transform training data and align features
dataset_transf_train = TR.transform(dataset_orig_train)


# #### Metric with the transformed training data


thresholds = [0.2, 0.3, 0.35, 0.4, 0.5]
for threshold in thresholds:
    
    # Transform training data and align features
    dataset_transf_train = TR.transform(dataset_orig_train, threshold=threshold)

    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    # Transformed training dataset
    print("Classification threshold = %f" % threshold)
    print("Difference in mean outcomes between unprivileged and privileged groups = %f"
          % metric_transf_train.mean_difference())


# Optimized preprocessing has reduced the disparity in favorable outcomes
# between the privileged and unprivileged groups (training data).




# Individual fairness metrics
print("Consistency of labels in transformed training dataset= %f"
      %metric_transf_train.consistency())
print("Consistency of labels in original training dataset= %f"
      %metric_orig_train.consistency())




# #### PCA Analysis of consistency



import pandas as pd

feat_cols = dataset_orig_train.feature_names

orig_df = pd.DataFrame(dataset_orig_train.features,columns=feat_cols)
orig_df['label'] = dataset_orig_train.labels
orig_df['label'] = orig_df['label'].apply(lambda i: str(i))

transf_df = pd.DataFrame(dataset_transf_train.features,columns=feat_cols)
transf_df['label'] = dataset_transf_train.labels
transf_df['label'] = transf_df['label'].apply(lambda i: str(i))



from sklearn.decomposition import PCA

orig_pca = PCA(n_components=3)
orig_pca_result = orig_pca.fit_transform(orig_df[feat_cols].values)

orig_df['pca-one'] = orig_pca_result[:,0]
orig_df['pca-two'] = orig_pca_result[:,1] 
orig_df['pca-three'] = orig_pca_result[:,2]

# Original training dataset
print('Explained variation per principal component:')
print(orig_pca.explained_variance_ratio_)





transf_pca = PCA(n_components=3)
transf_pca_result = transf_pca.fit_transform(transf_df[feat_cols].values)

transf_df['pca-one'] = transf_pca_result[:,0]
transf_df['pca-two'] = transf_pca_result[:,1] 
transf_df['pca-three'] = transf_pca_result[:,2]

# Transformed training dataset
print('Explained variation per principal component:')
print(transf_pca.explained_variance_ratio_)


# #### Load, clean up original test data and compute metric




# Testing Dataset shape
print(dataset_orig_test.features.shape)

metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
# Original test dataset
print("Difference in mean outcomes between unprivileged and privileged groups = %f"
      % metric_orig_test.mean_difference())


# #### Transform test data and compute metric




dataset_transf_test = TR.transform(dataset_orig_test, threshold=threshold)
metric_transf_test = BinaryLabelDatasetMetric(dataset_transf_test, 
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)





print("Consistency of labels in tranformed test dataset= %f"
      %metric_transf_test.consistency())





print("Consistency of labels in original test dataset= %f"
      %metric_orig_test.consistency())



def check_algorithm_success():
     """Transformed dataset consistency should be greater than original dataset."""
    assert metric_transf_test.consistency() > metric_orig_test.consistency(), \
        "Transformed dataset consistency should be greater than original dataset."

check_algorithm_success()    


#     References:
#     [1] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
#     "Learning Fair Representations."
#     International Conference on Machine Learning, 2013.






