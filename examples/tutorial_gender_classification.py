"""

Bias in Image based Automatic Gender Classification

Overview

Recent studies have shown that the machine learning models for gender
classification task from face images perform differently across groups defined 
by skin tone. In this tutorial, we will demonstrate the use of the aiflearn 
toolbox to study the the differential performance of a custom classifier.
 We use a bias mitigiating algorithm available in aiflearn with the aim of 
 improving a classfication model in terms of the fairness metrics. We will 
 work with the UTK dataset for this tutorial. This can be downloaded from here:
https://susanqq.github.io/UTKFace/

 In a nutshell, we will follow these steps:
  - Process images and load them as a aiflearn dataset
  - Learn a baseline classifier and obtain fairness metrics
  - Call the `Reweighing` algorithm to obtain obtain instance weights
  - Learn a new classifier with the instance weights and obtain updated fairness
    metrics
"""
# Call the import statements


import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import Markdown, display

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn
from torchsummary import summary

import pandas as pd
import sys
sys.path.append("../")

from aiflearn.datasets import BinaryLabelDataset
from aiflearn.metrics import BinaryLabelDatasetMetric
from aiflearn.metrics import ClassificationMetric
from aiflearn.algorithms.preprocessing.reweighing import Reweighing




np.random.seed(99)
torch.manual_seed(99)


# # Step 1: Load and Process Images
# The first step is to to download the images `Aligned&cropped` images at this
# location mentioned above.
# 
# After unzipping the downloaded file, point the location of the folder in the
# `image_dir` variable below. The file name has the following format
# `age`-`gender`-`race`-`date&time`.jpg
# 
#     age: indicates the age of the person in the picture and can range from 0 to 116.
#     gender: indicates the gender of the person and is either 0 (male) or 1 (female).
#     race: indicates the race of the person and can from 0 to 4, denoting White,
#     Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
#     date&time: indicates the date and time an image was collected in the UTK dataset.
# 
# For this tutorial we will restict the images to contrain `White` and `Others`
# races. We need to specify the unprivileged and previledged groups to obtain
# various metrics from the aiflearn toolbox. We set `White` as the previledged
# group and `Others` as the unpreviledged group for computing the results with
# gender as the outcome variable that need to be predicted. We set prediction
# as `female (1)` as the unfavorable label and `male (0)` as favorable label
# for the purpose of computing metrics and does not have any special meaning
# in the context of gender prediction.


races_to_consider = [0,4]
unprivileged_groups = [{'race': 4.0}]
privileged_groups = [{'race': 0.0}]
favorable_label = 0.0 
unfavorable_label = 1.0


# ### Update the `image_dir` with the downloaded and extracted images location
# and specify the desired image size. The images and loaded and resized usign
# opencv library. The following code creates three key numpy arrays each
# containing the raw images, the race attributes and the gender labels.



image_dir = '/path/to/UTKFace/'
img_size = 64



protected_race = []
outcome_gender = []
feature_image = []
feature_age = []

for i, image_path in enumerate(glob.glob(image_dir + "*.jpg")):
    try:
        age, gender, race = image_path.split('/')[-1].split("_")[:3]
        age = int(age)
        gender = int(gender)
        race = int(race)
        
        if race in races_to_consider:
            protected_race.append(race)
            outcome_gender.append(gender)
            feature_image.append(cv2.resize(cv2.imread(image_path), (img_size, img_size)))
            feature_age.append(age)
    except:
        print("Missing: " + image_path)

feature_image_mat = np.array(feature_image)
outcome_gender_mat =  np.array(outcome_gender)
protected_race_mat =  np.array(protected_race)
age_mat = np.array(feature_age)


# # Step 2: Learn a Baseline Classifier
# Lets build a simple convolutional neural network (CNN) with $3$ convolutional
# layers and $2$ fully connected layers using the `pytorch` framework.
# ![CNN](images/cnn_arch.png)
# Each convolutional layer is followed by a maxpool layer. The final layer
# provides the logits for the binary gender predicition task.


class ThreeLayerCNN(torch.nn.Module):
    """
    Input: 128x128 face image (eye aligned).
    Output: 1-D tensor with 2 elements. Used for binary classification.
    Parameters:
        Number of conv layers: 3
        Number of fully connected layers: 2       
    """
    def __init__(self):
        super(ThreeLayerCNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,6,5)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(6,16,5)
        self.conv3 = torch.nn.Conv2d(16,16,6)
        self.fc1 = torch.nn.Linear(16*4*4,120)
        self.fc2 = torch.nn.Linear(120,2)


    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(-1,16*4*4)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ## Split the dataset into train and test
# 
# Let us rescale the pixels to lie between $-1$ and $1$ and split the complete
# dataset into train and test sets.
# We use $70$-$30$ percentage for train and test, respectively.

# In[9]:


feature_image_mat_normed = 2.0 *feature_image_mat.astype('float32')/256.0 - 1.0

N = len(feature_image_mat_normed)
ids = np.random.permutation(N)
train_size=int(0.7 * N)
X_train = feature_image_mat_normed[ids[0:train_size]]
y_train = outcome_gender_mat[ids[0:train_size]]
X_test = feature_image_mat_normed[ids[train_size:]]
y_test = outcome_gender_mat[ids[train_size:]]

p_train = protected_race_mat[ids[0:train_size]]
p_test = protected_race_mat[ids[train_size:]]

age_train = age_mat[ids[0:train_size]]
age_test = age_mat[ids[train_size:]]


# Next, we will create the pytorch train and test data loaders after transposing
# and converting the images and labels. The batch size is set to $64$.


batch_size = 64

X_train = X_train.transpose(0,3,1,2)
X_test = X_test.transpose(0,3,1,2)

train = torch.utils.data.TensorDataset(Variable(torch.FloatTensor(X_train.astype('float32'))), Variable(torch.LongTensor(y_train.astype('float32'))))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test = torch.utils.data.TensorDataset(Variable(torch.FloatTensor(X_test.astype('float32'))), Variable(torch.LongTensor(y_test.astype('float32'))))
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)


# ## Create a Plain Model
# In the next few steps, we will create and intialize a model with the above
# described architecture and train it.



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ThreeLayerCNN().to(device)
summary(model, (3,img_size,img_size))


# ## Training the network
# Next,  we will train the model summarized above. `num_epochs` specifies the
# number of epochs used for  training. The learning rate is set to $0.001$.
# We will use the `Adam` otimizer to minimze the standard cross-entropy loss
# for classification tasks.



num_epochs = 5
learning_rate = 0.001
print_freq = 100

# Specify the loss and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start training the model
num_batches = len(train_loader)
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (idx+1) % print_freq == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1,
                     num_epochs, idx+1, num_batches, loss.item()))


# # Measure Fairness Metrics
# Let's get the predictions of this trained model on the test and use them to
# compute various fariness metrics available in the aiflearn toolbox.



# Run model on test set in eval mode.
model.eval()
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred += predicted.tolist()
y_pred = np.array(y_pred)


# The wrapper function defined below can be used to convert the numpy
# arrays and the related meta data into a aiflearn dataset. This will ease the
# process of computing metrics and comparing two datasets. The wrapper consumes
# the outcome array, the protected attribute array, information about
# unprivileged_groups and privileged_groups; and the favorable and unfavorable
# label to produce an instance of aiflearn's `BinaryLabelDataset`.



def dataset_wrapper(outcome, protected, unprivileged_groups, privileged_groups,
                          favorable_label, unfavorable_label):
    """ A wraper function to create aiflearn dataset from outcome and protected
    in numpy array format.
    """
    df = pd.DataFrame(data=outcome,
                      columns=['outcome'])
    df['race'] = protected
    
    dataset = BinaryLabelDataset(favorable_label=favorable_label,
                                       unfavorable_label=unfavorable_label,
                                       df=df,
                                       label_names=['outcome'],
                                       protected_attribute_names=['race'],
                                       unprivileged_protected_attributes=unprivileged_groups)
    return dataset



original_traning_dataset = dataset_wrapper(outcome=y_train, protected=p_train, 
                                                 unprivileged_groups=unprivileged_groups, 
                                                 privileged_groups=privileged_groups,
                                                 favorable_label=favorable_label,
                                          unfavorable_label=unfavorable_label)
original_test_dataset = dataset_wrapper(outcome=y_test, protected=p_test, 
                                              unprivileged_groups=unprivileged_groups, 
                                              privileged_groups=privileged_groups,
                                                 favorable_label=favorable_label,
                                          unfavorable_label=unfavorable_label)
plain_predictions_test_dataset = dataset_wrapper(outcome=y_pred, protected=p_test, 
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups,
                                                 favorable_label=favorable_label,
                                          unfavorable_label=unfavorable_label)


# #### Obtaining the Classification Metrics
# We use the `ClassificationMetric` class from the aiflearn toolbox for
# computing metrics based on two BinaryLabelDatasets. The first dataset is the
# original one and the second is the output of the classification transformer
# (or similar). Later on we will use `BinaryLabelDatasetMetric` which computes
# based on a single `BinaryLabelDataset`.



classified_metric_nodebiasing_test = ClassificationMetric(original_test_dataset, 
                                                 plain_predictions_test_dataset,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
TPR = classified_metric_nodebiasing_test.true_positive_rate()
TNR = classified_metric_nodebiasing_test.true_negative_rate()
bal_acc_nodebiasing_test = 0.5*(TPR+TNR)




# Plain model - without debiasing - classification metrics
print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
print("Test set: Statistical parity difference = %f" % classified_metric_nodebiasing_test.statistical_parity_difference())
print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
print("Test set: Theil index = %f" % classified_metric_nodebiasing_test.theil_index())
print("Test set: False negative rate difference = %f" % classified_metric_nodebiasing_test.false_negative_rate_difference())


# # Step 3: Apply the Reweighing algorithm to tranform the dataset
# Reweighing is a preprocessing technique that weights the examples in each
# (group, label) combination differently to ensure fairness before classification
# [1]. This is one of the very few pre-processing method we are aware of that
# could tractably be applied to multimedia data (since it does not work with
# the features).
# 
#     References:
#     [1] F. Kamiran and T. Calders,"Data Preprocessing Techniques for
#     Classification without Discrimination," Knowledge and Information Systems, 2012.




RW = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
RW.fit(original_traning_dataset)
transf_traning_dataset = RW.transform(original_traning_dataset)



metric_orig_train = BinaryLabelDatasetMetric(original_traning_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
metric_tranf_train = BinaryLabelDatasetMetric(transf_traning_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)



# Original training dataset
print("Difference in mean outcomes between privileged and unprivileged groups = %f" % metric_orig_train.mean_difference())
# Transformed training dataset
print("Difference in mean outcomes between privileged and unprivileged groups = %f" % metric_tranf_train.mean_difference())



metric_orig_test = BinaryLabelDatasetMetric(original_test_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
transf_test_dataset = RW.transform(original_test_dataset)
metric_transf_test = BinaryLabelDatasetMetric(transf_test_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
# Original testing dataset
print("Difference in mean outcomes between privileged and unprivileged groups = %f"
      % metric_orig_test.mean_difference())
# Transformed testing dataset
print("Difference in mean outcomes between privileged and unprivileged groups = %f"
      % metric_transf_test.mean_difference())


# # Step 4: Learn a New Classfier using the Instance Weights
# We can see that the reweighing was able to reduce the difference in mean
# outcomes between privileged and unprivileged groups. This was done by
# learning appropriate weights for each training instance. In the current step,
# we will use these learned instance weights to train a network. We  will create
# coustom pytorch loss called `InstanceWeighetedCrossEntropyLoss` that uses the
# instances weights to produce the loss value for a batch of data samples.



tranf_train = torch.utils.data.TensorDataset(Variable(torch.FloatTensor(X_train.astype('float32'))),
                                             Variable(torch.LongTensor(transf_traning_dataset.labels.astype('float32'))),
                                            Variable(torch.FloatTensor(transf_traning_dataset.instance_weights.astype('float32'))),)
tranf_train_loader = torch.utils.data.DataLoader(tranf_train, batch_size=64, shuffle=True)



class InstanceWeighetedCrossEntropyLoss(nn.Module):
    """Cross entropy loss with instance weights."""
    def __init__(self):
        super(InstanceWeighetedCrossEntropyLoss, self).__init__()

    def forward(self, logits, target, weights):
        loss = log_sum_exp(logits) - select_target_class(logits, target.squeeze(1))
        loss = loss * weights
        return loss.mean()

# Helper functions
def select_target_class(logits, target):
    batch_size, num_classes = logits.size()
    mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .to(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(mask)

def log_sum_exp(x):
    c, _ = torch.max(x, 1)
    y = c + torch.log(torch.exp(x - c.unsqueeze(dim=1).expand_as(x)).sum(1))
    return y





tranf_model = ThreeLayerCNN().to(device)




num_epochs = 5
learning_rate = 0.001
print_freq = 100

# Specify the loss and the optimizer
criterion = InstanceWeighetedCrossEntropyLoss()
optimizer = torch.optim.Adam(tranf_model.parameters(), lr=learning_rate)

# Start training the new model
num_batches = len(tranf_train_loader)
for epoch in range(num_epochs):
    for idx, (images, labels, weights) in enumerate(tranf_train_loader):

        images = images.to(device)
        labels = labels.to(device)
        
        outputs = tranf_model(images)
        loss = criterion(outputs, labels, weights)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (idx+1) % print_freq == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1,
                    num_epochs, idx+1, num_batches, loss.item()))




# Test the model
tranf_model.eval()
y_pred_transf = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = tranf_model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred_transf += predicted.tolist()
y_pred_transf = np.array(y_pred_transf)


# Let us repeat the same steps as before to convert the predictions into
# aiflearn dataset and obtain various metrics.



transf_predictions_test_dataset = dataset_wrapper(outcome=y_pred_transf, protected=p_test, 
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups,
                                                 favorable_label=favorable_label,
                                                  unfavorable_label=unfavorable_label
                                                 )



classified_metric_debiasing_test = ClassificationMetric(original_test_dataset, 
                                                 transf_predictions_test_dataset,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
TPR = classified_metric_debiasing_test.true_positive_rate()
TNR = classified_metric_debiasing_test.true_negative_rate()
bal_acc_debiasing_test = 0.5*(TPR+TNR)




# Plain model - without debiasing - classification metrics
print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
print("Test set: Statistical parity difference = %f" % classified_metric_nodebiasing_test.statistical_parity_difference())
print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())
print("Test set: False negative rate difference = %f" % classified_metric_nodebiasing_test.false_negative_rate_difference())

# Model - with debiasing - classification metrics
print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
print("Test set: Statistical parity difference = %f" % classified_metric_debiasing_test.statistical_parity_difference())
print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())
print("Test set: False negative rate difference = %f" % classified_metric_debiasing_test.false_negative_rate_difference())


# Let us break down these numbers by age to understand how these bias differs
# across age groups. For demonstration, we dividee all the samples into these
# age groups: 0-10, 10-20, 20-40, 40-60 and 60-150. For this we will create
# aiflearn datasets using the subset of samples that fall into each of the age
# groups. The plot below shows how the `Equal opportunity difference` metric
# varies across age groups before and after applying the bias mitigating
# reweighing algorithm.


# Metrics sliced by age
age_range_intervals = [0, 10, 20, 40, 60, 150]
nodebiasing_perf = []
debiasing_perf = []

for idx in range(len(age_range_intervals)-1):
    start = age_range_intervals[idx]
    end = age_range_intervals[idx+1]
    ids = np.where((age_test>start) & (age_test<end))
    true_dataset = dataset_wrapper(outcome=y_test[ids], protected=p_test[ids],
                                   unprivileged_groups=unprivileged_groups,
                                   privileged_groups=privileged_groups,
                                   favorable_label=favorable_label,
                                unfavorable_label=unfavorable_label)
    transf_pred_dataset = dataset_wrapper(outcome=y_pred_transf[ids], protected=p_test[ids],
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups,
                                          favorable_label=favorable_label,
                                          unfavorable_label=unfavorable_label)
    pred_dataset = dataset_wrapper(outcome=y_pred[ids], protected=p_test[ids],
                                   unprivileged_groups=unprivileged_groups,
                                   privileged_groups=privileged_groups,
                                   favorable_label=favorable_label,
                                   unfavorable_label=unfavorable_label)
 
    classified_metric_nodebiasing_test = ClassificationMetric(true_dataset, 
                                                 pred_dataset,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    classified_metric_debiasing_test = ClassificationMetric(true_dataset, 
                                                 transf_pred_dataset,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    nodebiasing_perf.append(classified_metric_nodebiasing_test.equal_opportunity_difference())
    debiasing_perf.append(classified_metric_debiasing_test.equal_opportunity_difference())

N = len(age_range_intervals)-1
fig, ax = plt.subplots()
ind = np.arange(N)
width = 0.35
p1 = ax.bar(ind, nodebiasing_perf, width, color='r')
p2 = ax.bar(ind + width, debiasing_perf, width,
            color='y')
ax.set_title('Equal opportunity difference by age group')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels([str(age_range_intervals[idx])+'-'+str(age_range_intervals[idx+1]) for idx in range(N)])

ax.legend((p1[0], p2[0]), ('Before', 'After'))
ax.autoscale_view()

plt.show()


# # Conclusions
# In this tutorial, we have examined fairness in the scenario of binary
# classification with face images. We discussed methods to process several
# attributes of images, outcome variables, and protected attributes and create
# aiflearn ready dataset objects on which many bias mititation algorithms can
# be easily applied and fairness metrics can be easliy computed. We used the
# reweighing algorithm with the aim of improving the algorithmic fairness of
# the learned classifiers. The empirical results show slight improvement in
# the case of debiased model over the vanilla model. When sliced by age group,
# the results appear to be mixed bag and thus has scope for further improvements
# by considering age group while learning models.











