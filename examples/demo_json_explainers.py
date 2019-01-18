#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load all necessary packages
import sys
sys.path.append("../")
from collections import OrderedDict
import json
from pprint import pprint
from aiflearn.datasets import GermanDataset
from aiflearn.metrics import BinaryLabelDatasetMetric
from aiflearn.explainers import MetricTextExplainer, MetricJSONExplainer
from IPython.display import JSON, display_json


# ##### Load dataset

# In[2]:


gd = GermanDataset()


# ##### Create metrics

# In[3]:


priv = [{'sex': 1}]
unpriv = [{'sex': 0}]
bldm = BinaryLabelDatasetMetric(gd, unprivileged_groups=unpriv, privileged_groups=priv)


# ##### Create explainers

# In[4]:


text_expl = MetricTextExplainer(bldm)
json_expl = MetricJSONExplainer(bldm)


# ##### Text explanations

# In[5]:


print(text_expl.num_positives())


# In[6]:


print(text_expl.mean_difference())


# In[7]:


print(text_expl.disparate_impact())


# ##### JSON Explanations

# In[8]:


def format_json(json_str):
    return json.dumps(json.loads(json_str, object_pairs_hook=OrderedDict), indent=2)


# In[9]:


print(format_json(json_expl.num_positives()))


# In[10]:


print(format_json(json_expl.mean_difference()))


# In[11]:


print(format_json(json_expl.disparate_impact()))


# In[ ]:




