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


gd = GermanDataset()


# ##### Create metrics




priv = [{'sex': 1}]
unpriv = [{'sex': 0}]
bldm = BinaryLabelDatasetMetric(gd, unprivileged_groups=unpriv, privileged_groups=priv)


# ##### Create explainers


text_expl = MetricTextExplainer(bldm)
json_expl = MetricJSONExplainer(bldm)


# ##### Text explanations


print(text_expl.num_positives())




print(text_expl.mean_difference())



print(text_expl.disparate_impact())


# ##### JSON Explanations

def format_json(json_str):
    return json.dumps(json.loads(json_str, object_pairs_hook=OrderedDict), indent=2)



print(format_json(json_expl.num_positives()))



print(format_json(json_expl.mean_difference()))



print(format_json(json_expl.disparate_impact()))




