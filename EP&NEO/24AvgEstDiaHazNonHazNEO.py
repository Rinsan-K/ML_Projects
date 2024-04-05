# ---------Importing Necessary Libraries--------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
# from yellowbrick.classifier import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier

# -------Loading 2 Datasets (Exoplanets and NEOs)------------

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

exoplanets_data = pd.read_csv('C:/SMEC/DataScience/MyProjects/ML_Projects/EP&NEO/cleaned_5250.csv')
neos_data = pd.read_csv('C:/SMEC/DataScience/MyProjects/ML_Projects/EP&NEO/neo_v2.csv')

# -----------Average estimated diameter of hazardous and non-hazardous NEOs--------

neos_data['average_diameter'] = (neos_data['est_diameter_min'] + neos_data['est_diameter_max']) / 2

# Average estimated diameter of hazardous NEOs
average_diameter_hazardous = neos_data[neos_data['hazardous'] == True]['average_diameter'].mean()

# Average estimated diameter of non-hazardous NEOs
average_diameter_non_hazardous = neos_data[neos_data['hazardous'] == False]['average_diameter'].mean()

print("Average Estimated Diameter of Hazardous NEOs:", average_diameter_hazardous)
print("Average Estimated Diameter of Non-Hazardous NEOs:", average_diameter_non_hazardous)