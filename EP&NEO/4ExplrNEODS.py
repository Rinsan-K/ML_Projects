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

#--------- Explore and preprocess the Near Earth Objects dataset--------------

# Handling Missing Values
neos_data.dropna(inplace=True)
neos_data.reset_index(drop=True, inplace=True)

# Data Type Transformation
neos_data['est_diameter_min'] = neos_data['est_diameter_min'].astype(float)
neos_data['est_diameter_max'] = neos_data['est_diameter_max'].astype(float)
neos_data['relative_velocity'] = neos_data['relative_velocity'].astype(float)
neos_data['miss_distance'] = neos_data['miss_distance'].astype(float)
neos_data['absolute_magnitude'] = neos_data['absolute_magnitude'].astype(float)

# Print 5 rows
print(neos_data.head())

# Print tail
print(neos_data.tail())

# Print Shape
print(neos_data.shape)

# Check for missing values
print(neos_data.isnull().sum())

neos_data.duplicated().sum()

# Simply replacing any missing values if needed
neos_data.fillna(method='ffill', inplace=True)

print(neos_data["orbiting_body"].value_counts())

print(neos_data["sentry_object"].value_counts())