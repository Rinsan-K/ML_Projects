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

# ---------Loading 2 Datasets (Exoplanets and NEOs)------------

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

exoplanets_data = pd.read_csv('C:/SMEC/DataScience/MyProjects/ML_Projects/EP&NEO/cleaned_5250.csv')
neos_data = pd.read_csv('C:/SMEC/DataScience/MyProjects/ML_Projects/EP&NEO/neo_v2.csv')

# ------------Explore and preprocess the Exoplanets dataset------------

# Handling Missing Values
exoplanets_data.dropna(inplace=True)
exoplanets_data.reset_index(drop=True, inplace=True)

# Data Type Transformation
exoplanets_data['orbital_period'] = exoplanets_data['orbital_period'].astype(float)  # Convert column to float data type
exoplanets_data['name'] = exoplanets_data['name'].astype(str)  # Convert column to string data type

# Print 5 rows
print(exoplanets_data.head())

# Print tail
print(exoplanets_data.tail())

# Print shape
print(exoplanets_data.shape)

# Check duplicates
exoplanets_data.duplicated().sum()

# Check missing values
print(exoplanets_data.isnull().sum())

# Simply replacing any missing values if needed
exoplanets_data.fillna(method='ffill', inplace=True)