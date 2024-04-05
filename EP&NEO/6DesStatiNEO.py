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

# --------Near Earth Objects--------------

# Central Tendency - Mode
neos_data_mode = neos_data['name'].mode()[0]
print("Mode of NEOs_data Name:", neos_data_mode)

# Dispersion - Range, IQR, Variance
neos_data_range = neos_data['miss_distance'].max() - neos_data['miss_distance'].min()
neos_data_iqr = neos_data['miss_distance'].quantile(0.75) - neos_data['miss_distance'].quantile(0.25)
neos_data_variance = neos_data['miss_distance'].var()
print("Range of Miss Distance (NEOs_data):", neos_data_range)
print("IQR of Miss Distance (NEOs_data):", neos_data_iqr)
print("Variance of Miss Distance (NEOs_data):", neos_data_variance)

# Shape - Skewness, Kurtosis
neos_data_skewness = neos_data['miss_distance'].skew()
neos_data_kurtosis = neos_data['miss_distance'].kurt()
print("Skewness of Miss Distance (NEOs_data):", neos_data_skewness)
print("Kurtosis of Miss Distance (NEOs_data):", neos_data_kurtosis)

# Categorical Variables - Frequencies for Hazardous & Non-Hazardous NEOs_data
hazardous_neos_data_counts = neos_data['hazardous'].value_counts()
print("Frequencies of Hazardous and Non-Hazardous NEOs_data:\n", hazardous_neos_data_counts)