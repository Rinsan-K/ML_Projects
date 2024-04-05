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

# ---------Descriptive Statistics---------
# ----------Exoplanets--------

# Central Tendency - Mode
exoplanets_data_mode = exoplanets_data['planet_type'].mode()[0]
print("Mode of Planet Type (Exoplanets_data):", exoplanets_data_mode)

# Dispersion - Range, IQR, Variance
exoplanets_data_range = exoplanets_data['distance'].max() - exoplanets_data['distance'].min()
exoplanets_data_iqr = exoplanets_data['distance'].quantile(0.75) - exoplanets_data['distance'].quantile(0.25)
exoplanets_data_variance = exoplanets_data['distance'].var()
print("Range of Distance (Exoplanets_data):", exoplanets_data_range)
print("IQR of Distance (Exoplanets_data):", exoplanets_data_iqr)
print("Variance of Distance (Exoplanets_data):", exoplanets_data_variance)

# Shape - Skewness, Kurtosis
exoplanets_data_skewness = exoplanets_data['distance'].skew()
exoplanets_data_kurtosis = exoplanets_data['distance'].kurt()
print("Skewness of Distance (Exoplanets_data):", exoplanets_data_skewness)
print("Kurtosis of Distance (Exoplanets_data):", exoplanets_data_kurtosis)

# Categorical Variables - Frequencies & Percentages for Planet Types
exoplanet_classes_counts = exoplanets_data['planet_type'].value_counts()
exoplanet_classes_percentages = exoplanets_data['planet_type'].value_counts(normalize=True) * 100
print("Frequencies of Planet Types:\n", exoplanet_classes_counts)
print("Percentages of Planet Types:\n", exoplanet_classes_percentages)