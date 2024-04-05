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

# ----------Statistical Analysis of the Relationship between Orbital Period and Exoplanet Discoveries------------

from scipy.stats import pearsonr

grouped_data = exoplanets_data.groupby('discovery_year').agg({'orbital_period': 'median', 'name': 'count'})

# Calculate the correlation coefficient and p-value
corr_coef, p_value = pearsonr(grouped_data['orbital_period'], grouped_data['name'])

print("Correlation coefficient:", corr_coef)
print("P-value:", p_value)

sns.regplot(x=grouped_data['orbital_period'], y=grouped_data['name'])
plt.show()