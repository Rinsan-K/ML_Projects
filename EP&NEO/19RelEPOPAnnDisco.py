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

# -----------Relationship Between Exoplanet Orbital Period and Annual Discoveries-------------

# Extracting the relevant columns
exoplanets_period = exoplanets_data[['discovery_year', 'orbital_period']]

median_period_by_year = exoplanets_period.groupby('discovery_year').median()

exoplanet_counts_by_year = exoplanets_data['discovery_year'].value_counts().sort_index()

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Median Orbital Period', color=color)
ax1.plot(median_period_by_year.index, median_period_by_year['orbital_period'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Number of Exoplanets Discovered', color=color)
ax2.plot(exoplanet_counts_by_year.index, exoplanet_counts_by_year.values, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Relationship between Orbital Period and Annual Exoplanet Discoveries')
plt.show()