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

# Assuming neos_data is your DataFrame containing the Near Earth Object dataset
num_df = neos_data.select_dtypes(include='number')
num_cols = num_df.columns[:5]  # Select the first 5 numerical columns

# -----------Exploring the Distribution of Numerical Variables in Near Earth Object Dataset------------


fig, axs = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(15, 5))
                                   # ncols=5
# Loop over the numerical columns and make a box plot for all
for i, num_col in (enumerate(num_cols)):
    sns.boxplot(x='hazardous', y=num_col, data=neos_data, ax=axs[i])
    axs[i].set_title(num_col)  # Set title for each subplot

fig.tight_layout()
plt.show()
