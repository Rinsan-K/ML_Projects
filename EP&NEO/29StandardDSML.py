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

x = exoplanets_data.drop(['name','planet_type','mass_wrt','radius_wrt','detection_method'],axis=1)
new_dms = exoplanets_data[['name','mass_wrt','radius_wrt','detection_method']]
X_ = pd.get_dummies(new_dms)
X = pd.concat([x,X_],axis=1)

#-------- Classification of Near Earth Objects as Hazardous or Non-Hazardous using Machine Learning Models--------
#-------Standardizing Dataset for Machine Learning using scikit-learn's StandardScaler() Function------

from sklearn.preprocessing import StandardScaler

y = exoplanets_data['planet_type']

X_train_neo, X_test_neo, y_train_neo, y_test_neo = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Standarization of the data
sc=StandardScaler()
X_train_neo_scaled=pd.DataFrame(sc.fit_transform(X_train_neo))
X_test_neo_scaled=pd.DataFrame(sc.transform(X_test_neo))

X_train_neo_scaled.columns=X.columns
print(X_train_neo_scaled)

X_train_neo=X_train_neo_scaled
X_test_neo=X_test_neo_scaled

X_train_neo.shape, y_train_neo.shape

X_test_neo.shape, y_test_neo.shape