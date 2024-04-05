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

# Handling Missing Values
exoplanets_data.dropna(inplace=True)
exoplanets_data.reset_index(drop=True, inplace=True)

# ----------Classification of Habitable Exoplanets by their Orbital Period and Mass using Machine Learning Models---------

x = exoplanets_data.drop(['name','planet_type','mass_wrt','radius_wrt','detection_method'],axis=1)
new_dms = exoplanets_data[['name','mass_wrt','radius_wrt','detection_method']]
X_ = pd.get_dummies(new_dms)
X = pd.concat([x,X_],axis=1)
Y = exoplanets_data['planet_type']

# --------Splitting and Scaling Exoplanet Data for Machine Learning Model Training--------------

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.35,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# ---------K-Nearest Neighbors Classification of Exoplanets Using Preprocessed Data and ROC Analysis---------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc

knn = KNeighborsClassifier().fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))

probs = knn.predict_proba(X_test)

unique_labels = np.unique(y_test)

fpr, tpr, threshold = roc_curve(y_test, probs[:, 1], pos_label=unique_labels[1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of KNN')
plt.show()