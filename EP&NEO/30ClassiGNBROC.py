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

x = exoplanets_data.drop(['name','planet_type','mass_wrt','radius_wrt','detection_method'],axis=1)
new_dms = exoplanets_data[['name','mass_wrt','radius_wrt','detection_method']]
X_ = pd.get_dummies(new_dms)
X = pd.concat([x,X_],axis=1)

#-------Standardizing Dataset for Machine Learning using scikit-learn's StandardScaler() Function------

from sklearn.preprocessing import StandardScaler

y = exoplanets_data['planet_type']

X_train_neo, X_test_neo, y_train_neo, y_test_neo = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Standarization of the data
sc = StandardScaler()
X_train_neo_scaled = pd.DataFrame(sc.fit_transform(X_train_neo), columns=X_train_neo.columns)
X_test_neo_scaled = pd.DataFrame(sc.transform(X_test_neo), columns=X_train_neo.columns)

X_train_neo_scaled.columns = X.columns

X_train_neo = X_train_neo_scaled
X_test_neo = X_test_neo_scaled

X_train_neo.shape, y_train_neo.shape

X_test_neo.shape, y_test_neo.shape

#----------- Classification using Gaussian Naive Bayes and ROC Analysis--------------

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# Convert target variable to numerical values
label_encoder = LabelEncoder()
y_train_neo = label_encoder.fit_transform(y_train_neo)
y_test_neo = label_encoder.transform(y_test_neo)

# Convert target variable to binary labels based on a threshold of 0.5
y_train_neo_class = (y_train_neo >= 0.5).astype(int)
y_test_neo_class = (y_test_neo >= 0.5).astype(int)

model = GaussianNB()
model.fit(X_train_neo, y_train_neo_class)

# Prediction on test set
y_pred_neo = model.predict(X_test_neo)

# Confusion matrix
cm = confusion_matrix(y_test_neo_class, y_pred_neo)

# Calculate the accuracy score
accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
accuracy = round(accuracy * 100, 2)
print("Gaussian Naive Bayes Accuracy Score: ", accuracy, "%\n")

# Plot confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix of Gaussian Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Generate ROC Curve
y_true_neos = y_test_neo_class
y_prob_neos = model.predict_proba(X_test_neo)[:, 1]
model_name = 'Gaussian Naive Bayes'

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true_neos, y_prob_neos)
area_under_curve = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6, 5))
plt.plot(false_positive_rate, true_positive_rate, 'b', label = 'AUC = %0.2f' % area_under_curve)
plt.legend()
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title(f'ROC Curve of {model_name}')
plt.show()
