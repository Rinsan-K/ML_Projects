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

# ---------Classification of Exoplanets using Logistic Regression and Confusion Matrix/ROC Curve Analysis-----------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Assuming y_train_neo is encoded as strings
label_encoder = LabelEncoder()
y_train_neo_encoded = label_encoder.fit_transform(y_train_neo)
y_test_neo_encoded = label_encoder.transform(y_test_neo)

# Convert target variable to binary labels based on a threshold of 0.5
y_train_binary = (y_train_neo_encoded >= 0.5).astype(int)
y_test_binary = (y_test_neo_encoded >= 0.5).astype(int)

model = LogisticRegression()
model.fit(X_train_neo, y_train_binary)

y_pred_neo = model.predict(X_test_neo)

cm = confusion_matrix(y_test_binary, y_pred_neo)

# Accuracy score
accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
accuracy = round(accuracy * 100, 2)

# Calculate the false positive rate, true positive rate, and AUC
fpr, tpr, thresholds = roc_curve(y_test_binary, model.predict_proba(X_test_neo)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve and confusion matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
ax1.plot([0, 1], [0, 1], 'r--')
ax1.set_title('ROC Curve of Logistic Regression')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right')

cm_plot = ConfusionMatrixDisplay(cm, display_labels=['Not Confirmed', 'Confirmed']).plot(ax=ax2)
cm_plot.ax_.set_title('Confusion Matrix for Logistic Regression')
cm_plot.ax_.set_xlabel('Predicted Labels')
cm_plot.ax_.set_ylabel('True Labels')

fig.tight_layout()
plt.show()

print("Logistic Regression Accuracy Score: ", accuracy, "%\n")