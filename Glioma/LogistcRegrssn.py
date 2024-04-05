import pandas as pd
from plotly import express
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

INFO = 'TCGA_InfoWithGrade.csv'
df = pd.read_csv(filepath_or_buffer=INFO)
df['grade'] = df['Grade'].map({0:'LGG', 1:'GMB'})
df.head()

express.pie(data_frame=df, names='grade',  color='grade')

express.histogram(data_frame=df, x='Age_at_diagnosis', color='Grade', facet_col='Grade')

from umap import UMAP
from plotly import express

columns = ['Gender', 'Age_at_diagnosis', 'Race', 'IDH1', 'TP53', 'ATRX','PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4','PDGFRA']
target = 'grade'

reducer = UMAP(n_components=2, random_state=2024, transform_seed=2024, verbose=True, n_jobs=1, n_epochs=200)
df[['x', 'y']] = pd.DataFrame(data=reducer.fit_transform(X=df[columns]))
express.scatter(data_frame=df, x='x', y='y', color=target, ).show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(df[columns], df['grade'], test_size=0.2, random_state=2024, stratify=df['grade'])

regression = LogisticRegression(max_iter=100000)

regression.fit(X=X_train, y=y_train)
print('accuracy: {:5.4f} '.format(regression.score(X=X_test, y=y_test)))
express.histogram(y=regression.coef_.tolist()[0], x=columns).show(validate=True)

from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=regression.predict(X=X_test)))

