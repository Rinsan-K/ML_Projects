import numpy as np
import pandas as pd
df = pd.read_csv('shein_mens_fashion.csv')

# Sku, Category and Url not needed
df = df.drop(columns = ['sku','url','category_name'])
print(df)

print(df['title'].nunique())
print(df['description'].nunique())
print(df['color'].nunique())
#High cardinality across all categorical fields, Ordinal encoding needed

from sklearn.preprocessing import OrdinalEncoder
cols = ['title','description','color']
Ord = OrdinalEncoder()
df[cols] = Ord.fit_transform(df[cols])
print(df)

df.isnull().values.any()
#No NaN values

df.columns

# Building the model (Random Forest)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

features = [ 'title', 'color', 'sale_price/amount', 'discount_percentage','category_id', 'description', 'reviews_count','average_rating']
y = df['retail_price/amount']
X = df[features]

X_train, X_val, y_train,y_val = train_test_split(X,y)

my_model = RandomForestRegressor(random_state = 0)
my_model.fit(X_train, y_train)

predictions = my_model.predict(X_val)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_val).round(2)))

# Version 2 : Making use of an XGB regressor

from xgboost import XGBRegressor

my_model_2 = XGBRegressor(random_state = 0)
my_model_2.fit(X_train , y_train, early_stopping_rounds=8 , eval_set=[(X_val, y_val)],verbose=False)
predictions2 = my_model_2.predict(X_val)
val = mean_absolute_error(predictions2, y_val).round(2)
print(val)