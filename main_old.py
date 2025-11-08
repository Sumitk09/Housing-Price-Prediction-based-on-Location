import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the dataset
housing = pd.read_csv("housing.csv")

# 2. Creating stratified test set
housing['income_category'] = pd.cut(housing['median_income'], bins=[0,1.5,3,4.5,6,np.inf], labels=[1,2,3,4,5])
split_data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split_data.split(housing, housing['income_category']):
    strat_train_set = housing.loc[train_index].drop('income_category', axis=1)
    strat_test_set = housing.loc[test_index].drop('income_category', axis=1)

# Will work on copy of training data
housing_train = strat_train_set.copy()

# 3. Seprate features and labeling data
housing_train_label = housing_train['median_house_value'].copy()
housing_train = housing_train.drop('median_house_value', axis=1)

# 4. seprate neumerical and categorial columns labels
num_attribute = housing_train.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribute = ["ocean_proximity"]

# 5. Creating Pipeline
# for Neumerical columns
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# for Categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Construct full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribute),
    ("cat", cat_pipeline, cat_attribute)
])

# 6. Transform/ Preparing the data
housing_prepared = full_pipeline.fit_transform(housing_train)

# 7. Train the model
# Linear Regression Model
lin_regression = LinearRegression()
lin_regression.fit(housing_prepared, housing_train_label)
lin_predict = lin_regression.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_train_label, lin_predict)
print(f"Root mean square error of Linear Regression model: {lin_rmse}")
lin_rmse = -cross_val_score(lin_regression, housing_prepared, housing_train_label, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_rmse).describe())


# Decision Tree Model
dec_regression = DecisionTreeRegressor()
dec_regression.fit(housing_prepared, housing_train_label)
dec_predict = dec_regression.predict(housing_prepared)
dec_rmse = root_mean_squared_error(housing_train_label, dec_predict)
print(f"Root mean square error of Decision Tree Regression model: {dec_rmse}")
dec_rmse = -cross_val_score(dec_regression, housing_prepared, housing_train_label, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(dec_rmse).describe())

# Random Forest Model
Random_Forest_regression = RandomForestRegressor()
Random_Forest_regression.fit(housing_prepared, housing_train_label)
Random_Forest_predict = Random_Forest_regression.predict(housing_prepared)
Random_Forest_rmse = root_mean_squared_error(housing_train_label, Random_Forest_predict)
print(f"Root mean square error of Random Forest model: {Random_Forest_rmse}")
Random_Forest_rmse = -cross_val_score(Random_Forest_regression, housing_prepared, housing_train_label, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(Random_Forest_rmse).describe())