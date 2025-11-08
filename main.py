import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipeline(num_attribute, cat_attribute):
    # Creating Pipeline for Numerical columns
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline for Categorical columns
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full preprocessing pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribute),
        ("cat", cat_pipeline, cat_attribute)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):
    # Load dataset
    housing = pd.read_csv("housing.csv")

    # Create income_category for stratified sampling only
    housing['income_category'] = pd.cut(housing['median_income'],
                                        bins=[0, 1.5, 3, 4.5, 6, np.inf],
                                        labels=[1, 2, 3, 4, 5])

    # Stratified split
    split_data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split_data.split(housing, housing['income_category']):
        strat_train_set = housing.loc[train_index].copy()
        strat_test_set = housing.loc[test_index].copy()

    # Drop income_category AFTER stratification — prevents leaking into training features
    strat_train_set = strat_train_set.drop("income_category", axis=1)
    strat_test_set = strat_test_set.drop("income_category", axis=1)

    # Save test set correctly — don't assign result of .to_csv()
    strat_test_set.to_csv('input.csv', index=False)

    # Use correct stratified training set to get features and labels
    housing_train_label = strat_train_set['median_house_value'].copy()
    housing_train_features = strat_train_set.drop('median_house_value', axis=1)

    # Identify numeric and categorical columns
    num_attribute = housing_train_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribute = ["ocean_proximity"]

    # Build and apply pipeline
    pipeline = build_pipeline(num_attribute, cat_attribute)
    housing_prepared = pipeline.fit_transform(housing_train_features)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_train_label)

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is Trained....!!!!")

else:
    # Inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")  # in real projects this can be done using API, Flask, Django,etc

    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Inference is complete. Results saved to output.csv")
