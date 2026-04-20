import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@st.cache_resource
def train_model(_df: pd.DataFrame):
    base_features   = ["property_tax_rate","carpet_area","num_bathrooms","num_rooms","Estimated Value","Year","month"]
    encoded_cols    = sorted([c for c in _df.columns if c not in base_features + ["Sale Price"]])
    feature_columns = [c for c in base_features + encoded_cols if c in _df.columns]
    X = _df[feature_columns];  y = _df["Sale Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    metrics = {
        "R² Score":   round(r2_score(y_test, y_pred), 4),
        "MAE":        round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE":       round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "Train Rows": int(len(X_train)),
        "Test Rows":  int(len(X_test)),
    }
    return mdl, feature_columns, metrics

def predict_price(input_data, mdl, feature_columns) -> float:
    filled = {col: input_data.get(col, 0) for col in feature_columns}
    return float(mdl.predict(pd.DataFrame([filled]))[0])
