"""Shared feature preparation for the analysis notebook and Streamlit app."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "bedrooms", "bathrooms", "car_spaces", "listed_area_sqm",
    "month_index", "bath_per_bed", "parking_per_bed", "area_per_bed",
    "area_reported",
]
CATEGORICAL = ["suburb", "dwelling_type"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        d["sale_date"] = pd.to_datetime(d["sale_date"], errors="raise")
        d["dwelling_type"] = d["property_type_listed"].replace({
            "Unit": "Apartment", "Studio": "Apartment"
        })
        for col in ("bedrooms", "bathrooms", "car_spaces", "listed_area_sqm"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d["month_index"] = (d["sale_date"].dt.year - 2026) * 12 + d["sale_date"].dt.month
        beds = d["bedrooms"].clip(lower=1)
        d["bath_per_bed"] = d["bathrooms"] / beds
        d["parking_per_bed"] = d["car_spaces"] / beds
        d["area_per_bed"] = d["listed_area_sqm"] / beds
        d["area_reported"] = d["listed_area_sqm"].notna().astype(int)
        return d[NUMERIC + CATEGORICAL]


def make_model(name):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    prep = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("category", categorical, CATEGORICAL),
    ], sparse_threshold=0)
    estimators = {
        "Ridge": Ridge(alpha=10.0),
        "Random forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.8,
            random_state=307, n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            min_samples_leaf=3, random_state=307,
        ),
    }
    pipeline = Pipeline([
        ("features", FeatureBuilder()),
        ("prep", prep),
        ("regressor", estimators[name]),
    ])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)
