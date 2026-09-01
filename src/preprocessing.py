"""
preprocessing.py
------------------
Cleaning + feature selection + scaling.
This is the "kitchen prep" stage - washing and cutting vegetables
before cooking (clustering).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# The beginner feature set recommended in the project guide.
# CUST_ID is deliberately excluded - it's an identifier, not behaviour.
DEFAULT_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PRC_FULL_PAYMENT",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicates.

    We use MEDIAN imputation (not mean) because financial columns like
    BALANCE and CREDIT_LIMIT are usually skewed by a few very rich
    customers - the median is not pulled around by those outliers the
    way the mean is.
    """
    df = df.copy()

    if "MINIMUM_PAYMENTS" in df.columns:
        df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(
            df["MINIMUM_PAYMENTS"].median()
        )
    if "CREDIT_LIMIT" in df.columns:
        df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())

    df = df.drop_duplicates()
    return df


def select_features(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """Pick the numerical behaviour columns we'll cluster on."""
    features = features or DEFAULT_FEATURES
    return df[features].copy()


def scale_features(X: pd.DataFrame):
    """Standardize features: z = (x - mean) / std_dev.

    WHY this matters: CREDIT_LIMIT might range 0-20,000 while
    PRC_FULL_PAYMENT ranges 0-1. Without scaling, CREDIT_LIMIT alone
    would dominate the distance calculation and the clustering would
    basically just be "who has a big credit limit" - ignoring every
    other behaviour.

    Returns:
        X_scaled (numpy array), fitted scaler (so you can reuse it later)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
