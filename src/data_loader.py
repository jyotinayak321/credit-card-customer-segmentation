"""
data_loader.py
---------------
Responsible ONLY for loading the CSV and doing a first inspection.
Keeping this separate from preprocessing/clustering follows the
"single responsibility" principle - each file does ONE job.
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the credit card dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick summary: shape, dtypes, missing values, duplicates.
    Always run this BEFORE touching the data - you must understand
    what you have before you clean or model it."""
    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)
    print("\nCOLUMN INFO:")
    print(df.info())
    print("\nMISSING VALUES PER COLUMN:")
    print(df.isnull().sum())
    print("\nDUPLICATE ROWS:", df.duplicated().sum())
    print("\nBASIC STATISTICS:")
    print(df.describe().T)


if __name__ == "__main__":
    df = load_data("../data/credit_card_customers.csv")
    inspect_data(df)
