import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from .config import ADULT_COLUMN_NAMES, ADULT_DATA_URL

def load_adult_dataset():
    df = pd.read_csv(
        ADULT_DATA_URL,
        header=None,
        names=ADULT_COLUMN_NAMES,
        na_values=" ?",
        skipinitialspace=True,
    )
    df = df.dropna()
    return df

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = MinMaxScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor

def train_test_split_adult(df, test_size=0.2, random_state=42):
    X = df.drop("income", axis=1)
    y = (df["income"] == ">50K").astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
