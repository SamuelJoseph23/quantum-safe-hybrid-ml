"""
Data utility functions for loading and preprocessing the Adult Income dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml

def load_and_preprocess_data(n_clients: int, split_strategy: str = "iid", non_iid_label_skew: bool = False):
    """
    Loads the Adult Income dataset, preprocesses it, and splits it for federated clients.
    
    Returns:
        client_datasets (list): A list of tuples (X_train, y_train) for each client.
        test_data (tuple): (X_test, y_test) for global evaluation.
    """
    print("Loading Adult Income dataset...")
    # Fetch dataset from OpenML (ID 1590 is the adult dataset)
    # as_frame=True returns pandas DataFrame
    adult = fetch_openml(data_id=1590, as_frame=True, parser='auto')
    X = adult.data
    y = adult.target

    # Simple preprocessing
    # 1. Drop rows with missing values
    X = X.dropna()
    y = y[X.index]

    # 2. Encode target labels (<=50K -> 0, >50K -> 1)
    # The dataset typically has strings like '<=50K', '>50K'
    y = (y == '>50K').astype(int)

    # 3. Select a subset of features for simplicity in this demo
    # We'll pick a mix of numerical and categorical features
    # For a real run, you'd likely One-Hot Encode categorical ones.
    # Here, we keep it simple: just numerical for the logistic regression baseline.
    numeric_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    X = X[numeric_features]

    # Split into train and test FIRST (avoid preprocessing leakage)
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Split training data among clients
    # We'll do a simple IID split (random shuffle is implicit in train_test_split)
    client_datasets = []
    chunk_size = len(X_train) // n_clients
    indices = np.arange(len(X_train))

    # Optional simple non-IID label skew for demo purposes:
    # sort by label and split contiguous blocks, producing label-imbalanced clients.
    if non_iid_label_skew or split_strategy.lower() in ("non-iid", "noniid", "label_skew"):
        indices = indices[np.argsort(np.array(y_train))]
    
    for i in range(n_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        idx = indices[start:end]
        X_c = X_train[idx]
        y_c = np.array(y_train)[idx]
        client_datasets.append((X_c, y_c)) # y already numpy

    print(f"Data loaded. {n_clients} clients, {chunk_size} samples per client.")
    return client_datasets, (X_test, y_test)

if __name__ == "__main__":
    # Quick test
    clients, test = load_and_preprocess_data(3)
    print("Test set shape:", test[0].shape)
