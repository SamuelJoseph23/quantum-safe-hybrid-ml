"""
Data utility functions for loading and preprocessing the Adult Income dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

def load_and_preprocess_data(n_clients: int, split_strategy: str = "iid", non_iid_label_skew: bool = False):
    """
    Loads the Adult Income dataset, preprocesses it, and splits it for federated clients.
    
    Args:
        n_clients: Number of federated clients
        split_strategy: "iid" or "non-iid" for data distribution
        non_iid_label_skew: Whether to apply label-based skew (deprecated, use split_strategy)
    
    Returns:
        client_datasets (list): A list of tuples (X_train, y_train) for each client.
        test_data (tuple): (X_test, y_test) for global evaluation.
    """
    if n_clients < 1:
        raise ValueError("Number of clients must be at least 1")
    
    print("Loading Adult Income dataset...")
    try:
        # Fetch dataset from OpenML (ID 1590 is the adult dataset)
        adult = fetch_openml(data_id=1590, as_frame=True, parser='auto')
        X = adult.data
        y = adult.target
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from OpenML: {e}") from e

    # 1. Handle missing values - important to do before encoding
    # Some categorical columns might have '?' or NaN
    X = X.replace('?', np.nan)
    # Simple strategy: drop rows with NaNs for this demo
    clean_mask = X.notna().all(axis=1)
    X = X[clean_mask]
    y = y[clean_mask]

    # 2. Encode target labels (<=50K -> 0, >50K -> 1)
    # The dataset typically has strings like '<=50K', '>50K', '<=50K.', '>50K.'
    y = y.astype(str).str.replace('.', '', regex=False).str.strip()
    y = (y == '>50K').astype(int)

    # 3. Feature Selection & Encoding
    # We'll use a mix of numerical and categorical features
    # Standard names for Adult dataset from OpenML 1590
    potential_numeric = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'educational-num', 'education-num']
    potential_categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender']
    
    # Filter only existing columns
    numeric_features = [f for f in potential_numeric if f in X.columns]
    categorical_features = [f for f in potential_categorical if f in X.columns]
    
    print(f"Using numeric features: {numeric_features}")
    print(f"Using categorical features: {categorical_features}")

    X_num = X[numeric_features].astype(float)
    X_cat = X[categorical_features].astype(str)
    
    # One-Hot Encoding for categorical features
    X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
    
    # Combine back
    X_processed = pd.concat([X_num, X_cat_encoded], axis=1)

    # Split into train and test FIRST (avoid preprocessing leakage)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale all features (fit on train only)
    # Even OHE features benefit from scaling in SGD if regularization is used
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Split training data among clients
    client_datasets = []
    chunk_size = len(X_train) // n_clients
    
    if chunk_size == 0:
        raise ValueError(f"Not enough training samples ({len(X_train)}) for {n_clients} clients")
    
    indices = np.arange(len(X_train))

    if non_iid_label_skew or split_strategy.lower() in ("non-iid", "noniid", "label_skew"):
        indices = indices[np.argsort(np.array(y_train))]
    
    for i in range(n_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_clients - 1 else len(X_train)
        idx = indices[start:end]
        X_c = X_train[idx]
        y_c = np.array(y_train)[idx]
        client_datasets.append((X_c, y_c))

    print(f"Data loaded. {n_clients} clients, ~{chunk_size} samples per client.")
    print(f"Total training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    return client_datasets, (X_test, y_test)

if __name__ == "__main__":
    # Quick test
    clients, test = load_and_preprocess_data(3)
    print("Test set shape:", test[0].shape)
