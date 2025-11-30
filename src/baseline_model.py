from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from .data_preprocessing import (
    load_adult_dataset,
    build_preprocessor,
    train_test_split_adult,
)

def run_baseline():
    df = load_adult_dataset()
    X_train, X_test, y_train, y_test = train_test_split_adult(df)

    preprocessor = build_preprocessor(X_train)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Baseline accuracy: {acc:.4f}")
    print(report)

    return clf, acc, report
