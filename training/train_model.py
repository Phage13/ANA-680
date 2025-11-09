import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Paths

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(REPO_ROOT, "data", "StudentsPerformance.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "student_app", "model.pkl")


def main():
    # Load Dataset
    df = pd.read_csv(DATA_PATH)

    # Features: math, reading, writing scores
    X = df[["math score", "reading score", "writing score"]].values

    # Target: Race/Ethnicity
    y = df["race/ethnicity"].astype("category")
    classes = list(y.cat.categories)   # store class labels
    y_codes = y.cat.codes.values       # convert to numeric codes

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_codes, test_size=0.2, random_state=42, stratify=y_codes
    )

    # Build Pipeline: Scale + Logistic Regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate

    acc = pipeline.score(X_test, y_test)
    print(f"Validation accuracy: {acc:.3f}")

    # Save Pipeline and Class Labels together
    payload = {
        "pipeline": pipeline,
        "classes": classes
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)

if __name__ == "__main__":
    main()