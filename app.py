import os

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)

# ==================== APP TITLE ====================

st.title("Bank Customer Churn Prediction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== STEP 1: DOWNLOAD TEST DATA ====================

st.subheader("Step 1: Download Test Data")

test_file = os.path.join(BASE_DIR, "test_data.csv")

if os.path.exists(test_file):
    with open(test_file, "rb") as f:
        st.download_button(
            "Download Test Data (CSV)",
            f,
            "test_data.csv",
            "text/csv",
        )
else:
    st.info("test_data.csv not found (this is okay if teacher did not require it).")

# ==================== STEP 2: UPLOAD DATA ====================

st.subheader("Step 2: Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain 'Exited' column)",
    type=["csv"],
)

# ==================== STEP 3: SELECT MODEL ====================

st.subheader("Step 3: Select Machine Learning Model")

model_name = st.selectbox(
    "Choose a model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ),
)

# ==================== STEP 4: PROCESS ====================

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Ensure target column exists
    if "Exited" not in df.columns:
        st.error("Dataset must contain an 'Exited' column.")
        st.stop()

    # Drop rows where Exited is NaN
    df = df.dropna(subset=["Exited"])

    # Drop ID-like columns if present
    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Check that dataset has at least two classes
    if y.nunique() < 2:
        st.error(
            "The dataset contains only one class in 'Exited'. "
            "Model training requires at least two classes."
        )
        st.stop()

    # Encode categorical columns
    if "Geography" in X.columns:
        X["Geography"] = LabelEncoder().fit_transform(X["Geography"])
    if "Gender" in X.columns:
        X["Gender"] = LabelEncoder().fit_transform(X["Gender"])

    # ==================== TRAIN-TEST SPLIT ====================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Safety check
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        st.error(
            "After splitting, train or test has only one class in 'Exited'. "
            "Please use a larger dataset with both 0 and 1 values in 'Exited'."
        )
        st.stop()

    # ==================== SCALING ====================

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==================== MODEL TRAINING ====================

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    elif model_name == "Naive Bayes":
        model = GaussianNB()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    elif model_name == "XGBoost":
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # ==================== METRICS ====================

    st.subheader(f"Evaluation Metrics ({model_name})")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    st.write("AUC:", round(roc_auc_score(y_test, y_prob), 4))
    st.write("Precision:", round(precision_score(y_test, y_pred), 4))
    st.write("Recall:", round(recall_score(y_test, y_pred), 4))
    st.write("F1 Score:", round(f1_score(y_test, y_pred), 4))
    st.write("MCC:", round(matthews_corrcoef(y_test, y_pred), 4))

    # ==================== CONFUSION MATRIX ====================

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"],
    )
    st.dataframe(cm_df)
else:
    st.info("Please upload a dataset to continue.")
