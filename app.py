import streamlit as st
import pandas as pd

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
    roc_auc_score
)

# -------------------- UI --------------------
st.title("Bank Customer Churn Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain 'Exited' column)",
    type=["csv"]
)

model_name = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -------------------- PROCESS --------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Exited" not in df.columns:
        st.error("CSV must contain 'Exited' column")
    else:
        # Drop ID columns if present
        for col in ["RowNumber", "CustomerId", "Surname"]:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        X = df.drop("Exited", axis=1)
        y = df["Exited"]

        # Encode categorical columns
        if "Geography" in X.columns:
            X["Geography"] = LabelEncoder().fit_transform(X["Geography"])
        if "Gender" in X.columns:
            X["Gender"] = LabelEncoder().fit_transform(X["Gender"])

        # Train-test split (ASSIGNMENT REQUIRED)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Scaling (needed for LR, KNN, NB)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -------------------- MODEL SELECTION --------------------
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
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # -------------------- METRICS --------------------
        st.subheader(f"Evaluation Metrics ({model_name})")

        st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        st.write("AUC:", round(roc_auc_score(y_test, y_prob), 4))
        st.write("Precision:", round(precision_score(y_test, y_pred), 4))
        st.write("Recall:", round(recall_score(y_test, y_pred), 4))
        st.write("F1 Score:", round(f1_score(y_test, y_pred), 4))
        st.write("MCC:", round(matthews_corrcoef(y_test, y_pred), 4))

        # -------------------- CONFUSION MATRIX --------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted No Churn", "Predicted Churn"],
            index=["Actual No Churn", "Actual Churn"]
        )

        st.dataframe(cm_df)

else:
    st.info("Please upload test data CSV file.")









