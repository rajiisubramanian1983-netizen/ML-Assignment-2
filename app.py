import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

st.title("Bank Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    X = df.drop(columns=["Exited"], errors="ignore")

    if "Geography" in X.columns:
        X["Geography"] = LabelEncoder().fit_transform(X["Geography"])
    if "Gender" in X.columns:
        X["Gender"] = LabelEncoder().fit_transform(X["Gender"])

    X = X.fillna(0)

    base = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base, "model", "logistic_regression.pkl"))
    scaler = joblib.load(os.path.join(base, "model", "scaler.pkl"))

    X_scaled = scaler.transform(X)
    df["Churn_Prediction"] = model.predict(X_scaled)

    st.success("Prediction completed")
    st.dataframe(df.head())

    st.download_button(
        "Download predictions",
        df.to_csv(index=False),
        "churn_predictions.csv",
        "text/csv"
    )
else:
    st.info("Please upload a CSV file.")
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
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader(f"Evaluation Metrics ({model_name})")
st.write("Accuracy:", round(acc, 4))
st.write("AUC:", round(auc, 4))
st.write("Precision:", round(prec, 4))
st.write("Recall:", round(rec, 4))
st.write("F1 Score:", round(f1, 4))
st.write("MCC:", round(mcc, 4))

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








