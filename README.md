1. Problem Statement
The objective of this project is to predict whether a bank customer will churn (leave the bank) based on their demographic, account, and transaction-related information using machine learning classification models.
This is a binary classification problem in the banking domain, where the target variable indicates whether a customer has exited the bank.
2. Dataset Description
Dataset Source: Kaggle – Bank Customer Churn Dataset
Domain: Banking
Problem Type: Binary Classification
Dataset Characteristics:
Number of instances: ~10,000
Number of features: 13+
Target Variable: Exited
1 → Customer churned
0 → Customer did not churn
Features Used:
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
The dataset is moderately imbalanced, hence robust metrics such as AUC and Matthews Correlation Coefficient (MCC) are used for evaluation.
3. Machine Learning Models Used
The following six classification models were implemented on the same dataset:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes (Gaussian)
Random Forest (Ensemble)
XGBoost (Ensemble)
4. Evaluation Metrics
Each model was evaluated using the following metrics:
Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)
5. Model Comparison Table
ML Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.805	0.771	0.589	0.143	0.229	0.216
Decision Tree	0.776	0.664	0.453	0.476	0.464	0.323
KNN	0.835	0.772	0.662	0.385	0.487	0.418
Naive Bayes	0.829	0.814	0.755	0.235	0.359	0.357
Random Forest (Ensemble)	0.864	0.846	0.782	0.459	0.578	0.529
XGBoost (Ensemble)	0.847	0.833	0.678	0.471	0.556	0.478
6. Model Performance Observations						
ML Model	Observation					
---------	-------------					
Logistic Regression	Provides a strong baseline with high accuracy but very low recall due to class imbalance. The low MCC indicates limited correlation in identifying churn customers.					
Decision Tree	Captures non-linear relationships in the data and improves recall compared to Logistic Regression, but may overfit and shows moderate MCC.					
KNN	Performance is sensitive to feature scaling. It achieves balanced precision and recall but does not significantly outperform tree-based models.					
Naive Bayes	Assumes feature independence, resulting in lower precision but higher recall. Useful for identifying churners but less accurate overall.					
Random Forest (Ensemble)	Demonstrates improved generalization and balanced performance across all metrics, with higher AUC and MCC compared to individual models.					
XGBoost (Ensemble)	Achieves the best overall performance with the highest AUC, F1 score, and MCC, effectively handling class imbalance and complex feature interactions.					

##  Live Demo

 Click here to access the Live Demo](https://ml-assignment-2-dqbjzzz5knncletkmm6udm.streamlit.app/)
##  GitHub Repository

 [ML-Assignment-2 on GitHub](https://github.com/rajiisubramanian1983-netizen/ML-Assignment-2)

 ScreenShot

 <img width="940" height="461" alt="image" src="https://github.com/user-attachments/assets/6f0cca87-9434-4170-9b5b-69e005f75363" />


