import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("/content/fraudTest.csv")
test_df = pd.read_csv("/content/fraudTrain.csv")

train_df = train_df.dropna()
test_df = test_df.dropna()

print("Train DataFrame Columns:", train_df.columns)
print("Test DataFrame Columns:", test_df.columns)

non_numeric_cols_to_drop = [
    'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
    'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num'
]

X_train = train_df.drop(columns=["is_fraud"] + non_numeric_cols_to_drop, errors='ignore')
y_train = train_df["is_fraud"]

X_test = test_df.drop(columns=["is_fraud"] + non_numeric_cols_to_drop, errors='ignore')
y_test = test_df["is_fraud"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Model Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sample_transaction = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(sample_transaction)

if prediction[0] == 1:
    print("\nOutput: Fraudulent Transaction")
else:
    print("\nOutput: Legitimate Transaction")