import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("spam.csv", encoding="latin-1")


df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Dataset Shape:", df.shape)
print(df.head())


df['label'] = df['label'].map({'ham': 0, 'spam': 1})


X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)


models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": LinearSVC()
}

results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

 
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
print("\nModel Comparison:")
print(results_df)



best_model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('model', MultinomialNB())
])

best_model.fit(X_train, y_train)

test_messages = [
    "Congratulations! You won a free lottery ticket. Call now!",
    "Hi, are we meeting tomorrow for class?"
]

predictions = best_model.predict(test_messages)

print("\nExample Predictions:")
for msg, pred in zip(test_messages, predictions):
    print(msg, " --> ", "SPAM" if pred == 1 else "HAM")