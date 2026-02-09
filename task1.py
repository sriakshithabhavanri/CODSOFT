

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv(
    "/content/train_data.txt",
    sep=" ::: ",
    engine="python",
    header=None,
    names=["id", "title", "genre", "description"]
)

X = train_data["description"]   
y = train_data["genre"]        

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

new_plot = [
    "A fearless warrior begins an epic quest filled with magic, ancient prophecies, deadly monsters, and massive battles to defeat an evil king and save the fantasy kingdom"
]




new_plot_tfidf = tfidf.transform(new_plot)
prediction = model.predict(new_plot_tfidf)

print("\nPredicted Genre:", prediction[0])