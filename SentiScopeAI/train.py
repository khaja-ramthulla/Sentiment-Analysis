
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

df = pd.read_csv("dataset/IMDB Dataset.csv")

# Remove missing values
df = df.dropna()

# Features and labels
X = df["review"]
y = df["sentiment"]

print(f"Total Reviews: {len(df)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training model...")

model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    random_state=42
)

model.fit(X_train_vec, y_train)

print("Evaluating...")

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# Save model
joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("\nModel Saved Successfully!")
print("Files created:")
print(" - model.pkl")
print(" - vectorizer.pkl")
