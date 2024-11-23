import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("malware_signatures.csv")  # Replace with your dataset file
print(data.info())
print(data.head())

# Check for class balance
sns.countplot(x="label", data=data)
plt.title("Class Distribution")
plt.show()

# Handle missing values, if any
if data.isnull().sum().any():
    print("Missing values detected, handling...")
    data = data.dropna()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Static feature extraction
tfidf_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),  # Character-level n-grams
    max_features=5000,  # Limit feature space for efficiency
)

# Convert signatures into TF-IDF features
X_tfidf = tfidf_vectorizer.fit_transform(data["signature"])


# Compute entropy as an additional feature
def compute_entropy(string):
    import math

    probabilities = [string.count(char) / len(string) for char in set(string)]
    return -sum(p * math.log2(p) for p in probabilities)


data["entropy"] = data["signature"].apply(compute_entropy)

# Combine features (TF-IDF and entropy)
X_features = np.hstack((X_tfidf.toarray(), data["entropy"].values.reshape(-1, 1)))

# Labels
y = data["label"].map({"malware": 1, "benign": 0}).values

import json


# Example: Extract system calls from JSON logs
def extract_dynamic_features(log):
    try:
        log_data = json.loads(log)
        api_calls = log_data.get("api_calls", [])
        return " ".join(api_calls)
    except json.JSONDecodeError:
        return ""


# Assuming `dynamic_log` is a column with behavioral logs in JSON format
if "dynamic_log" in data.columns:
    data["dynamic_features"] = data["dynamic_log"].apply(extract_dynamic_features)

# Combine static and dynamic features
if "dynamic_features" in data.columns:
    data["combined_features"] = data["signature"] + " " + data["dynamic_features"]
else:
    data["combined_features"] = data["signature"]

from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

# Dimensionality reduction using PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_features)

# Optional: Feature selection using a tree-based model
selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold="mean")
X_selected = selector.fit_transform(X_features, y)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Tokenize text
tokenizer = Tokenizer(num_words=10000)  # Top 10,000 words
tokenizer.fit_on_texts(data["combined_features"])

# Convert to sequences
X_sequences = tokenizer.texts_to_sequences(data["combined_features"])
X_padded = pad_sequences(X_sequences, maxlen=500)  # Pad to fixed length

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Define the model
model = Sequential(
    [
        Embedding(input_dim=10000, output_dim=128, input_length=500),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),  # Binary classification
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_padded, y, epochs=10, batch_size=32, validation_split=0.2)

import random


def mutate_signature(signature):
    # Example: Randomly replace characters in the signature
    chars = list(signature)
    for _ in range(random.randint(1, 3)):
        idx = random.randint(0, len(chars) - 1)
        chars[idx] = random.choice("abcdef0123456789")  # Hexadecimal chars
    return "".join(chars)


# Apply mutation to malware samples
augmented_data = data.copy()
augmented_data["signature"] = augmented_data["signature"].apply(
    lambda x: mutate_signature(x) if random.random() > 0.5 else x
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Define the pipeline
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5))),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=20)),
    ]
)

# Train the pipeline
pipeline.fit(data["combined_features"], y)

# Save the pipeline
joblib.dump(pipeline, "malware_pipeline.pkl")
