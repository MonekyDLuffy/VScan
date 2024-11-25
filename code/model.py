import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loadind dataset from csv
data = pd.read_csv("malware_signatures.csv")
print(data.info())
print(data.head())

# Checking for class balance
sns.countplot(x="label", data=data)
plt.title("Class Distribution")
plt.show()

# removing null values
if data.isnull().sum().any():
    print("Missing values detected, handling...")
    data = data.dropna()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# extracting features
tfidf_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=5000,
)

# TF-IDF vectorizing
X_tfidf = tfidf_vectorizer.fit_transform(data["signature"])


def compute_entropy(string):
    import math

    probabilities = [string.count(char) / len(string) for char in set(string)]
    return -sum(p * math.log2(p) for p in probabilities)


data["entropy"] = data["signature"].apply(compute_entropy)

X_features = np.hstack((X_tfidf.toarray(), data["entropy"].values.reshape(-1, 1)))

# output label
y = data["label"].map({"malware": 1, "benign": 0}).values

import json


# extracting features to json
def extract_dynamic_features(log):
    try:
        log_data = json.loads(log)
        api_calls = log_data.get("api_calls", [])
        return " ".join(api_calls)
    except json.JSONDecodeError:
        return ""


if "dynamic_log" in data.columns:
    data["dynamic_features"] = data["dynamic_log"].apply(extract_dynamic_features)

if "dynamic_features" in data.columns:
    data["combined_features"] = data["signature"] + " " + data["dynamic_features"]
else:
    data["combined_features"] = data["signature"]

from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

# Using PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_features)

selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold="mean")
X_selected = selector.fit_transform(X_features, y)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Tokenizing text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data["combined_features"])

# Converting tokens to sequences
X_sequences = tokenizer.texts_to_sequences(data["combined_features"])
X_padded = pad_sequences(X_sequences, maxlen=500)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential(
    [
        Embedding(input_dim=10000, output_dim=128, input_length=500),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ]
)

# Compiling model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training model
history = model.fit(X_padded, y, epochs=10, batch_size=32, validation_split=0.2)

import random


def mutate_signature(signature):
    chars = list(signature)
    for _ in range(random.randint(1, 3)):
        idx = random.randint(0, len(chars) - 1)
        chars[idx] = random.choice("abcdef0123456789")
    return "".join(chars)


# Applying mutation analysis
augmented_data = data.copy()
augmented_data["signature"] = augmented_data["signature"].apply(
    lambda x: mutate_signature(x) if random.random() > 0.5 else x
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5))),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=20)),
    ]
)

# Training the pipeline
pipeline.fit(data["combined_features"], y)

# Saving the pipeline
joblib.dump(pipeline, "malware_pipeline.pkl")
