import pandas as pd
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# Load the data
dataset = load_dataset("alex-miller/crs-2014-2023-housing-similarity", split="train")
dat = dataset.to_pandas()

# Load keywords
keywords = pd.read_csv("~/git/HFHI-NLP/input/keywords.csv")

# Filter keywords for relevant terms
relevant_terms = ["housing", "informal settlement", "informal settlements", "shelter", "shelters", "slum", "slums"]
keywords = keywords[keywords["literal_translation"].isin(relevant_terms)]

# Filter for specific languages
languages = ["English", "French", "Spanish"]
keywords = keywords[keywords["language"].isin(languages)]

# Create regex pattern for keywords
keyword_regex = "|".join([fr"\b{re.escape(word)}\b" for word in keywords["word"]])

# Apply keyword matching
dat["keyword"] = dat["text"].str.lower().str.contains(keyword_regex, regex=True, na=False)

# Define target condition
dat["target"] = ((dat["sector_code"].isin([16030, 16040])) | dat["keyword"]).astype(int)

# Variables
y = dat.pop('target').values.astype(int)
dat.pop('text')
dat.pop('sector_code')
dat.pop('keyword')
X = dat.values.astype(float)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# Fit
rf_class = RandomForestClassifier(n_estimators=100, random_state=0)
rf_class.fit(X_train, y_train)

# Predict
y_pred_rf = rf_class.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-squared (R²)
acc = accuracy_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
rec = recall_score(y_test, y_pred_rf)
pre = precision_score(y_test, y_pred_rf)

print("\nRandom Forest Classifier:")
print("Accuracy: ", acc)
print("F1: ", f1)
print("Accuracy: ", rec)
print("F1: ", pre)

# Print the results
print(confusion_matrix(y_test, y_pred_rf))