import pandas as pd
import re
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score

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
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest Classifier:")
print("MSE: ", mse_rf)
print("R²: ", r2_rf)

# Plot the results
plt.scatter(y_test, y_pred_rf, label='Random Forest Classifier', color='orange', alpha=0.75)
plt.axline((0, 0), slope=1, alpha=0.75)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.title("Model evaluation")
plt.show()