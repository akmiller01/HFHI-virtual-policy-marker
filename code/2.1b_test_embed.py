import pandas as pd
import re
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data
dataset = load_dataset("alex-miller/crs-2014-2023-housing-similarity", split="train")
dat = dataset.to_pandas()

# Load keywords
keywords = pd.read_csv("input/keywords.csv")

# Create regex pattern for keywords
keyword_regex = "|".join([fr"\b{re.escape(word)}\b" for word in keywords["word"]])

# Apply keyword matching
print("Finding keywords...")
dat["keyword"] = dat["text"].str.lower().str.contains(keyword_regex, regex=True, na=False)

# Define target condition
dat["target"] = ((dat["sector_code"].isin([16030, 16040])) | dat["keyword"]).astype(int)

# Variables
y = dat.pop('target').values.astype(int)
dat.pop('text')
dat.pop('sector_code')

X = dat.values.astype(float)

# Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
X_train = X_test = X
y_train = y_test = y

# Remove split keyword search
X_train_keyword = X_train[:,X_train.shape[1] - 1]
X_test_keyword = X_test[:,X_test.shape[1] - 1]
X_train = X_train[:,range(X_train.shape[1] -1)]
X_test = X_test[:,range(X_test.shape[1] -1)]

# Fit
print("Fitting model...")
xgb_class = xgb.XGBClassifier(n_estimators=100)
xgb_class.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_class.predict(X_test)
xgb_or_kw = np.any(np.array([y_pred_xgb, X_test_keyword]), axis=0).astype(int)

# Calculate metrics
acc = accuracy_score(y_test, y_pred_xgb)
f1 = f1_score(y_test, y_pred_xgb)
rec = recall_score(y_test, y_pred_xgb)
pre = precision_score(y_test, y_pred_xgb)

print("\nXGBoost Classifier:")
print("Accuracy: ", acc)
print("F1: ", f1)
print("Recall: ", rec)
print("Precision: ", pre)
print(confusion_matrix(y_test, y_pred_xgb))

k_acc = accuracy_score(y_test, X_test_keyword)
k_f1 = f1_score(y_test, X_test_keyword)
k_rec = recall_score(y_test, X_test_keyword)
k_pre = precision_score(y_test, X_test_keyword)

print("\nKeyword search:")
print("Accuracy: ", k_acc)
print("F1: ", k_f1)
print("Recall: ", k_rec)
print("Precision: ", k_pre)
print(confusion_matrix(y_test, X_test_keyword))


both_acc = accuracy_score(y_test, xgb_or_kw)
both_f1 = f1_score(y_test, xgb_or_kw)
both_rec = recall_score(y_test, xgb_or_kw)
both_pre = precision_score(y_test, xgb_or_kw)

print("\nBoth together:")
print("Accuracy: ", both_acc)
print("F1: ", both_f1)
print("Recall: ", both_rec)
print("Precision: ", both_pre)
print(confusion_matrix(y_test, xgb_or_kw))


# Print the results
cm = confusion_matrix(y_test, xgb_or_kw)
labels = ['Not relevant', 'Relevant']
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()