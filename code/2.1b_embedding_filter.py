import pandas as pd
import re
import numpy as np
from datasets import load_dataset
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
keyword_results = dat.pop('keyword').values.astype(int)
dat.pop('text')
dat.pop('sector_code')

X = dat.values.astype(float)

# Fit
print("Fitting model...")
xgb_class = xgb.XGBClassifier(n_estimators=100)
xgb_class.fit(X, y)

# Predict
y_pred_xgb = xgb_class.predict(X)
xgb_or_kw = np.any(np.array([y_pred_xgb, keyword_results]), axis=0).astype(int)

# Calculate metrics
acc = accuracy_score(y, y_pred_xgb)
f1 = f1_score(y, y_pred_xgb)
rec = recall_score(y, y_pred_xgb)
pre = precision_score(y, y_pred_xgb)

print("\nXGBoost Classifier:")
print("Accuracy: ", acc)
print("F1: ", f1)
print("Recall: ", rec)
print("Precision: ", pre)
print(confusion_matrix(y, y_pred_xgb))

k_acc = accuracy_score(y, keyword_results)
k_f1 = f1_score(y, keyword_results)
k_rec = recall_score(y, keyword_results)
k_pre = precision_score(y, keyword_results)

print("\nKeyword search:")
print("Accuracy: ", k_acc)
print("F1: ", k_f1)
print("Recall: ", k_rec)
print("Precision: ", k_pre)
print(confusion_matrix(y, keyword_results))

both_acc = accuracy_score(y, xgb_or_kw)
both_f1 = f1_score(y, xgb_or_kw)
both_rec = recall_score(y, xgb_or_kw)
both_pre = precision_score(y, xgb_or_kw)

print("\nBoth together:")
print("Accuracy: ", both_acc)
print("F1: ", both_f1)
print("Recall: ", both_rec)
print("Precision: ", both_pre)
print(confusion_matrix(y, xgb_or_kw))

# Save the results
cm = confusion_matrix(y, xgb_or_kw)
labels = ['Not relevant', 'Relevant']
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('output/Figure_1.png')

# Upload
cols_to_remove = dataset.column_names
cols_to_remove.remove('text')
cols_to_remove.remove('sector_code')
dataset = dataset.remove_columns(cols_to_remove)
dataset = dataset.add_column("selected", xgb_or_kw)
dataset = dataset.filter(lambda e: e['selected'] == 1 or e['sector_code'] in [16030, 16040])
dataset = dataset.remove_columns('selected')
dataset.push_to_hub('alex-miller/crs-2014-2023-housing-selection')