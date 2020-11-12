# Databricks notebook source
# MAGIC %md
# MAGIC #### Data Source: [Kaggle Credit Card Faud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# COMMAND ----------

df = pd.read_csv("../data/creditcard.csv")

# COMMAND ----------

df.head(10)

# COMMAND ----------

df.isna().sum()

# COMMAND ----------

df["Class"].value_counts()

# COMMAND ----------

fig , ax = plt.subplots(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Count of Fraud cases")
plt.show()

# COMMAND ----------

df[["Class", "Time", "Amount"]].describe()

# COMMAND ----------

df["Time"].hist(bins=20)

# COMMAND ----------

fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x="Class", data=df, hue="Time")
plt.title("Fraud cases based on Time")
plt.show()

# COMMAND ----------

fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x="Class", data=df, hue="Amount")
plt.title("Fraud cases based on Amount")
plt.show()

# COMMAND ----------

df["Amount"].hist(bins=20)

# COMMAND ----------

pd.plotting.scatter_matrix(df[["Class", "Time", "Amount"]], alpha=0.2)

# COMMAND ----------

pd.plotting.scatter_matrix(df[["Class", "V1", "V2", "V3","V4"]], alpha=0.3, c="yellow")

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# COMMAND ----------

random_seed = 111

# COMMAND ----------

raw_df = pd.read_csv("../data/creditcard.csv")
y = raw_df["Class"]
X = raw_df.drop("Class", axis=1)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed, stratify=y)
clf = GaussianNB()

# COMMAND ----------

y_train.sum() / y_train.count()

# COMMAND ----------

y_test.sum() / y_test.count()

# COMMAND ----------

print(f"Training size: {y_train.count()}, Test set size: {y_test.count()}")

# COMMAND ----------

clf.fit(X_train, y_train)

# COMMAND ----------

y_pred = clf.predict(X_test)

# COMMAND ----------

clf.predict_proba(X_test)[:, 1] # positive class predict probability

# COMMAND ----------

roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

# COMMAND ----------

recall_score(y_test, y_pred)

# COMMAND ----------

y_pred.sum()

# COMMAND ----------

