from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time 
from imblearn.over_sampling import SMOTE
from itertools import product
import os

path = r'C:\Users\prabh\Desktop\BCIs\Project 5\dataset.csv'

df = pd.read_csv(path)

X = df.drop(['Class'], axis=1)
y = df['Class']

classes = y.value_counts()
classes.plot(kind='bar')

plt.xlabel('Class')
plt.ylabel('Quantity')
plt.title('Quantity of Each Class')

rus = SMOTE(random_state=42)
X, y = rus.fit_resample(X, y)

resampled_classes = y.value_counts()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=classes.index, y=classes.values)
plt.title('Original Classes')
plt.xlabel('Class')
plt.ylabel('Quantity')

plt.subplot(1, 2, 2)    
sns.barplot(x=resampled_classes.index, y=resampled_classes.values)
plt.title('Resampled Classes')
plt.xlabel('Class')
plt.ylabel('Quantity')

plt.tight_layout()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42, shuffle=True)

model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=False)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))