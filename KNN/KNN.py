from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time 
from imblearn.over_sampling import SMOTE
from itertools import product

df = pd.read_csv(r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\responder_erp_features_final_KNN.csv')

print('nan values: ')
print(df.isna().sum().sort_values())

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
print('scaling done')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42, shuffle=True)
print('splitting done')

param_grids = {
    'et': {
        'max_depth': [10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]
    },
    'knn': {
        'n_neighbors': [1, 3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    },
    'logreg': {
        'C': [0.001, 0.01, 0.1, 1, 10]
    },
    'randomforest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    'svc': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
}

models = {
    'knn': KNeighborsClassifier()
}

# Perform model training and evaluation
trainacc = {}
testacc = {}
params = {}
best_estimator = {}

for name, model in models.items():
    if name in param_grids:
        start = time.time()

        tuning = GridSearchCV(model, param_grid=param_grids[name], cv=5)
        tuning.fit(X_train, y_train)

        y_pred = tuning.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        testacc[name] = test_acc

        y_pred = tuning.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        trainacc[name] = accuracy

        best_estimator[name] = tuning.best_estimator_
        params[name] = tuning.best_params_
        
        print(f"Model: {name}")
        print(f"Training Accuracy: {accuracy}")
        print(f"Test Accuracy: {test_acc}")

        # Print feature importances for applicable models
        if hasattr(tuning.best_estimator_, 'feature_importances_'):
            print(f"Feature Importances for {name}:")
            importances = tuning.best_estimator_.feature_importances_
            feature_names = X.columns
            feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
            print(feature_importances)
            feature_importances.plot(kind='bar', x='Feature', y='Importance')
            plt.title(f'Feature Importances for {name}')
            

        end = time.time()
        elapsed = end - start
        print(f"Elapsed time for {name}: {elapsed} seconds\n")

print("Training accuracies: ", trainacc)
print("Test accuracies: ", testacc)
print("Best estimators: ", best_estimator)
print("Best parameters: ", params)
