import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

# Load the dataset
df = pd.read_csv("C:\\Users\\prabh\\Desktop\\BCIs\\Datasets\\Schizophrenia\\responder_erp_features.csv")

# Drop the target column for calculations
features_df = df.drop(['Class'], axis=1)

# Function to calculate and collect outliers
def calculate_and_collect_outliers(df):
    stats = df.describe().transpose()
    stats['Q1'] = df.quantile(0.25)
    stats['Q2'] = stats['50%']
    stats['Q3'] = df.quantile(0.75)
    stats['IQR'] = stats['Q3'] - stats['Q1']
    stats['Lower Bound'] = stats['Q1'] - 1.5 * stats['IQR']
    stats['Upper Bound'] = stats['Q3'] + 1.5 * stats['IQR']
    
    outlier_indices = set()
    
    for feature in df.columns:
        lower_bound = stats.at[feature, 'Lower Bound']
        upper_bound = stats.at[feature, 'Upper Bound']
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
        outlier_indices.update(outliers.index)
    
    return outlier_indices

# Calculate and collect outlier indices
outlier_indices = calculate_and_collect_outliers(features_df)

# 1. Capping (Winsorizing)
winsorized_df = features_df.copy()
for column in winsorized_df.columns:
    winsorized_df[column] = winsorize(winsorized_df[column], limits=[0.05, 0.05])

# Add back the target column
winsorized_df['Class'] = df['Class']

# Save to CSV
winsorized_df.to_csv(r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\responder_erp_features_winsorized.csv', index=False)
print("Winsorized dataset saved.")

# 2. Log Transformation
log_transformed_df = features_df.copy()
for column in log_transformed_df.columns:
    log_transformed_df[column] = np.log(log_transformed_df[column] + 1)

# Add back the target column
log_transformed_df['Class'] = df['Class']

# Save to CSV
log_transformed_df.to_csv(r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\responder_erp_features_log_transformed.csv', index=False)
print("Log transformed dataset saved.")

# 3. Imputation (Replacing with Median)
imputed_df = features_df.copy()
for column in imputed_df.columns:
    upper_threshold = imputed_df[column].quantile(0.99)
    median = imputed_df[column].median()
    imputed_df[column] = np.where(imputed_df[column] > upper_threshold, median, imputed_df[column])

# Add back the target column
imputed_df['Class'] = df['Class']

# Save to CSV
imputed_df.to_csv(r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\responder_erp_features_imputed.csv', index=False)
print("Imputed dataset saved.")

# 4. Robust Scaler
robust_scaled_df = features_df.copy()
scaler = RobustScaler()
robust_scaled_df = pd.DataFrame(scaler.fit_transform(robust_scaled_df), columns=robust_scaled_df.columns)

# Add back the target column
robust_scaled_df['Class'] = df['Class']

# Save to CSV
robust_scaled_df.to_csv(r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\responder_erp_features_robust_scaled.csv', index=False)
print("Robust scaled dataset saved.")

# 5. Isolation Forest
isolation_forest_df = features_df.copy()
clf = IsolationForest(contamination=0.1, random_state=42)
outlier_pred = clf.fit_predict(isolation_forest_df)

# Replace outliers with NaN
isolation_forest_df.loc[outlier_pred == -1, :] = np.nan

# Fill NaN values with median
isolation_forest_df = isolation_forest_df.fillna(isolation_forest_df.median())

# Add back the target column
isolation_forest_df['Class'] = df['Class']

# Save to CSV
isolation_forest_df.to_csv(r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\responder_erp_features_isolation_forest.csv', index=False)
print("Isolation Forest processed dataset saved.")
