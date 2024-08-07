import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("C:\\Users\\prabh\\Desktop\\BCIs\\Datasets\\Schizophrenia\\responder_erp_features_outliered.csv")

# Check for missing values
print('nan values: ')
print(df.isna().sum().sort_values())

# Compute the correlation matrix
corr_matrix = df.drop(['Class'], axis=1).corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')

# Correlation with the target variable
corr_with_target = df.corr()['Class'].sort_values(ascending=False)

# Plot the correlation with the target variable
plt.figure(figsize=(8, 6))
corr_with_target.drop('Class').plot(kind='bar')
plt.title('Correlation of Features with Target Variable')
plt.ylabel('Correlation coefficient')
plt.xlabel('Features')

# Identify highly correlated features
threshold = 0.8  # Set a threshold for considering features as highly correlated
high_corr = corr_matrix.abs() > threshold
high_corr_pairs = [(i, j) for i in range(high_corr.shape[0]) for j in range(i+1, high_corr.shape[1]) if high_corr.iloc[i, j]]

# Print highly correlated pairs
print("Highly correlated feature pairs:")
for i, j in high_corr_pairs:
    print(f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]}")

# Apply PCA to highly correlated feature pairs and replace original features
pca = PCA(n_components=1)
features_to_drop = set()
pca_feature_names = []

for i, j in high_corr_pairs:
    feature1 = corr_matrix.columns[i]
    feature2 = corr_matrix.columns[j]
    
    # Apply PCA to the pair
    features = df[[feature1, feature2]]
    principal_component = pca.fit_transform(features)
    pc_feature_name = f"{feature1}_{feature2}_PC"
    pca_feature_names.append(pc_feature_name)
    
    # Add the principal component to the DataFrame
    df[pc_feature_name] = principal_component
    
    # Add original features to drop list
    features_to_drop.update([feature1, feature2])

# Drop the original features that were replaced by principal components
df_reduced = df.drop(columns=features_to_drop)

# Compute correlation with the target variable for the new DataFrame
corr_with_target_reduced = df_reduced.corr()['Class'].sort_values(ascending=False)

# Determine a threshold to remove features with low correlation to the target variable
correlation_threshold = 0.05  # You can adjust this threshold based on your needs

# Identify features to drop based on low correlation with the target variable
low_corr_features = corr_with_target_reduced[corr_with_target_reduced.abs() < correlation_threshold].index

# Combine low correlation features to drop
features_to_drop = set(low_corr_features)

# Drop low correlation features
df_final = df_reduced.drop(columns=features_to_drop, errors='ignore')

# Print the final DataFrame
print("\nFeatures to be dropped:")
print(features_to_drop)

print("\nFinal DataFrame:")
print(df_final.head())

# Plot the correlation matrix heatmap of the final DataFrame
final_corr_matrix = df_final.drop(['Class'], axis=1).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(final_corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Final Correlation Matrix Heatmap')

# Save the final DataFrame to a CSV file
output_file = r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\responder_erp_features_final.csv'
df_final.to_csv(output_file, index=False)

print(f"Dataset saved to {output_file}")