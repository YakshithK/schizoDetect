import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:\\Users\\prabh\\Desktop\\BCIs\\Datasets\\Schizophrenia\\responder_erp_features_outliered.csv")

# Function to calculate and collect indices of outliers
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
        print(f"\nFeature: {feature}")
        print(f"Min: {stats.at[feature, 'min']}")
        print(f"Max: {stats.at[feature, 'max']}")
        print(f"Q1: {stats.at[feature, 'Q1']}")
        print(f"Q2 (Median): {stats.at[feature, 'Q2']}")
        print(f"Q3: {stats.at[feature, 'Q3']}")
        
        lower_bound = stats.at[feature, 'Lower Bound']
        upper_bound = stats.at[feature, 'Upper Bound']
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
        
        print(f"Outliers: {outliers.values}")
        
        # Add outlier indices to the set
        outlier_indices.update(outliers.index)
    
    return outlier_indices

# Drop the target column for calculations
features_df = df.drop(['Class'], axis=1)

# Plot the original data
plt.figure(figsize=(15, 10))
sns.boxplot(data=features_df)
plt.xticks(rotation=90)
plt.title('Box Plot of Features (Original Data)')
plt.show()

# Calculate and collect outlier indices
outlier_indices = calculate_and_collect_outliers(features_df)

# Drop the rows with outliers
df_cleaned = df.drop(index=outlier_indices)

# Plot the cleaned data
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_cleaned.drop(['Class'], axis=1))
plt.xticks(rotation=90)
plt.title('Box Plot of Features (Cleaned Data)')
plt.show()

output_file = r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\responder_erp_features_outliered.csv'
df_cleaned.to_csv(output_file,index=False)

print(f"Dataset saved to {output_file}")