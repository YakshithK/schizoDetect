import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os

# Function to remove features with low correlation with the target
def remove_low_corr_features(df, threshold):
    corr_with_target = df.corr()['Class'].sort_values(ascending=False)
    low_corr_features = corr_with_target[corr_with_target.abs() < threshold].index
    return df.drop(columns=low_corr_features, errors='ignore'), low_corr_features

# Directory containing the CSV files
input_directory = "C:\\Users\\prabh\\Desktop\\BCIs\\Datasets\\Schizophrenia\\files2"

# Output path template
output_path_template = r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\files3\responder_erp_features_final_{}_{}_{}.csv'

# Iterate over each CSV file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_directory, filename)
        
        # Load the dataset
        df = pd.read_csv(filepath)

        # Check for missing values
        print(f'\nProcessing file: {filename}')
        print('NaN values: ')
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

        # Strategy 1: PCA
        df_pca = df.copy()
        pca = PCA(n_components=1)
        pca_feature_names = []
        features_to_drop_pca = set()

        for i, j in high_corr_pairs:
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]

            # Apply PCA to the pair
            features = df_pca[[feature1, feature2]]
            principal_component = pca.fit_transform(features)
            pc_feature_name = f"{feature1}_{feature2}_PC"
            pca_feature_names.append(pc_feature_name)

            # Add the principal component to the DataFrame
            df_pca[pc_feature_name] = principal_component

            # Add original features to drop list
            features_to_drop_pca.update([feature1, feature2])

        # Drop the original features that were replaced by principal components
        df_pca = df_pca.drop(columns=features_to_drop_pca)

        # Save the PCA modified DataFrame
        output_file_pca = output_path_template.format('PCA', threshold, filename.split('.')[0])
        df_pca.to_csv(output_file_pca, index=False)
        print(f"PCA modified dataset saved to {output_file_pca}")

        # Strategy 2: Averaging
        df_avg = df.copy()
        features_to_drop_avg = set()
        average_feature_names = []

        for i, j in high_corr_pairs:
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]

            # Compute the average of the pair
            avg_feature_name = f"{feature1}_{feature2}_AVG"
            df_avg[avg_feature_name] = df_avg[[feature1, feature2]].mean(axis=1)
            average_feature_names.append(avg_feature_name)

            # Add original features to drop list
            features_to_drop_avg.update([feature1, feature2])

        # Drop the original features that were replaced by average
        df_avg = df_avg.drop(columns=features_to_drop_avg)

        # Save the Average modified DataFrame
        output_file_avg = output_path_template.format('AVG', threshold, filename.split('.')[0])
        df_avg.to_csv(output_file_avg, index=False)
        print(f"Averaging modified dataset saved to {output_file_avg}")

        # Strategy 3: Dropping one feature
        df_drop = df.copy()
        features_to_drop_drop = set()

        for i, j in high_corr_pairs:
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]

            # Drop one of the highly correlated features (feature2 in this case)
            features_to_drop_drop.add(feature2)

        # Drop the identified features
        df_drop = df_drop.drop(columns=features_to_drop_drop)

        # Save the Dropped features modified DataFrame
        output_file_drop = output_path_template.format('Drop', threshold, filename.split('.')[0])
        df_drop.to_csv(output_file_drop, index=False)
        print(f"Dropping one feature modified dataset saved to {output_file_drop}")

        # Apply and save each method with low correlation feature removal
        for method, df_method in [('PCA', df_pca), ('AVG', df_avg), ('Drop', df_drop)]:
            for threshold in np.arange(0.01, 0.11, 0.01):
                df_final, low_corr_features = remove_low_corr_features(df_method, threshold)

                print(f"\nMethod: {method}, Threshold: {threshold}")
                print("\nFeatures to be dropped:")
                print(low_corr_features)

                print("\nFinal DataFrame:")
                print(df_final.head())

                # Plot the correlation matrix heatmap of the final DataFrame
                final_corr_matrix = df_final.drop(['Class'], axis=1).corr()
                plt.figure(figsize=(12, 8))
                sns.heatmap(final_corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title(f'Final Correlation Matrix Heatmap ({method}, Threshold={threshold})')
                

                # Save the final DataFrame to a CSV file
                output_file_final = output_path_template.format(method, threshold, filename.split('.')[0])
                df_final.to_csv(output_file_final, index=False)
                print(f"Dataset saved to {output_file_final}")
