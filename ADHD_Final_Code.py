import mne
import numpy as np
import pandas as pd
import os
import networkx as nx
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.inspection import permutation_importance

output_directory = r"D:\Amir\MSU\Spring 2024\Network Flow & Dynamic Programming\Project\Codes\Correct version\CSV Feature Files"
data_path = r'D:\Amir\MSU\Spring 2024\Network Flow & Dynamic Programming\Project\Data\12012023Signals\Epochs_RESTdata\*.fif'

def load_epochs_data(file_path):
    """
    Load epochs from a single .fif file.
    """
    return mne.read_epochs(file_path, preload=True)

def load_features(directory):
    """
    Load and concatenate features from CSV files stored in the directory.
    Checks if CSV files exist and handles the case where no files are available.
    """
    # Collect all CSV files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    # Create a list of DataFrames
    df_list = [pd.read_csv(file) for file in files]
    
    # Check if the list is empty
    if not df_list:
        print("No CSV files found in the directory.")
        return None  # or return pd.DataFrame() to return an empty DataFrame
    
    # Concatenate all DataFrames into a single DataFrame
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def prepare_data(features_df):
    """
    Prepares features and labels for training from the DataFrame.
    """
    features = features_df.iloc[:, 1:-1].values  # Exclude the Patient_ID and ADHD columns
    labels = (features_df['ADHD'] == 'Y').astype(int).values  # Convert labels to 0 and 1

    X_augmented = np.vstack([features, -features])  # Example augmentation: invert values
    y_augmented = np.hstack([labels, labels])

    X_combined, y_combined = shuffle(X_augmented, y_augmented)
    return X_combined, y_combined, features_df.iloc[:, 1:-1].values, labels  # Ensure X is a NumPy array

def perform_nested_cross_validation(X, y, param_grid, n_splits=5):
    outer_kf = KFold(n_splits=n_splits, shuffle=True)
    outer_accuracy_scores = []
    outer_precision_scores = []
    outer_recall_scores = []
    outer_f1_scores = []

    for train_val_index, test_index in outer_kf.split(X):
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]


        # Define the scoring metrics for cross-validation
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }

        # Initialize the RandomForestClassifier
        rf_classifier = RandomForestClassifier()

        # Set up GridSearchCV with the scoring dictionary
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)

        # Fit the model using GridSearchCV
        grid_search.fit(X_train_val, y_train_val)

        # Best model after grid search
        best_rf_model = grid_search.best_estimator_




        # Evaluate on the outer test set
        y_test_pred = best_rf_model.predict(X_test)
        
        outer_accuracy_scores.append(accuracy_score(y_test, y_test_pred))
        outer_precision_scores.append(precision_score(y_test, y_test_pred))
        outer_recall_scores.append(recall_score(y_test, y_test_pred))
        outer_f1_scores.append(f1_score(y_test, y_test_pred))

    # Calculate average metrics
    avg_accuracy = np.mean(outer_accuracy_scores)
    avg_precision = np.mean(outer_precision_scores)
    avg_recall = np.mean(outer_recall_scores)
    avg_f1 = np.mean(outer_f1_scores)

    return avg_accuracy, avg_precision, avg_recall, avg_f1, outer_accuracy_scores, outer_precision_scores, outer_recall_scores, outer_f1_scores, best_rf_model

# Load and prepare the data
features_df = load_features(output_directory)
if features_df is None or features_df.empty:
    print("No data available for further processing.")
else:
    # Proceed with data processing
    X_combined, y_combined, X, y = prepare_data(features_df)
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform nested cross-validation
    avg_accuracy, avg_precision, avg_recall, avg_f1, accuracy_scores, precision_scores, recall_scores, f1_scores, best_rf_model = perform_nested_cross_validation(X_combined, y_combined, param_grid)
    
    print(f"Average Accuracy: {avg_accuracy:.2f}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1 Score: {avg_f1:.2f}")

    # Feature Importance using the best model from the last fold
    best_rf_model.fit(X_combined, y_combined)  # Fit model on full data
    feature_importances = best_rf_model.feature_importances_
    feature_names = features_df.columns[1:-1]  # Exclude Patient_ID and ADHD columns





# Rename columns to the appropriate sleep stage names
rename_dict = {
    'Sleep_ID_1': 'Wake',
    'Sleep_ID_2': 'Sleep Stage 1',
    'Sleep_ID_3': 'Sleep Stage 2',
    'Sleep_ID_4': 'Sleep Stage 3-4',
    'Sleep_ID_5': 'Rapid Eye Movement'
}
features_df.rename(columns=rename_dict, inplace=True)

# Updated feature names after renaming
feature_names = list(features_df.columns[1:-1])

# Enhanced Plotting of Feature Importances (Random Forest)
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.title("Feature Importance for ADHD Prediction", fontsize=18, fontweight='bold')
plt.xlabel("Importance Score", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Annotating the bars with the exact importance values
for index, value in enumerate(feature_importances):
    plt.text(value + 0.01, index, f'{value:.4f}', va='center', fontsize=12)

plt.show()

# Enhanced Permutation Importance Plot
perm_importance = permutation_importance(best_rf_model, X_combined, y_combined, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
}).sort_values(by='Importance Mean', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance Mean', y='Feature', data=perm_importance_df, palette='magma', ci=None)
plt.errorbar(x=perm_importance_df['Importance Mean'], y=np.arange(len(perm_importance_df)), 
             xerr=perm_importance_df['Importance Std'], fmt='none', c='black', capsize=5)

# Annotating the bars with the exact mean importance values and standard deviations
for index, value in enumerate(perm_importance_df['Importance Mean']):
    std_dev = perm_importance_df['Importance Std'].iloc[index]
    plt.text(value + 0.01, index, f'{value:.4f} ± {std_dev:.4f}', va='center', fontsize=12)

plt.title("Permutation Feature Importance for ADHD Prediction", fontsize=18, fontweight='bold')
plt.xlabel("Importance Mean", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

    # Enhanced Plotting of Feature Importances (Random Forest)
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.title("Feature Importance for ADHD Prediction", fontsize=18, fontweight='bold')
plt.xlabel("Importance Score", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Enhanced Permutation Importance Plot
perm_importance = permutation_importance(best_rf_model, X_combined, y_combined, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
}).sort_values(by='Importance Mean', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance Mean', y='Feature', data=perm_importance_df, palette='magma', ci=None)
plt.errorbar(x=perm_importance_df['Importance Mean'], y=np.arange(len(perm_importance_df)), 
             xerr=perm_importance_df['Importance Std'], fmt='none', c='black', capsize=5)
plt.title("Permutation Feature Importance for ADHD Prediction", fontsize=18, fontweight='bold')
plt.xlabel("Importance Mean", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()



# Plot cross-validation results
metrics = ['accuracy', 'precision', 'recall', 'f1']
scores = [accuracy_scores, precision_scores, recall_scores, f1_scores]
colors = ['blue', 'red', 'green', 'purple']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
n_splits=5
for idx, metric in enumerate(metrics):
    axes[idx].bar(range(1, n_splits+1), scores[idx], color=colors[idx])
    axes[idx].set_title(f'{metric.capitalize()} per Fold')
    axes[idx].set_xlabel('Fold Number')
    axes[idx].set_ylabel(metric.capitalize())
    axes[idx].set_xticks(range(1, n_splits+1))
    avg_score = np.mean(scores[idx])
    axes[idx].axhline(avg_score, color='k', linestyle='--', label="Average")
    axes[idx].legend(loc='lower right')

plt.tight_layout()
plt.show()






