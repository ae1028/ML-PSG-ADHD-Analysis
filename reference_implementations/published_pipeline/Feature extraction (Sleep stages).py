import mne
import numpy as np
import pandas as pd
import os
import networkx as nx
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

output_directory = r"D:\Amir\MSU\Spring 2024\Network Flow & Dynamic Programming\Project\Codes\Correct version\CSV Feature Files"
data_path = r'D:\Amir\MSU\Spring 2024\Network Flow & Dynamic Programming\Project\Data\12012023Signals\Epochs_RESTdata\*.fif'

def load_epochs_data(file_path):
    """
    Load epochs from a single .fif file.
    """
    return mne.read_epochs(file_path, preload=True)

def compute_avg_shortest_paths(epochs_data, patient_ids, sleep_ids):
    avg_shortest_paths_by_patient = {}
    all_shortest_paths = {sleep_id: [] for sleep_id in sleep_ids}  # Dictionary to store all paths per sleep_id

    adjacency_matrices = []  # This was missing; add it here.

    for patient_id in patient_ids:
        patient_epochs = epochs_data[epochs_data.metadata['ID'] == patient_id]
        shortest_paths_for_patient = []

        for sleep_id in sleep_ids:
            sleep_epochs = patient_epochs[patient_epochs.events[:, 2] == sleep_id]
            adjacency_matrices = []

            for epoch in sleep_epochs.iter_evoked():
                correlation_matrix = np.abs(np.corrcoef(epoch.data))
                adjacency_matrices.append(correlation_matrix)

            if adjacency_matrices:
                average_adjacency_matrix = np.mean(adjacency_matrices, axis=0)
                graph = nx.from_numpy_array(average_adjacency_matrix)
                shortest_path_lengths = dict(nx.shortest_path_length(graph, weight='weight'))
                total_paths = sum(len(v) for v in shortest_path_lengths.values())
                total_length = sum(sum(v.values()) for v in shortest_path_lengths.values())
                avg_shortest_path_length = total_length / total_paths
                shortest_paths_for_patient.append(avg_shortest_path_length)
                all_shortest_paths[sleep_id].append(avg_shortest_path_length)
            else:
                shortest_paths_for_patient.append(np.nan)  # Use NaN for missing data

        avg_shortest_paths_by_patient[patient_id] = shortest_paths_for_patient

    # Now, calculate the overall average for each sleep ID
    averages_per_sleep_id = {sleep_id: np.nanmean(all_shortest_paths[sleep_id]) for sleep_id in sleep_ids}

    # Replace NaN values with the average for their sleep ID
    for patient_id, shortest_paths in avg_shortest_paths_by_patient.items():
        avg_shortest_paths_by_patient[patient_id] = [averages_per_sleep_id[sleep_id] if np.isnan(path) else path
                                                     for sleep_id, path in zip(sleep_ids, shortest_paths)]

    last_correlation_matrix = adjacency_matrices[-1] if adjacency_matrices else None
    last_average_adjacency_matrix = average_adjacency_matrix if adjacency_matrices else None
    # Convert the dictionary to a numpy array to return
    return np.array(list(avg_shortest_paths_by_patient.values())), patient_ids, sleep_ids, last_correlation_matrix, last_average_adjacency_matrix

def batch_process_and_save_features(data_path, output_directory, batch_size=10):
    """
    Processes .fif files in batches, calculates features, and saves them to CSV files.
    """
    raw_files = sorted(glob(data_path))
    num_batches = (len(raw_files) + batch_size - 1) // batch_size
    print(f"Batch processing {len(raw_files)} files in {num_batches} batches.")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize the variables to avoid UnboundLocalError
    patient_ids, sleep_ids, avg_shortest_paths = None, None, None
    last_correlation_matrix, last_average_adjacency_matrix = None, None

    # Only proceed if there are files to process
    if len(raw_files) != 0:
        for i in range(num_batches):
            batch_files = raw_files[i * batch_size:(i + 1) * batch_size]
            epochs_list = [load_epochs_data(f) for f in batch_files]
            concatenated_epochs = mne.concatenate_epochs(epochs_list)

            patient_ids = np.unique(concatenated_epochs.metadata['ID'])
            sleep_ids = np.unique(concatenated_epochs.events[:, 2])
            avg_shortest_paths, patient_ids, sleep_ids, last_correlation_matrix, last_average_adjacency_matrix = compute_avg_shortest_paths(concatenated_epochs, patient_ids, sleep_ids)

            feature_df = pd.DataFrame(avg_shortest_paths, index=patient_ids, columns=[f'Sleep_ID_{int(sleep_id)}' for sleep_id in sleep_ids])
            feature_df.reset_index(inplace=True)
            feature_df.rename(columns={'index': 'Patient_ID'}, inplace=True)
            feature_df['ADHD'] = [concatenated_epochs.metadata[concatenated_epochs.metadata['ID'] == pid]['ADHD'].iloc[0] for pid in patient_ids]

            output_file = os.path.join(output_directory, f'features_batch_{i+1}.csv')
            feature_df.to_csv(output_file, index=False)
            print(f"Saved features to {output_file}")

            # Clean up to save memory
            del epochs_list, concatenated_epochs, feature_df

    # Return these values regardless of whether files were processed
    return patient_ids, sleep_ids, avg_shortest_paths, last_correlation_matrix, last_average_adjacency_matrix

patient_ids, sleep_ids, avg_shortest_paths, last_correlation_matrix, last_average_adjacency_matrix = batch_process_and_save_features(data_path, output_directory)

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

    X_combined, y_combined = shuffle(X_augmented, y_augmented, random_state=42)
    return X_combined, y_combined, features_df.iloc[:, 1:-1], labels  # also return original X, y for plotting

def perform_nested_cross_validation(X, y, param_grid, n_splits=5):
    outer_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    outer_accuracy_scores = []
    outer_precision_scores = []
    outer_recall_scores = []
    outer_f1_scores = []

    for train_index, test_index in outer_kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define the scoring metrics for cross-validation
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }

        # Initialize the RandomForestClassifier
        rf_classifier = RandomForestClassifier(random_state=42)

        # Set up GridSearchCV with the scoring dictionary
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=n_splits, scoring=scoring, refit='accuracy', n_jobs=-1)

        # Fit the model using GridSearchCV
        grid_search.fit(X_train, y_train)

        # Best model after grid search
        best_rf_model = grid_search.best_estimator_

        # Predict on the test set using the best model
        y_pred = best_rf_model.predict(X_test)

        # Calculate evaluation metrics
        outer_accuracy_scores.append(accuracy_score(y_test, y_pred))
        outer_precision_scores.append(precision_score(y_test, y_pred))
        outer_recall_scores.append(recall_score(y_test, y_pred))
        outer_f1_scores.append(f1_score(y_test, y_pred))

    # Calculate average metrics
    avg_accuracy = np.mean(outer_accuracy_scores)
    avg_precision = np.mean(outer_precision_scores)
    avg_recall = np.mean(outer_recall_scores)
    avg_f1 = np.mean(outer_f1_scores)

    return avg_accuracy, avg_precision, avg_recall, avg_f1, outer_accuracy_scores, outer_precision_scores, outer_recall_scores, outer_f1_scores

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
    avg_accuracy, avg_precision, avg_recall, avg_f1, accuracy_scores, precision_scores, recall_scores, f1_scores = perform_nested_cross_validation(X_combined, y_combined, param_grid)
    
    print(f"Average Accuracy: {avg_accuracy:.2f}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1 Score: {avg_f1:.2f}")

    # Plot cross-validation results
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    scores = [accuracy_scores, precision_scores, recall_scores, f1_scores]
    colors = ['blue', 'red', 'green', 'purple']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        axes[idx].bar(range(1, 6), scores[idx], color=colors[idx])
        axes[idx].set_title(f'{metric.capitalize()} per Fold')
        axes[idx].set_xlabel('Fold Number')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].set_xticks(range(1, 6))
        avg_score = np.mean(scores[idx])
        axes[idx].axhline(avg_score, color='k', linestyle='--', label="Average")
        axes[idx].legend(loc='lower right')

    plt.tight_layout()
    plt.show()

def count_epochs_by_group(data_path):
    raw_files = sorted(glob(data_path))
    adhd_epochs = 0
    control_epochs = 0

    for file_path in raw_files:
        epochs = load_epochs_data(file_path)
        print(f"Processing file: {file_path}")
        print(f"Metadata columns: {epochs.metadata.columns}")
        print(epochs.metadata.head())  # Print the first few rows of metadata to verify its contents
        
        if 'ADHD' in epochs.metadata.columns:
            adhd_count = np.sum(epochs.metadata['ADHD'] == 'Y')
            control_count = np.sum(epochs.metadata['ADHD'] == 'N')
            adhd_epochs += adhd_count
            control_epochs += control_count
            print(f"ADHD epochs in this file: {adhd_count}")
            print(f"Control epochs in this file: {control_count}")
        else:
            print(f"Metadata does not contain 'ADHD' column for file {file_path}")

    return adhd_epochs, control_epochs

# Counting the number of epochs for ADHD and control groups
adhd_epochs, control_epochs = count_epochs_by_group(data_path)
print(f"Number of ADHD epochs: {adhd_epochs}")
print(f"Number of control epochs: {control_epochs}")
