#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclear Reactor Data Processing Pipeline
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Warning: imblearn not installed. Will not be able to use SMOTE.")
import joblib
try:
    from tqdm import tqdm
except ImportError:
    # Define a simple replacement for tqdm if it's not available
    def tqdm(iterable, **kwargs):
        return iterable

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'sklearn', 'joblib'
    ]
    
    optional_packages = [
        'tensorflow', 'tensorflow-macos', 'xgboost', 'keras_tuner', 'imblearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install required dependencies using: pip install -r requirements.txt")
        return False
    
    missing_optional = []
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)
    
    if missing_optional:
        print(f"Warning: Optional packages not found: {', '.join(missing_optional)}")
        print("These are needed for model training but not for data processing.")
    
    print("All required dependencies are installed.")
    return True

def read_and_combine_data(folder_path):
    """
    Read all CSV files from the specified folder and combine them into a single DataFrame
    with memory optimization.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        
    Returns:
        DataFrame: Combined data from all CSV files
    """
    print("Reading and combining data from CSV files (memory-optimized)...")
    
    # Get list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Initialize empty list to store dataframes
    dfs = []
    
    # Define dtypes for columns to optimize memory usage
    # Assuming all feature columns can be stored as float32 instead of float64
    # First, read a sample file to get column names
    sample_file = os.path.join(folder_path, csv_files[0])
    sample_df = pd.read_csv(sample_file, nrows=5)
    
    # Create dtype dictionary - use float32 for numeric columns, save ~50% memory
    dtypes = {}
    for col in sample_df.columns:
        if col == 'Time':
            dtypes[col] = 'float32'
        elif sample_df[col].dtype == 'float64' or sample_df[col].dtype == 'int64':
            if 'label' in col.lower():  # Labels are small integers
                dtypes[col] = 'int8'
            else:
                dtypes[col] = 'float32'
    
    print("Using optimized dtypes to reduce memory usage")
    
    # Process files in chunks to reduce peak memory usage
    chunk_size = max(1, len(csv_files) // 3)  # Process in 3 chunks
    combined_df = pd.DataFrame()
    
    for chunk_start in range(0, len(csv_files), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(csv_files))
        print(f"Processing files {chunk_start+1} to {chunk_end} of {len(csv_files)}")
        
        chunk_dfs = []
        
        # Read each CSV file in this chunk and append to the list
        for file in tqdm(csv_files[chunk_start:chunk_end], desc="Reading files"):
            file_path = os.path.join(folder_path, file)
            
            # Extract scenario information from filename
            scenario_id = file.split('.')[0]  # Remove .csv extension
            
            # Read the CSV file with optimized dtypes
            df = pd.read_csv(file_path, dtype=dtypes)
            
            # Add scenario ID column
            df['scenario_id'] = scenario_id
            
            # Append to list
            chunk_dfs.append(df)
        
        # Combine dataframes for this chunk
        chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
        
        # Append to combined dataframe
        if combined_df.empty:
            combined_df = chunk_combined
        else:
            combined_df = pd.concat([combined_df, chunk_combined], ignore_index=True)
        
        # Clear memory
        del chunk_dfs, chunk_combined
    
    print(f"Successfully combined {len(csv_files)} CSV files.")
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    return combined_df

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, converting data types,
    and performing other necessary transformations with memory optimization.
    
    Args:
        df (DataFrame): Raw combined data
        
    Returns:
        DataFrame: Preprocessed data
    """
    print("Preprocessing data (memory-optimized)...")
    
    # Modify the dataframe in place instead of creating a copy
    # Convert 'Time' column to float32 (instead of float64) to save memory
    df['Time'] = df['Time'].astype('float32')
    
    # Identify feature and label columns
    feature_cols = list(df.columns[1:54])  # Columns 2-54 are features
    label_cols = list(df.columns[54:59])   # Columns 55-59 are labels
    
    # Verify column counts
    print(f"Number of feature columns: {len(feature_cols)}")
    print(f"Number of label columns: {len(label_cols)}")
    
    # Convert all feature columns to float32 instead of float64 to save memory
    for col in feature_cols:
        df[col] = df[col].astype('float32')
    
    # Handle missing values
    # Process missing values in feature columns
    for col in feature_cols:
        if df[col].isnull().any():
            print(f"Interpolating missing values in column: {col}")
            df[col] = df[col].interpolate(method='linear').fillna(
                method='ffill').fillna(method='bfill')
    
    # For labels, use mode (most frequent value) as they are categorical
    # Keep labels as integers to save memory
    for col in label_cols:
        if df[col].isnull().any():
            print(f"Filling missing values in label column: {col}")
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
        # Ensure labels are int8 (since they're small integers 0-5)
        df[col] = df[col].astype('int8')
    
    # Check for any remaining NaN values
    if df.isnull().any().any():
        print("Warning: DataFrame still contains NaN values after preprocessing")
        print(df.isnull().sum())
    else:
        print("No NaN values found after preprocessing.")
    
    return df

def engineer_features(df):
    """
    Engineer additional features to improve model performance with optimized memory usage.
    
    Args:
        df (DataFrame): Preprocessed data
        
    Returns:
        DataFrame: Data with engineered features
    """
    print("Engineering additional features (memory-optimized)...")
    
    # Identify feature columns (excluding Time, scenario_id, and label columns)
    feature_cols = list(df.columns[1:54])  # Columns 2-54 are features
    label_cols = list(df.columns[54:59])
    
    # Convert data to float32 instead of float64 to reduce memory usage
    for col in feature_cols + ['Time']:
        df[col] = df[col].astype('float32')
    
    # Reduce the number of engineered features to save memory
    # Only use the most important feature types and window sizes
    
    # Process in chunks by scenario_id to reduce memory footprint
    unique_scenarios = df['scenario_id'].unique()
    chunk_size = max(1, len(unique_scenarios) // 4)  # Process scenarios in 4 chunks
    
    # Initialize dataframe to store results
    result_df = pd.DataFrame()
    
    # Process scenarios in chunks
    for chunk_idx in range(0, len(unique_scenarios), chunk_size):
        print(f"Processing chunk {chunk_idx//chunk_size + 1} of {(len(unique_scenarios)-1)//chunk_size + 1}")
        
        # Get scenarios for this chunk
        chunk_scenarios = unique_scenarios[chunk_idx:chunk_idx+chunk_size]
        chunk_df = df[df['scenario_id'].isin(chunk_scenarios)].copy()
        
        # Group by scenario_id to process each scenario separately
        scenario_groups = chunk_df.groupby('scenario_id')
        
        # Initialize list to store processed dataframes for this chunk
        processed_dfs = []
        
        # Process each scenario in this chunk
        for scenario_id, scenario_df in tqdm(scenario_groups, desc="Processing scenarios"):
            # Sort by Time within each scenario
            scenario_df = scenario_df.sort_values('Time')
            
            # 1. Reduced set of rolling statistics - only use window size 10 and only mean/std
            window_size = 10
            # Create all the derived features at once to avoid fragmentation
            # 1. Rolling statistics
            rolling_mean = scenario_df[feature_cols].rolling(window=window_size, min_periods=1).mean()
            rolling_mean.columns = [f'{col}_rolling_mean_{window_size}' for col in rolling_mean.columns]
            
            # Rolling standard deviation
            rolling_std = scenario_df[feature_cols].rolling(window=window_size, min_periods=1).std().fillna(0)
            rolling_std.columns = [f'{col}_rolling_std_{window_size}' for col in rolling_std.columns]
            
            # 2. Exponential moving average
            span = 10
            ema = scenario_df[feature_cols].ewm(span=span, min_periods=1).mean()
            ema.columns = [f'{col}_ema_{span}' for col in ema.columns]
            
            # 3. Trend indicators (first-order differences)
            diff_df = scenario_df[feature_cols].diff().fillna(0)
            diff_df.columns = [f'{col}_diff' for col in diff_df.columns]
            
            # 4. Time-based feature
            time_since_start = pd.DataFrame({
                'time_since_start': scenario_df['Time'] - scenario_df['Time'].min()
            }, index=scenario_df.index)
            
            # Combine all new features in one concat operation instead of individual assignments
            # This avoids the DataFrame fragmentation warning
            scenario_df = pd.concat(
                [scenario_df, rolling_mean, rolling_std, ema, diff_df, time_since_start], 
                axis=1
            )
            
            # 5. Simplified time-to-failure indicators (only for first component to save memory)
            label_col = label_cols[0]  # Only use the first label column
            component = label_col.replace('_label', '')
            
            # Create time-to-failure feature
            max_label = scenario_df[label_col].max()
            if max_label > 0:
                time_range = scenario_df['Time'].max() - scenario_df['Time'].min()
                if time_range > 0:
                    max_deg_time = scenario_df.loc[scenario_df[label_col] == max_label, 'Time'].max()
                    if not pd.isna(max_deg_time):
                        # Create as separate DataFrame then concat to avoid fragmentation
                        time_to_failure = pd.DataFrame({
                            f'{component}_time_to_failure': 1 - ((max_deg_time - scenario_df['Time']) / time_range).clip(0, 1)
                        }, index=scenario_df.index)
                        
                        # Add to scenario_df
                        scenario_df = pd.concat([scenario_df, time_to_failure], axis=1)
            
            # Convert all new columns to float32
            for col in scenario_df.columns:
                if col not in df.columns and scenario_df[col].dtype == 'float64':
                    scenario_df[col] = scenario_df[col].astype('float32')
            
            # Add to list of processed dataframes
            processed_dfs.append(scenario_df)
            
            # Clear memory
            del rolling_mean, rolling_std, ema, diff_df
        
        # Combine all processed dataframes for this chunk
        chunk_result = pd.concat(processed_dfs, ignore_index=True)
        
        # Append to the result dataframe
        result_df = pd.concat([result_df, chunk_result], ignore_index=True)
        
        # Clear memory
        del processed_dfs, chunk_df, chunk_result
    
    # Report the number of engineered features
    original_feature_count = len(feature_cols)
    new_feature_count = result_df.shape[1] - df.shape[1]
    print(f"Added {new_feature_count} new engineered features.")
    print(f"Original feature count: {original_feature_count}")
    print(f"Total feature count after engineering: {original_feature_count + new_feature_count}")
    
    return result_df

def balance_classes(X, y):
    """
    Balance classes using improved approach for multi-label data:
    1. Apply SMOTE for each label separately 
    2. Combine the results to ensure all classes are balanced
    
    Args:
        X (DataFrame): Feature matrix
        y (DataFrame): Target labels
        
    Returns:
        tuple: Balanced X and y
    """
    print("Balancing classes using enhanced multi-label SMOTE approach...")
    
    # Check for NaN values in X
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in features. Filling with mean...")
        # Fill NaN values with column means
        X = X.fillna(X.mean())
    
    # Check for NaN values in y
    if y.isna().any().any():
        print("Warning: Found NaN values in labels. Filling with mode...")
        # Fill NaN values with column modes
        for col in y.columns:
            y[col] = y[col].fillna(y[col].mode()[0])
    
    try:
        # Two-step approach:
        # 1. For problematic labels like LPT, apply separate balancing
        # 2. For the rest, apply multi-label SMOTE with combined labels
        
        # Copy the original data
        X_balanced = X.copy()
        y_balanced = y.copy()
        
        # Apply separate SMOTE for each label independently
        print("\nApplying separate SMOTE for each label:")
        for label_col in y.columns:
            # Get initial class distribution
            initial_dist = y_balanced[label_col].value_counts().sort_index()
            min_class_count = initial_dist.min()
            
            # Check if all classes have at least a minimum number of samples
            min_required = 100  # Minimum number of samples per class to ensure good balance
            if min_class_count < min_required:
                print(f"\nBalancing {label_col} (min class count: {min_class_count})...")
                
                try:
                    # Apply SMOTE to just this label
                    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count-1))
                    X_temp, y_temp = smote.fit_resample(X_balanced, y_balanced[label_col])
                    
                    # Create a new dataframe with balanced data for this label
                    X_balanced = pd.DataFrame(X_temp, columns=X_balanced.columns)
                    
                    # Update the balanced target dataframe with this column
                    y_balanced = pd.DataFrame(y_balanced, columns=y.columns)
                    y_balanced[label_col] = y_temp
                    
                    # Print new distribution
                    new_dist = y_balanced[label_col].value_counts().sort_index()
                    print(f"  Initial distribution: {initial_dist.to_dict()}")
                    print(f"  Balanced distribution: {new_dist.to_dict()}")
                    
                except Exception as e:
                    print(f"  Error balancing {label_col}: {str(e)}")
                    print(f"  Proceeding with original distribution for {label_col}")
        
        # Apply multi-label SMOTE to ensure all combinations are represented
        print("\nApplying multi-label SMOTE to balance label combinations...")
        
        try:
            # Convert all labels to integers and create a combined stratification key
            y_int = y_balanced.astype(int)
            y_combined = y_int.apply(lambda row: ','.join(row.astype(str)), axis=1)
            
            # Get sample counts per combined class
            print("Top 10 class combinations before final balancing:")
            print(y_combined.value_counts().head(10))
            
            # Apply SMOTE to the combined representation
            try:
                # Check if we have enough samples in each class combination
                if y_combined.value_counts().min() >= 5:
                    smote = SMOTE(random_state=42)
                    X_resampled, y_combined_resampled = smote.fit_resample(X_balanced, y_combined)
                    
                    # Convert back to multi-output format
                    y_resampled = pd.DataFrame([list(map(int, comb.split(','))) for comb in y_combined_resampled],
                                            columns=y.columns)
                    
                    print("\nFinal SMOTE successfully applied to balance label combinations")
                    X_balanced, y_balanced = X_resampled, y_resampled
                else:
                    # If some combinations are too rare, skip the final multi-label SMOTE
                    print("\nSkipping final multi-label SMOTE due to rare class combinations")
                    print(f"Minimum combination count: {y_combined.value_counts().min()}")
            except Exception as e:
                print(f"Error in final multi-label SMOTE: {str(e)}")
                print("Using the results from individual label balancing instead")
        except Exception as e:
            print(f"Error preparing for multi-label SMOTE: {str(e)}")
        
        print(f"\nShape before balancing: X {X.shape}, y {y.shape}")
        print(f"Shape after balancing: X {X_balanced.shape}, y {y_balanced.shape}")
        
        # Display final class distribution
        print("\nLabel distribution before balancing:")
        for col in y.columns:
            print(f"{col}:", y[col].value_counts().sort_index().to_dict())
        
        print("\nLabel distribution after balancing:")
        for col in y_balanced.columns:
            print(f"{col}:", y_balanced[col].value_counts().sort_index().to_dict())
        
        return X_balanced, y_balanced
    
    except Exception as e:
        print(f"Error in balancing process: {str(e)}")
        print("Proceeding with original imbalanced data.")
        return X, y

def create_sequences(X, y, seq_length=10, is_test_or_val=False):
    """
    Create sequences for time-series modeling with optimized memory usage
    and improved diversity for test and validation sets.
    
    Args:
        X (DataFrame): Feature matrix
        y (DataFrame): Target labels
        seq_length (int): Length of sequences
        is_test_or_val (bool): Whether this is for test/validation data, affects sampling strategy
        
    Returns:
        tuple: Sequence X and y
    """
    print(f"Creating sequences with length {seq_length} (memory-optimized)...")
    
    # Adjust step size based on whether this is for training or test/validation
    if is_test_or_val:
        # Use a smaller step size for test/validation to ensure more diversity
        step_size = 5  # Smaller step size for test/validation creates more diverse sequences
        max_sequences_per_scenario = 20  # Limit sequences per scenario for test/val
    else:
        # For training, can use larger step size to reduce computational load
        step_size = 15
        max_sequences_per_scenario = None  # No limit per scenario for training
    
    max_sequences = 5000  # Overall maximum sequence count
    
    # Check if scenario_id is in X, if not we'll create simple sliding window sequences
    if 'scenario_id' not in X.columns:
        print("Warning: 'scenario_id' column not found. Creating simple sliding window sequences.")
        
        # Convert to numpy arrays for efficiency
        X_array = X.values
        y_array = y.values
        
        # Pre-allocate arrays with estimated size to avoid append operations
        n_samples = X_array.shape[0]
        estimated_seq_count = min(max_sequences, (n_samples - seq_length + 1) // step_size)
        
        # Use numpy pre-allocation for better memory efficiency
        n_features = X_array.shape[1]
        n_targets = y_array.shape[1]
        
        X_seq = np.zeros((estimated_seq_count, seq_length, n_features), dtype=np.float32)
        y_seq = np.zeros((estimated_seq_count, n_targets), dtype=np.float32)
        
        # Create sliding window sequences
        seq_count = 0
        for i in range(0, n_samples - seq_length + 1, step_size):
            if seq_count >= estimated_seq_count:
                break
                
            X_seq[seq_count] = X_array[i:i+seq_length]
            y_seq[seq_count] = y_array[i+seq_length-1]  # Use the label of the last timestep
            seq_count += 1
            
            # Limit the number of sequences to avoid memory issues
            if seq_count >= max_sequences:
                print(f"Warning: Limiting to {seq_count} sequences to avoid memory issues.")
                break
                
        # Trim arrays to actual size used
        X_seq = X_seq[:seq_count]
        y_seq = y_seq[:seq_count]
    else:
        # Group by scenario_id more efficiently
        scenario_ids = X['scenario_id'].values
        unique_scenarios = np.unique(scenario_ids)
        
        # Convert to numpy arrays for efficiency
        X_drop = X.drop(columns=['scenario_id'])  # Remove scenario_id column before conversion
        X_array = X_drop.values.astype(np.float32)
        y_array = y.values.astype(np.float32)
        
        # Get label columns to track diversity
        # We'll use this to ensure we have a diverse set of sequences from different classes
        label_cols = y.columns.tolist()
        
        # Temporary storage for sequences with different labels
        sequences = []
        
        # Track label distribution for diversity (only for test/validation)
        if is_test_or_val:
            label_distributions = {}
            for i, label in enumerate(label_cols):
                label_distributions[label] = {}
        
        # Process each scenario
        for scenario_id in tqdm(unique_scenarios, desc="Creating sequences"):
            # Get indices for this scenario
            scenario_mask = scenario_ids == scenario_id
            indices = np.where(scenario_mask)[0]
            indices = sorted(indices)  # Ensure time order
            
            # Skip scenarios with fewer points than sequence length
            if len(indices) < seq_length:
                continue
            
            # Get labels for this scenario to track diversity
            scenario_y = y_array[indices]
            
            # Track sequences created for this scenario
            scenario_sequences = 0
            
            # Sample using different strategies based on whether this is test/validation
            if is_test_or_val:
                # For test/validation, we want to ensure diverse labels
                # Try to extract sequences with different ending labels
                
                # Calculate possible start indices
                max_start_idx = len(indices) - seq_length + 1
                
                # Choose several starting positions throughout the scenario instead of fixed steps
                # This helps capture more diverse labels
                if max_start_idx > 1:
                    # Create multiple start points to ensure diversity
                    num_samples = min(max_start_idx, 10)  # Sample up to 10 starting points
                    start_indices = np.linspace(0, max_start_idx-1, num_samples).astype(int)
                else:
                    start_indices = [0]
                
                for start_idx in start_indices:
                    if start_idx >= max_start_idx:  # Safety check
                        continue
                        
                    end_idx = start_idx + seq_length - 1
                    seq_indices = indices[start_idx:start_idx+seq_length]
                    
                    # Get labels at the end of this sequence
                    end_labels = scenario_y[end_idx]
                    
                    # Create the sequence
                    X_seq_i = X_array[seq_indices]
                    y_seq_i = end_labels  # Use end labels
                    
                    # Track distribution of labels
                    for i, label in enumerate(label_cols):
                        label_val = int(end_labels[i])
                        if label_val not in label_distributions[label]:
                            label_distributions[label][label_val] = 0
                        label_distributions[label][label_val] += 1
                    
                    # Store sequence
                    sequences.append((X_seq_i, y_seq_i))
                    scenario_sequences += 1
                    
                    # Limit sequences per scenario for better balance in test/validation
                    if max_sequences_per_scenario and scenario_sequences >= max_sequences_per_scenario:
                        break
            else:
                # For training, use traditional step-based sequencing
                # Create sequences with step size
                for i in range(0, len(indices) - seq_length + 1, step_size):
                    seq_indices = indices[i:i+seq_length]
                    
                    # Create the sequence
                    X_seq_i = X_array[seq_indices]
                    y_seq_i = y_array[seq_indices[-1]]  # Use the label of the last timestep
                    
                    # Store sequence
                    sequences.append((X_seq_i, y_seq_i))
                    scenario_sequences += 1
                    
                    # Check if we've reached the maximum number of sequences for the dataset
                    if len(sequences) >= max_sequences:
                        break
            
            # Break outer loop if max sequences reached
            if len(sequences) >= max_sequences:
                print(f"Warning: Limiting to {len(sequences)} sequences to avoid memory issues.")
                break
        
        # Convert sequences to arrays
        if sequences:
            # Get shapes from the first sequence
            first_X, first_y = sequences[0]
            n_features = first_X.shape[1]
            n_targets = first_y.shape[0]
            
            # Pre-allocate arrays
            n_sequences = len(sequences)
            X_seq = np.zeros((n_sequences, seq_length, n_features), dtype=np.float32)
            y_seq = np.zeros((n_sequences, n_targets), dtype=np.float32)
            
            # Fill arrays
            for i, (X_seq_i, y_seq_i) in enumerate(sequences):
                X_seq[i] = X_seq_i
                y_seq[i] = y_seq_i
        else:
            print("Warning: No sequences created. Using fallback approach.")
            # Fallback approach - create at least one sequence
            X_seq = np.zeros((1, seq_length, X_array.shape[1]), dtype=np.float32)
            y_seq = np.zeros((1, y_array.shape[1]), dtype=np.float32)
        
        # Print label distribution for test/validation
        if is_test_or_val:
            print("\nLabel distribution in sequences:")
            for label, dist in label_distributions.items():
                print(f"  {label}: {dist}")
    
    print(f"Created {len(X_seq)} sequences.")
    print(f"Sequence shape: X {X_seq.shape}, y {y_seq.shape}")
    
    # For test/val data, perform an additional check to ensure class diversity
    if is_test_or_val:
        print("\nVerifying label diversity in sequences:")
        for i, label in enumerate(label_cols if 'label_cols' in locals() else range(y_seq.shape[1])):
            unique_vals, counts = np.unique(y_seq[:, i], return_counts=True)
            if len(unique_vals) == 1:
                print(f"  WARNING: {label} has only one class in sequences: {unique_vals[0]}")
            else:
                print(f"  {label}: {dict(zip(unique_vals, counts))}")
    
    return X_seq, y_seq

def visualize_data_distribution(df):
    """
    Visualize the distribution of labels in the dataset.
    
    Args:
        df (DataFrame): Processed data
    """
    print("Visualizing label distribution...")
    
    # Identify label columns
    label_cols = list(df.columns[54:59])
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(label_cols), 1, figsize=(12, 15))
    
    # Plot the distribution of each label
    for i, col in enumerate(label_cols):
        sns.countplot(x=col, data=df, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('Degradation Level')
        axes[i].set_ylabel('Count')
        
        # Add count labels on top of bars
        for p in axes[i].patches:
            axes[i].annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 10), 
                            textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.close()
    
    print("Label distribution visualization saved as 'label_distribution.png'")

def stratified_multi_label_scenario_split(df, label_cols, test_size=0.3, val_size=0.5, random_state=42):
    """
    Perform a manual stratified split of scenarios ensuring each label class is proportionally represented
    in train, validation, and test sets.
    
    Args:
        df (DataFrame): The dataframe containing the data
        label_cols (list): List of label column names
        test_size (float): Proportion of data for test+validation (default: 0.3)
        val_size (float): Proportion of test_size to use for validation (default: 0.5)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Lists of scenario IDs for train, validation, and test sets
    """
    print("Performing manual stratified multi-label split...")
    
    # Group by scenario_id and calculate the most common class for each label
    scenario_groups = df.groupby('scenario_id')
    scenario_labels = {}
    
    for scenario_id, group in tqdm(scenario_groups, desc="Analyzing scenarios"):
        # For each label column, find the most frequent class in this scenario
        scenario_label_values = {}
        for label_col in label_cols:
            # Get the mode (most common class)
            most_common_class = group[label_col].mode()[0]
            scenario_label_values[label_col] = most_common_class
        
        scenario_labels[scenario_id] = scenario_label_values
    
    # Convert the dictionary to a DataFrame for easier manipulation
    scenario_label_df = pd.DataFrame.from_dict(scenario_labels, orient='index')
    
    # Create scenarios grouped by the most problematic label (LPT_label)
    # This will ensure that all classes of LPT_label are represented in each split
    target_label = 'LPT_label'
    print(f"Stratifying primarily based on {target_label}")
    
    # Group scenarios by their most common LPT_label value
    lpt_class_scenarios = {}
    for i in range(6):  # 0-5 degradation levels
        lpt_class_scenarios[i] = scenario_label_df[scenario_label_df[target_label] == i].index.tolist()
    
    # Set the random seed for reproducibility
    np.random.seed(random_state)
    
    # Calculate the number of scenarios to use for each split
    train_size = 1 - test_size
    val_size_of_test = val_size
    test_size_of_test = 1 - val_size
    
    # Initialize the scenario lists for each split
    train_scenarios = []
    val_scenarios = []
    test_scenarios = []
    
    print("\nDistributing scenarios by class:")
    # For each class of the target label (LPT_label)
    for class_value, scenarios in lpt_class_scenarios.items():
        # Shuffle the scenarios for this class
        np.random.shuffle(scenarios)
        n_scenarios = len(scenarios)
        
        if n_scenarios == 0:
            print(f"  Class {class_value}: No scenarios found")
            continue
        
        # Calculate split sizes for this class
        n_train = max(int(n_scenarios * train_size), 1)  # Ensure at least 1 scenario in train
        n_val = max(int((n_scenarios - n_train) * val_size_of_test), 1) if n_scenarios - n_train > 1 else 0
        n_test = n_scenarios - n_train - n_val
        
        # Adjust if we need at least 1 in each split
        if n_test == 0 and n_val > 1:
            n_val -= 1
            n_test = 1
        
        # Split into train, validation, and test
        class_train = scenarios[:n_train]
        class_val = scenarios[n_train:n_train+n_val]
        class_test = scenarios[n_train+n_val:]
        
        # Add to the overall lists
        train_scenarios.extend(class_train)
        val_scenarios.extend(class_val)
        test_scenarios.extend(class_test)
        
        print(f"  Class {class_value}: {n_scenarios} scenarios → Train: {len(class_train)}, Val: {len(class_val)}, Test: {len(class_test)}")
    
    # Secondary balancing by other labels
    # We'll look at the distribution of the other labels and make small adjustments
    # to ensure a better balance across all labels
    
    # For each secondary label column (except the primary target_label)
    secondary_labels = [col for col in label_cols if col != target_label]
    
    for label_col in secondary_labels:
        print(f"\nChecking {label_col} distribution in splits...")
        
        # Check the distribution of classes in each split
        train_dist = {}
        val_dist = {}
        test_dist = {}
        
        # Calculate current distributions
        for class_val in range(6):  # 0-5 degradation levels
            train_count = sum(1 for s in train_scenarios if scenario_label_df.loc[s, label_col] == class_val)
            val_count = sum(1 for s in val_scenarios if scenario_label_df.loc[s, label_col] == class_val)
            test_count = sum(1 for s in test_scenarios if scenario_label_df.loc[s, label_col] == class_val)
            
            train_dist[class_val] = train_count
            val_dist[class_val] = val_count
            test_dist[class_val] = test_count
            
            total = train_count + val_count + test_count
            if total > 0:
                print(f"  Class {class_val}: Total {total} → Train: {train_count}, Val: {val_count}, Test: {test_count}")
            
            # Check for any missing classes in splits
            if train_count == 0 or val_count == 0 or test_count == 0:
                if total > 0:
                    print(f"  Warning: Class {class_val} is missing from one or more splits")
                    
                    # Try to balance by moving scenarios between splits
                    moved = False
                    
                    # If no scenarios in train, try to move some from validation or test
                    if train_count == 0:
                        if val_count > 1:
                            # Find a scenario in val with this class value
                            for s in val_scenarios[:]:  # Copy to avoid modification during iteration
                                if scenario_label_df.loc[s, label_col] == class_val:
                                    val_scenarios.remove(s)
                                    train_scenarios.append(s)
                                    print(f"    Moved scenario {s} from validation to train")
                                    moved = True
                                    break
                        elif test_count > 1:
                            # Find a scenario in test with this class value
                            for s in test_scenarios[:]:  # Copy to avoid modification during iteration
                                if scenario_label_df.loc[s, label_col] == class_val:
                                    test_scenarios.remove(s)
                                    train_scenarios.append(s)
                                    print(f"    Moved scenario {s} from test to train")
                                    moved = True
                                    break
                    
                    # If no scenarios in validation, try to move some from train or test
                    if val_count == 0 and not moved:
                        if train_count > 1:
                            # Find a scenario in train with this class value
                            for s in train_scenarios[:]:  # Copy to avoid modification during iteration
                                if scenario_label_df.loc[s, label_col] == class_val:
                                    train_scenarios.remove(s)
                                    val_scenarios.append(s)
                                    print(f"    Moved scenario {s} from train to validation")
                                    moved = True
                                    break
                        elif test_count > 1:
                            # Find a scenario in test with this class value
                            for s in test_scenarios[:]:  # Copy to avoid modification during iteration
                                if scenario_label_df.loc[s, label_col] == class_val:
                                    test_scenarios.remove(s)
                                    val_scenarios.append(s)
                                    print(f"    Moved scenario {s} from test to validation")
                                    moved = True
                                    break
                    
                    # If no scenarios in test, try to move some from train or validation
                    if test_count == 0 and not moved:
                        if train_count > 1:
                            # Find a scenario in train with this class value
                            for s in train_scenarios[:]:  # Copy to avoid modification during iteration
                                if scenario_label_df.loc[s, label_col] == class_val:
                                    train_scenarios.remove(s)
                                    test_scenarios.append(s)
                                    print(f"    Moved scenario {s} from train to test")
                                    moved = True
                                    break
                        elif val_count > 1:
                            # Find a scenario in val with this class value
                            for s in val_scenarios[:]:  # Copy to avoid modification during iteration
                                if scenario_label_df.loc[s, label_col] == class_val:
                                    val_scenarios.remove(s)
                                    test_scenarios.append(s)
                                    print(f"    Moved scenario {s} from validation to test")
                                    moved = True
                                    break
    
    # Final split sizes
    print("\nFinal split sizes:")
    print(f"  Train: {len(train_scenarios)} scenarios")
    print(f"  Validation: {len(val_scenarios)} scenarios")
    print(f"  Test: {len(test_scenarios)} scenarios")
    print(f"  Total: {len(train_scenarios) + len(val_scenarios) + len(test_scenarios)} scenarios")
    
    # Verify label distributions in each split
    print("\nVerifying final label distribution in train/val/test splits:")
    splits = {
        'Train': train_scenarios,
        'Validation': val_scenarios,
        'Test': test_scenarios
    }
    
    for split_name, scenarios in splits.items():
        split_data = df[df['scenario_id'].isin(scenarios)]
        print(f"\n{split_name} set ({len(scenarios)} scenarios, {len(split_data)} samples):")
        
        for label_col in label_cols:
            class_counts = split_data[label_col].value_counts().sort_index().to_dict()
            print(f"  {label_col}: {class_counts}")
    
    return train_scenarios, val_scenarios, test_scenarios

def verify_label_distribution(train_df, val_df, test_df, label_cols):
    """
    Verify that each class for each label appears in each data split.
    
    Args:
        train_df (DataFrame): Training data
        val_df (DataFrame): Validation data
        test_df (DataFrame): Test data
        label_cols (list): List of label column names
        
    Returns:
        bool: True if distribution is valid, False otherwise
    """
    print("\nVerifying class representation in all splits...")
    valid_distribution = True
    
    for label_col in label_cols:
        train_classes = set(train_df[label_col].unique())
        val_classes = set(val_df[label_col].unique())
        test_classes = set(test_df[label_col].unique())
        
        # Identify all unique classes across all splits
        all_classes = train_classes.union(val_classes).union(test_classes)
        
        # Check if any split is missing a class
        missing_in_train = all_classes - train_classes
        missing_in_val = all_classes - val_classes
        missing_in_test = all_classes - test_classes
        
        if missing_in_train or missing_in_val or missing_in_test:
            valid_distribution = False
            print(f"ISSUE: {label_col} has incomplete class representation:")
            if missing_in_train:
                print(f"  - Classes missing in training set: {missing_in_train}")
            if missing_in_val:
                print(f"  - Classes missing in validation set: {missing_in_val}")
            if missing_in_test:
                print(f"  - Classes missing in test set: {missing_in_test}")
        else:
            print(f"✓ {label_col}: All {len(all_classes)} classes represented in all splits")
            
            # Show class distribution in each split
            train_counts = train_df[label_col].value_counts().sort_index().to_dict()
            val_counts = val_df[label_col].value_counts().sort_index().to_dict()
            test_counts = test_df[label_col].value_counts().sort_index().to_dict()
            
            print(f"  - Train: {train_counts}")
            print(f"  - Val: {val_counts}")
            print(f"  - Test: {test_counts}")
    
    return valid_distribution

def main():
    """
    Main function to execute the data processing pipeline.
    """
    # Check dependencies
    if not check_dependencies():
        return
    
    # Set path to the folder containing CSV files
    data_folder = "TrainingData"
    
    # 1. Read and combine data from all CSV files
    combined_df = read_and_combine_data(data_folder)
    
    # 2. Preprocess data
    processed_df = preprocess_data(combined_df)
    
    # 3. Engineer additional features
    enhanced_df = engineer_features(processed_df)
    
    # 4. Visualize data distribution
    visualize_data_distribution(enhanced_df)
    
    # 5. Prepare data for modeling
    # Separate features and labels
    label_cols = list(enhanced_df.columns[54:59])
    feature_cols = [col for col in enhanced_df.columns if col not in label_cols and col != 'scenario_id']
    
    # Store scenario_id separately for use in sequence creation
    X = enhanced_df[feature_cols]
    scenario_ids_col = enhanced_df['scenario_id'] if 'scenario_id' in enhanced_df.columns else None
    y = enhanced_df[label_cols]
    
    # 6. Split data with stratification to ensure proper label distribution
    if scenario_ids_col is not None:
        # Use the new stratified multi-label split function
        train_scenarios, val_scenarios, test_scenarios = stratified_multi_label_scenario_split(
            enhanced_df, label_cols, test_size=0.3, val_size=0.5, random_state=42
        )
        
        # Filter the data based on scenario IDs
        train_df = enhanced_df[enhanced_df['scenario_id'].isin(train_scenarios)]
        val_df = enhanced_df[enhanced_df['scenario_id'].isin(val_scenarios)]
        test_df = enhanced_df[enhanced_df['scenario_id'].isin(test_scenarios)]
        
        print(f"Training set size: {train_df.shape}")
        print(f"Validation set size: {val_df.shape}")
        print(f"Test set size: {test_df.shape}")
        
        # Verify that each class for each label appears in each split
        is_valid = verify_label_distribution(train_df, val_df, test_df, label_cols)
        if not is_valid:
            print("WARNING: Some classes are missing from one or more splits!")
            print("Consider adjusting the stratification approach or using a different random seed.")
        
        # 7. Prepare features and labels for each set
        X_train = train_df[feature_cols]
        y_train = train_df[label_cols]
        X_train_with_scenario = train_df[feature_cols + ['scenario_id']]
        
        X_val = val_df[feature_cols]
        y_val = val_df[label_cols]
        X_val_with_scenario = val_df[feature_cols + ['scenario_id']]
        
        X_test = test_df[feature_cols]
        y_test = test_df[label_cols]
        X_test_with_scenario = test_df[feature_cols + ['scenario_id']]
    else:
        # If no scenario_id, use stratification on combined labels
        # Create a combined stratification column for multi-label classification
        y_combined = y.apply(lambda row: '_'.join([f"{col}_{int(row[col])}" for col in y.columns]), axis=1)
        
        # Split with stratification
        X_train, X_temp, y_train, y_temp, y_train_combined, y_temp_combined = train_test_split(
            X, y, y_combined, test_size=0.3, random_state=42, stratify=y_combined
        )
        
        # Further split temp into validation and test
        X_val, X_test, y_val, y_test, y_val_combined, y_test_combined = train_test_split(
            X_temp, y_temp, y_temp_combined, test_size=0.5, random_state=42, stratify=y_temp_combined
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Validation set size: {X_val.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Create dummy dataframes for verification
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Verify that each class for each label appears in each split
        is_valid = verify_label_distribution(train_df, val_df, test_df, label_cols)
        if not is_valid:
            print("WARNING: Some classes are missing from one or more splits!")
            print("Consider adjusting the stratification approach or using a different random seed.")
        
        X_train_with_scenario = X_train.copy()
        X_val_with_scenario = X_val.copy()
        X_test_with_scenario = X_test.copy()
    
    # 8. Apply SMOTE balancing to address class imbalance
    try:
        X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)
        print("Successfully applied SMOTE to balance training data")
    except Exception as e:
        print(f"Failed to apply SMOTE: {str(e)}")
        print("Proceeding with original imbalanced data")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # 9. Create sequences for LSTM model
    # For training data - now using the balanced data for training sequences
    if scenario_ids_col is not None:
        # For balanced data, we need to add back the scenario_id (any value is fine since we're just using it for sequencing)
        X_train_balanced_with_scenario = X_train_balanced.copy()
        if isinstance(X_train_balanced, pd.DataFrame) and 'scenario_id' not in X_train_balanced.columns:
            # Add a dummy scenario ID (since SMOTE doesn't preserve the original scenario grouping)
            X_train_balanced_with_scenario['scenario_id'] = 'balanced_scenario'
        
        X_train_seq, y_train_seq = create_sequences(
            X_train_balanced_with_scenario.reset_index(drop=True), 
            y_train_balanced.reset_index(drop=True), 
            seq_length=20,
            is_test_or_val=False  # Training data
        )
    else:
        X_train_seq, y_train_seq = create_sequences(
            X_train_balanced.reset_index(drop=True), 
            y_train_balanced.reset_index(drop=True), 
            seq_length=20,
            is_test_or_val=False  # Training data
        )
    
    # For validation data
    if scenario_ids_col is not None:
        X_val_seq, y_val_seq = create_sequences(
            X_val_with_scenario.reset_index(drop=True), 
            y_val.reset_index(drop=True), 
            seq_length=20,
            is_test_or_val=True  # Validation data - need diverse classes
        )
    else:
        X_val_seq, y_val_seq = create_sequences(
            X_val.reset_index(drop=True), 
            y_val.reset_index(drop=True), 
            seq_length=20,
            is_test_or_val=True  # Validation data - need diverse classes
        )
    
    # For test data
    if scenario_ids_col is not None:
        X_test_seq, y_test_seq = create_sequences(
            X_test_with_scenario.reset_index(drop=True), 
            y_test.reset_index(drop=True), 
            seq_length=20,
            is_test_or_val=True  # Test data - need diverse classes
        )
    else:
        X_test_seq, y_test_seq = create_sequences(
            X_test.reset_index(drop=True), 
            y_test.reset_index(drop=True), 
            seq_length=20,
            is_test_or_val=True  # Test data - need diverse classes
        )
    
    # Verify sequence label distributions
    print("\nVerifying sequence label distributions:")
    for label_idx, label_col in enumerate(label_cols):
        print(f"\n{label_col} distribution in sequences:")
        train_labels = y_train_seq[:, label_idx]
        val_labels = y_val_seq[:, label_idx]
        test_labels = y_test_seq[:, label_idx]
        
        unique_train = np.unique(train_labels, return_counts=True)
        unique_val = np.unique(val_labels, return_counts=True)
        unique_test = np.unique(test_labels, return_counts=True)
        
        print(f"  - Train: {dict(zip(unique_train[0].astype(int), unique_train[1]))}")
        print(f"  - Val: {dict(zip(unique_val[0].astype(int), unique_val[1]))}")
        print(f"  - Test: {dict(zip(unique_test[0].astype(int), unique_test[1]))}")
    
    # 10. Save processed data
    output_folder = "processed_data"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save tabular data for XGBoost
    joblib.dump((X_train_balanced, y_train_balanced), os.path.join(output_folder, 'tabular_train.pkl'))
    joblib.dump((X_val, y_val), os.path.join(output_folder, 'tabular_val.pkl'))
    joblib.dump((X_test, y_test), os.path.join(output_folder, 'tabular_test.pkl'))
    
    # Save sequence data for LSTM
    joblib.dump((X_train_seq, y_train_seq), os.path.join(output_folder, 'sequence_train.pkl'))
    joblib.dump((X_val_seq, y_val_seq), os.path.join(output_folder, 'sequence_val.pkl'))
    joblib.dump((X_test_seq, y_test_seq), os.path.join(output_folder, 'sequence_test.pkl'))
    
    # Save feature and label column names for reference
    joblib.dump(feature_cols, os.path.join(output_folder, 'feature_cols.pkl'))
    joblib.dump(label_cols, os.path.join(output_folder, 'label_cols.pkl'))
    
    # Save the full processed dataset
    enhanced_df.to_pickle(os.path.join(output_folder, 'full_processed_dataset.pkl'))
    
    # Create label distribution visualization for the final split
    fig, axes = plt.subplots(len(label_cols), 3, figsize=(15, 20))
    splits = {'Train': y_train_balanced, 'Validation': y_val, 'Test': y_test}
    
    for i, label_col in enumerate(label_cols):
        for j, (split_name, split_data) in enumerate(splits.items()):
            if isinstance(split_data, pd.DataFrame):
                # For pandas DataFrame (tabular data)
                counts = split_data[label_col].value_counts().sort_index()
                sns.barplot(x=counts.index, y=counts.values, ax=axes[i, j])
            else:
                # For numpy array (from sequences)
                if j == 0:  # Training data
                    unique, counts = np.unique(y_train_seq[:, i], return_counts=True)
                elif j == 1:  # Validation data
                    unique, counts = np.unique(y_val_seq[:, i], return_counts=True)
                else:  # Test data
                    unique, counts = np.unique(y_test_seq[:, i], return_counts=True)
                    
                sns.barplot(x=unique.astype(int), y=counts, ax=axes[i, j])
            
            axes[i, j].set_title(f'{split_name} - {label_col}')
            axes[i, j].set_xlabel('Class')
            axes[i, j].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('stratified_label_distribution.png')
    plt.close()
    
    print("Data processing complete. Results saved to the 'processed_data' folder.")
    print("Label distribution visualization saved as 'stratified_label_distribution.png'")

if __name__ == "__main__":
    main()