#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclear Reactor Model Comprehensive Evaluation

This script provides a more reliable and transparent evaluation of all models:
1. Evaluates XGBoost on the full tabular test set
2. Evaluates LSTM on the full sequence test set
3. Evaluates the hybrid model on a representative subset
4. Provides detailed analysis of which approach works best for each label
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import joblib
import gc
import random

def load_data_and_models():
    """
    Load all test data and trained models.
    """
    print("Loading all test data and models...")
    
    # Load test data
    processed_folder = "processed_data"
    X_test_tab, y_test_tab = joblib.load(os.path.join(processed_folder, 'tabular_test.pkl'))
    X_test_seq, y_test_seq = joblib.load(os.path.join(processed_folder, 'sequence_test.pkl'))
    
    # Load feature and label columns
    feature_cols = joblib.load(os.path.join(processed_folder, 'feature_cols.pkl'))
    label_cols = joblib.load(os.path.join(processed_folder, 'label_cols.pkl'))
    
    # Load trained models
    try:
        xgb_models = joblib.load(os.path.join('models', 'xgboost_models.pkl'))
        print(f"XGBoost models loaded successfully for {len(xgb_models)} labels")
    except Exception as e:
        print(f"Error loading XGBoost models: {e}")
        xgb_models = None
    
    try:
        lstm_model = load_model(os.path.join('models', 'lstm_model.keras'))
        print("LSTM model loaded successfully")
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        lstm_model = None
    
    try:
        hybrid_weights = joblib.load(os.path.join('models', 'hybrid_weights.pkl'))
        print(f"Hybrid weights loaded successfully for {len(hybrid_weights)} labels")
    except Exception as e:
        print(f"Error loading hybrid weights: {e}")
        hybrid_weights = None
    
    # Print data shapes for verification
    print(f"Tabular test data: {X_test_tab.shape} samples, labels: {y_test_tab.shape}")
    print(f"Sequence test data: {X_test_seq.shape} sequences, labels: {y_test_seq.shape}")
    
    return (X_test_tab, y_test_tab, X_test_seq, y_test_seq, 
            feature_cols, label_cols, xgb_models, lstm_model, hybrid_weights)

def normalize_sequence_data(X_seq):
    """
    Normalize sequence data for LSTM.
    """
    feature_means = np.mean(X_seq, axis=(0, 1), keepdims=True)
    feature_stds = np.std(X_seq, axis=(0, 1), keepdims=True) + 1e-8  # Add epsilon to avoid division by zero
    X_seq_norm = (X_seq - feature_means) / feature_stds
    X_seq_norm = np.nan_to_num(X_seq_norm, nan=0.0, posinf=0.0, neginf=0.0)  # Replace any NaN or inf values
    return X_seq_norm

def evaluate_xgboost_full(X_test_tab, y_test_tab, xgb_models, label_cols):
    """
    Evaluate XGBoost on the full tabular test set.
    """
    print("\n=== Evaluating XGBoost on full tabular test set ===")
    
    if xgb_models is None:
        print("XGBoost models not available.")
        return None
    
    # Dictionary to store results
    results = {
        'accuracy': {},
        'predictions': {},
        'probabilities': {}
    }
    
    # Evaluate each label
    for label in label_cols:
        if label not in xgb_models:
            print(f"No XGBoost model available for {label}")
            continue
        
        print(f"Evaluating {label}...")
        
        # Get predictions
        model = xgb_models[label]
        y_pred = model.predict(X_test_tab)
        y_prob = model.predict_proba(X_test_tab)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test_tab[label], y_pred)
        print(f"XGBoost accuracy for {label}: {accuracy:.4f}")
        
        # Store results
        results['accuracy'][label] = accuracy
        results['predictions'][label] = y_pred
        results['probabilities'][label] = y_prob
    
    # Calculate overall accuracy
    if results['accuracy']:
        overall_acc = np.mean(list(results['accuracy'].values()))
        results['accuracy']['overall'] = overall_acc
        print(f"XGBoost overall accuracy: {overall_acc:.4f}")
    
    return results

def evaluate_lstm_full(X_test_seq, y_test_seq, lstm_model, label_cols):
    """
    Evaluate LSTM on the full sequence test set.
    """
    print("\n=== Evaluating LSTM on full sequence test set ===")
    
    if lstm_model is None:
        print("LSTM model not available.")
        return None
    
    # Dictionary to store results
    results = {
        'accuracy': {},
        'predictions': {},
        'probabilities': {}
    }
    
    # Normalize sequence data
    X_test_seq_norm = normalize_sequence_data(X_test_seq)
    
    # Print test label distribution to verify diversity
    print("Test label distribution:")
    for i, label in enumerate(label_cols):
        unique_vals, counts = np.unique(y_test_seq[:, i], return_counts=True)
        print(f"{label}: {dict(zip(unique_vals, counts))}")
    
    # Get model predictions
    print("Getting LSTM predictions...")
    lstm_probs = lstm_model.predict(X_test_seq_norm, batch_size=32)
    
    # Ensure probs is a list of arrays
    if not isinstance(lstm_probs, list):
        print("Single output model detected, converting to list format")
        lstm_probs = [lstm_probs]
    
    # Print prediction distribution
    print("\nLSTM prediction distribution:")
    for i, label in enumerate(label_cols):
        if i < len(lstm_probs):
            pred_classes = np.argmax(lstm_probs[i], axis=1)
            unique_preds, pred_counts = np.unique(pred_classes, return_counts=True)
            print(f"{label}: {dict(zip(unique_preds, pred_counts))}")
    
    # Evaluate each label
    for i, label in enumerate(label_cols):
        if i >= len(lstm_probs):
            print(f"No LSTM output available for {label}")
            continue
        
        print(f"Evaluating {label}...")
        
        # Get predictions for this label
        y_prob = lstm_probs[i]
        y_pred = np.argmax(y_prob, axis=1)
        
        # Get true labels for this label
        y_true = y_test_seq[:, i]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"LSTM accuracy for {label}: {accuracy:.4f}")
        
        # Calculate per-class accuracies for deeper insight
        class_accuracies = {}
        unique_classes = np.unique(y_true)
        for cls in unique_classes:
            mask = (y_true == cls)
            if np.sum(mask) > 0:  # Avoid division by zero
                cls_acc = np.sum(y_pred[mask] == cls) / np.sum(mask)
                class_accuracies[int(cls)] = float(cls_acc)
        
        print(f"  Per-class accuracy: {class_accuracies}")
        
        # Store results
        results['accuracy'][label] = accuracy
        results['predictions'][label] = y_pred
        results['probabilities'][label] = y_prob
    
    # Calculate overall accuracy
    if results['accuracy']:
        overall_acc = np.mean(list(results['accuracy'].values()))
        results['accuracy']['overall'] = overall_acc
        print(f"LSTM overall accuracy: {overall_acc:.4f}")
    
    return results

def create_sequence_subset(X_test_tab, y_test_tab, feature_cols, sequence_length=20, n_samples=50):
    """
    Create a small subset of sequence data from tabular data for hybrid evaluation.
    This is needed because we don't know the exact mapping between tabular and sequence data.
    
    Args:
        X_test_tab: Tabular test features
        y_test_tab: Tabular test labels
        feature_cols: Feature column names
        sequence_length: Length of sequences to create
        n_samples: Number of sequences to create
        
    Returns:
        X_seq_subset: Subset of sequence data
        y_seq_subset: Corresponding labels
    """
    print("\n=== Creating sequence subset for hybrid evaluation ===")
    
    # Get unique scenario IDs if available
    if 'scenario_id' in X_test_tab.columns:
        scenario_ids = X_test_tab['scenario_id'].unique()
        print(f"Found {len(scenario_ids)} unique scenarios")
        
        # Select a subset of scenarios
        n_scenarios = min(n_samples, len(scenario_ids))
        selected_scenarios = np.random.choice(scenario_ids, n_scenarios, replace=False)
        
        # Create sequences from each selected scenario
        X_seq_subset = []
        y_seq_subset = []
        
        for scenario_id in selected_scenarios:
            # Get data for this scenario
            scenario_data = X_test_tab[X_test_tab['scenario_id'] == scenario_id].copy()
            scenario_labels = y_test_tab[X_test_tab['scenario_id'] == scenario_id].copy()
            
            # Sort by time if available
            if 'Time' in scenario_data.columns:
                scenario_data = scenario_data.sort_values('Time')
                scenario_labels = scenario_labels.iloc[scenario_data.index]
            
            # Ensure we have enough data points
            if len(scenario_data) >= sequence_length:
                # Select a random starting point
                max_start = len(scenario_data) - sequence_length
                start_idx = random.randint(0, max_start)
                
                # Extract sequence
                seq_data = scenario_data.iloc[start_idx:start_idx+sequence_length]
                seq_features = seq_data[feature_cols].values
                
                # Use the label from the last timestep
                seq_label = scenario_labels.iloc[start_idx+sequence_length-1].values
                
                X_seq_subset.append(seq_features)
                y_seq_subset.append(seq_label)
        
        X_seq_subset = np.array(X_seq_subset)
        y_seq_subset = np.array(y_seq_subset)
    
    else:
        # If no scenario_id is available, create sequences from random starting points
        print("No scenario_id found, creating sequences from random starting points")
        
        # Determine how many sequences we can create
        max_sequences = (len(X_test_tab) - sequence_length) // sequence_length
        n_sequences = min(n_samples, max_sequences)
        
        # Create random starting indices
        max_start_idx = len(X_test_tab) - sequence_length * n_sequences
        start_idx = random.randint(0, max_start_idx)
        
        # Initialize arrays
        X_seq_subset = np.zeros((n_sequences, sequence_length, len(feature_cols)))
        y_seq_subset = np.zeros((n_sequences, y_test_tab.shape[1]))
        
        # Create sequences
        for i in range(n_sequences):
            seq_start = start_idx + i * sequence_length
            seq_end = seq_start + sequence_length
            
            # Extract features
            seq_features = X_test_tab.iloc[seq_start:seq_end][feature_cols].values
            
            # Use label from the last timestep
            seq_label = y_test_tab.iloc[seq_end-1].values
            
            X_seq_subset[i] = seq_features
            y_seq_subset[i] = seq_label
    
    print(f"Created {len(X_seq_subset)} sequences with shape {X_seq_subset.shape}")
    return X_seq_subset, y_seq_subset

def evaluate_hybrid_model(X_test_seq, y_test_seq, xgb_models, lstm_model, hybrid_weights, label_cols):
    """
    Evaluate the hybrid model using sequence test data.
    
    For the hybrid model, we'll:
    1. Use LSTM on the full sequences
    2. Extract the last timestep of each sequence for XGBoost
    3. Combine predictions using the hybrid weights
    """
    print("\n=== Evaluating hybrid model on sequence test data ===")
    
    if xgb_models is None or lstm_model is None or hybrid_weights is None:
        print("One or more models or weights not available.")
        return None
    
    # Dictionary to store results
    results = {
        'accuracy': {},
        'predictions': {}
    }
    
    # Get LSTM predictions
    X_test_seq_norm = normalize_sequence_data(X_test_seq)
    lstm_probs = lstm_model.predict(X_test_seq_norm, batch_size=32)
    
    # Ensure lstm_probs is a list
    if not isinstance(lstm_probs, list):
        lstm_probs = [lstm_probs]
    
    # Extract last timestep features for XGBoost
    X_last_timestep = X_test_seq[:, -1, :]
    
    # Evaluate each label
    for i, label in enumerate(label_cols):
        if label not in xgb_models or i >= len(lstm_probs):
            print(f"Missing model for {label}, skipping")
            continue
        
        print(f"Evaluating hybrid model for {label}...")
        
        # Get XGBoost predictions
        xgb_prob = xgb_models[label].predict_proba(X_last_timestep)
        
        # Get LSTM predictions
        lstm_prob = lstm_probs[i]
        
        # Ensure compatible shapes
        if xgb_prob.shape[1] != lstm_prob.shape[1]:
            print(f"Warning: Shape mismatch for {label} - XGBoost: {xgb_prob.shape}, LSTM: {lstm_prob.shape}")
            
            # Use minimum number of classes
            min_classes = min(xgb_prob.shape[1], lstm_prob.shape[1])
            
            # Truncate both arrays
            xgb_prob = xgb_prob[:, :min_classes]
            lstm_prob = lstm_prob[:, :min_classes]
            
            # Renormalize
            xgb_prob = xgb_prob / np.sum(xgb_prob, axis=1, keepdims=True)
            lstm_prob = lstm_prob / np.sum(lstm_prob, axis=1, keepdims=True)
        
        # Get weight for this label
        weight = hybrid_weights.get(label, 0.5)
        print(f"Using weight of {weight:.2f} for XGBoost, {1-weight:.2f} for LSTM")
        
        # Combine predictions
        hybrid_prob = weight * xgb_prob + (1 - weight) * lstm_prob
        hybrid_pred = np.argmax(hybrid_prob, axis=1)
        
        # Get true labels
        y_true = y_test_seq[:, i]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, hybrid_pred)
        print(f"Hybrid model accuracy for {label}: {accuracy:.4f}")
        
        # Store results
        results['accuracy'][label] = accuracy
        results['predictions'][label] = hybrid_pred
        
        # Also calculate individual model accuracies for comparison
        xgb_pred = np.argmax(xgb_prob, axis=1)
        lstm_pred = np.argmax(lstm_prob, axis=1)
        
        xgb_acc = accuracy_score(y_true, xgb_pred)
        lstm_acc = accuracy_score(y_true, lstm_pred)
        
        print(f"On this subset - XGBoost: {xgb_acc:.4f}, LSTM: {lstm_acc:.4f}, Hybrid: {accuracy:.4f}")
        
        # Check if evaluation is valid (has multiple classes in test set)
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 1:
            print(f"WARNING: Test data contains only one class ({unique_classes[0]}) for {label}")
            print("This could lead to artificially high accuracy. Consider re-running data processing.")
        
        # Calculate if hybrid is better than the individual models
        if accuracy > max(xgb_acc, lstm_acc):
            print(f"✓ Hybrid model improves performance for {label}")
        else:
            print(f"✗ Hybrid model does not improve performance for {label}")
    
    # Calculate overall accuracy
    if results['accuracy']:
        overall_acc = np.mean(list(results['accuracy'].values()))
        results['accuracy']['overall'] = overall_acc
        print(f"Hybrid model overall accuracy: {overall_acc:.4f}")
    
    return results

def create_comparison_summary(xgb_results, lstm_results, hybrid_results, label_cols):
    """
    Create a comprehensive comparison of all three models.
    """
    print("\n=== Model Comparison Summary ===")
    
    # Create output directory
    output_dir = "comprehensive_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for comparison table
    labels = list(label_cols) + ['overall']
    
    xgb_acc = []
    lstm_acc = []
    hybrid_acc = []
    best_model = []
    
    for label in labels:
        # Get accuracies with fallback to NaN
        xgb_accuracy = xgb_results['accuracy'].get(label, np.nan) if xgb_results else np.nan
        lstm_accuracy = lstm_results['accuracy'].get(label, np.nan) if lstm_results else np.nan
        hybrid_accuracy = hybrid_results['accuracy'].get(label, np.nan) if hybrid_results else np.nan
        
        xgb_acc.append(xgb_accuracy)
        lstm_acc.append(lstm_accuracy)
        hybrid_acc.append(hybrid_accuracy)
        
        # Determine best model
        if not np.isnan(xgb_accuracy) and not np.isnan(lstm_accuracy) and not np.isnan(hybrid_accuracy):
            accuracies = [xgb_accuracy, lstm_accuracy, hybrid_accuracy]
            models = ['XGBoost', 'LSTM', 'Hybrid']
            best_idx = np.argmax(accuracies)
            best_model.append(models[best_idx])
        else:
            best_model.append('N/A')
    
    # Create DataFrame
    summary_df = pd.DataFrame({
        'Label': labels,
        'XGBoost': xgb_acc,
        'LSTM': lstm_acc,
        'Hybrid': hybrid_acc,
        'Best Model': best_model
    })
    
    # Save to CSV
    summary_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Display summary
    print("\nModel Comparison Table:")
    print(summary_df)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(labels))
    width = 0.25
    
    # Plot only non-NaN values
    xgb_mask = ~np.isnan(xgb_acc)
    lstm_mask = ~np.isnan(lstm_acc)
    hybrid_mask = ~np.isnan(hybrid_acc)
    
    plt.bar(x[xgb_mask] - width, np.array(xgb_acc)[xgb_mask], width, label='XGBoost')
    plt.bar(x[lstm_mask], np.array(lstm_acc)[lstm_mask], width, label='LSTM')
    plt.bar(x[hybrid_mask] + width, np.array(hybrid_acc)[hybrid_mask], width, label='Hybrid')
    
    plt.xlabel('Label')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    # Add value annotations
    for i, model in enumerate(['XGBoost', 'LSTM', 'Hybrid']):
        values = [xgb_acc, lstm_acc, hybrid_acc][i]
        mask = [xgb_mask, lstm_mask, hybrid_mask][i]
        offset = (i-1) * width
        
        for j, v in enumerate(np.array(values)[mask]):
            plt.text(x[mask][j] + offset, v + 0.01, f'{v:.3f}', 
                     ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=100)
    plt.close()
    
    print(f"Comparison results saved to {output_dir}")

def create_confusion_matrices(model_results, y_test, label_cols, model_name, output_dir):
    """
    Create confusion matrices for a model.
    """
    if model_results is None or 'predictions' not in model_results:
        return
    
    for i, label in enumerate(label_cols):
        if label not in model_results['predictions']:
            continue
        
        # Get predictions and true values
        y_pred = model_results['predictions'][label]
        
        # Get true labels
        if isinstance(y_test, pd.DataFrame):
            y_true = y_test[label].values
        else:
            y_true = y_test[:, i]
        
        # Create confusion matrix
        classes = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix - {label}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_cm_{label}.png'), dpi=100)
        plt.close()

def main():
    """
    Main function to run comprehensive evaluation.
    """
    # Create output directory
    output_dir = "comprehensive_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data and models
    X_test_tab, y_test_tab, X_test_seq, y_test_seq, feature_cols, label_cols, xgb_models, lstm_model, hybrid_weights = load_data_and_models()
    
    # Evaluate XGBoost on full tabular data
    xgb_results = evaluate_xgboost_full(X_test_tab, y_test_tab, xgb_models, label_cols)
    
    # Evaluate LSTM on full sequence data
    lstm_results = evaluate_lstm_full(X_test_seq, y_test_seq, lstm_model, label_cols)
    
    # Evaluate hybrid model
    hybrid_results = evaluate_hybrid_model(X_test_seq, y_test_seq, xgb_models, lstm_model, hybrid_weights, label_cols)
    
    # Create confusion matrices
    create_confusion_matrices(xgb_results, y_test_tab, label_cols, 'XGBoost', output_dir)
    create_confusion_matrices(lstm_results, y_test_seq, label_cols, 'LSTM', output_dir)
    create_confusion_matrices(hybrid_results, y_test_seq, label_cols, 'Hybrid', output_dir)
    
    # Create comparison summary
    create_comparison_summary(xgb_results, lstm_results, hybrid_results, label_cols)
    
    print("\nComprehensive evaluation complete!")

if __name__ == "__main__":
    main()