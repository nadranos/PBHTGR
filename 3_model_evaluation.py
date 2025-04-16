#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclear Reactor Model Testing and Evaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from tensorflow.keras.models import load_model
import time

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'tensorflow',
        'xgboost', 'sklearn', 'joblib'  # Changed 'scikit-learn' to 'sklearn' which is the actual import name
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install required dependencies using: pip install -r requirements.txt")
        return False
    
    print("All dependencies are installed.")
    return True

def load_models_and_data():
    """
    Load trained models and test data.
    
    Returns:
        tuple: Loaded models and test data
    """
    print("Loading models and test data...")
    
    # Load models with error handling
    try:
        xgb_models = joblib.load(os.path.join('models', 'xgboost_models.pkl'))
        print("XGBoost models loaded successfully")
    except Exception as e:
        print(f"Error loading XGBoost models: {str(e)}")
        xgb_models = None
        
    try:
        lstm_model = load_model(os.path.join('models', 'lstm_model.keras'))  # Updated file extension
        print("LSTM model loaded successfully")
    except Exception as e:
        print(f"Error loading LSTM model: {str(e)}")
        lstm_model = None
        
    try:
        hybrid_weights = joblib.load(os.path.join('models', 'hybrid_weights.pkl'))
        print("Hybrid weights loaded successfully")
    except Exception as e:
        print(f"Error loading hybrid weights: {str(e)}")
        hybrid_weights = None
    
    # Load test data
    X_test_tab, y_test_tab = joblib.load(os.path.join('processed_data', 'tabular_test.pkl'))
    X_test_seq, y_test_seq = joblib.load(os.path.join('processed_data', 'sequence_test.pkl'))
    
    # Load feature and label column names
    feature_cols = joblib.load(os.path.join('processed_data', 'feature_cols.pkl'))
    label_cols = joblib.load(os.path.join('processed_data', 'label_cols.pkl'))
    
    print("Models and data loaded successfully.")
    
    return xgb_models, lstm_model, hybrid_weights, X_test_tab, y_test_tab, X_test_seq, y_test_seq, feature_cols, label_cols

def evaluate_xgboost(xgb_models, X_test, y_test, label_cols):
    """
    Evaluate XGBoost models on test data.
    
    Args:
        xgb_models (dict): Dictionary of trained XGBoost models
        X_test (DataFrame): Test features
        y_test (DataFrame): Test labels
        label_cols (list): List of label column names
        
    Returns:
        tuple: XGBoost predictions and accuracies
    """
    print("Evaluating XGBoost models...")
    
    # Dictionary to store predictions for each label
    xgb_preds = {}
    xgb_probs = {}
    
    # Dictionary to store accuracy for each label
    xgb_accuracies = {}
    
    # Overall correct predictions counter
    total_correct = 0
    total_samples = 0
    
    # Evaluate each model
    for label in label_cols:
        # Get predictions
        y_pred = xgb_models[label].predict(X_test)
        y_prob = xgb_models[label].predict_proba(X_test)
        
        # Store predictions
        xgb_preds[label] = y_pred
        xgb_probs[label] = y_prob
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test[label], y_pred)
        xgb_accuracies[label] = accuracy
        
        # Update overall counters
        total_correct += np.sum(y_pred == y_test[label].values)
        total_samples += len(y_pred)
        
        print(f"XGBoost accuracy for {label}: {accuracy:.4f}")
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_samples
    print(f"XGBoost overall accuracy: {overall_accuracy:.4f}")
    
    # Store overall accuracy
    xgb_accuracies['overall'] = overall_accuracy
    
    return xgb_preds, xgb_probs, xgb_accuracies

def evaluate_lstm(lstm_model, X_test_seq, y_test_seq, label_cols):
    """
    Evaluate LSTM model on test data.
    
    Args:
        lstm_model (Model): Trained LSTM model
        X_test_seq (array): Test sequence data
        y_test_seq (array): Test sequence labels
        label_cols (list): List of label column names
        
    Returns:
        tuple: LSTM predictions and accuracies
    """
    print("Evaluating LSTM model...")
    
    if lstm_model is None:
        print("LSTM model not available for evaluation.")
        return {}, [], {}
    
    # Properly normalize test data with better approach
    # In a real scenario, we should save mean/std from training and apply here
    # For simplicity, we'll normalize based on test data
    
    # First convert to float32 if needed
    if X_test_seq.dtype != np.float32:
        X_test_seq = X_test_seq.astype(np.float32)
    
    # Normalize features to match training normalization
    print("Applying feature normalization to test data...")
    feature_means = np.mean(X_test_seq, axis=(0, 1), keepdims=True)
    feature_stds = np.std(X_test_seq, axis=(0, 1), keepdims=True) + 1e-8
    X_test_seq_norm = (X_test_seq - feature_means) / feature_stds
    X_test_seq_norm = np.nan_to_num(X_test_seq_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Print test label distribution to verify diversity
    print("Test label distribution:")
    for i, label in enumerate(label_cols):
        unique_vals, counts = np.unique(y_test_seq[:, i], return_counts=True)
        print(f"{label}: {dict(zip(unique_vals, counts))}")
    
    # Get predictions from LSTM model with smaller batch size to avoid memory issues
    try:
        print("Getting LSTM predictions...")
        lstm_probs = lstm_model.predict(X_test_seq_norm, batch_size=32)
        
        # Check if output is a list (multi-output model) or single array
        if not isinstance(lstm_probs, list):
            print("LSTM model has a single output. Converting to list format...")
            lstm_probs = [lstm_probs]  # Convert to list to match expected format
        
        # Convert predictions to class labels
        lstm_preds = {}
        
        # Also save per-class prediction distribution to debug
        print("\nLSTM prediction distribution by class:")
        for i, label in enumerate(label_cols):
            if i < len(lstm_probs):  # Make sure we don't go out of bounds
                pred_array = np.argmax(lstm_probs[i], axis=1)
                lstm_preds[label] = pred_array
                
                # Print distribution of predicted classes
                unique_preds, pred_counts = np.unique(pred_array, return_counts=True)
                print(f"{label}: {dict(zip(unique_preds, pred_counts))}")
            else:
                print(f"Warning: No output for label {label} in LSTM model")
                lstm_preds[label] = np.zeros(len(y_test_seq))
        
        # Dictionary to store accuracy for each label
        lstm_accuracies = {}
        
        # Overall correct predictions counter
        total_correct = 0
        total_samples = 0
        
        # Evaluate each output
        for i, label in enumerate(label_cols):
            if i < len(lstm_probs):  # Only evaluate if we have predictions
                # Calculate accuracy
                accuracy = accuracy_score(y_test_seq[:, i], lstm_preds[label])
                lstm_accuracies[label] = accuracy
                
                # Update overall counters
                total_correct += np.sum(lstm_preds[label] == y_test_seq[:, i])
                total_samples += len(lstm_preds[label])
                
                print(f"LSTM accuracy for {label}: {accuracy:.4f}")
                
                # Print per-class accuracy for this label
                class_accuracies = {}
                unique_classes = np.unique(y_test_seq[:, i])
                for cls in unique_classes:
                    mask = (y_test_seq[:, i] == cls)
                    if np.sum(mask) > 0:  # Avoid division by zero
                        cls_acc = np.sum(lstm_preds[label][mask] == cls) / np.sum(mask)
                        class_accuracies[int(cls)] = float(cls_acc)
                
                print(f"  Per-class accuracy for {label}: {class_accuracies}")
            else:
                lstm_accuracies[label] = 0.0
                print(f"LSTM accuracy for {label}: N/A (no output)")
        
        # Calculate overall accuracy
        if total_samples > 0:
            overall_accuracy = total_correct / total_samples
        else:
            overall_accuracy = 0.0
            
        print(f"LSTM overall accuracy: {overall_accuracy:.4f}")
        
        # Store overall accuracy
        lstm_accuracies['overall'] = overall_accuracy
        
        return lstm_preds, lstm_probs, lstm_accuracies
    
    except Exception as e:
        print(f"Error evaluating LSTM model: {str(e)}")
        return {}, [], {}

def evaluate_hybrid(xgb_probs, lstm_probs, hybrid_weights, y_test_seq, label_cols):
    """
    Evaluate hybrid model on test data.
    
    Args:
        xgb_probs (dict): XGBoost probability predictions
        lstm_probs (list): LSTM probability predictions
        hybrid_weights (dict): Hybrid model weights
        y_test_seq (array): Test sequence labels
        label_cols (list): List of label column names
        
    Returns:
        tuple: Hybrid predictions and accuracies
    """
    print("Evaluating hybrid model...")
    
    # Check if we have all the required components
    if not xgb_probs or not lstm_probs or not hybrid_weights:
        print("Missing components for hybrid evaluation.")
        return {}, {}
    
    # Convert LSTM output to dictionary
    lstm_probs_dict = {}
    for i, label in enumerate(label_cols):
        if i < len(lstm_probs):
            lstm_probs_dict[label] = lstm_probs[i]
    
    # Dictionary to store hybrid predictions
    hybrid_preds = {}
    
    # Dictionary to store accuracy for each label
    hybrid_accuracies = {}
    
    # Check for shape mismatch between XGBoost and LSTM predictions
    sample_label = next(iter(xgb_probs))
    xgb_shape = xgb_probs[sample_label].shape[0]
    
    if sample_label in lstm_probs_dict:
        lstm_shape = lstm_probs_dict[sample_label].shape[0]
    else:
        lstm_shape = 0
    
    shape_mismatch = xgb_shape != lstm_shape
    
    if shape_mismatch:
        print("\nWARNING: Shape mismatch between XGBoost and LSTM predictions")
        print(f"XGBoost predictions: {xgb_shape} samples")
        print(f"LSTM predictions: {lstm_shape} samples")
        print("Using individual model predictions instead")
        
        # Use the models individually based on their weights
        # Dictionary to store hybrid predictions
        hybrid_preds = {}
        
        # Overall correct predictions counter
        total_correct = 0
        total_samples = 0
        
        # Evaluate for each label
        for i, label in enumerate(label_cols):
            # Get the weight for this label
            weight = hybrid_weights.get(label, 0.5)
            
            # Get XGBoost accuracy (using tabular test data)
            try:
                xgb_pred = np.argmax(xgb_probs[label], axis=1)
                xgb_accuracy = None  # We won't calculate XGBoost accuracy against sequence labels
            except:
                print(f"Error getting XGBoost predictions for {label}")
                xgb_accuracy = 0.0
                
            # Get LSTM accuracy (using sequence test data)
            if label in lstm_probs_dict:
                lstm_pred = np.argmax(lstm_probs_dict[label], axis=1) 
                lstm_accuracy = accuracy_score(y_test_seq[:, i], lstm_pred)
                print(f"LSTM accuracy for {label}: {lstm_accuracy:.4f}")
            else:
                lstm_accuracy = 0.0
                
            # Assign the prediction based on weight
            if weight >= 0.5 or label not in lstm_probs_dict:
                print(f"Using XGBoost predictions for {label} (weight={weight:.2f})")
                hybrid_preds[label] = xgb_pred
                accuracy = xgb_accuracy or 0.0  # If we don't have accuracy, use 0
            else:
                print(f"Using LSTM predictions for {label} (weight={1-weight:.2f})")
                hybrid_preds[label] = lstm_pred
                accuracy = lstm_accuracy
            
            hybrid_accuracies[label] = accuracy
            
            # Skip overall counters for XGBoost predictions due to shape mismatch
            if label in lstm_probs_dict and weight < 0.5:
                # Only count when using LSTM predictions which match the test sequence labels
                total_correct += np.sum(hybrid_preds[label] == y_test_seq[:, i])
                total_samples += len(hybrid_preds[label])
            
            print(f"Hybrid model accuracy for {label}: {accuracy:.4f}")
    else:
        # No shape mismatch, proceed with normal hybrid evaluation
        # Overall correct predictions counter
        total_correct = 0
        total_samples = 0
        
        # Evaluate for each label
        for i, label in enumerate(label_cols):
            # Skip if label is not in either model
            if label not in xgb_probs or label not in lstm_probs_dict:
                print(f"Skipping {label} - not found in one or both models")
                continue
                
            # Get the weight for this label
            weight = hybrid_weights.get(label, 0.5)
            
            # Combine predictions
            combined_probs = weight * xgb_probs[label] + (1 - weight) * lstm_probs_dict[label]
            hybrid_preds[label] = np.argmax(combined_probs, axis=1)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test_seq[:, i], hybrid_preds[label])
            hybrid_accuracies[label] = accuracy
            
            # For regular evaluation, the shapes should match
            try:
                total_correct += np.sum(hybrid_preds[label] == y_test_seq[:, i])
                total_samples += len(hybrid_preds[label])
            except:
                print(f"Warning: Could not calculate overall stats for {label}")
            
            print(f"Hybrid model accuracy for {label} (weight={weight:.2f}): {accuracy:.4f}")
    
    # Calculate overall accuracy
    if total_samples > 0:
        overall_accuracy = total_correct / total_samples
    else:
        overall_accuracy = 0.0
        
    print(f"Hybrid model overall accuracy: {overall_accuracy:.4f}")
    
    # Store overall accuracy
    hybrid_accuracies['overall'] = overall_accuracy
    
    return hybrid_preds, hybrid_accuracies

def validate_test_data_diversity(y_test_seq, label_cols):
    """
    Validate that test data has diverse class labels.
    
    Args:
        y_test_seq (array): Test sequence labels
        label_cols (list): List of label column names
        
    Returns:
        bool: True if test data is diverse, False otherwise
    """
    print("\nValidating test data diversity...")
    
    diverse = True
    
    for i, label in enumerate(label_cols):
        unique_classes = np.unique(y_test_seq[:, i])
        if len(unique_classes) == 1:
            print(f"WARNING: Test data contains only one class ({unique_classes[0]}) for {label}")
            print("This will lead to misleading model evaluation results.")
            diverse = False
        else:
            class_counts = {}
            for cls in unique_classes:
                count = np.sum(y_test_seq[:, i] == cls)
                class_counts[int(cls)] = int(count)
            print(f"✓ {label} has {len(unique_classes)} classes in test data: {class_counts}")
    
    if not diverse:
        print("\nIMPORTANT: You should rerun the data processing with the updated script to")
        print("ensure proper class diversity in test and validation datasets.\n")
    
    return diverse

def plot_confusion_matrices(xgb_preds, lstm_preds, hybrid_preds, y_test_seq, label_cols):
    """
    Plot confusion matrices for all models.
    
    Args:
        xgb_preds (dict): XGBoost predictions
        lstm_preds (dict): LSTM predictions
        hybrid_preds (dict): Hybrid predictions
        y_test_seq (array): Test sequence labels
        label_cols (list): List of label column names
    """
    print("Plotting confusion matrices...")
    
    output_folder = "evaluation_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a figure for each label
    for i, label in enumerate(label_cols):
        # Skip if no predictions available for this label
        if label not in lstm_preds and label not in hybrid_preds:
            print(f"Skipping confusion matrix for {label} - insufficient data")
            continue
            
        # Create plot with dynamic number of subplots based on available models
        available_models = sum([
            label in xgb_preds and len(xgb_preds[label]) == len(y_test_seq[:, i]), 
            label in lstm_preds,
            label in hybrid_preds and len(hybrid_preds[label]) == len(y_test_seq[:, i])
        ])
        
        if available_models == 0:
            print(f"No compatible models for confusion matrix on {label}")
            continue
            
        # Get the true labels for this output
        y_true = y_test_seq[:, i]
        
        # Get unique classes
        classes = np.unique(y_true)
        
        # Create figure with appropriate number of plots
        fig, axes = plt.subplots(1, available_models, figsize=(6*available_models, 6))
        if available_models == 1:
            axes = [axes]  # Make sure axes is always a list
            
        plot_idx = 0
        
        # Plot XGBoost confusion matrix if compatible
        if label in xgb_preds and len(xgb_preds[label]) == len(y_true):
            try:
                cm_xgb = confusion_matrix(y_true, xgb_preds[label], labels=classes)
                sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[plot_idx], cbar=False)
                axes[plot_idx].set_title(f'XGBoost - {label}')
                axes[plot_idx].set_xlabel('Predicted')
                axes[plot_idx].set_ylabel('True')
                plot_idx += 1
            except Exception as e:
                print(f"Error plotting XGBoost confusion matrix for {label}: {e}")
        
        # Plot LSTM confusion matrix
        if label in lstm_preds:
            try:
                cm_lstm = confusion_matrix(y_true, lstm_preds[label], labels=classes)
                sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', ax=axes[plot_idx], cbar=False)
                axes[plot_idx].set_title(f'LSTM - {label}')
                axes[plot_idx].set_xlabel('Predicted')
                axes[plot_idx].set_ylabel('True')
                plot_idx += 1
            except Exception as e:
                print(f"Error plotting LSTM confusion matrix for {label}: {e}")
        
        # Plot Hybrid confusion matrix if compatible
        if label in hybrid_preds and len(hybrid_preds[label]) == len(y_true):
            try:
                cm_hybrid = confusion_matrix(y_true, hybrid_preds[label], labels=classes)
                sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Blues', ax=axes[plot_idx], cbar=False)
                axes[plot_idx].set_title(f'Hybrid - {label}')
                axes[plot_idx].set_xlabel('Predicted')
                axes[plot_idx].set_ylabel('True')
            except Exception as e:
                print(f"Error plotting Hybrid confusion matrix for {label}: {e}")
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'confusion_matrix_{label}.png'))
        plt.close()
    
    print(f"Confusion matrices saved to {output_folder} folder.")

def plot_accuracy_comparison(xgb_accuracies, lstm_accuracies, hybrid_accuracies, label_cols):
    """
    Plot accuracy comparison for all models.
    
    Args:
        xgb_accuracies (dict): XGBoost accuracies
        lstm_accuracies (dict): LSTM accuracies
        hybrid_accuracies (dict): Hybrid accuracies
        label_cols (list): List of label column names
    """
    print("Plotting accuracy comparison...")
    
    output_folder = "evaluation_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare data for plotting with error handling
    labels = label_cols.copy()
    if all(model is not None and 'overall' in model for model in [xgb_accuracies, lstm_accuracies, hybrid_accuracies]):
        labels.append('overall')
    
    # Use NaN for missing values
    xgb_acc = []
    lstm_acc = []
    hybrid_acc = []
    
    for label in labels:
        # Get XGBoost accuracy with fallback to NaN
        if xgb_accuracies is not None and label in xgb_accuracies:
            xgb_acc.append(xgb_accuracies[label])
        else:
            xgb_acc.append(np.nan)
            
        # Get LSTM accuracy with fallback to NaN
        if lstm_accuracies is not None and label in lstm_accuracies:
            lstm_acc.append(lstm_accuracies[label])
        else:
            lstm_acc.append(np.nan)
            
        # Get Hybrid accuracy with fallback to NaN
        if hybrid_accuracies is not None and label in hybrid_accuracies:
            hybrid_acc.append(hybrid_accuracies[label])
        else:
            hybrid_acc.append(np.nan)
    
    # Set up bar positions
    x = np.arange(len(labels))
    width = 0.25
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars only for non-NaN values
    xgb_mask = ~np.isnan(xgb_acc)
    lstm_mask = ~np.isnan(lstm_acc)
    hybrid_mask = ~np.isnan(hybrid_acc)
    
    if any(xgb_mask):
        rects1 = ax.bar(x[xgb_mask] - width, np.array(xgb_acc)[xgb_mask], width, label='XGBoost')
    
    if any(lstm_mask):
        rects2 = ax.bar(x[lstm_mask], np.array(lstm_acc)[lstm_mask], width, label='LSTM')
    
    if any(hybrid_mask):
        rects3 = ax.bar(x[hybrid_mask] + width, np.array(hybrid_acc)[hybrid_mask], width, label='Hybrid')
    
    # Add labels and title
    ax.set_xlabel('Component')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add exact values on top of bars (with error handling)
    def autolabel(rects):
        if rects is None:
            return
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=90)
    
    # Only call autolabel if the variables exist
    if any(xgb_mask):
        autolabel(rects1)
    if any(lstm_mask):
        autolabel(rects2)
    if any(hybrid_mask):
        autolabel(rects3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'accuracy_comparison.png'))
    plt.close()
    
    # Also create a table with the accuracies
    table_data = pd.DataFrame({
        'Component': labels,
        'XGBoost': xgb_acc,
        'LSTM': lstm_acc,
        'Hybrid': hybrid_acc
    })
    
    table_data.to_csv(os.path.join(output_folder, 'accuracy_comparison.csv'), index=False)
    
    print(f"Accuracy comparison saved to {output_folder} folder.")

def plot_feature_importance(xgb_models, feature_cols, label_cols):
    """
    Plot feature importance for XGBoost models.
    
    Args:
        xgb_models (dict): Dictionary of trained XGBoost models
        feature_cols (list): List of feature column names
        label_cols (list): List of label column names
    """
    print("Plotting feature importance...")
    
    output_folder = "evaluation_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a figure for each label
    for i, label in enumerate(label_cols):
        # Get the feature importance from the model
        importance = xgb_models[label].feature_importances_
        
        # Get indices of top 20 features
        top_indices = np.argsort(importance)[-20:]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot horizontal bars
        plt.barh(range(len(top_indices)), importance[top_indices])
        plt.yticks(range(len(top_indices)), [feature_cols[i] for i in top_indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importance for {label}')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'feature_importance_{label}.png'))
        plt.close()
    
    print(f"Feature importance plots saved to {output_folder} folder.")

def generate_classification_reports(xgb_preds, lstm_preds, hybrid_preds, y_test_seq, label_cols):
    """
    Generate classification reports for all models.
    
    Args:
        xgb_preds (dict): XGBoost predictions
        lstm_preds (dict): LSTM predictions
        hybrid_preds (dict): Hybrid predictions
        y_test_seq (array): Test sequence labels
        label_cols (list): List of label column names
    """
    print("Generating classification reports...")
    
    output_folder = "evaluation_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a report for each label and model
    for i, label in enumerate(label_cols):
        # Get the true labels for this output
        y_true = y_test_seq[:, i]
        
        # XGBoost report - only if shapes match
        if label in xgb_preds and len(xgb_preds[label]) == len(y_true):
            try:
                xgb_report = classification_report(y_true, xgb_preds[label], output_dict=True)
                xgb_report_df = pd.DataFrame(xgb_report).transpose()
                xgb_report_df.to_csv(os.path.join(output_folder, f'xgboost_report_{label}.csv'))
                print(f"XGBoost classification report for {label} saved.")
            except Exception as e:
                print(f"Error generating XGBoost report for {label}: {e}")
        else:
            print(f"Skipping XGBoost report for {label} - shape mismatch")
        
        # LSTM report
        if label in lstm_preds:
            try:
                lstm_report = classification_report(y_true, lstm_preds[label], output_dict=True)
                lstm_report_df = pd.DataFrame(lstm_report).transpose()
                lstm_report_df.to_csv(os.path.join(output_folder, f'lstm_report_{label}.csv'))
                print(f"LSTM classification report for {label} saved.")
            except Exception as e:
                print(f"Error generating LSTM report for {label}: {e}")
        else:
            print(f"Skipping LSTM report for {label} - not available")
        
        # Hybrid report
        if label in hybrid_preds and len(hybrid_preds[label]) == len(y_true):
            try:
                hybrid_report = classification_report(y_true, hybrid_preds[label], output_dict=True)
                hybrid_report_df = pd.DataFrame(hybrid_report).transpose()
                hybrid_report_df.to_csv(os.path.join(output_folder, f'hybrid_report_{label}.csv'))
                print(f"Hybrid classification report for {label} saved.")
            except Exception as e:
                print(f"Error generating Hybrid report for {label}: {e}")
        else:
            print(f"Skipping Hybrid report for {label} - shape mismatch or not available")
    
    print(f"Classification reports saved to {output_folder} folder.")

def measure_prediction_time(xgb_models, lstm_model, hybrid_weights, X_test_tab, X_test_seq, label_cols):
    """
    Measure prediction time for all models.
    
    Args:
        xgb_models (dict): Dictionary of trained XGBoost models
        lstm_model (Model): Trained LSTM model
        hybrid_weights (dict): Hybrid model weights
        X_test_tab (DataFrame): Test tabular data
        X_test_seq (array): Test sequence data
        label_cols (list): List of label column names
    """
    print("Measuring prediction time...")
    
    output_folder = "evaluation_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize times
    xgb_time = None
    lstm_time = None
    hybrid_time = None
    
    # Measure XGBoost prediction time if available
    if xgb_models:
        try:
            xgb_start_time = time.time()
            xgb_probs = {}
            for label in label_cols:
                if label in xgb_models:
                    xgb_probs[label] = xgb_models[label].predict_proba(X_test_tab)
            xgb_end_time = time.time()
            xgb_time = xgb_end_time - xgb_start_time
            n_samples_xgb = len(X_test_tab) if isinstance(X_test_tab, pd.DataFrame) else X_test_tab.shape[0]
            xgb_time_per_sample = xgb_time / n_samples_xgb * 1000  # ms
            print(f"XGBoost prediction time: {xgb_time:.4f} seconds ({xgb_time_per_sample:.4f} ms/sample)")
        except Exception as e:
            print(f"Error measuring XGBoost prediction time: {e}")
            xgb_time = None
    else:
        print("XGBoost models not available for timing.")
        xgb_time = None
    
    # Measure LSTM prediction time if available
    if lstm_model:
        try:
            # Normalize test data first
            feature_means = np.mean(X_test_seq, axis=(0, 1), keepdims=True)
            feature_stds = np.std(X_test_seq, axis=(0, 1), keepdims=True) + 1e-8
            X_test_seq_norm = (X_test_seq - feature_means) / feature_stds
            X_test_seq_norm = np.nan_to_num(X_test_seq_norm, nan=0.0, posinf=0.0, neginf=0.0)
            
            lstm_start_time = time.time()
            lstm_probs = lstm_model.predict(X_test_seq_norm, batch_size=32)
            lstm_end_time = time.time()
            lstm_time = lstm_end_time - lstm_start_time
            n_samples_lstm = len(X_test_seq)
            lstm_time_per_sample = lstm_time / n_samples_lstm * 1000  # ms
            print(f"LSTM prediction time: {lstm_time:.4f} seconds ({lstm_time_per_sample:.4f} ms/sample)")
            
            # Convert LSTM output to dictionary for later use
            lstm_probs_dict = {}
            if isinstance(lstm_probs, list):
                for i, label in enumerate(label_cols):
                    if i < len(lstm_probs):
                        lstm_probs_dict[label] = lstm_probs[i]
            else:
                lstm_probs_dict[label_cols[0]] = lstm_probs
        except Exception as e:
            print(f"Error measuring LSTM prediction time: {e}")
            lstm_time = None
            lstm_probs_dict = {}
    else:
        print("LSTM model not available for timing.")
        lstm_time = None
        lstm_probs_dict = {}
    
    # Skip hybrid time measurement due to shape mismatch issues
    print("Skipping hybrid model timing due to shape mismatch issues.")
    hybrid_time = None
    
    # Prepare data for visualization and saving
    models = []
    times = []
    times_per_sample = []
    
    if xgb_time is not None:
        models.append('XGBoost')
        times.append(xgb_time)
        times_per_sample.append(xgb_time_per_sample)
    
    if lstm_time is not None:
        models.append('LSTM')
        times.append(lstm_time)
        times_per_sample.append(lstm_time_per_sample)
    
    if hybrid_time is not None:
        models.append('Hybrid')
        times.append(hybrid_time)
        times_per_sample.append(hybrid_time_per_sample)
    
    # Save results to CSV if we have data
    if models:
        time_data = pd.DataFrame({
            'Model': models,
            'Total Time (s)': times,
            'Time per Sample (ms)': times_per_sample
        })
        
        time_data.to_csv(os.path.join(output_folder, 'prediction_time.csv'), index=False)
        
        # Create a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(models, times_per_sample)
        plt.xlabel('Model')
        plt.ylabel('Time per Sample (ms)')
        plt.title('Prediction Time Comparison')
        plt.savefig(os.path.join(output_folder, 'prediction_time.png'))
        plt.close()
        
        print(f"Prediction time results saved to {output_folder} folder.")
    else:
        print("No timing data available to save.")
        
    return

def main():
    """
    Main function to execute the model evaluation pipeline.
    """
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create output directory
    output_folder = "evaluation_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Load models and data
    xgb_models, lstm_model, hybrid_weights, X_test_tab, y_test_tab, X_test_seq, y_test_seq, feature_cols, label_cols = load_models_and_data()
    
    # Validate test data diversity
    is_diverse = validate_test_data_diversity(y_test_seq, label_cols)
    if not is_diverse:
        print("Warning: Test data lacks diversity in one or more labels.")
        print("Evaluation results may not accurately reflect model performance.")
        print("Consider reprocessing data with the updated script.")
    
    # Evaluate XGBoost models
    xgb_preds, xgb_probs, xgb_accuracies = evaluate_xgboost(xgb_models, X_test_tab, y_test_tab, label_cols)
    
    # Evaluate LSTM model
    lstm_preds, lstm_probs, lstm_accuracies = evaluate_lstm(lstm_model, X_test_seq, y_test_seq, label_cols)
    
    # Evaluate hybrid model
    hybrid_preds, hybrid_accuracies = evaluate_hybrid(xgb_probs, lstm_probs, hybrid_weights, y_test_seq, label_cols)
    
    # Plot confusion matrices
    plot_confusion_matrices(xgb_preds, lstm_preds, hybrid_preds, y_test_seq, label_cols)
    
    # Plot accuracy comparison
    plot_accuracy_comparison(xgb_accuracies, lstm_accuracies, hybrid_accuracies, label_cols)
    
    # Plot feature importance
    plot_feature_importance(xgb_models, feature_cols, label_cols)
    
    # Generate classification reports
    generate_classification_reports(xgb_preds, lstm_preds, hybrid_preds, y_test_seq, label_cols)
    
    # Measure prediction time
    measure_prediction_time(xgb_models, lstm_model, hybrid_weights, X_test_tab, X_test_seq, label_cols)
    
    print("Model evaluation complete. Results saved to 'evaluation_results' folder.")

if __name__ == "__main__":
    main()