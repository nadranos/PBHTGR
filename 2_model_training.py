#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclear Reactor Model Training and Cross-Validation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from tqdm import tqdm
import time

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'tensorflow',
        'xgboost', 'sklearn', 'keras_tuner', 'imblearn', 'joblib'
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

def check_gpu():
    """
    Check if GPU is available for TensorFlow.
    """
    print("Checking GPU availability...")
    
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available. TensorFlow will use GPU acceleration.")
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Limit TensorFlow memory usage
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Limit TensorFlow to use only 70% of GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 3)]
                    )
                print("GPU memory limited to 3GB")
            except RuntimeError as e:
                print(f"GPU memory limitation failed: {e}")
                
        return True
    else:
        print("No GPU found. TensorFlow will use CPU.")
        return False

def load_data(folder_path):
    """
    Load the processed data.
    
    Args:
        folder_path (str): Path to the folder containing processed data
        
    Returns:
        tuple: Loaded data for training, validation, and testing
    """
    print("Loading processed data...")
    
    # Load tabular data for XGBoost
    X_train_tab, y_train_tab = joblib.load(os.path.join(folder_path, 'tabular_train.pkl'))
    X_val_tab, y_val_tab = joblib.load(os.path.join(folder_path, 'tabular_val.pkl'))
    X_test_tab, y_test_tab = joblib.load(os.path.join(folder_path, 'tabular_test.pkl'))
    
    # Load sequence data for LSTM
    X_train_seq, y_train_seq = joblib.load(os.path.join(folder_path, 'sequence_train.pkl'))
    X_val_seq, y_val_seq = joblib.load(os.path.join(folder_path, 'sequence_val.pkl'))
    X_test_seq, y_test_seq = joblib.load(os.path.join(folder_path, 'sequence_test.pkl'))
    
    # Load feature and label column names
    feature_cols = joblib.load(os.path.join(folder_path, 'feature_cols.pkl'))
    label_cols = joblib.load(os.path.join(folder_path, 'label_cols.pkl'))
    
    print("Data loading complete.")
    print(f"Tabular data shapes - Train: {X_train_tab.shape}, Val: {X_val_tab.shape}, Test: {X_test_tab.shape}")
    print(f"Sequence data shapes - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    
    return (X_train_tab, y_train_tab, X_val_tab, y_val_tab, X_test_tab, y_test_tab,
            X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq,
            feature_cols, label_cols)

def train_xgboost(X_train, y_train, X_val, y_val, label_cols):
    """
    Train an XGBoost model for multi-output classification with memory optimization.
    
    Args:
        X_train (DataFrame): Training features
        y_train (DataFrame): Training labels
        X_val (DataFrame): Validation features
        y_val (DataFrame): Validation labels
        label_cols (list): List of label column names
        
    Returns:
        dict: Dictionary of trained XGBoost models
    """
    print("Training XGBoost models (memory-optimized)...")
    
    # We'll train one XGBoost model for each output label
    xgb_models = {}
    
    # Reduced parameters for grid search to save memory
    param_grid = {
        'max_depth': [3, 5],              # Reduced from [3, 5, 7]
        'learning_rate': [0.1],           # Only use 0.1 instead of [0.1, 0.01]
        'n_estimators': [100],            # Only use 100 instead of [100, 200]
        'subsample': [0.8],               # Only use 0.8 instead of [0.8, 1.0]
        'colsample_bytree': [0.8]         # Only use 0.8 instead of [0.8, 1.0]
    }
    
    # Convert data to float32 to save memory if needed
    if X_train.values.dtype == np.float64:
        print("Converting training data to float32 to save memory")
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
    
    # Train a model for each label
    for i, label in enumerate(label_cols):
        print(f"\nTraining XGBoost model for {label}...")
        
        # Get the number of classes for this label
        num_classes = len(np.unique(y_train[label]))
        print(f"Number of classes for {label}: {num_classes}")
        
        # Use a memory-optimized approach for XGBoost
        # Initialize XGBoost classifier with tree_method=hist for better memory efficiency
        if num_classes <= 2:
            xgb_clf = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist',        # Use histogram-based algorithm (more memory efficient)
                grow_policy='lossguide',   # Memory efficient growth policy
                max_bin=256               # Reduced from default for memory efficiency
            )
        else:
            xgb_clf = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                random_state=42,
                use_label_encoder=False, 
                eval_metric='mlogloss',
                tree_method='hist',        # Use histogram-based algorithm (more memory efficient)
                grow_policy='lossguide',   # Memory efficient growth policy
                max_bin=256               # Reduced from default for memory efficiency
            )
        
        # Create grid search with cross-validation
        # Use fewer workers (n_jobs=2) and pre_dispatch='2*n_jobs' to limit memory usage
        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            cv=3,
            n_jobs=2,             # Reduced from -1 (all cores) to limit memory usage
            pre_dispatch='2*n_jobs',  # Limit number of parallel jobs
            verbose=1,
            scoring='accuracy'
        )
        
        # Free some memory before fitting
        import gc
        gc.collect()
        
        # Fit grid search
        start_time = time.time()
        grid_search.fit(X_train, y_train[label])
        end_time = time.time()
        
        print(f"Grid search complete in {end_time - start_time:.2f} seconds.")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Clean up grid_search to free memory
        del grid_search
        gc.collect()
        
        # Evaluate on validation set
        y_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val[label], y_pred)
        print(f"Validation accuracy for {label}: {val_accuracy:.4f}")
        
        # Store the model
        xgb_models[label] = best_model
        
        # Save model immediately to free memory later
        output_folder = "models"
        os.makedirs(output_folder, exist_ok=True)
        joblib.dump(best_model, os.path.join(output_folder, f'xgboost_model_{label}.pkl'))
    
    # Save the model dictionary
    joblib.dump(xgb_models, os.path.join(output_folder, 'xgboost_models.pkl'))
    
    print("\nXGBoost training complete. Models saved to 'models/xgboost_models.pkl'")
    
    return xgb_models

def build_lstm_model(input_shape, num_classes_per_label):
    """
    Build an LSTM model for sequence data.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, features)
        num_classes_per_label (list): Number of classes for each label
        
    Returns:
        Model: Compiled Keras LSTM model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer with Layer Normalization and Dropout
    x = LSTM(64, return_sequences=True)(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Second LSTM layer
    x = LSTM(64)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Dense layer
    x = Dense(32, activation='relu')(x)
    
    # Output layers (one for each label)
    outputs = []
    for num_classes in num_classes_per_label:
        output = Dense(num_classes, activation='softmax')(x)
        outputs.append(output)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with Adam optimizer and categorical crossentropy loss
    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    
    loss_functions = ['categorical_crossentropy'] * len(outputs)
    loss_weights = [1.0] * len(outputs)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_functions,
        loss_weights=loss_weights,
        metrics=['accuracy']
    )
    
    return model

class LSTMHyperModel(kt.HyperModel):
    """
    Memory-optimized hypermodel class for Keras Tuner to optimize LSTM model hyperparameters.
    """
    def __init__(self, input_shape, num_classes_per_label):
        self.input_shape = input_shape
        self.num_classes_per_label = num_classes_per_label
    
    def build(self, hp):
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # First LSTM layer - reduced parameter range
        lstm_units_1 = hp.Int('lstm_units_1', min_value=16, max_value=64, step=16)  # Reduced range
        x = LSTM(lstm_units_1, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.3, step=0.1))(x)
        
        # Second LSTM layer - reduced parameter range
        lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=64, step=16)  # Reduced range
        x = LSTM(lstm_units_2)(x)
        x = LayerNormalization()(x)
        x = Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.3, step=0.1))(x)
        
        # Dense layer - reduced parameter range
        dense_units = hp.Int('dense_units', min_value=16, max_value=32, step=16)  # Reduced range
        x = Dense(dense_units, activation='relu')(x)
        
        # Output layers (one for each label)
        outputs = []
        for i, num_classes in enumerate(self.num_classes_per_label):
            output = Dense(num_classes, activation='softmax', name=f'output_{i}')(x)
            outputs.append(output)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with Adam optimizer and categorical crossentropy loss
        # Using fixed learning rate to reduce hyperparameter search space
        learning_rate = 0.001
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        
        loss_functions = ['categorical_crossentropy'] * len(outputs)
        loss_weights = [1.0] * len(outputs)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=['accuracy']
        )
        
        return model

def train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, label_cols):
    """
    Train an LSTM model for multi-output classification with memory optimization.
    
    Args:
        X_train_seq (array): Training sequence data
        y_train_seq (array): Training labels
        X_val_seq (array): Validation sequence data
        y_val_seq (array): Validation labels
        label_cols (list): List of label column names
        
    Returns:
        Model: Trained LSTM model
    """
    print("Training LSTM model (memory-optimized)...")
    
    # Force using float32 to save memory
    if X_train_seq.dtype == np.float64:
        print("Converting sequence data to float32 to save memory")
        X_train_seq = X_train_seq.astype(np.float32)
        X_val_seq = X_val_seq.astype(np.float32)
    
    # Normalize features to prevent numerical instability
    print("Normalizing features to improve LSTM performance...")
    
    # Calculate mean and std per feature
    feature_means = np.mean(X_train_seq, axis=(0, 1), keepdims=True)
    feature_stds = np.std(X_train_seq, axis=(0, 1), keepdims=True) + 1e-8  # Add epsilon to prevent division by zero
    
    # Apply normalization
    X_train_seq = (X_train_seq - feature_means) / feature_stds
    X_val_seq = (X_val_seq - feature_means) / feature_stds
    
    # Replace NaN and inf values if any
    X_train_seq = np.nan_to_num(X_train_seq, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_seq = np.nan_to_num(X_val_seq, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get input shape
    seq_length, num_features = X_train_seq.shape[1], X_train_seq.shape[2]
    print(f"Sequence length: {seq_length}, Number of features: {num_features}")
    
    # Print some statistics about the training labels
    print("Training label distribution statistics:")
    for i, label in enumerate(label_cols):
        unique_vals, counts = np.unique(y_train_seq[:, i], return_counts=True)
        print(f"{label}: {dict(zip(unique_vals, counts))}")
    
    # Print validation label distribution
    print("\nValidation label distribution statistics:")
    for i, label in enumerate(label_cols):
        unique_vals, counts = np.unique(y_val_seq[:, i], return_counts=True)
        print(f"{label}: {dict(zip(unique_vals, counts))}")
    
    # Fix the number of classes to exactly 6 (0-5) for all labels
    # This ensures consistency across datasets
    num_classes = 6
    print(f"Using fixed {num_classes} classes (0-5) for all labels")
    
    # Convert labels to one-hot encoding with exactly 6 classes
    y_train_seq_onehot = []
    for i in range(y_train_seq.shape[1]):
        y_train_seq_onehot.append(tf.keras.utils.to_categorical(
            y_train_seq[:, i], num_classes=num_classes))
        print(f"One-hot encoded shape for label {i}: {y_train_seq_onehot[i].shape}")
    
    y_val_seq_onehot = []
    for i in range(y_val_seq.shape[1]):
        y_val_seq_onehot.append(tf.keras.utils.to_categorical(
            y_val_seq[:, i], num_classes=num_classes))
    
    # Check GPU availability and set memory limits
    gpu_available = check_gpu()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Skip hyperparameter tuning to save memory - use predefined model
    print("Using improved LSTM model architecture with regularization...")
    
    # Input layer with normalization
    inputs = Input(shape=(seq_length, num_features))
    
    # First LSTM layer with regularization
    x = LSTM(32, return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(1e-5),
             recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
             bias_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Second LSTM layer
    x = LSTM(16, 
             kernel_regularizer=tf.keras.regularizers.l2(1e-5),
             recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
             bias_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output branches - one per label with a small dense layer before each output
    outputs = []
    for i in range(len(label_cols)):
        # Add a small hidden layer for each output branch to help with feature extraction
        hidden = Dense(8, activation='relu', 
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                      name=f'hidden_{i}')(x)
        output = Dense(num_classes, activation='softmax', name=f'output_{i}')(hidden)
        outputs.append(output)
    
    # Create model
    best_model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with Adam optimizer and categorical crossentropy loss
    # Use a reduced learning rate and gradient clipping for stability
    optimizer = Adam(learning_rate=0.0005, clipnorm=0.5)
    loss_functions = ['categorical_crossentropy'] * len(outputs)
    loss_weights = [1.0] * len(outputs)
    
    # Need to provide a list of metrics for each output
    metrics = [['accuracy'] for _ in range(len(outputs))]
    
    best_model.compile(
        optimizer=optimizer,
        loss=loss_functions,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    # Define early stopping with longer patience to allow better convergence
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    # Use reduced batch size to improve learning dynamics
    batch_size = 16
    
    # Use enough epochs to allow convergence
    max_epochs = 50
    
    # Train the model
    print("Training the LSTM model...")
    history = best_model.fit(
        X_train_seq,
        y_train_seq_onehot,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_val_seq, y_val_seq_onehot),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Saving plot uses less memory with reduced figure size
    plt.figure(figsize=(8, 6))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy (average over all outputs)
    plt.subplot(2, 1, 2)
    acc_metrics = [metric for metric in history.history.keys() if 'accuracy' in metric and 'val' not in metric]
    val_acc_metrics = [metric for metric in history.history.keys() if 'val' in metric and 'accuracy' in metric]
    
    if acc_metrics and val_acc_metrics:  # Only calculate if metrics exist
        avg_acc = np.mean([history.history[metric] for metric in acc_metrics], axis=0)
        avg_val_acc = np.mean([history.history[metric] for metric in val_acc_metrics], axis=0)
        
        plt.plot(avg_acc, label='Training Accuracy')
        plt.plot(avg_val_acc, label='Validation Accuracy')
        plt.title('LSTM Model Accuracy (Average)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('lstm_training_history.png', dpi=100)  # Lower DPI to save memory
    plt.close()
    
    # Force garbage collection
    gc.collect()
    
    # Save the model
    output_folder = "models"
    os.makedirs(output_folder, exist_ok=True)
    best_model.save(os.path.join(output_folder, 'lstm_model'))
    
    print("LSTM training complete. Model saved to 'models/lstm_model'")
    
    # Evaluate on validation set and print per-output accuracy
    print("Validation accuracies:")
    
    # Use smaller batch size for prediction to save memory
    val_pred = best_model.predict(X_val_seq, batch_size=32)
    
    for i, label in enumerate(label_cols):
        val_pred_classes = np.argmax(val_pred[i], axis=1)
        val_true_classes = np.argmax(y_val_seq_onehot[i], axis=1)
        val_accuracy = accuracy_score(val_true_classes, val_pred_classes)
        print(f"  {label}: {val_accuracy:.4f}")
    
    return best_model

def create_hybrid_model(xgb_models, lstm_model, X_val_tab, X_val_seq, y_val, label_cols):
    """
    Create a hybrid model by combining XGBoost and LSTM predictions with memory optimization.
    
    Args:
        xgb_models (dict): Dictionary of trained XGBoost models
        lstm_model (Model): Trained LSTM model
        X_val_tab (DataFrame): Validation tabular data
        X_val_seq (array): Validation sequence data
        y_val (DataFrame): Validation labels
        label_cols (list): List of label column names
        
    Returns:
        dict: Dictionary containing hybrid model weights
    """
    print("Creating hybrid model (memory-optimized)...")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Make sure all data is float32 to save memory
    if X_val_tab.values.dtype == np.float64:
        print("Converting validation data to float32 to save memory")
        X_val_tab = X_val_tab.astype(np.float32)
    
    if X_val_seq.dtype == np.float64:
        print("Converting sequence validation data to float32 to save memory")
        X_val_seq = X_val_seq.astype(np.float32)
    
    # Process each label separately and free memory between labels
    hybrid_weights = {}
    
    # Use fewer weight steps to save memory and computation
    weight_steps = np.arange(0, 1.01, 0.1)  # Changed from 0.05 to 0.1
    
    for i, label in enumerate(label_cols):
        print(f"Processing hybrid weights for {label} ({i+1}/{len(label_cols)})...")
        
        # Get predictions from XGBoost model for this label
        print(f"Getting XGBoost predictions for {label}...")
        xgb_pred = xgb_models[label].predict_proba(X_val_tab)
        
        # Get predictions from LSTM model in batches to save memory
        print(f"Getting LSTM predictions for {label} with batch processing...")
        lstm_pred = lstm_model.predict(X_val_seq, batch_size=32)[i]
        
        # Get ground truth label
        y_true_label = y_val[label].values
        
        print(f"Finding optimal weight for {label}...")
        best_accuracy = 0
        best_weight = 0.5
        
        # Grid search for weights with fewer steps
        for weight in weight_steps:
            # Combine predictions with current weight
            combined_probs = weight * xgb_pred + (1 - weight) * lstm_pred
            combined_preds = np.argmax(combined_probs, axis=1)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_true_label, combined_preds)
            
            # Update best weight if accuracy improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = weight
        
        print(f"Best weight for {label}: {best_weight:.2f} (Accuracy: {best_accuracy:.4f})")
        hybrid_weights[label] = best_weight
        
        # Free memory
        del xgb_pred, lstm_pred
        gc.collect()
    
    # Save hybrid model weights
    output_folder = "models"
    os.makedirs(output_folder, exist_ok=True)
    joblib.dump(hybrid_weights, os.path.join(output_folder, 'hybrid_weights.pkl'))
    
    print("Hybrid model creation complete. Weights saved to 'models/hybrid_weights.pkl'")
    
    return hybrid_weights

def main():
    """
    Main function to execute the model training pipeline with memory optimization.
    """
    # Import gc at the beginning for memory management
    import gc
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Set environment variables for TensorFlow memory optimization
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # More efficient memory allocator
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'      # Prevent TensorFlow from allocating all GPU memory
    
    # Load processed data
    data_folder = "processed_data"
    (X_train_tab, y_train_tab, X_val_tab, y_val_tab, X_test_tab, y_test_tab,
     X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq,
     feature_cols, label_cols) = load_data(data_folder)
    
    # Free memory of test data until needed for evaluation
    del X_test_tab, y_test_tab, X_test_seq, y_test_seq
    gc.collect()
    
    # Train XGBoost model
    xgb_models = train_xgboost(X_train_tab, y_train_tab, X_val_tab, y_val_tab, label_cols)
    
    # Force garbage collection before LSTM training
    gc.collect()
    
    # Train LSTM model
    lstm_model = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, label_cols)
    
    # Force garbage collection before hybrid model creation
    gc.collect()
    
    # Create hybrid model
    hybrid_weights = create_hybrid_model(xgb_models, lstm_model, X_val_tab, X_val_seq, y_val_tab, label_cols)
    
    print("Model training complete. All models saved to the 'models' directory.")

if __name__ == "__main__":
    main()