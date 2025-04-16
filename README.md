# Nuclear Reactor Simulation Data Analysis

This project analyzes time-series data from nuclear reactor simulations to predict component degradation levels using machine learning techniques.

## Project Structure

The project is divided into three main components:

1. **Data Processing Pipeline** (`1_data_processing.py`)
   - Reads all CSV files from the "TrainingData" folder
   - Handles missing values and performs data type conversions
   - Engineers additional features for better model performance
   - Balances classes using SMOTE
   - Creates sequences for time-series modeling
   - Splits data into training, validation, and test sets

2. **Model Training** (`2_model_training.py`)
   - Trains XGBoost models for tabular data
   - Trains LSTM models for sequential time-series data
   - Creates a hybrid model combining XGBoost and LSTM predictions
   - Performs hyperparameter optimization

3. **Model Evaluation** (`3_model_evaluation.py`)
   - Evaluates all models on test data
   - Generates confusion matrices and classification reports
   - Compares model performance
   - Visualizes feature importance (for XGBoost)
   - Measures prediction time

## Dataset

The dataset consists of 65 CSV files, each representing a distinct nuclear reactor simulation scenario. Each file contains:
- Time column (timestamps)
- 53 feature columns (already scaled/normalized)
- 5 label columns (categorical degradation levels for 5 components, integers between 0-5)

## Requirements

- Python 3.9.6
- Required packages are listed in `requirements.txt`

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts in sequence:

1. Data processing:
```bash
python 1_data_processing.py
```

2. Model training (choose one):
```bash
python 2_model_training.py  # Train all models
python train_lstm_only.py   # Train only LSTM (faster)
```

3. Model evaluation (choose one):
```bash
python 3_model_evaluation.py          # Basic evaluation
python comprehensive_evaluation.py     # Detailed evaluation of all models
python optimize_hybrid_weights.py      # Optimize and evaluate hybrid model weights
```

## Results

After running all scripts, you'll find:
- Processed data in the `processed_data` folder
- Trained models in the `models` folder
- Evaluation results in the `evaluation_results` folder, including:
  - Confusion matrices
  - Accuracy comparisons
  - Feature importance plots
  - Classification reports
  - Prediction time measurements
