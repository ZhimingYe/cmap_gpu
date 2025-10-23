# %%
import cupy as cp
import cudf
import numpy as np
import pandas as pd
from cuml.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Union
import warnings

def scale_data_by_column(train_set, test_set):
    """
    Perform min-max normalization on the training and test sets by column (only process numeric columns, skip 'label' column)
    """
    # Make sure to use pandas for processing
    if isinstance(train_set, cudf.DataFrame):
        tmp_train = train_set.to_pandas()
        tmp_test = test_set.to_pandas()
    else:
        tmp_train = train_set.copy()
        tmp_test = test_set.copy()
    
    # Normalize only columns that are not 'label'
    feature_cols = [col for col in tmp_train.columns if col != 'label']
    
    for col in feature_cols:
        # Force conversion to float type
        tmp_train[col] = pd.to_numeric(tmp_train[col], errors='coerce').astype(np.float32)
        tmp_test[col] = pd.to_numeric(tmp_test[col], errors='coerce').astype(np.float32)
        
        min_val = tmp_train[col].min()
        max_val = tmp_train[col].max()
        
        if max_val - min_val != 0:
            # Min-Max normalization: (x - min) / (max - min)
            tmp_train[col] = (tmp_train[col] - min_val) / (max_val - min_val)
            tmp_test[col] = (tmp_test[col] - min_val) / (max_val - min_val)
        else:
            # If the column is a constant, set to 0
            tmp_train[col] = 0.0
            tmp_test[col] = 0.0
    
    return [tmp_train, tmp_test]


def tune_svm_parameters(train_set, test_set, scale=True, class_weight=True,
                       kernel='rbf', verbose=False, cross_para=[4, 6, 8, 10]):
    """
    SVM parameter tuning (GPU acceleration)
    """
    parameter = {}
    
    # Data normalization
    if scale:
        scale_data = scale_data_by_column(train_set, test_set)
        tmp_train = scale_data[0]
        tmp_test = scale_data[1]
    else:
        print("Please set Scale TRUE")
        tmp_train = train_set.copy()
        tmp_test = test_set.copy()
    
    # Label encoding (convert string labels to numeric)
    label_encoder = LabelEncoder()
    original_labels = tmp_train['label'].values
    encoded_labels = label_encoder.fit_transform(original_labels)
    
    # Calculate class weights
    if class_weight:
        unique_labels, label_counts = np.unique(encoded_labels, return_counts=True)
        num_samples = len(encoded_labels)
        num_classes = len(unique_labels)
        wts = {}
        for label, count in zip(unique_labels, label_counts):
            wts[int(label)] = num_samples / (count * num_classes)
        
        if verbose and num_classes > 2:
            print("Note: cuML SVC ignores sample_weight for multi-class classification")
    else:
        wts = None
    
    # Define parameter search range (the same as R code)
    cost_range = [2**i for i in range(-5, 16, 2)]
    gamma_range = [2**i for i in range(-15, 4, 2)]
    
    # Prepare features - ensure all columns are float32 type
    feature_cols = [col for col in tmp_train.columns if col != 'label']
    X_train_pd = tmp_train[feature_cols].copy()
    
    # Ensure all feature columns are float32
    for col in X_train_pd.columns:
        X_train_pd[col] = X_train_pd[col].astype(np.float32)
    
    y_train = encoded_labels.astype(np.int32)
    
    # Tune parameters for each cross-validation fold number
    for cross_i in cross_para:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting grid search with {cross_i}-fold cross-validation")
            print(f"Total combinations: {len(cost_range)} x {len(gamma_range)} = {len(cost_range) * len(gamma_range)}")
            print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
            print(f"{'='*60}")
        
        np.random.seed(321)
        
        best_score = -np.inf
        best_cost = None
        best_gamma = None
        
        total_combinations = len(cost_range) * len(gamma_range)
        current_combination = 0
        
        # Grid search
        for cost in cost_range:
            for gamma in gamma_range:
                current_combination += 1
                
                try:
                    cv_scores = []
                    
                    # K-fold cross-validation
                    skf = StratifiedKFold(n_splits=cross_i, shuffle=True, random_state=321)
                    
                    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_pd, y_train)):
                        # Get training and validation data
                        X_tr_pd = X_train_pd.iloc[train_idx].copy()
                        X_val_pd = X_train_pd.iloc[val_idx].copy()
                        y_tr = y_train[train_idx]
                        y_val = y_train[val_idx]
                        
                        # Ensure types are correct again
                        for col in X_tr_pd.columns:
                            X_tr_pd[col] = X_tr_pd[col].astype(np.float32)
                            X_val_pd[col] = X_val_pd[col].astype(np.float32)
                        
                        # Convert to cudf - explicitly specify dtype
                        X_tr = cudf.DataFrame()
                        for col in X_tr_pd.columns:
                            X_tr[col] = cudf.Series(X_tr_pd[col].values, dtype=np.float32)
                        
                        X_val = cudf.DataFrame()
                        for col in X_val_pd.columns:
                            X_val[col] = cudf.Series(X_val_pd[col].values, dtype=np.float32)
                        
                        # Ensure y is also the correct type
                        y_tr = y_tr.astype(np.int32)
                        y_val = y_val.astype(np.int32)
                        
                        # Train GPU-accelerated SVM model
                        model = SVC(
                            C=float(cost),
                            gamma=float(gamma),
                            kernel=kernel,
                            probability=False
                        )
                        
                        model.fit(X_tr, y_tr)
                        
                        # Compute validation accuracy
                        score = model.score(X_val, y_val)
                        cv_scores.append(float(score))
                    
                    # Calculate mean accuracy
                    avg_score = np.mean(cv_scores)
                    
                    # Update best parameters
                    if avg_score > best_score:
                        best_score = avg_score
                        best_cost = cost
                        best_gamma = gamma
                        
                        if verbose:
                            print(f"[{current_combination}/{total_combinations}] ✓ New best: C={cost:.6f}, gamma={gamma:.6f}, CV score={avg_score:.4f}")
                
                except Exception as e:
                    if verbose:
                        print(f"[{current_combination}/{total_combinations}] ✗ Failed: C={cost:.6f}, gamma={gamma:.6f}, Error: {str(e)}")
                    continue
        
        # Save optimal parameters
        parameter[f'cross_{cross_i}'] = {
            'cost': best_cost,
            'gamma': best_gamma,
            'score': best_score
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Best parameters for {cross_i}-fold CV:")
            print(f"  Cost: {best_cost}")
            print(f"  Gamma: {best_gamma}")
            print(f"  CV Score: {best_score:.4f}")
            print(f"{'='*60}\n")
    
    return parameter


def PredictDomain(train_set, test_set, scale=True, class_weight=True,
                 cost=1, gamma=None, kernel='rbf', st_svm=False, verbose=False):
    """
    Domain prediction using SVM (GPU acceleration)
    """
    # Data normalization
    if scale:
        scale_data = scale_data_by_column(train_set, test_set)
        tmp_train = scale_data[0]
        tmp_test = scale_data[1]
    else:
        tmp_train = train_set.copy()
        tmp_test = test_set.copy()
    
    # Label encoding
    label_encoder = LabelEncoder()
    original_train_labels = tmp_train['label'].values
    encoded_train_labels = label_encoder.fit_transform(original_train_labels)
    
    # Prepare features
    feature_cols = [col for col in tmp_train.columns if col != 'label']
    X_train_pd = tmp_train[feature_cols].copy()
    X_test_pd = tmp_test[feature_cols].copy() if 'label' not in tmp_test.columns else tmp_test[feature_cols].copy()
    
    # Ensure all features are float32
    for col in feature_cols:
        X_train_pd[col] = X_train_pd[col].astype(np.float32)
        X_test_pd[col] = X_test_pd[col].astype(np.float32)
    
    # Convert to cudf
    X_train = cudf.DataFrame()
    for col in X_train_pd.columns:
        X_train[col] = cudf.Series(X_train_pd[col].values, dtype=np.float32)
    
    X_test = cudf.DataFrame()
    for col in X_test_pd.columns:
        X_test[col] = cudf.Series(X_test_pd[col].values, dtype=np.float32)
    
    y_train = encoded_train_labels.astype(np.int32)
    
    # Set default gamma value
    if gamma is None:
        gamma = 1.0 / len(feature_cols)
    
    # Train GPU-accelerated SVM model
    np.random.seed(321)
    svmfit = SVC(
        C=float(cost),
        gamma=float(gamma),
        kernel=kernel,
        probability=False
    )
    
    svmfit.fit(X_train, y_train)
    
    # Make predictions (encoded labels)
    pred_st_svm_encoded = svmfit.predict(X_train)
    pred_sc_svm_encoded = svmfit.predict(X_test)
    
    # Convert to numpy array
    if hasattr(pred_st_svm_encoded, 'to_numpy'):
        pred_st_svm_encoded = pred_st_svm_encoded.to_numpy()
    elif hasattr(pred_st_svm_encoded, 'values'):
        pred_st_svm_encoded = pred_st_svm_encoded.values
    elif isinstance(pred_st_svm_encoded, cp.ndarray):
        pred_st_svm_encoded = cp.asnumpy(pred_st_svm_encoded)
    
    pred_st_svm_encoded = np.array(pred_st_svm_encoded, dtype=np.int32)
    
    if hasattr(pred_sc_svm_encoded, 'to_numpy'):
        pred_sc_svm_encoded = pred_sc_svm_encoded.to_numpy()
    elif hasattr(pred_sc_svm_encoded, 'values'):
        pred_sc_svm_encoded = pred_sc_svm_encoded.values
    elif isinstance(pred_sc_svm_encoded, cp.ndarray):
        pred_sc_svm_encoded = cp.asnumpy(pred_sc_svm_encoded)
    
    pred_sc_svm_encoded = np.array(pred_sc_svm_encoded, dtype=np.int32)
    
    # Decode to original labels
    pred_st_svm_np = label_encoder.inverse_transform(pred_st_svm_encoded)
    pred_sc_svm_np = label_encoder.inverse_transform(pred_sc_svm_encoded)
    
    # Detailed output
    if verbose:
        print("\n" + "="*60)
        print("Prediction Results")
        print("="*60)
        
        print("\nThe prediction table of training data:")
        unique, counts = np.unique(pred_st_svm_np, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")
        
        # Calculate training accuracy
        train_accuracy = np.mean(pred_st_svm_np == original_train_labels)
        print(f"\nThe accuracy of spatial data: {train_accuracy:.6f}")
        
        print("\nThe prediction table of test data:")
        unique, counts = np.unique(pred_sc_svm_np, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")
        
        # If the test set has labels, also calculate the accuracy
        if 'label' in tmp_test.columns:
            test_labels = tmp_test['label'].values
            test_accuracy = np.mean(pred_sc_svm_np == test_labels)
            print(f"\nThe accuracy of test data: {test_accuracy:.6f}")
        
        print("="*60 + "\n")
    
    # Return result
    if st_svm:
        return pred_st_svm_np
    else:
        return pred_sc_svm_np


