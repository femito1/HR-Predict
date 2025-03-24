import torch
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import os
import pickle


class PyTorchGradientBoosting:
    """
    A PyTorch-based implementation of Gradient Boosting with Decision Trees
    """
    
    def __init__(self, n_estimators=500, learning_rate=0.1, max_depth=6, 
                 subsample=1.0, random_state=None, reg_lambda=1.0, reg_alpha=0.1,
                 lr_schedule=None, objective='binary:logistic', class_weight=None):
        """
        Initialize the gradient boosting model

        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages/trees to be used
        learning_rate : float
            Step size shrinkage used to prevent overfitting
        max_depth : int
            Maximum depth of individual regression trees
        subsample : float
            Fraction of samples to be used for fitting the individual trees
        random_state : int
            Random seed for reproducibility
        reg_lambda : float
            L2 regularization term on weights
        reg_alpha : float
            L1 regularization term on weights
        lr_schedule : str or callable, optional
            Learning rate schedule, one of:
            - 'constant': No change to learning rate
            - 'linear_decay': Linearly decrease learning rate to 1/10 of initial value
            - 'exponential_decay': Exponentially decrease learning rate
            - callable: Custom function that takes (iteration, total_iterations) and returns a multiplier
        objective : str
            Objective function to optimize. Supported values:
            - 'binary:logistic': Binary classification with logistic loss
            - 'reg:squarederror': Regression with squared error
            - 'binary:hinge': Binary classification with hinge loss
        class_weight : str or dict, optional
            Class weight for balancing the data
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.lr_schedule = lr_schedule
        self.objective = objective
        self.class_weight = class_weight
        
        self.trees = []
        self.feature_importances_ = None
        self.base_score = None
        
        # Set random seed for PyTorch
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
    
    def _gradient(self, y_true, y_pred):
        """Calculate gradient of loss function w.r.t predicted values"""
        if self.objective == 'binary:logistic':
            return y_pred - y_true
    
    def _hessian(self, y_pred):
        """Calculate hessian of loss function w.r.t predicted values"""
        if self.objective == 'binary:logistic':
            return y_pred * (1 - y_pred)

    def _get_lr_multiplier(self, iteration, total_iterations):
        """Get learning rate multiplier based on schedule"""
        if self.lr_schedule is None or self.lr_schedule == 'constant':
            return 1.0
        
        if callable(self.lr_schedule):
            return self.lr_schedule(iteration, total_iterations)
        
        if self.lr_schedule == 'linear_decay':
            # Linearly decay to 1/10 of the initial learning rate
            return 1.0 - 0.9 * (iteration / total_iterations)
        
        if self.lr_schedule == 'exponential_decay':
            # Exponential decay
            decay_rate = 0.1  # End with 1/10 of initial learning rate
            return np.exp(np.log(decay_rate) * iteration / total_iterations)
        
        return 1.0  # Default fallback
    
    def _get_loss_function(self):
        """Return appropriate loss function based on objective"""
        if self.objective == 'binary:logistic':
            return torch.nn.BCELoss()
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=True, eval_metric=None):
        """
        Fit the gradient boosting model
        
        Parameters:
        -----------
        X : torch.Tensor
            Features of training data
        y : torch.Tensor
            Target values
        eval_set : tuple or list of tuples, optional
            (X_val, y_val) for tracking validation performance or
            [(X_val1, y_val1), (X_val2, y_val2), ...] for multiple validation sets
        early_stopping_rounds : int, optional
            Stops training if validation score doesn't improve for this many rounds
        verbose : bool
            Whether to print progress during training
        eval_metric : callable, optional
            Custom evaluation metric, should take (y_true, y_pred) and return a score
            where higher is better
        
        Returns:
        --------
        self : object
        """
        X = torch.as_tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        y = torch.as_tensor(y, dtype=torch.float32) if not isinstance(y, torch.Tensor) else y
        
        # Support for multiple validation sets
        validation_sets = []
        if eval_set is not None:
            if isinstance(eval_set[0], (list, tuple)) and isinstance(eval_set[0][0], (np.ndarray, torch.Tensor)):
                # Multiple validation sets
                for val_X, val_y in eval_set:
                    val_X = torch.as_tensor(val_X, dtype=torch.float32) if not isinstance(val_X, torch.Tensor) else val_X
                    val_y = torch.as_tensor(val_y, dtype=torch.float32) if not isinstance(val_y, torch.Tensor) else val_y
                    validation_sets.append((val_X, val_y))
            else:
                # Single validation set
                X_val, y_val = eval_set
                X_val = torch.as_tensor(X_val, dtype=torch.float32) if not isinstance(X_val, torch.Tensor) else X_val
                y_val = torch.as_tensor(y_val, dtype=torch.float32) if not isinstance(y_val, torch.Tensor) else y_val
                validation_sets.append((X_val, y_val))
        
        self.base_score = y.mean().item()
        y_pred = torch.full_like(y, self.base_score)
        
        # Create device to run computations on (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        y = y.to(device)
        y_pred = y_pred.to(device)
        
        # Move validation sets to device
        for i in range(len(validation_sets)):
            validation_sets[i] = (validation_sets[i][0].to(device), validation_sets[i][1].to(device))
        
        best_val_loss = float('inf')
        no_improve_rounds = 0
        train_losses = []
        val_losses = [[] for _ in range(len(validation_sets))]
        
        criterion = self._get_loss_function()
        
        # Apply class weighting if specified
        if self.class_weight == 'balanced':
            # Calculate balanced weights
            n_samples = len(y)
            n_classes = 2  # Binary classification
            classes = torch.unique(y)
            class_weights = torch.zeros(n_classes, device=y.device)
            
            for c in classes:
                class_weights[int(c)] = n_samples / (n_classes * (y == c).sum())
            
            # Create sample weights (1 for majority class, higher for minority class)
            sample_weights = torch.ones_like(y, dtype=torch.float32)
            for c in classes:
                sample_weights[y == c] = class_weights[int(c)]
        else:
            sample_weights = None
        
        # Main boosting loop
        for i in range(self.n_estimators):
            # Apply learning rate schedule
            current_lr = self.learning_rate * self._get_lr_multiplier(i, self.n_estimators)
            
            # Calculate gradients and hessians for current predictions
            with torch.no_grad():
                if self.objective == 'binary:logistic':
                    sig_preds = torch.sigmoid(y_pred)
                    gradients = self._gradient(y, sig_preds)
                    hessians = self._hessian(y)
                    
                    # Apply sample weights if they exist
                    if sample_weights is not None:
                        gradients = gradients * sample_weights
                        hessians = hessians * sample_weights
                else:
                    gradients = self._gradient(y, y_pred)
                    hessians = self._hessian(y)
            
            # Subsample the data if specified
            if self.subsample < 1.0:
                n_samples = int(self.subsample * len(X))
                # Use PyTorch's built-in random permutation
                subsample_indices = torch.randperm(len(X), device=device)[:n_samples]
                X_subset = X[subsample_indices]
                gradients_subset = gradients[subsample_indices]
                # Move to CPU for sklearn
                X_subset_np = X_subset.cpu().numpy() 
                gradients_subset_np = gradients_subset.cpu().numpy()
                weights = hessians[subsample_indices].cpu().numpy() + self.reg_lambda
            else:
                X_subset_np = X.cpu().numpy()
                gradients_subset_np = gradients.cpu().numpy()
                weights = hessians.cpu().numpy() + self.reg_lambda
            
            # Fit a regression tree to the negative gradients
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                random_state=self.random_state
            )
            # Fit to negative gradients (same as minimizing loss)
            tree.fit(X_subset_np, -gradients_subset_np, sample_weight=weights)
            
            # Add the tree to our ensemble
            self.trees.append(tree)
            
            # Update predictions with current learning rate
            X_np = X.cpu().numpy()
            tree_preds = torch.tensor(tree.predict(X_np), dtype=torch.float32, device=device)
            y_pred += current_lr * tree_preds
            
            # Calculate training loss
            with torch.no_grad():
                if self.objective == 'binary:logistic':
                    sig_preds = torch.sigmoid(y_pred)
                    train_loss = criterion(sig_preds, y).item()
                else:
                    train_loss = criterion(y_pred, y).item()
                train_losses.append(train_loss)
            
            # Evaluate on validation sets if provided
            if validation_sets:
                all_val_losses = []
                for i, (X_val, y_val) in enumerate(validation_sets):
                    with torch.no_grad():
                        val_raw_preds = self.predict_proba_raw(X_val)
                        val_sig_preds = torch.sigmoid(val_raw_preds)
                        val_loss = criterion(val_sig_preds, y_val).item()
                        val_losses[i].append(val_loss)
                        
                        # Custom evaluation metric if provided
                        if eval_metric is not None:
                            val_score = eval_metric(y_val.cpu().numpy(), val_sig_preds.cpu().numpy())
                            if verbose and (i + 1) % 10 == 0:
                                print(f"Validation set {i+1}: metric = {val_score:.6f}")
                
                    all_val_losses.append(val_loss)
                
                # Use first validation set for early stopping by default
                current_val_loss = all_val_losses[0]
                
                # Check for early stopping
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"Iteration {i+1}/{self.n_estimators}: Train Loss: {train_loss:.6f}")
                
                if early_stopping_rounds is not None and no_improve_rounds >= early_stopping_rounds:
                    if verbose:
                        print(f"Early stopping at iteration {i+1}")
                    # Remove the trees that didn't improve validation performance
                    self.trees = self.trees[:-no_improve_rounds]
                    break
            elif verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{self.n_estimators}: Train Loss: {train_loss:.6f}")
        
        # Calculate and store feature importances
        self._calculate_feature_importances()
        
        # Store loss history
        self.train_losses_ = train_losses
        if validation_sets:
            self.val_losses_ = val_losses
        
        return self
    
    def _calculate_feature_importances(self):
        """Calculate feature importances from the trained trees"""
        if not self.trees:
            return
            
        self.feature_importances_ = np.mean([
            tree.feature_importances_ for tree in self.trees
        ], axis=0)
    
    def predict_proba_raw(self, X):
        """Get raw predictions (before sigmoid)"""
        X = torch.as_tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        device = X.device
        
        y_pred = torch.full((len(X),), self.base_score, dtype=torch.float32, device=device)
        X_np = X.cpu().numpy()
        
        # Add predictions from each tree - more efficient with a sum
        tree_preds_sum = sum(
            self.learning_rate * torch.tensor(
                tree.predict(X_np), dtype=torch.float32, device=device
            ) for tree in self.trees
        )
        
        return y_pred + tree_preds_sum
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        raw_predictions = self.predict_proba_raw(X)
        return torch.sigmoid(raw_predictions)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        probabilities = self.predict_proba_calibrated(X)
        return (probabilities >= threshold).to(torch.int)
    
    def plot_learning_curve(self):
        """Plot the learning curves for training and validation"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses_, label='Training Loss')
        if hasattr(self, 'val_losses_'):
            for i, val_losses in enumerate(self.val_losses_):
                plt.plot(val_losses, label=f'Validation Loss (Set {i+1})')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Log Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_feature_importance(self, feature_names=None):
        """Plot feature importances"""
        if self.feature_importances_ is None:
            raise ValueError("Model has not been trained yet.")
        
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(len(self.feature_importances_))]
        
        # Use numpy's built-in argsort for getting sorted indices
        indices = np.argsort(self.feature_importances_)[::-1]
        sorted_importance = self.feature_importances_[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_importance)), sorted_importance, align='center')
        plt.yticks(range(len(sorted_importance)), sorted_names)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

    def cross_validate(self, X, y, n_splits=5, shuffle=True, random_state=None,
                     verbose=True, return_models=False):
        """
        Perform k-fold cross-validation
        
        Parameters:
        -----------
        X : numpy.ndarray or torch.Tensor
            Training data
        y : numpy.ndarray or torch.Tensor
            Target values
        n_splits : int
            Number of folds for cross-validation
        shuffle : bool
            Whether to shuffle the data before splitting
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        return_models : bool
            If True, returns the trained models for each fold
            
        Returns:
        --------
        scores : dict
            Dictionary containing validation metrics for each fold
        models : list, optional
            List of trained models if return_models=True
        """
        from sklearn.model_selection import StratifiedKFold
        import copy
        
        X = X.numpy() if isinstance(X, torch.Tensor) else X
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        fold_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'log_loss': []
        }
        
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            if verbose:
                print(f"\nFold {fold+1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create a fresh copy of the model for each fold
            model = copy.deepcopy(self)
            
            # Train the model
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=20,
                verbose=verbose
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val).cpu().numpy()
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            fold_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            fold_scores['precision'].append(precision_score(y_val, y_pred))
            fold_scores['recall'].append(recall_score(y_val, y_pred))
            fold_scores['f1'].append(f1_score(y_val, y_pred))
            fold_scores['auc'].append(roc_auc_score(y_val, y_pred_proba))
            fold_scores['log_loss'].append(log_loss(y_val, y_pred_proba))
            
            if verbose:
                print(f"Fold {fold+1} - Accuracy: {fold_scores['accuracy'][-1]:.4f}, "
                      f"AUC: {fold_scores['auc'][-1]:.4f}")
            
            if return_models:
                fold_models.append(model)
        
        # Calculate and display average scores
        if verbose:
            print("\nCross-Validation Results:")
            for metric, scores in fold_scores.items():
                print(f"{metric.capitalize()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        if return_models:
            return fold_scores, fold_models
        return fold_scores

    def evaluate(self, X, y, threshold=0.5, calibrate=False):
        """
        Evaluate model performance with multiple metrics
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            Test target values
        threshold : float
            Classification threshold
            
        Returns:
        --------
        metrics : dict
            Dictionary of performance metrics
        """
        if calibrate:
            y_pred_proba = self.predict_proba_calibrated(X).cpu().numpy()
        else:
            y_pred_proba = self.predict_proba(X).cpu().numpy()
        y_pred = (y_pred_proba >= threshold).astype(int)
        y_true = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba)
        }

    def plot_confusion_matrix(self, X, y, threshold=0.5, normalize=True, figsize=(10, 8)):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            Test target values
        threshold : float
            Classification threshold
        normalize : bool
            Whether to normalize the confusion matrix
        figsize : tuple
            Figure size
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        y_pred = self.predict(X, threshold=threshold).cpu().numpy()
        y_true = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                    cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, X, y, figsize=(10, 8)):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        y_pred_proba = self.predict_proba(X).cpu().numpy()
        y_true = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_precision_recall_curve(self, X, y, figsize=(10, 8)):
        """Plot Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        y_pred_proba = self.predict_proba(X).cpu().numpy()
        y_true = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.show()

    def calibrate_probabilities(self, X_cal, y_cal, method='isotonic'):
        """
        Calibrate the probability outputs using a held-out calibration set
        
        Parameters:
        -----------
        X_cal : array-like
            Calibration features
        y_cal : array-like
            Calibration target values
        method : str, default='isotonic'
            The method to use for calibration. Can be 'sigmoid' (Platt scaling) 
            or 'isotonic' (non-parametric isotonic regression)
            
        Returns:
        --------
        self : object
            Returns self
        """
        
        # Create a scikit-learn compatible wrapper for our model
        # Convert inputs to numpy if they're tensors
        X_cal_np = X_cal.cpu().numpy() if isinstance(X_cal, torch.Tensor) else X_cal
        y_cal_np = y_cal.cpu().numpy() if isinstance(y_cal, torch.Tensor) else y_cal
        
        # Ensure y is binary and consists of integers
        y_cal_np = y_cal_np.astype(int)
        
        # Create a direct calibration map instead of using CalibratedClassifierCV
        if method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            
            # Get raw predictions on calibration data
            X_tensor = torch.tensor(X_cal_np, dtype=torch.float32)
            raw_probs = self.predict_proba(X_tensor).cpu().numpy()
            
            # Fit isotonic regression directly
            self.calibrator.fit(raw_probs, y_cal_np)
            self.is_calibrated = True
            self.calibration_method = 'isotonic'
            
        elif method == 'sigmoid':
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression(C=1.0, solver='lbfgs')
            
            # Get raw predictions on calibration data
            X_tensor = torch.tensor(X_cal_np, dtype=torch.float32)
            raw_probs = self.predict_proba(X_tensor).cpu().numpy()
            
            # Reshape to 2D array expected by LogisticRegression
            raw_probs = raw_probs.reshape(-1, 1)
            
            # Fit logistic regression directly
            self.calibrator.fit(raw_probs, y_cal_np)
            self.is_calibrated = True
            self.calibration_method = 'sigmoid'
        
        return self

    def predict_proba_calibrated(self, X):
        """
        Get calibrated probability predictions
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        probabilities : ndarray
            Calibrated probabilities for the positive class
        """
        if not hasattr(self, 'calibrator') or not hasattr(self, 'is_calibrated'):
            raise ValueError("Model is not calibrated. Call calibrate_probabilities first.")
        
        # Convert to numpy if tensor
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        
        # Get raw probabilities from our model
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        raw_probs = self.predict_proba(X_tensor).cpu().numpy()
        
        # Apply calibration based on method
        if self.calibration_method == 'isotonic':
            # Isotonic regression takes raw probabilities directly
            calibrated_probs = self.calibrator.transform(raw_probs)
            
        elif self.calibration_method == 'sigmoid':
            # Logistic regression needs reshaped input
            raw_probs = raw_probs.reshape(-1, 1)
            calibrated_probs = self.calibrator.predict_proba(raw_probs)[:, 1]
        
        return torch.tensor(calibrated_probs, dtype=torch.float32)

    def plot_calibration_curve(self, X, y, n_bins=10, figsize=(10, 8)):
        """
        Plot the calibration curve to visualize probability calibration
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            True target values
        n_bins : int
            Number of bins for calibration curve
        figsize : tuple
            Figure size
        """
        if not hasattr(self, 'is_calibrated'):
            uncalibrated_probs = self.predict_proba(X).cpu().numpy()
            label_uncalibrated = 'Uncalibrated'
            
            plt.figure(figsize=figsize)
            prob_true, prob_pred = calibration_curve(y, uncalibrated_probs, n_bins=n_bins)
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=label_uncalibrated)
            
        else:
            # Get both uncalibrated and calibrated probabilities
            uncalibrated_probs = self.predict_proba(X).cpu().numpy()
            calibrated_probs = self.predict_proba_calibrated(X)
            
            plt.figure(figsize=figsize)
            
            # Plot uncalibrated curve
            prob_true_uncal, prob_pred_uncal = calibration_curve(y, uncalibrated_probs, n_bins=n_bins)
            plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', linewidth=1, label='Uncalibrated')
            
            # Plot calibrated curve
            prob_true_cal, prob_pred_cal = calibration_curve(y, calibrated_probs, n_bins=n_bins)
            plt.plot(prob_pred_cal, prob_true_cal, marker='o', linewidth=1, label='Calibrated')
        
        # Plot diagonal perfect calibration line
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()

    def save_model(self, filepath):
        """
        Save model to file including calibration
        
        Parameters:
        -----------
        filepath : str
            Path to save the model to
        """
        # We need to temporarily remove the calibrator as it may 
        # contain unpicklable objects
        if hasattr(self, 'calibrator'):
            calibrator = self.calibrator
            delattr(self, 'calibrator')
            is_calibrated = self.is_calibrated
            
            # Save the main model
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            # Save the calibrator separately
            calibrator_path = filepath + '.calibrator'
            joblib.dump(calibrator, calibrator_path)
            
            # Restore calibrator to model
            self.calibrator = calibrator
            self.is_calibrated = is_calibrated
        else:
            # Regular save
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath):
        """
        Load model from file including calibration if available
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
            
        Returns:
        --------
        model : PyTorchGradientBoosting
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        # Check if calibrator exists and load it
        calibrator_path = filepath + '.calibrator'
        if os.path.exists(calibrator_path):
            model.calibrator = joblib.load(calibrator_path)
            model.is_calibrated = True
        
        return model

    def calibrate_threshold(self, X_val, y_val, metric='f1', thresholds=None):
        """
        Find the optimal classification threshold based on a validation set
        
        Parameters:
        -----------
        X_val : array-like
            Validation features
        y_val : array-like
            Validation target values
        metric : str
            Metric to optimize: 'f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy'
        thresholds : array-like, optional
            Thresholds to try. Default is np.arange(0.01, 1.0, 0.01)
            
        Returns:
        --------
        best_threshold : float
            The threshold that maximizes the specified metric
        best_score : float
            The score achieved with the best threshold
        threshold_metrics : dict
            Dictionary with metrics at each threshold value
        """
        if thresholds is None:
            thresholds = np.arange(0.01, 1.0, 0.01)
        
        # Get probability predictions
        y_pred_proba = self.predict_proba(X_val).cpu().numpy()
        y_true = y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val
        
        # Dictionary to store scores for each metric
        threshold_metrics = {
            'thresholds': thresholds,
            'f1': [],
            'precision': [],
            'recall': [],
            'accuracy': [],
            'balanced_accuracy': []
        }
        
        # Calculate metrics for each threshold
        for t in thresholds:
            y_pred = (y_pred_proba >= t).astype(int)
            
            threshold_metrics['f1'].append(f1_score(y_true, y_pred))
            threshold_metrics['precision'].append(precision_score(y_true, y_pred))
            threshold_metrics['recall'].append(recall_score(y_true, y_pred))
            threshold_metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            threshold_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_true, y_pred))
        
        # Find the threshold that maximizes the chosen metric
        best_idx = np.argmax(threshold_metrics[metric])
        best_threshold = thresholds[best_idx]
        best_score = threshold_metrics[metric][best_idx]
        
        return best_threshold, best_score, threshold_metrics

    def plot_threshold_metrics(self, threshold_metrics):
        """Plot metrics across different threshold values"""
        metrics = ['f1', 'precision', 'recall', 'accuracy', 'balanced_accuracy']
        thresholds = threshold_metrics['thresholds']
        
        plt.figure(figsize=(12, 8))
        for metric in metrics:
            plt.plot(thresholds, threshold_metrics[metric], label=metric.capitalize())
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Classification Metrics by Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
