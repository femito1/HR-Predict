import torch
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, balanced_accuracy_score
import matplotlib.pyplot as plt


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
        probabilities = self.predict_proba(X)
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

    def evaluate(self, X, y, threshold=0.5):
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

    def save_model(self, filepath):
        """
        Save model to file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model to
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        if filepath.endswith('.pkl'):
            print(f"Model saved to {filepath}")
        else:
            print(f"Model saved to {filepath}.pkl")

    @classmethod
    def load_model(cls, filepath):
        """
        Load model from file
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
            
        Returns:
        --------
        model : PyTorchGradientBoosting
            Loaded model
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__} instance")
        
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

    def compute_shap_values(self, X, feature_names=None, n_samples=100):
        """
        Compute SHAP values for the input data
        """
        import shap
        
        # Convert to numpy if tensor
        X = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        
        # Create a function that returns the model predictions
        def model_predict(X_samples):
            X_tensor = torch.tensor(X_samples, dtype=torch.float32)
            return self.predict_proba(X_tensor).cpu().numpy()
        
        # Create explainer with the model function
        explainer = shap.KernelExplainer(model_predict, X)
        
        # Calculate SHAP values
        shap_values = explainer(X)
        
        # Store feature names for later use
        if feature_names is not None:
            self.feature_names = feature_names
        
        return shap_values, explainer

    def explain_prediction(self, X, sample_idx=0, feature_names=None, top_n=5, plot=True):
        """
        Explain a single prediction using SHAP values
        
        Parameters:
        -----------
        X : array-like
            Sample features
        sample_idx : int
            Index of the sample to explain
        feature_names : list, optional
            Names of features
        top_n : int
            Number of top features to highlight
        plot : bool
            Whether to show plots
            
        Returns:
        --------
        explanation : dict
            Dictionary with detailed explanation
        """
        import numpy as np
        import matplotlib.pyplot as plt

        shap_values, _ = self.compute_shap_values(X, feature_names)
        # Calculate average effect direction for each feature
        feature_effects = []
        for i, feature in enumerate(feature_names):
            # Calculate correlation between feature and SHAP value
            if i < shap_values.shape[1]:  # Make sure feature index is valid
                shap_impact = shap_values[:, i]
                feature_values = X[:, i]
                print('feature values', feature_values)
                print('shap impact', shap_impact)
                print('after shap impact, before correlation')
                correlation = np.corrcoef(feature_values, shap_impact.values)[0, 1]
                print('after correlation')
                
                # Determine effect direction
                if abs(correlation) < 0.1:
                    effect = "Minimal/Non-linear"
                elif correlation > 0:
                    effect = "↑ Increases attrition"
                else:
                    effect = "↓ Decreases attrition"
                
                # Calculate average absolute SHAP value (importance)
                importance = np.mean(np.abs(shap_impact.values))
                
                feature_effects.append({
                    'Feature': feature,
                    'Effect Direction': effect,
                    'Importance': importance,
                    'Correlation': correlation
                })
        print('hello1')
        # Create DataFrame and sort by importance
        effect_df = pd.DataFrame(feature_effects)
        print(effect_df)
        effect_df = effect_df.sort_values('Importance', ascending=False)
        print('hello2')
        # Display the table
        print("\nSummary of How Features Affect Attrition:")
        print(effect_df.head(15))  # Show top 15 features

        # Create a more visual summary for key features
        plt.figure(figsize=(12, 8))
        top_features = effect_df.head(15)['Feature'].tolist()

        for i, feature in enumerate(top_features):
            idx = feature_names.index(feature)
            corr = effect_df[effect_df['Feature'] == feature]['Correlation'].values[0]
            
            # Determine color based on correlation
            if abs(corr) < 0.1:
                color = 'gray'
                effect = "Minimal/Non-linear effect"
            elif corr > 0:
                color = 'red'
                effect = "Increases attrition risk"
            else:
                color = 'green'
                effect = "Decreases attrition risk"
            
            plt.barh(i, effect_df[effect_df['Feature'] == feature]['Importance'].values[0], color=color)
            plt.text(0.01, i, f"{feature}: {effect}", va='center')

            plt.yticks([]) 
            plt.xlabel('Feature Importance')
            plt.title('How Top Features Affect Employee Attrition')
            plt.tight_layout()
            plt.show()
    
    def suggest_improvements(self, employee: pd.DataFrame):
        means = {'satisfaction_level': 0.4400980117614114,
                'last_evaluation': 0.7181125735088211,
                'average_montly_hours': 207.41921030523662}
        suggestions = []


        employee_features = dict(zip(employee[['satisfaction_level', 'last_evaluation', 'average_montly_hours']].columns.tolist(), *employee[['satisfaction_level', 'last_evaluation', 'average_montly_hours']].values.tolist()))
        employee = employee.to_numpy()
        if employee_features['satisfaction_level'] < means['satisfaction_level']:
            suggestions.append('It seems that this employee is not satisfied with their job, as their satisfaction level is lower than expected. We recommend that HR and managers should talk to them to improve their situation.')
        if employee_features['last_evaluation'] > means['last_evaluation']:
            suggestions.append('This employee is performing very well. We have found that when employees have a very high last evaluation score, they are more likely to leave the company. We recommend that they get rewarded for their performance.')
        if employee_features['average_montly_hours'] > means['average_montly_hours']:
            suggestions.append('This employee is working a lot of hours. Work-life balance is important for employee retention, a few less hours a month could help them to be happier at work. Reconsidering your current resource planning and expanding your current workforce could be a good idea.')
        return suggestions
    
    def plot_what_if(self, X, feature_index, feature_name=None, range_min=None, 
                    range_max=None, num_points=20, sample_idx=0):
        """
        Create a what-if plot showing how changing a feature affects the prediction
        
        Parameters:
        -----------
        X : array-like
            Sample features
        feature_index : int
            Index of the feature to vary
        feature_name : str, optional
            Name of the feature
        range_min, range_max : float, optional
            Min and max values for the feature (defaults to sensible range)
        num_points : int
            Number of points to evaluate
        sample_idx : int
            Index of the sample to use as base
        """
        # Extract single sample
        if len(X.shape) > 1 and X.shape[0] > 1:
            x_base = X[sample_idx:sample_idx+1].clone() if isinstance(X, torch.Tensor) else X[sample_idx:sample_idx+1].copy()
        else:
            x_base = X.clone() if isinstance(X, torch.Tensor) else X.copy()
        
        # Determine feature range
        if range_min is None or range_max is None:
            if isinstance(X, torch.Tensor):
                all_values = X[:, feature_index].cpu().numpy()
            else:
                all_values = X[:, feature_index]
            std_dev = np.std(all_values)
            mean_val = np.mean(all_values)
            if range_min is None:
                range_min = mean_val - 2 * std_dev
            if range_max is None:
                range_max = mean_val + 2 * std_dev
        
        # Generate range of values
        feature_values = np.linspace(range_min, range_max, num_points)
        probabilities = []
        
        # Evaluate model for each value
        for value in feature_values:
            x_modified = x_base.clone() if isinstance(x_base, torch.Tensor) else x_base.copy()
            if isinstance(x_modified, torch.Tensor):
                x_modified[0, feature_index] = value
            else:
                x_modified[0, feature_index] = value
            
            probabilities.append(self.predict_proba(x_modified).item())
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(feature_values, probabilities, marker='o')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        plt.xlabel(f"{feature_name if feature_name else f'Feature {feature_index}'}")
        plt.ylabel('Probability of Leaving')
        plt.title(f"What-If Analysis: Impact of {feature_name if feature_name else f'Feature {feature_index}'}")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {
            'feature': feature_name if feature_name else f'Feature {feature_index}',
            'values': feature_values,
            'probabilities': probabilities
        }
