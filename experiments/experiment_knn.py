from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .base_experiment import BaseExperiment  # Relative import
from .utils import safe_rmsle, save_model, manage_resources, load_state, save_state  # Relative import
import logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

class KNNExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "KNN"
        
    def create_model(self):
        """Define and return a KNeighborsRegressor model with hyperparameter tuning."""
        # Define the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ])
        
        # Define hyperparameter grid
        param_distributions = {
            'knn__n_neighbors': [5, 10, 15, 20, 25, 30],
            'knn__weights': ['uniform', 'distance'],
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        # Define cross-validation strategy
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        
        # Define RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=20,
            scoring='neg_mean_squared_log_error',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        return random_search

    def run(self):
        """Execute the KNN regression experiment."""
        try:
            X, y = self.load_data()
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            
            # Validate data
            self.logger.info("Validating training data...")
            validate_data(X_train, y_train, stage="Training")
            self.logger.info("Validating validation data...")
            validate_data(X_val, y_val, stage="Validation")
            
            # Create and train the model
            model = self.create_model()
            self.logger.info("Starting model training with hyperparameter tuning...")
            
            with manage_resources():
                model.fit(X_train, y_train)
                
            self.logger.info("Model training completed.")
            
            # Make predictions and evaluate
            predictions = model.predict(X_val)
            score = safe_rmsle(y_val, predictions)
            self.logger.info(f"KNN RMSLE: {score:.4f}")
            
            # Prepare model information
            model_info = self.create_model_info(model, X, y, score)
            
            # Save the model
            save_model(model_info, self.name, self.output_dir)
            self.logger.info("KNN model saved successfully.")
            
            return model_info
            
        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'model': None,
                'features': [],
                'score': float('inf'),
                'status': 'failed',
                'error': error_msg
            }

def validate_data(X, y, stage=""):
    """Validate input data for KNN training."""
    if X is None or y is None:
        raise ValueError(f"{stage} data is None")
    
    if len(X) != len(y):
        raise ValueError(f"{stage} X and y have different lengths: {len(X)} vs {len(y)}")
    
    # Check if X is DataFrame or ndarray
    if isinstance(X, pd.DataFrame):
        if X.isna().any().any():
            raise ValueError(f"{stage} X contains NaN values")
    else:
        if np.isnan(X).any():
            raise ValueError(f"{stage} X contains NaN values")

    # Handle y as Series or ndarray
    if isinstance(y, pd.Series) and y.isna().any():
        raise ValueError(f"{stage} y contains NaN values")
    elif isinstance(y, np.ndarray) and np.isnan(y).any():
        raise ValueError(f"{stage} y contains NaN values")
    
    if (y < 0).any():
        raise ValueError(f"{stage} y contains negative values")

    logger.info(f"{stage} data validation passed: X shape={X.shape}, y shape={y.shape}")

def run_knn(output_dir):
    experiment = KNNExperiment(output_dir)
    model_info = experiment.run()
    
    # Load existing state
    state = load_state(output_dir)
    
    if model_info['status'] == 'success':
        # Append as a dict with 'name' and 'score'
        state['completed'].append({
            'name': model_info['name'],
            'score': model_info['score']
        })
    elif model_info['status'] == 'failed':
        state['failed'].append((model_info['name'], model_info['error']))
    else:
        state['crashed'].append(model_info['name'])
    
    # Save updated state
    save_state(output_dir, state['completed'], state['failed'], state['crashed'])
    
    return model_info

if __name__ == "__main__":
    run_knn('output')
