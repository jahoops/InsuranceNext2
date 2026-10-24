import joblib
import numpy as np
import pandas as pd
import logging
import json
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from .base_experiment import BaseExperiment, feature_importances  # Add this import
from .utils import FeatureNamePreserver, save_model, save_state, load_state  # Ensure correct import
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm  # Import tqdm for progress bar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "RandomForest"
        
    def create_model(self):
        return Pipeline([
            ('names', FeatureNamePreserver()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('feature_selector', SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42))),
            ('scaler', RobustScaler(quantile_range=(5, 95))),
            ('rf', RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            ))
        ])
    
    def run(self):
        """Run the RandomForest experiment with data sampling to reduce memory usage."""
        try:
            start_time = time.time()
            
            logger.info("Loading data...")
            X, y = self.load_data()
            self.logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")

            # Ensure target values are non-negative
            if (y < 0).any():
                raise ValueError("Target values contain negative values, which are not allowed for mean_squared_log_error.")

            # Remove 'id' column if present
            if 'id' in X.columns:
                X = X.drop(columns=['id'])
                logger.info("'id' column removed from features")

            # Sample to avoid excessive memory usage
            MAX_SIZE = 50000
            if len(X) > MAX_SIZE:
                X = X.sample(n=MAX_SIZE, random_state=42)
                y = y.sample(n=MAX_SIZE, random_state=42)
            logger.info("Data sampling completed.")

            # Feature Engineering (if any)
            # ...existing code...
            
            # Define parameter distributions
            param_distributions = {
                'rf__n_estimators': [100, 200, 300, 500],
                'rf__max_depth': [None, 10, 20, 30],
                'rf__min_samples_split': [2, 5, 10],
                'rf__min_samples_leaf': [1, 2, 4],
                'rf__max_features': ['sqrt', 'log2', 0.5]  # Added 0.5 as an option
            }
            logger.info("Parameter distributions defined.")
            
            # Use Repeated K-Fold
            rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
            
            # Log features before starting RandomizedSearchCV
            logger.info(f"Features before RandomizedSearchCV: {X.columns.tolist()}")
            logger.info(f"First few rows of features:\n{X.head()}")
            
            # Calculate total iterations
            n_iter = 50  # number of random parameter combinations
            n_splits = 5  # from RepeatedKFold(n_splits=5, n_repeats=3)
            n_repeats = 3
            total_fits = n_iter * n_splits * n_repeats
            
            self.logger.info(f"Starting RandomForest search with:")
            self.logger.info(f"- {n_iter} parameter combinations")
            self.logger.info(f"- {n_splits} splits x {n_repeats} repeats = {n_splits * n_repeats} CV iterations per combination")
            self.logger.info(f"- Total: {total_fits} model fits")
            self.logger.info(f"- Estimated time: {(total_fits * 15.3 / 6):.1f} minutes (based on 15.3min per 6 fits)")
            
            # Randomized search over parameter space
            search = RandomizedSearchCV(
                self.create_model(),
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring='neg_mean_squared_log_error',
                cv=rkf,
                verbose=2,
                random_state=42,
                n_jobs=-1
            )
            
            search_start = time.time()
            search.fit(X, y)
            search_time = time.time() - search_start
            
            self.logger.info(f"RandomForest search completed in {search_time/60:.1f} minutes")
            self.logger.info(f"Best parameters: {search.best_params_}")
            
            self.logger.info(f"Total run time so far: ~{time.time() - start_time:.2f} seconds.")
            
            logger.info("RandomizedSearchCV completed.")

            # Best model
            best_pipeline = search.best_estimator_
            best_score_msle = -search.best_score_
            msle = best_score_msle if best_score_msle > 0 else 0.0
            final_rmsle = np.sqrt(msle)
            self.logger.info(f"Best RandomForest RMSLE: {final_rmsle:.4f}")
            
            model_info = {
                'name': self.name,
                'model': best_pipeline,
                'features': X.columns.tolist(),  # Features used
                'score': final_rmsle,
                'status': 'success',
                'error': None,
                'version': str(self.output_dir),
                'sample_size': len(X)
            }
            
            save_model(model_info, self.name, self.output_dir)
            
            self.logger.info("RandomForest model saved successfully")
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

def run_random_forest(output_dir):
    experiment = RandomForestExperiment(output_dir)
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
    run_random_forest('output')