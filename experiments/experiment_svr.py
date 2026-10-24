import joblib
import os
import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold
from sklearn.compose import TransformedTargetRegressor
from .base_experiment import BaseExperiment, feature_importances  # Add this import
from .utils import save_model, FeatureNamePreserver, load_state, save_state  # Add save_model, FeatureNamePreserver, load_state, save_state import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVRExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "SVR"
        
    def create_model(self):
        return Pipeline([
            ('names', FeatureNamePreserver()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
            ('feature_scaler', RobustScaler(quantile_range=(5, 95))),
            ('target_pipeline', TransformedTargetRegressor(
                regressor=SVR(
                    kernel='rbf',
                    C=1.0,
                    epsilon=0.1,
                    gamma='scale',
                    cache_size=4000,
                    max_iter=5000,
                    tol=1e-4,
                    verbose=False
                ),
                transformer=StandardScaler()
            ))
        ])

    def run(self):
        """Override run to implement chunked training"""
        try:
            # Load existing experiment state
            state = load_state(self.output_dir)

            X, y = self.load_data()
            self.logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")
            
            # Ensure target values are non-negative
            if (y < 0).any():
                raise ValueError("Target values contain negative values, which are not allowed for mean_squared_log_error.")
            
            # More aggressive sampling for SVR
            if len(X) > 10000:
                self.logger.info("Large dataset detected, using stratified sampling")
                X_sample, _, y_sample, _ = train_test_split(
                    X, y, 
                    train_size=5000,     # Reduced sample size
                    random_state=42,
                    stratify=pd.qcut(y, q=10, labels=False, duplicates='drop')
                )
                X, y = X_sample, y_sample
            
            self.logger.info(f"Training SVR with {len(X)} samples")
            
            # Create base pipeline
            pipeline = self.create_model()
            
            # Define parameter distributions
            param_distributions = {
                'poly__degree': [1, 2, 3],
                'target_pipeline__regressor__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'target_pipeline__regressor__C': [0.01, 0.1, 1, 10, 100, 500],
                'target_pipeline__regressor__epsilon': [0.01, 0.05, 0.1, 0.2],
                'target_pipeline__regressor__gamma': ['scale', 'auto'],
            }
            
            # Use Repeated K-Fold
            rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
            
            # Randomized search over parameter space
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_distributions,
                n_iter=10, scoring='neg_mean_squared_log_error',
                cv=rkf, verbose=2, random_state=42, n_jobs=-1
            )
            search.fit(X, y)
            
            # Best model
            best_pipeline = search.best_estimator_
            best_score_msle = -search.best_score_
            msle = best_score_msle if best_score_msle > 0 else 0.0
            final_rmsle = np.sqrt(msle)
            self.logger.info(f"Best SVR RMSLE: {final_rmsle:.4f}")
            
            model_info = {
                'name': self.name,
                'model': best_pipeline,
                'features': X.columns.tolist(),
                'score': final_rmsle,
                'status': 'success',
                'error': None,
                'version': str(self.output_dir),
                'sample_size': len(X)
            }
            
            self.logger.info(f"Saving SVR model with score: {final_rmsle:.4f}")
            save_model(model_info, self.name, self.output_dir)
            
            # Update state
            state['completed'].append({
                'name': model_info['name'],
                'score': model_info['score']
            })
            save_state(self.output_dir, state['completed'], state['failed'], state['crashed'])
            
            self.logger.info("SVR model saved successfully")
            return model_info
            
        except Exception as e:
            error_msg = f"{self.name} run failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'model': None,
                'features': [],
                'score': float('inf'),
                'status': 'failed',
                'error': error_msg
            }

def run_svr(output_dir):
    experiment = SVRExperiment(output_dir)
    return experiment.run()

if __name__ == "__main__":
    result = run_svr('output')
    logger.info(result)