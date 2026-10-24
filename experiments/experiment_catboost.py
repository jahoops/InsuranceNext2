from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from .base_experiment import BaseExperiment
from .utils import FeatureNamePreserver, save_model, save_state, load_state
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# Suppress specific TensorFlow warnings if TensorFlow is used elsewhere
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Unregister existing TensorFlow plugins if registered
try:
    tf.experimental.unregister_fft_plugin('cuFFT')
    logging.info("Successfully unregistered existing TensorFlow plugins in CatBoost.")
except AttributeError:
    logging.info("No existing TensorFlow plugins to unregister in CatBoost.")

# Re-register TensorFlow plugins
try:
    tf.experimental.register_fft_plugin('cuFFT')
    logging.info("Successfully registered TensorFlow plugins in CatBoost.")
except AttributeError as e:
    logging.error(f"TensorFlow plugin registration methods not found in CatBoost: {e}")
except Exception as e:
    logging.error(f"Failed to register TensorFlow plugins in CatBoost: {e}")

logger = logging.getLogger(__name__)

class CatBoostExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "CatBoost"
        
    def create_model(self):
        # Temporarily disable GPU usage to prevent worker crashes
        return Pipeline([
            ('names', FeatureNamePreserver()),
            ('catboost', CatBoostRegressor(
                iterations=500,            # Reduced from 1000
                learning_rate=0.05,
                depth=6,
                loss_function='RMSE',
                random_state=42,
                verbose=0,
                task_type='CPU'           # Switch to CPU to avoid GPU memory issues
            ))
        ])
    
    def run(self):
        try:
            X, y = self.load_data()
            self.logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")
            
            # Identify categorical features if any
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Define smaller parameter distributions to reduce memory usage
            param_distributions = {
                'catboost__iterations': [300, 500],       # Fewer iterations
                'catboost__learning_rate': [0.01, 0.05],
                'catboost__depth': [4, 6],
                'catboost__l2_leaf_reg': [1, 3],
                'catboost__border_count': [32, 64]
            }
            
            rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
            
            search = RandomizedSearchCV(
                self.create_model(),
                param_distributions=param_distributions,
                n_iter=5,                                 # Fewer random draws
                scoring='neg_mean_squared_log_error',
                cv=rkf,
                verbose=2,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X, y)
            
            best_pipeline = search.best_estimator_
            best_score_msle = -search.best_score_
            msle = best_score_msle if best_score_msle > 0 else 0.0
            final_rmsle = np.sqrt(msle)
            self.logger.info(f"Best CatBoost RMSLE: {final_rmsle:.4f}")
            
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
            
            self.logger.info(f"Saving CatBoost model with score: {final_rmsle:.4f}")
            save_model(model_info, self.name, self.output_dir)
            
            self.logger.info("CatBoost model saved successfully")
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

def run_catboost(output_dir):
    experiment = CatBoostExperiment(output_dir)
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
    run_catboost('output')
