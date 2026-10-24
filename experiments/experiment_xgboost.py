import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from .base_experiment import BaseExperiment
from .utils import FeatureNamePreserver, save_model, save_state, load_state, load_data

logger = logging.getLogger(__name__)

class XGBRegressorWrapper(XGBRegressor, BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _more_tags(self):
        return {
            'non_deterministic': True,
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'binary_only': False,
            'requires_fit': True,
            'preserves_dtype': [np.float64],
            'requires_y': True,
            'pairwise': False,
        }

class XGBoostExperiment(BaseExperiment):
    @property
    def name(self):
        return "XGBoost"
        
    def create_model(self):
        model = Pipeline([
            ('names', FeatureNamePreserver()),
            ('xgb', XGBRegressor(random_state=42))
        ])
        return model
        
    def run(self):
        try:
            # Use shared load_data utility
            X_train, y_train = self.load_data()
            
            if (y_train < 0).any():
                raise ValueError("Target contains negative values, not allowed for MSLE.")

            param_distributions = {
                'xgb__n_estimators': [100, 200, 300],
                'xgb__learning_rate': [0.01, 0.05, 0.1],
                'xgb__max_depth': [3, 6, 9],
                'xgb__min_child_weight': [1, 3, 5],
                'xgb__gamma': [0, 0.1, 0.2]
            }

            search = RandomizedSearchCV(
                self.create_model(),
                param_distributions=param_distributions,
                n_iter=10,
                cv=RepeatedKFold(n_splits=5, n_repeats=1, random_state=42),
                scoring='neg_mean_squared_log_error',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

            search.fit(X_train, y_train)

            best_pipeline = search.best_estimator_
            best_score_msle = -search.best_score_  # Negative MSLE used in scoring
            final_rmsle = np.sqrt(best_score_msle) if best_score_msle > 0 else 0

            logger.info(f"[{self.name}] Best RMSLE: {final_rmsle:.4f}")

            model_info = {
                'name': self.name,
                'model': best_pipeline,
                'features': X_train.columns.tolist(),
                'score': final_rmsle,
                'status': 'success',
                'error': None,
                'version': str(self.output_dir),
                'sample_size': len(X_train)
            }

            # Save the model the same way RandomForest does
            save_model(model_info, self.name, self.output_dir)
            logger.info(f"[{self.name}] Model saved successfully")
            return model_info
        
        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'model': None,
                'features': [],
                'score': float('inf'),
                'status': 'failed',
                'error': error_msg
            }

def run_xgboost(output_dir):
    experiment = XGBoostExperiment(output_dir)
    model_info = experiment.run()
    
    # Load existing state, just like RandomForest
    state = load_state(output_dir)
    
    if model_info['status'] == 'success':
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
    run_xgboost('output')