from pathlib import Path
import psutil
import time
import logging
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm
import numpy as np
from datetime import timedelta

import pandas as pd

from base_experiment import BaseExperiment, feature_importances
from utils import (
    FeatureNamePreserver,
    save_model,
    save_state,
    load_state
)

logger = logging.getLogger(__name__)

class LinearExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "Linear"
        
    def log_memory_usage(self, context=None):
        """Log memory usage with optional context."""
        process = psutil.Process()
        mem_info = process.memory_info()
        if context:
            self.logger.info(f"{context} - Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
        else:
            self.logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

    def create_model(self):
        return LinearRegression()

    def run(self):
        """Train and save the best linear model, then update experiment state."""
        try:
            state = load_state(self.output_dir)
            best_model, best_model_name, final_rmsle = self.create_model()
            
            # Build model info object
            model_info = {
                'name': best_model_name,
                'model': best_model,
                'features': getattr(best_model, 'feature_names_in_', X.columns.tolist()),
                'score': final_rmsle,
                'status': 'success' if final_rmsle < float('inf') else 'failed',
                'error': None,
                'version': str(self.output_dir),
                'sample_size': len(X)  # Assuming X is loaded in create_model
            }
            
            # Save the model
            save_model(model_info, best_model_name, self.output_dir)
            logger.info(f"[{best_model_name}] Model saved successfully")
            
            # Update experiment state
            if model_info['status'] == 'success':
                state['completed'].append({
                    'name': model_info['name'],
                    'score': model_info['score']
                })
            else:
                state['failed'].append((model_info['name'], model_info['error']))
            
            save_state(self.output_dir, state['completed'], state['failed'], state['crashed'])
            return model_info
        
        except Exception as e:
            error_msg = f"{self.name} run failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'model': None,
                'features': [],
                'score': float('inf'),
                'status': 'failed',
                'error': error_msg
            }

if __name__ == "__main__":
    # Define the output directory for this experiment
    output_dir = Path('/home/USER/Kaggle/InsuranceNext/output/v1/Linear')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run the experiment
    experiment = LinearExperiment(output_dir)
    model_info = experiment.run()

    # Optional: Print or log the model information
    print(model_info)
