import os
import logging

# Disable oneDNN optimizations to suppress related warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from .base_experiment import BaseExperiment, feature_importances  # Add this import
from .utils import safe_rmsle, save_model, manage_resources, load_state, save_state  # Changed to relative import
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Unregister existing TensorFlow plugins if registered
# Removed references to cuDNN / cuBLAS or tf.experimental.* plugin calls, if any

logger = logging.getLogger(__name__)

class NeuralNetExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "NeuralNet"
        
    def create_model(self):
        """Define and compile the neural network model."""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(len(self.load_data()[0].columns),)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])
        
        return model

    def run(self):
        """Execute the neural network experiment."""
        try:
            # Ensure TensorFlow uses GPU if available
            if tf.config.list_physical_devices('GPU'):
                self.logger.info("Using GPU for TensorFlow")
            else:
                self.logger.info("No GPU found, using CPU for TensorFlow")
            
            X, y = self.load_data()
            
            # Ensure target values are non-negative
            if (y < 0).any():
                raise ValueError("Target values contain negative values, which are not allowed for mean_squared_log_error.")
            
            # Log transform target variable
            y_log = np.log1p(y)
            
            # Pre-select features if desired:
            selected_cols = SelectKBest(f_regression, k=50).fit_transform(X, y_log)
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                selected_cols, y_log, test_size=0.2, random_state=42, shuffle=True
            )
            
            # Validate data
            self.logger.info("Validating training data...")
            validate_data(X_train, y_train, stage="Training")
            self.logger.info("Validating validation data...")
            validate_data(X_val, y_val, stage="Validation")
            
            # Create model
            model = self.create_model()
            self.logger.info("Neural network model created.")
            
            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, mode='min')
            checkpoint = ModelCheckpoint(
                filepath=str(self.output_dir / 'best_neural_net_model.keras'),  # Changed extension to .keras
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
            csv_logger = CSVLogger(str(self.output_dir / 'neural_net_training_log.csv'), append=True)
            
            # Train the model
            self.logger.info("Starting training of the neural network...")
            with manage_resources():
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=128,
                    callbacks=[early_stop, reduce_lr, checkpoint, csv_logger],
                    verbose=1,
                    shuffle=True
                )
            
            # Load the best model
            best_model = self.load_best_model()
            
            # Make predictions and evaluate
            predictions_log = best_model.predict(X_val).flatten()
            predictions = np.expm1(predictions_log)
            y_val_actual = np.expm1(y_val)
            score = safe_rmsle(y_val_actual, predictions)
            self.logger.info(f"NeuralNet RMSLE: {score:.4f}")
            
            # Prepare model information
            model_info = {
                'name': self.name,  # Model name
                'model': best_model,  # Trained model
                'features': X.columns.tolist(),  # Feature list
                'score': score,  # RMSLE score
                'status': 'success',
                'error': None,
                'version': str(self.output_dir),  # Version information
                'sample_size': len(X)
            }
            
            # Save the model information
            save_model(model_info, self.name, self.output_dir)
            self.logger.info("Neural network model saved successfully.")
            
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
    
    def load_best_model(self):
        """Load the best saved neural network model."""
        model_path = self.output_dir / 'best_neural_net_model.keras'  # Changed extension to .keras
        if not model_path.exists():
            raise FileNotFoundError(f"Best neural network model not found at {model_path}")
        model = tf.keras.models.load_model(str(model_path), custom_objects={'safe_rmsle': safe_rmsle})
        self.logger.info("Best neural network model loaded.")
        return model

def validate_data(X, y, stage=""):
    """Validate input data for neural network training."""
    if X is None or y is None:
        raise ValueError(f"{stage} data is None")
    
    if len(X) != len(y):
        raise ValueError(f"{stage} X and y have different lengths: {len(X)} vs {len(y)}")
    
    # Fix: Check if X is DataFrame or ndarray
    if isinstance(X, pd.DataFrame):
        if X.isna().any().any():
            raise ValueError(f"{stage} X contains NaN values")
    else:
        if np.isnan(X).any():
            raise ValueError(f"{stage} X contains NaN values")

    # Already handling y as Series or ndarray
    if isinstance(y, pd.Series) and y.isna().any():
        raise ValueError(f"{stage} y contains NaN values")
    elif isinstance(y, np.ndarray) and np.isnan(y).any():
        raise ValueError(f"{stage} y contains NaN values")
    
    if (y < 0).any():
        raise ValueError(f"{stage} y contains negative values")

    logger.info(f"{stage} data validation passed: X shape={X.shape}, y shape={y.shape}")

def run_neural_net(output_dir):
    experiment = NeuralNetExperiment(output_dir)
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
    run_neural_net('output')