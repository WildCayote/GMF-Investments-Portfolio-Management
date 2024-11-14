import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging

class ForecastModel:
    def __init__(self, data_frame, column_name='ClosePrice', logger=None):
        """
        Initialize the ForecastModel.

        Parameters:
        data_frame (pd.DataFrame): Input time series data.
        column_name (str): The column name to be forecasted.
        logger (logging.Logger, optional): Logger instance for tracking events and errors.
        """
        self.data_frame = data_frame
        self.column_name = column_name
        self.logger = logger or logging.getLogger(__name__)
        self.training_set = None
        self.testing_set = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess_data(self, train_ratio=0.75):
        """
        Prepares and scales data for training and testing. Resamples data to fill missing dates.

        Parameters:
        train_ratio (float): Fraction of data to be used for training.
        """
        try:
            self.data_frame.index = pd.to_datetime(self.data_frame.index)
            if self.data_frame.index.is_monotonic_decreasing:
                self.data_frame = self.data_frame.sort_index()
                self.logger.info("Data sorted to chronological order.")

            self.data_frame = self.data_frame.resample('D').ffill().dropna()

            split_point = int(len(self.data_frame) * train_ratio)
            self.training_set = self.data_frame[:split_point]
            self.testing_set = self.data_frame[split_point:]
            self.logger.info(f"Data split into {len(self.training_set)} training and {len(self.testing_set)} testing samples.")

            train_scaled = self.scaler.fit_transform(self.training_set[[self.column_name]]).astype(np.float32)
            test_scaled = self.scaler.transform(self.testing_set[[self.column_name]]).astype(np.float32)

            self.training_set[self.column_name] = train_scaled.flatten()
            self.testing_set[self.column_name] = test_scaled.flatten()
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise

    def train_auto_arima(self):
        """Fits an Auto-ARIMA model to the training data."""
        try:
            self.logger.info("Fitting Auto-ARIMA model.")
            arima_model = pm.auto_arima(self.training_set[self.column_name], seasonal=False, trace=True, error_action='ignore')
            self.models['AutoARIMA'] = arima_model
            self.logger.info(f"Auto-ARIMA model fitted: {arima_model.get_params()}")
        except Exception as e:
            self.logger.error(f"Auto-ARIMA model training failed: {e}")
            raise

    def train_seasonal_arima(self, season_length=5):
        """Fits a seasonal ARIMA model to the training data."""
        try:
            self.logger.info("Fitting seasonal ARIMA model.")
            sarima_model = pm.auto_arima(self.training_set[self.column_name], seasonal=True, m=season_length, trace=True, error_action='ignore')
            self.models['SARIMA'] = sarima_model
            self.logger.info(f"SARIMA model fitted: {sarima_model.get_params()}")
        except Exception as e:
            self.logger.error(f"SARIMA model training failed: {e}")
            raise

    def _generate_sequences(self, data, seq_length=60):
        """Generate data sequences for LSTM training."""
        sequences, labels = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            labels.append(data[i + seq_length])
        return np.array(sequences), np.array(labels)

    def train_lstm(self, seq_length=60, epochs=50, batch_size=32):
        """
        Trains an LSTM model on the time series data.

        Parameters:
        seq_length (int): Length of each input sequence.
        epochs (int): Number of epochs for training.
        batch_size (int): Size of each training batch.
        """
        try:
            self.logger.info("Training LSTM model.")
            data = self.training_set[self.column_name].values.reshape(-1, 1)
            X_train, y_train = self._generate_sequences(data, seq_length)

            lstm_model = Sequential([
                Input(shape=(seq_length, 1)),
                LSTM(64, activation='relu', return_sequences=True),
                Dropout(0.25),
                LSTM(64, activation='relu'),
                Dropout(0.25),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

            history = lstm_model.fit(
                X_train, y_train, epochs=epochs, batch_size=batch_size,
                validation_split=0.15, callbacks=[early_stopping], verbose=1
            )

            self.models['LSTM'] = {'model': lstm_model, 'history': history, 'seq_length': seq_length}
            self.plot_training_loss(history)
        except Exception as e:
            self.logger.error(f"LSTM model training failed: {e}")
            raise

    def plot_training_loss(self, history):
        """Plots training and validation loss."""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss', color='navy')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='salmon')
        plt.title('LSTM Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        self.logger.info("Training loss plotted.")

    def generate_predictions(self):
        """Generates predictions for all trained models."""
        try:
            for model_name, model_data in self.models.items():
                if model_name in ['AutoARIMA', 'SARIMA']:
                    self.predictions[model_name] = model_data.predict(n_periods=len(self.testing_set))
                elif model_name == 'LSTM':
                    lstm_model = model_data['model']
                    seq_length = model_data['seq_length']
                    recent_data = np.array(self.training_set[self.column_name][-seq_length:]).reshape(-1, 1)
                    forecast = []

                    for _ in range(len(self.testing_set)):
                        pred = lstm_model.predict(recent_data.reshape(1, seq_length, 1))
                        forecast.append(pred[0, 0])
                        recent_data = np.append(recent_data[1:], pred[0, 0]).reshape(-1, 1)

                    self.predictions['LSTM'] = np.array(forecast)
            self.logger.info("Predictions generated.")
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            raise

    def evaluate_models(self):
        """Evaluates each model and logs performance metrics."""
        results = []
        for model_name, prediction in self.predictions.items():
            actual = self.testing_set[self.column_name].values
            mae = mean_absolute_error(actual, prediction)
            rmse = np.sqrt(mean_squared_error(actual, prediction))
            mape = np.mean(np.abs((actual - prediction) / actual)) * 100
            self.metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
            results.append([model_name, mae, rmse, mape])

        metric_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "MAPE"])
        print("\nModel Performance Metrics:\n", metric_df)
        self.logger.info("Model evaluation completed.")

    def plot_predictions(self):
        """Plots actual versus predicted values."""
        plt.figure(figsize=(14, 6))
        plt.plot(self.testing_set.index, self.testing_set[self.column_name], label='Actual', color='teal', linewidth=2)
        colors = ['orange', 'purple', 'green']

        for idx, (model_name, pred) in enumerate(self.predictions.items()):
            plt.plot(self.testing_set.index, pred, label=f'{model_name} Prediction', linestyle='--', color=colors[idx])

        plt.title('Actual vs Predicted Values')
        plt.xlabel('Date')
        plt.ylabel(self.column_name)
        plt.legend()
        plt.show()
        self.logger.info("Prediction plot generated.")
    
    def save_optimal_model(self, directory, chosen_model='LSTM'):
        """
        Saves the chosen model for future use.

        Parameters:
        directory (str): Directory path to save the model.
        chosen_model (str): Name of the model to be saved.
        """
        try:
            if chosen_model in self.models:
                model_data = self.models[chosen_model]
                if chosen_model == 'LSTM':
                    model_data['model'].save(f'{directory}/{chosen_model}_best_model.h5')
                else:
                    joblib.dump(model_data, f'{directory}/{chosen_model}_best_model.pkl')
                self.logger.info(f"{chosen_model} model saved.")
            else:
                self.logger.error(f"Model '{chosen_model}' not found.")
        except Exception as e:
            self.logger.error(f"Error saving {chosen_model} model: {e}")
            raise
