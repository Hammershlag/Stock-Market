import sys
import warnings
import argparse

import numpy as np  # For numerical computations

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf  # For building and training machine learning models
import matplotlib.pyplot as plt  # For plotting data and visualizations
import seaborn as sns  # For enhanced visualizations
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import MinMaxScaler  # For normalizing data
from datetime import timedelta  # For date manipulations
import yfinance as yf  # For downloading stock market data
sns.set()


def get_params():
    """Parse command line arguments and set random seeds"""
    parser = argparse.ArgumentParser(description='Train Stock Market Predictor')
    parser.add_argument('--symbol', type=str, default="GC=F", help='Symbol of Stock to use')
    parser.add_argument('--period', type=str, default="1y", help='Data period to download')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--sims', type=int, default=3, help='Number of Simulations')
    parser.add_argument('--seed', type=int, default=14, help='Random seed for reproducibility')
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    return args


def download_data(symbol, period):
    """Download stock data using yfinance"""
    print(f"Downloading data for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    df.reset_index(inplace=True)

    if df.empty:
        print(f"No data found for symbol {symbol}")
        sys.exit(1)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df


def create_sequences(data, seq_length):
    """Create sequences for LSTM training with multiple features"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        # Only predict close price (first feature)
        y.append(data[i + seq_length, 0])  # Close price is first column

    X = np.array(X)
    y = np.array(y)

    print(f"Created sequences: X shape={X.shape}, y shape={y.shape}")
    return X, y


class StockPredictor:
    def __init__(self, sequence_length, num_features=4, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self._build_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True,
                                 input_shape=(self.sequence_length, self.num_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)  # Output only close price prediction
        ])
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, verbose=0):
        # Ensure proper shapes
        if len(y_train.shape) == 2:
            y_train = y_train.reshape(-1, 1)

        print(f"Training with X shape: {X_train.shape}, y shape: {y_train.shape}")

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True
        )
        return history

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def predict_future(self, last_sequence, steps, scalers):
        """Predict future values step by step using all features"""
        predictions = []
        current_sequence = np.array(last_sequence)

        # Ensure we have the right sequence length
        if len(current_sequence) != self.sequence_length:
            if len(current_sequence) > self.sequence_length:
                current_sequence = current_sequence[-self.sequence_length:]
            else:
                # Pad with the last values if sequence is too short
                pad_length = self.sequence_length - len(current_sequence)
                last_values = current_sequence[-1] if len(current_sequence) > 0 else np.zeros(self.num_features)
                padding = np.tile(last_values, (pad_length, 1))
                current_sequence = np.vstack([padding, current_sequence])

        for _ in range(steps):
            # Reshape for model input: (batch_size, sequence_length, features)
            input_seq = current_sequence.reshape(1, self.sequence_length, self.num_features)

            # Predict next close price
            next_close_pred = self.model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(next_close_pred)

            # For other features, use simple heuristics based on recent data
            recent_data = current_sequence[-5:]  # Last 5 time steps

            # High: typically close to or slightly above the predicted close
            recent_high_close_ratio = np.mean(recent_data[:, 1] / recent_data[:, 0])  # high/close ratio
            next_high_pred = next_close_pred * recent_high_close_ratio

            # Low: typically close to or slightly below the predicted close
            recent_low_close_ratio = np.mean(recent_data[:, 2] / recent_data[:, 0])  # low/close ratio
            next_low_pred = next_close_pred * recent_low_close_ratio

            # Volume: use moving average of recent volumes
            next_volume_pred = np.mean(recent_data[:, 3])

            # Create next step with all predicted features: [close, high, low, volume]
            next_step = np.array([next_close_pred, next_high_pred, next_low_pred, next_volume_pred])

            # Update sequence for next prediction
            current_sequence = np.vstack([current_sequence[1:], next_step.reshape(1, -1)])

        return np.array(predictions)


def calculate_accuracy(real, predict):
    """Calculate prediction accuracy"""
    real = np.array(real)
    predict = np.array(predict)
    mape = np.mean(np.abs((real - predict) / real)) * 100
    return max(0, 100 - mape)


def forecast_single(normalized_data, sequence_length, epoch, learning_rate, test_size, scalers):
    """Run one forecast simulation using close, high, low, and volume data"""
    # Prepare training data
    X, y = create_sequences(normalized_data, sequence_length)

    print(f"Training data shape: X={X.shape}, y={y.shape}")

    # Create and train model with 4 features
    predictor = StockPredictor(sequence_length, num_features=4, learning_rate=learning_rate)

    # Train with progress bar
    history = predictor.train(X, y, epochs=epoch, batch_size=16, verbose=0)

    # Generate predictions for existing data
    train_predictions = predictor.predict(X)

    # Get the last sequence for future predictions
    last_sequence = normalized_data[-sequence_length:]
    print(f"Last sequence shape: {last_sequence.shape}")

    # Predict future values
    future_predictions = predictor.predict_future(last_sequence, test_size, scalers)

    # Combine all predictions
    all_predictions = np.concatenate([
        np.full(sequence_length, np.nan),  # No predictions for first sequence_length points
        train_predictions.flatten(),
        future_predictions
    ])

    # Denormalize predictions (only close prices)
    valid_mask = ~np.isnan(all_predictions)
    denormalized_predictions = np.full_like(all_predictions, np.nan)

    if np.any(valid_mask):
        valid_predictions = all_predictions[valid_mask].reshape(-1, 1)
        denormalized_valid = scalers['close'].inverse_transform(valid_predictions).flatten()
        denormalized_predictions[valid_mask] = denormalized_valid

    return denormalized_predictions, history.history['loss'][-1]


def run_predictions(params):
    """Main prediction function that returns all necessary data for graphing"""
    # Download data
    df = download_data(params.symbol, params.period)

    # Check required columns
    required_columns = ['Close', 'High', 'Low', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: Missing data columns for {params.symbol}: {missing_columns}")
        print("Required columns: Close, High, Low, Volume")
        sys.exit(1)

    # Prepare all price data
    close_prices = df['Close'].values.reshape(-1, 1)
    high_prices = df['High'].values.reshape(-1, 1)
    low_prices = df['Low'].values.reshape(-1, 1)
    volumes = df['Volume'].values.reshape(-1, 1)

    # Scale all features separately for better normalization
    close_scaler = MinMaxScaler()
    high_scaler = MinMaxScaler()
    low_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()

    normalized_close = close_scaler.fit_transform(close_prices)
    normalized_high = high_scaler.fit_transform(high_prices)
    normalized_low = low_scaler.fit_transform(low_prices)
    normalized_volume = volume_scaler.fit_transform(volumes)

    # Combine features: [close, high, low, volume]
    normalized_data = np.hstack([
        normalized_close,
        normalized_high,
        normalized_low,
        normalized_volume
    ])

    print(f"Combined data shape: {normalized_data.shape}")
    print(f"Features: Close Price, High Price, Low Price, Volume")

    # Parameters
    simulation_size = params.sims
    sequence_length = 10
    test_size = 30
    learning_rate = 0.001

    print(f"Using {len(normalized_data)} data points for training")

    # Create scalers dictionary for easy passing
    scalers = {
        'close': close_scaler,
        'high': high_scaler,
        'low': low_scaler,
        'volume': volume_scaler
    }

    # Run multiple simulations
    print("Running simulations...")
    results = []
    losses = []

    for i in range(simulation_size):
        print(f'Simulation {i + 1}/{simulation_size}')
        pred, loss = forecast_single(normalized_data, sequence_length, params.epochs,
                                     learning_rate, test_size, scalers)
        results.append(pred)
        losses.append(loss)

    # Prepare dates for plotting
    dates = pd.to_datetime(df['Date']).tolist()
    for i in range(test_size):
        dates.append(dates[-1] + timedelta(days=1))

    # Filter valid results
    accepted_results = []
    original_close = df['Close'].values

    for i, r in enumerate(results):
        # Remove NaN values for comparison
        valid_predictions = r[~np.isnan(r)]
        if len(valid_predictions) > 0:
            # Check if predictions are reasonable
            min_reasonable = np.min(original_close) * 0.5
            max_reasonable = np.max(original_close) * 3

            if (np.min(valid_predictions) > min_reasonable and
                    np.max(valid_predictions) < max_reasonable):
                accepted_results.append(r)

    print(f"Accepted results: {len(accepted_results)}/{simulation_size}")

    # Calculate accuracies for training portion
    accuracies = []
    if len(accepted_results) > 0:
        for r in accepted_results:
            # Compare only the training portion (excluding NaN and future predictions)
            train_end = len(original_close)
            valid_mask = ~np.isnan(r[:train_end])
            if np.sum(valid_mask) > 0:
                acc = calculate_accuracy(
                    original_close[valid_mask],
                    r[:train_end][valid_mask]
                )
                accuracies.append(acc)

    # Return all data needed for graphing and analysis
    return {
        'symbol': params.symbol,
        'dates': dates,
        'original_close': original_close,
        'accepted_results': accepted_results,
        'accuracies': accuracies,
        'losses': losses,
        'df': df,
        'test_size': test_size,
        'simulation_size': simulation_size,
        'features_used': ['Close Price', 'High Price', 'Low Price', 'Volume'],
        'scalers': scalers  # Include scalers for potential future use
    }


def create_graphs(results_data):
    """Create graphs and print summary statistics"""
    symbol = results_data['symbol']
    dates = results_data['dates']
    original_close = results_data['original_close']
    accepted_results = results_data['accepted_results']
    accuracies = results_data['accuracies']
    losses = results_data['losses']
    df = results_data['df']
    test_size = results_data['test_size']
    simulation_size = results_data['simulation_size']

    if len(accepted_results) > 0:
        # Create the plot
        plt.figure(figsize=(15, 8))

        # Plot historical data
        historical_dates = dates[:len(original_close)]
        plt.plot(historical_dates, original_close,
                 label='Historical Price', color='black', linewidth=2)

        # Plot predictions
        colors = plt.cm.tab10(np.linspace(0, 1, len(accepted_results)))
        for no, (r, color) in enumerate(zip(accepted_results, colors)):
            plt.plot(dates[:len(r)], r,
                     label=f'Forecast {no + 1}', color=color, alpha=0.7)

        # Add vertical line to separate historical and future data
        plt.axvline(x=historical_dates[-1], color='red', linestyle='--', alpha=0.5,
                    label='Prediction Start')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'Stock: {symbol} - Average Accuracy: {np.mean(accuracies):.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Print summary statistics
        print(f"\nSummary:")
        print(f"Symbol: {symbol}")
        print(f"Features used: {', '.join(results_data.get('features_used', ['Close Price']))}")
        print(f"Training Period: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Average Training Accuracy: {np.mean(accuracies):.2f}%")
        print(f"Average Training Loss: {np.mean(losses):.6f}")

        # Show future predictions
        future_start_idx = len(original_close)
        print(f"\nFuture Predictions (next {test_size} days):")
        future_dates = dates[future_start_idx:future_start_idx + test_size]

        for i, date in enumerate(future_dates[:10]):  # Show first 10 days
            future_preds = [r[future_start_idx + i] for r in accepted_results
                            if len(r) > future_start_idx + i and not np.isnan(r[future_start_idx + i])]
            if future_preds:
                avg_pred = np.mean(future_preds)
                std_pred = np.std(future_preds)
                print(f"{date.strftime('%Y-%m-%d')}: ${avg_pred:.2f} ± ${std_pred:.2f}")

    else:
        print("No valid predictions generated. Try adjusting parameters:")
        print("- Increase epochs (--epochs)")
        print("- Change the time period (--period)")
        print("- Try a different stock symbol (--symbol)")

    print("\nTraining complete!")


def main():
    """Main function with clean structure"""
    # Get parameters
    params = get_params()

    # Run predictions and get results
    results_data = run_predictions(params)

    # Create graphs and display results
    create_graphs(results_data)


if __name__ == "__main__":
    main()