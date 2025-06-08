import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import warnings
from datetime import datetime, timedelta
import queue

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Set dark theme for matplotlib
plt.style.use('dark_background')


class ModernButton(tk.Button):
    """Custom button with modern styling"""

    def __init__(self, parent, **kwargs):
        # Extract custom parameters
        bg_color = kwargs.pop('bg_color', '#3B82F6')
        hover_color = kwargs.pop('hover_color', '#2563EB')
        text_color = kwargs.pop('text_color', 'white')

        super().__init__(parent, **kwargs)

        self.bg_color = bg_color
        self.hover_color = hover_color

        self.configure(
            bg=bg_color,
            fg=text_color,
            font=('Segoe UI', 12, 'bold'),
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=12,
            cursor='hand2'
        )

        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)

    def _on_enter(self, e):
        self.configure(bg=self.hover_color)

    def _on_leave(self, e):
        self.configure(bg=self.bg_color)


class ModernEntry(tk.Frame):
    """Custom entry with modern styling and labels"""

    def __init__(self, parent, label_text, **kwargs):
        super().__init__(parent, bg='#1F2937')

        # Label
        self.label = tk.Label(
            self,
            text=label_text,
            bg='#1F2937',
            fg='#D1D5DB',
            font=('Segoe UI', 10)
        )
        self.label.pack(anchor='w', pady=(0, 5))

        # Entry
        self.entry = tk.Entry(
            self,
            bg='#374151',
            fg='white',
            font=('Segoe UI', 11),
            relief='flat',
            borderwidth=2,
            highlightthickness=1,
            highlightcolor='#3B82F6',
            highlightbackground='#4B5563',
            **kwargs
        )
        self.entry.pack(fill='x', ipady=8)

    def get(self):
        return self.entry.get()

    def set(self, value):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(value))


class ModernCombobox(tk.Frame):
    """Custom combobox with modern styling"""

    def __init__(self, parent, label_text, values, **kwargs):
        super().__init__(parent, bg='#1F2937')

        # Label
        self.label = tk.Label(
            self,
            text=label_text,
            bg='#1F2937',
            fg='#D1D5DB',
            font=('Segoe UI', 10)
        )
        self.label.pack(anchor='w', pady=(0, 5))

        # Combobox with custom styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'Modern.TCombobox',
            fieldbackground='#374151',
            background='#374151',
            foreground='white',
            borderwidth=1,
            relief='flat'
        )

        self.combobox = ttk.Combobox(
            self,
            values=values,
            state='readonly',
            style='Modern.TCombobox',
            font=('Segoe UI', 11),
            **kwargs
        )
        self.combobox.pack(fill='x', ipady=8)

    def get(self):
        return self.combobox.get()

    def set(self, value):
        self.combobox.set(value)


# Your existing prediction classes and functions
class StockPredictor:
    def __init__(self, sequence_length, num_features=2, learning_rate=0.001):
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
            tf.keras.layers.Dense(1)
        ])
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, verbose=0):
        if len(y_train.shape) == 2:
            y_train = y_train.reshape(-1, 1)

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

    def predict_future(self, last_sequence, steps, close_scaler, volume_scaler):
        predictions = []
        current_sequence = np.array(last_sequence)

        if len(current_sequence) != self.sequence_length:
            if len(current_sequence) > self.sequence_length:
                current_sequence = current_sequence[-self.sequence_length:]
            else:
                pad_length = self.sequence_length - len(current_sequence)
                last_values = current_sequence[-1] if len(current_sequence) > 0 else np.zeros(self.num_features)
                padding = np.tile(last_values, (pad_length, 1))
                current_sequence = np.vstack([padding, current_sequence])

        for _ in range(steps):
            input_seq = current_sequence.reshape(1, self.sequence_length, self.num_features)
            next_close_pred = self.model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(next_close_pred)

            recent_volumes = current_sequence[-5:, 1]
            next_volume_pred = np.mean(recent_volumes)
            next_step = np.array([next_close_pred, next_volume_pred])
            current_sequence = np.vstack([current_sequence[1:], next_step.reshape(1, -1)])

        return np.array(predictions)


def download_data(symbol, period):
    """Download stock data using yfinance"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    df.reset_index(inplace=True)

    if df.empty:
        raise ValueError(f"No data found for symbol {symbol}")

    return df


def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])

    return np.array(X), np.array(y)


def forecast_single(normalized_data, sequence_length, epoch, learning_rate, test_size, close_scaler, volume_scaler):
    """Run one forecast simulation"""
    X, y = create_sequences(normalized_data, sequence_length)

    predictor = StockPredictor(sequence_length, num_features=2, learning_rate=learning_rate)
    history = predictor.train(X, y, epochs=epoch, batch_size=16, verbose=0)

    train_predictions = predictor.predict(X)
    last_sequence = normalized_data[-sequence_length:]
    future_predictions = predictor.predict_future(last_sequence, test_size, close_scaler, volume_scaler)

    all_predictions = np.concatenate([
        np.full(sequence_length, np.nan),
        train_predictions.flatten(),
        future_predictions
    ])

    valid_mask = ~np.isnan(all_predictions)
    denormalized_predictions = np.full_like(all_predictions, np.nan)

    if np.any(valid_mask):
        valid_predictions = all_predictions[valid_mask].reshape(-1, 1)
        denormalized_valid = close_scaler.inverse_transform(valid_predictions).flatten()
        denormalized_predictions[valid_mask] = denormalized_valid

    return denormalized_predictions, history.history['loss'][-1]


class StockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Predictor - AI Neural Network")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0F172A')
        self.root.resizable(True, True)

        # Set window icon (optional)
        try:
            self.root.iconbitmap('icon.ico')  # Add an icon file if you have one
        except:
            pass

        # Queue for thread communication
        self.result_queue = queue.Queue()
        self.root.after(100, self.check_queue)

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Main container with padding
        main_frame = tk.Frame(self.root, bg='#0F172A')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_frame = tk.Frame(main_frame, bg='#0F172A')
        title_frame.pack(fill='x', pady=(0, 30))

        title_label = tk.Label(
            title_frame,
            text="Artificial Neural Network for Stock Trend Prediction",
            font=('Segoe UI', 24, 'bold'),
            fg='white',
            bg='#0F172A'
        )
        title_label.pack()

        subtitle_label = tk.Label(
            title_frame,
            text="Advanced LSTM Deep Learning Model for Financial Forecasting",
            font=('Segoe UI', 12),
            fg='#94A3B8',
            bg='#0F172A'
        )
        subtitle_label.pack(pady=(5, 0))

        # Main content area
        content_frame = tk.Frame(main_frame, bg='#0F172A')
        content_frame.pack(fill='both', expand=True)

        # Left panel - Controls
        left_panel = tk.Frame(content_frame, bg='#1F2937', relief='solid', borderwidth=1)
        left_panel.pack(side='left', fill='y', padx=(0, 20), pady=10, ipadx=20, ipady=20)

        # Configuration Button
        self.config_btn = ModernButton(
            left_panel,
            text="⚙️ Configuration",
            command=self.toggle_configuration,
            bg_color='#3B82F6',
            hover_color='#2563EB'
        )
        self.config_btn.pack(fill='x', pady=(0, 20))

        # Configuration Frame (Initially Hidden) - This needs to be created BEFORE predict button
        self.config_frame = tk.Frame(left_panel, bg='#1F2937')
        # Don't pack it yet, just create it

        # Stock Symbol
        self.stock_entry = ModernEntry(self.config_frame, "Stock Symbol")
        self.stock_entry.pack(fill='x', pady=(0, 15))
        self.stock_entry.set("MSFT")

        # Days
        self.days_entry = ModernEntry(self.config_frame, "Days")
        self.days_entry.pack(fill='x', pady=(0, 15))
        self.days_entry.set("0")

        # Months
        self.months_entry = ModernEntry(self.config_frame, "Months")
        self.months_entry.pack(fill='x', pady=(0, 15))
        self.months_entry.set("1")

        # Years
        self.years_entry = ModernEntry(self.config_frame, "Years")
        self.years_entry.pack(fill='x', pady=(0, 15))
        self.years_entry.set("0")

        # Max Period
        # self.max_entry = ModernEntry(self.config_frame, "Max Period (leave empty or 'max')")
        # self.max_entry.pack(fill='x', pady=(0, 15))
        # self.max_entry.set("")

        # Epochs
        self.epochs_entry = ModernEntry(self.config_frame, "Number of Epochs")
        self.epochs_entry.pack(fill='x', pady=(0, 15))
        self.epochs_entry.set("50")

        # Simulations
        self.sims_entry = ModernEntry(self.config_frame, "Number of Simulations")
        self.sims_entry.pack(fill='x', pady=(0, 20))
        self.sims_entry.set("3")

        # Predict Button - Now comes AFTER configuration frame creation
        self.predict_btn = ModernButton(
            left_panel,
            text="🚀 PREDICT",
            command=self.start_prediction,
            bg_color='#3B82F6',
            hover_color='#2563EB'
        )
        self.predict_btn.pack(fill='x', pady=(0, 20))

        # Progress bar
        self.progress = ttk.Progressbar(
            left_panel,
            mode='indeterminate',
            style='Modern.Horizontal.TProgressbar'
        )
        self.progress.pack(fill='x', pady=(0, 20))

        # Status label
        self.status_label = tk.Label(
            left_panel,
            text="Ready to predict",
            font=('Segoe UI', 10),
            fg='#94A3B8',
            bg='#1F2937'
        )
        self.status_label.pack()

        # Predicted Price Display
        self.price_frame = tk.Frame(left_panel, bg='#065F46', relief='solid', borderwidth=1)
        self.price_frame.pack(fill='x', pady=(20, 0), padx=5)

        price_title = tk.Label(
            self.price_frame,
            text="💰 Predicted Price",
            font=('Segoe UI', 12, 'bold'),
            fg='white',
            bg='#065F46'
        )
        price_title.pack(pady=(10, 5))

        self.price_label = tk.Label(
            self.price_frame,
            text="$0.00",
            font=('Segoe UI', 32, 'bold'),
            fg='#10B981',
            bg='#065F46'
        )
        self.price_label.pack(pady=(0, 10))

        # Right panel - Chart
        right_panel = tk.Frame(content_frame, bg='#1F2937', relief='solid', borderwidth=1)
        right_panel.pack(side='right', fill='both', expand=True, pady=10, padx=5)

        chart_title = tk.Label(
            right_panel,
            text="📈 Price Trend Analysis",
            font=('Segoe UI', 16, 'bold'),
            fg='white',
            bg='#1F2937'
        )
        chart_title.pack(pady=(15, 10))

        # Chart area
        self.setup_chart(right_panel)

    def toggle_configuration(self):
        """Toggle visibility of configuration frame"""
        if self.config_frame.winfo_ismapped():
            self.config_frame.pack_forget()
            self.config_btn.configure(text="⚙️ Configuration")
        else:
            # Pack the configuration frame BEFORE the predict button
            self.config_frame.pack(before=self.predict_btn, fill='x', pady=(0, 15))
            self.config_btn.configure(text="⚙️ Hide Configuration")

    def setup_chart(self, parent):
        """Setup the matplotlib chart"""
        # Create figure with dark theme
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor='#1F2937')
        self.ax = self.fig.add_subplot(111, facecolor='#1F2937')

        # Style the axes
        self.ax.spines['bottom'].set_color('#94A3B8')
        self.ax.spines['top'].set_color('#94A3B8')
        self.ax.spines['left'].set_color('#94A3B8')
        self.ax.spines['right'].set_color('#94A3B8')
        self.ax.tick_params(colors='#94A3B8')
        self.ax.xaxis.label.set_color('#94A3B8')
        self.ax.yaxis.label.set_color('#94A3B8')

        # Initial empty plot
        self.ax.text(0.5, 0.5, 'Click "PREDICT" to generate forecast\n\n📊 Configure parameters and start analysis',
                     transform=self.ax.transAxes, ha='center', va='center',
                     fontsize=14, color='#94A3B8')

        self.ax.set_title('Stock Price Prediction', color='white', fontsize=16, pad=20)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=15, pady=(0, 15))

    def start_prediction(self):
        """Start prediction in a separate thread"""
        try:
            symbol = self.stock_entry.get().upper()

            # Calculate period from days, months, years inputs
            days = int(self.days_entry.get() or 0)
            months = int(self.months_entry.get() or 0)
            years = int(self.years_entry.get() or 0)
            # max_period = self.max_entry.get().strip()

            # Determine period
            # if max_period and max_period.lower() == 'max':
            #     period = 'max'
            if days == 0 and months == 0 and years == 0:
                period = '1y'  # Default
            else:
                # Convert to yfinance period format
                total_days = days + (months * 30) + (years * 365)
                if total_days <= 7:
                    period = f"{total_days}d"
                elif total_days <= 60:
                    period = f"{int(total_days / 7)}wk"
                elif total_days <= 730:
                    period = f"{int(total_days / 30)}mo"
                else:
                    period = f"{int(total_days / 365)}y"

            try:
                epochs = int(self.epochs_entry.get())
                sims = int(self.sims_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for epochs and simulations")
                return

            if not symbol:
                messagebox.showerror("Error", "Please enter a stock symbol")
                return

            if epochs < 1 or epochs > 1000:
                messagebox.showerror("Error", "Epochs must be between 1 and 1000")
                return

            if sims < 1 or sims > 10:
                messagebox.showerror("Error", "Simulations must be between 1 and 10")
                return

            # Disable button and start progress
            self.predict_btn.configure(state='disabled', text='Processing...')
            self.progress.start(10)
            self.status_label.configure(text="Downloading data...")

            # Start prediction thread
            thread = threading.Thread(
                target=self.run_prediction,
                args=(symbol, period, epochs, sims)
            )
            thread.daemon = True
            thread.start()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for epochs and simulations")

    def run_prediction(self, symbol, period, epochs, sims):
        """Run prediction in background thread"""
        try:
            self.root.after(0, lambda: self.status_label.configure(text="Downloading data..."))

            # Download data
            df = download_data(symbol, period)

            if 'Volume' not in df.columns:
                self.result_queue.put(('error', f'No volume data available for {symbol}'))
                return

            self.root.after(0, lambda: self.status_label.configure(text="Preparing data..."))

            # Prepare data
            close_prices = df['Close'].values.reshape(-1, 1)
            volumes = df['Volume'].values.reshape(-1, 1)

            close_scaler = MinMaxScaler()
            volume_scaler = MinMaxScaler()

            normalized_close = close_scaler.fit_transform(close_prices)
            normalized_volume = volume_scaler.fit_transform(volumes)
            normalized_data = np.hstack([normalized_close, normalized_volume])

            if normalized_data.shape[1] != 2:
                raise ValueError("Normalized data must have two features: [close, volume]")

            # Parameters
            sequence_length = 10
            test_size = 30
            learning_rate = 0.001

            # Run simulations
            results = []
            losses = []

            for i in range(sims):
                # noinspection PyUnresolvedReferences
                self.root.after(0,
                                lambda i=i: self.status_label.configure(
                                    text=f"Running simulation {i + 1}/{sims}..."))

                pred, loss = forecast_single(
                    normalized_data, sequence_length, epochs,
                    learning_rate, test_size, close_scaler, volume_scaler
                )

                results.append(pred)
                losses.append(loss)

            self.root.after(0, lambda: self.status_label.configure(text="Processing results..."))

            # Process results
            dates = pd.to_datetime(df['Date']).tolist()
            original_close = df['Close'].values

            # Extend dates for future predictions
            for i in range(test_size):
                dates.append(dates[-1] + timedelta(days=1))

            # Filter results
            accepted_results = []
            for r in results:
                if not np.all(np.isnan(r)):
                    min_reasonable = np.min(original_close) * 0.5
                    max_reasonable = np.max(original_close) * 3
                    valid_predictions = r[~np.isnan(r)]

                    if len(valid_predictions) > 0:
                        if (np.min(valid_predictions) > min_reasonable and
                                np.max(valid_predictions) < max_reasonable):
                            accepted_results.append(r)

            if not accepted_results:
                self.result_queue.put(('error', 'No valid predictions generated. Try different parameters.'))
                return

            # Calculate final predicted price
            future_start_idx = len(original_close)
            future_predictions = []
            for result in accepted_results:
                if len(result) > future_start_idx and not np.isnan(result[future_start_idx]):
                    future_predictions.append(result[future_start_idx])

            predicted_price = np.mean(future_predictions) if future_predictions else original_close[-1]

            # Send results to main thread
            result_data = {
                'symbol': symbol,
                'dates': dates,
                'original_close': original_close,
                'accepted_results': accepted_results,
                'predicted_price': predicted_price,
                'df': df
            }

            self.result_queue.put(('success', result_data))

        except Exception as e:
            self.result_queue.put(('error', str(e)))

    def check_queue(self):
        """Check for results from background thread"""
        try:
            result_type, data = self.result_queue.get_nowait()

            if result_type == 'success':
                self.display_results(data)
            else:  # error
                messagebox.showerror("Prediction Error", data)
                self.reset_ui()

        except queue.Empty:
            pass

        # Continue checking
        self.root.after(100, self.check_queue)

    def display_results(self, data):
        """Display prediction results"""
        # Update predicted price
        self.price_label.configure(text=f"${data['predicted_price']:.2f}")

        # Clear and update chart
        self.ax.clear()

        # Plot historical data
        historical_dates = data['dates'][:len(data['original_close'])]
        self.ax.plot(historical_dates, data['original_close'],
                     label='Actual Price', color='#3B82F6', linewidth=2, marker='o', markersize=3)

        # Plot predictions
        colors = ['#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4']
        for i, result in enumerate(data['accepted_results'][:5]):  # Show max 5 predictions
            color = colors[i % len(colors)]
            valid_mask = ~np.isnan(result)
            if np.any(valid_mask):
                plot_dates = [data['dates'][j] for j in range(len(result)) if valid_mask[j]]
                plot_values = result[valid_mask]
                self.ax.plot(plot_dates, plot_values,
                             label=f'Prediction {i + 1}', color=color, linewidth=2,
                             linestyle='--', alpha=0.8)

        # Add vertical line to separate historical and future data
        self.ax.axvline(x=historical_dates[-1], color='#EF4444', linestyle=':', alpha=0.7,
                        label='Prediction Start')

        # Styling
        self.ax.set_title(f'{data["symbol"]} - Stock Price Prediction', color='white', fontsize=16, pad=20)
        self.ax.set_xlabel('Date', color='#94A3B8')
        self.ax.set_ylabel('Price ($)', color='#94A3B8')
        self.ax.legend(facecolor='#374151', edgecolor='#94A3B8', labelcolor='white')
        self.ax.grid(True, alpha=0.3, color='#94A3B8')

        # Format x-axis
        self.fig.autofmt_xdate()

        # Update canvas
        self.canvas.draw()

        # Reset UI
        self.reset_ui()
        self.status_label.configure(text=f"Prediction completed for {data['symbol']}")

    def reset_ui(self):
        """Reset UI elements after prediction"""
        self.predict_btn.configure(state='normal', text='🚀 PREDICT')
        self.progress.stop()


def main():
    """Main function to run the application"""
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create and run the application
    root = tk.Tk()
    app = StockPredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()