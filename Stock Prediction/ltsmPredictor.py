import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_stock_data(ticker):
    api_key = 'UBZZZOJ2YBZPAITQ'
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
    file_to_save = 'stock_data/stock_market_data-%s.csv'%ticker
    
    os.makedirs('stock_data', exist_ok=True)

    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            if 'Time Series (Daily)' not in data:
                print("Error: Could not retrieve data from API")
                print("Response:", data)
                raise ValueError("API call failed or invalid ticker symbol")
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                            float(v['4. close']),float(v['1. open'])]
                df.loc[len(df)] = data_row
        print('Data saved to : %s'%file_to_save)        
        df.to_csv(file_to_save, index=False)
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    numeric_columns = ['Low', 'High', 'Close', 'Open']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    df = df.sort_values('Date')
    df = df.reset_index(drop=True)

    return df

def calc_mid_prices(df):
    high_prices = df.loc[:,'High'].values
    low_prices = df.loc[:,'Low'].values
    mid_prices = (high_prices + low_prices) / 2.0
    mid_prices = mid_prices.astype(np.float64)
    
    if np.any(np.isnan(mid_prices)) or np.any(np.isinf(mid_prices)):
        print("Warning: Found NaN or Inf values in mid prices")
        mid_prices = mid_prices[~(np.isnan(mid_prices) | np.isinf(mid_prices))]

    print(f"Total data points: {len(mid_prices)}")
    return mid_prices

def split_train_test(data, ratio=0.8):
    total_size = len(data)
    train_size = int(total_size * ratio)
    print(f"Using {train_size} points for training, {total_size - train_size} for testing")
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def scale_data(mid_prices, ratio):
    train_data, test_data = split_train_test(mid_prices, ratio)
    
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)
    
    # Fit scaler only on training data
    scaler.fit(train_data)
    train_data_scaled = scaler.transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    train_data_scaled = train_data_scaled.reshape(-1)
    test_data_scaled = test_data_scaled.reshape(-1)

    return train_data_scaled, test_data_scaled, scaler

def apply_smoothing(data, gamma=0.1):
    smoothed_data = data.copy()
    EMA = smoothed_data[0]
    smoothed_data[0] = EMA
    
    for ti in range(1, len(smoothed_data)):
        EMA = gamma*smoothed_data[ti] + (1-gamma)*EMA
        smoothed_data[ti] = EMA
    return smoothed_data

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def create_lstm_model(num_unrollings, num_nodes, dropout, learning_rate):
    model = keras.Sequential()
    
    model.add(keras.layers.LSTM(num_nodes[0], 
                               return_sequences=True, 
                               dropout=dropout,
                               recurrent_dropout=dropout,
                               input_shape=(num_unrollings, 1)))
    
    model.add(keras.layers.LSTM(num_nodes[1], 
                               return_sequences=True, 
                               dropout=dropout,
                               recurrent_dropout=dropout))
    
    model.add(keras.layers.LSTM(num_nodes[2], 
                               dropout=dropout,
                               recurrent_dropout=dropout))
    
    model.add(keras.layers.Dense(1, activation='linear'))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])
    
    print("Model created successfully!")
    model.summary()
    return model

def train_model(model, X_train, Y_train, epochs, batch_size):
    print(f"Training data shape: X={X_train.shape}, y={Y_train.shape}")

    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print("Warning: Found NaN or Inf in X_train")
        return None

    if np.any(np.isnan(Y_train)) or np.any(np.isinf(Y_train)):
        print("Warning: Found NaN or Inf in y_train")
        return None

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=3, min_lr=1e-7)
    ]

    print("Training the model...")
    history = model.fit(X_train, Y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=0.2,
                       callbacks=callbacks,
                       verbose=1)

    print("Training completed!")
    return history

def make_predictions(model, test_data, num_unrollings):
    X_test, y_test = create_sequences(test_data, num_unrollings)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    predictions = model.predict(X_test, verbose=0)
    predictions = predictions.flatten()
    
    mse = mean_squared_error(y_test, predictions)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Test MSE (normalized): {mse:.5f}")
    
    return predictions, mse

def predict_from_date_single_step(model, df, scaler, prediction_date, num_unrollings=60, days_ahead=60):
    """
    Single-step prediction: Each prediction uses real historical data, not previous predictions.
    This is more realistic and prevents error compounding.
    """
    # Convert prediction date to datetime
    pred_date = pd.to_datetime(prediction_date)
    
    # Find the index of the prediction date
    date_mask = df['Date'] == pred_date
    if not date_mask.any():
        # Find the closest date
        df['date_diff'] = abs(df['Date'] - pred_date)
        closest_idx = df['date_diff'].idxmin()
        actual_date = df.loc[closest_idx, 'Date']
        print(f"Exact date {prediction_date} not found. Using closest date: {actual_date.strftime('%Y-%m-%d')}")
        pred_start_idx = closest_idx
    else:
        pred_start_idx = df[date_mask].index[0]
        actual_date = pred_date
    
    # Check if we have enough historical data for the sequence
    if pred_start_idx < num_unrollings:
        raise ValueError(f"Not enough historical data. Need at least {num_unrollings} days before {prediction_date}")
    
    # Check if we have enough future data to compare against
    end_idx = pred_start_idx + days_ahead
    if end_idx > len(df):
        available_days = len(df) - pred_start_idx
        print(f"Warning: Only {available_days} days of actual data available after {prediction_date}")
        days_ahead = available_days
        end_idx = len(df)
    
    # Get ALL historical data (we'll use real data for each prediction)
    all_historical_df = df.iloc[:end_idx].copy()
    all_high_prices = all_historical_df['High'].values
    all_low_prices = all_historical_df['Low'].values
    all_mid_prices = (all_high_prices + all_low_prices) / 2.0
    
    # Scale all historical data
    all_mid_prices_scaled = scaler.transform(all_mid_prices.reshape(-1, 1)).flatten()
    
    # Get actual future prices for comparison
    if end_idx > pred_start_idx + 1:
        future_df = df.iloc[pred_start_idx + 1:end_idx].copy()
        future_high = future_df['High'].values
        future_low = future_df['Low'].values
        actual_future_prices = (future_high + future_low) / 2.0
    else:
        actual_future_prices = np.array([])
    
    # Single-step predictions using real historical data
    predictions = []
    
    print(f"Making single-step predictions using real historical sequences...")
    
    for i in range(days_ahead):
        # For each prediction, use the last num_unrollings days of REAL data
        # This ensures each prediction starts from the right sequence
        sequence_end_idx = pred_start_idx + i  # End at the day we want to predict FROM
        sequence_start_idx = sequence_end_idx - num_unrollings + 1  # Start num_unrollings days before
        
        if sequence_end_idx < len(all_mid_prices_scaled) and sequence_start_idx >= 0:
            # Use real historical data for the sequence
            input_sequence = all_mid_prices_scaled[sequence_start_idx:sequence_end_idx + 1]
            
            if len(input_sequence) == num_unrollings:
                # Reshape for LSTM input
                sequence_reshaped = input_sequence.reshape(1, num_unrollings, 1)
                
                # Make prediction for the NEXT day
                pred = model.predict(sequence_reshaped, verbose=0)
                predictions.append(pred[0, 0])
            else:
                print(f"Warning: Insufficient data for prediction {i+1}, got {len(input_sequence)} instead of {num_unrollings}")
                break
        else:
            print(f"Reached end of available data at prediction {i+1}")
            break
    
    # Convert predictions back to original price scale
    predictions_actual = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    # Get the actual price on the prediction start date
    start_actual_price_unscaled = all_mid_prices[pred_start_idx]
    
    print(f"Actual price on {prediction_date}: ${start_actual_price_unscaled:.2f}")
    print(f"First prediction (day after): ${predictions_actual[0]:.2f}" if len(predictions_actual) > 0 else "No predictions")
    
    # The first prediction should be close to the start price since it's predicting the next day
    if len(predictions_actual) > 0:
        print(f"Day-to-day change: ${predictions_actual[0] - start_actual_price_unscaled:.2f}")
    
    # Generate date range for predictions (starting the day after prediction_date)
    start_date = actual_date + pd.Timedelta(days=1)
    pred_dates = pd.date_range(start=start_date, periods=len(predictions_actual), freq='D')
    
    # Calculate metrics if we have actual data to compare
    if len(actual_future_prices) > 0:
        comparison_length = min(len(predictions_actual), len(actual_future_prices))
        mse = mean_squared_error(actual_future_prices[:comparison_length], predictions_actual[:comparison_length])
        mae = mean_absolute_error(actual_future_prices[:comparison_length], predictions_actual[:comparison_length])
        
        # Check for systematic bias
        bias = np.mean(predictions_actual[:comparison_length] - actual_future_prices[:comparison_length])
        print(f"Prediction bias (negative = underpredicting): ${bias:.2f}")
    else:
        mse = mae = None
        bias = None
    
    results = {
        'prediction_start_date': actual_date.strftime('%Y-%m-%d'),
        'predicted_prices': predictions_actual,
        'actual_prices': actual_future_prices[:len(predictions_actual)] if len(actual_future_prices) > 0 else [],
        'prediction_dates': pred_dates,
        'start_price': start_actual_price_unscaled,
        'mse': mse,
        'mae': mae,
        'bias': bias,
        'days_predicted': len(predictions_actual),
        'method': 'single_step'
    }
    
    return results

def plot_date_prediction(prediction_results, df, prediction_date, train_data, scaler):
    """
    Plot 1: Full actual vs prediction (all dates)
    Plot 2: Detailed view (just the 60 prediction days)
    """
    pred_prices = prediction_results['predicted_prices']
    actual_prices = prediction_results['actual_prices']
    pred_dates = prediction_results['prediction_dates']
    start_price = prediction_results['start_price']
    mse = prediction_results['mse']
    mae = prediction_results['mae']
    bias = prediction_results.get('bias', None)
    
    # Convert all data back to actual prices for full plot
    train_actual = scaler.inverse_transform(train_data.reshape(-1, 1)).flatten()
    all_mid_prices = (df['High'] + df['Low']) / 2.0
    
    # Find where prediction starts in the full dataset
    pred_start_date = pd.to_datetime(prediction_date)
    pred_start_idx = df[df['Date'] == pred_start_date].index[0] if not df[df['Date'] == pred_start_date].empty else \
                     df.iloc[(df['Date'] - pred_start_date).abs().argsort()[:1]].index[0]
    
    # Plot 1: Full view - All actual data + predictions overlay
    plt.figure(figsize=(18, 9))
    
    # Plot all historical actual prices
    plt.plot(df['Date'], all_mid_prices, 'b-', label='Actual Stock Prices', alpha=0.8, linewidth=1.5)
    
    # Add continuity point - show the starting price
    plt.plot(pred_start_date, start_price, 'ko', markersize=8, label=f'Prediction Start (${start_price:.2f})')
    
    # Overlay predictions on top
    plt.plot(pred_dates, pred_prices, 'r-', label='LSTM Predictions', linewidth=3)
    
    # Add vertical line at prediction start
    plt.axvline(x=pred_start_date, color='black', linestyle=':', alpha=0.7, linewidth=2, label='Prediction Date')
    
    title = f'LSTM Stock Prediction - Full Historical View'
    if mse is not None and mae is not None:
        title += f' (MSE: {mse:.2f}, MAE: {mae:.2f}'
        if bias is not None:
            title += f', Bias: {bias:.2f}'
        title += ')'
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Detailed view - Just the 60 prediction days with continuity
    plt.figure(figsize=(18, 9))
    
    # Show the starting point for continuity
    plt.plot(pred_start_date, start_price, 'ko', markersize=10, 
             label=f'Start Point (${start_price:.2f})', zorder=5)
    
    # Connect start point to first prediction with a dotted line
    if len(pred_prices) > 0:
        plt.plot([pred_start_date, pred_dates[0]], [start_price, pred_prices[0]], 
                'r:', linewidth=2, alpha=0.7, label='Transition')
    
    # Plot predictions
    plt.plot(pred_dates, pred_prices, 'r-', label='LSTM Predictions', linewidth=3, marker='o', markersize=4)
    
    # Plot actual prices if available
    if len(actual_prices) > 0:
        actual_dates = pred_dates[:len(actual_prices)]
        plt.plot(actual_dates, actual_prices, 'g-', label='Actual Prices', linewidth=3, marker='s', markersize=4)
        
        # Add horizontal reference lines at +/- 8% from the FINAL actual price
        final_actual_price = actual_prices[-1]  # Last actual price in the sequence
        upper_8_percent = final_actual_price * 1.08
        lower_8_percent = final_actual_price * 0.92
        
        # Extend lines across the entire prediction period
        plt.axhline(y=upper_8_percent, color='gray', linestyle='--', alpha=0.6, 
                   label=f'+8% from final actual (${upper_8_percent:.2f})')
        plt.axhline(y=lower_8_percent, color='gray', linestyle='--', alpha=0.6, 
                   label=f'-8% from final actual (${lower_8_percent:.2f})')
        
        # Add text annotations for the reference lines
        plt.text(pred_dates[-1], upper_8_percent + (final_actual_price * 0.01), '+8%', 
                ha='right', va='bottom', color='gray', fontsize=10)
        plt.text(pred_dates[-1], lower_8_percent - (final_actual_price * 0.01), '-8%', 
                ha='right', va='top', color='gray', fontsize=10)
    
    detail_title = f'LSTM Predictions - 60 Day Detailed View from {prediction_date}'
    if mse is not None and mae is not None:
        detail_title += f' (MSE: {mse:.2f}, MAE: {mae:.2f}'
        if bias is not None:
            bias_text = "underpredicting" if bias < 0 else "overpredicting"
            detail_title += f', {bias_text} by ${abs(bias):.2f}'
        detail_title += ')'
    
    plt.title(detail_title, fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print summary with bias analysis
    print(f"\nPrediction Summary:")
    print(f"Start Date: {prediction_results['prediction_start_date']}")
    print(f"Days Predicted: {prediction_results['days_predicted']}")
    print(f"Starting Price: ${start_price:.2f}")
    print(f"First Prediction: ${pred_prices[0]:.2f}" if len(pred_prices) > 0 else "No predictions")
    print(f"Price Gap: ${abs(pred_prices[0] - start_price):.2f}" if len(pred_prices) > 0 else "")
    
    if len(actual_prices) > 0:
        print(f"First Actual Price: ${actual_prices[0]:.2f}")
        print(f"Final Predicted Price: ${pred_prices[-1]:.2f}")
        print(f"Final Actual Price: ${actual_prices[-1]:.2f}")
    
    if mse is not None:
        print(f"Overall MSE: {mse:.2f}")
        print(f"Overall MAE: {mae:.2f}")
        
    if bias is not None:
        if abs(bias) > 0.5:  # Significant bias threshold
            bias_direction = "underpredicting" if bias < 0 else "overpredicting"
            print(f"⚠️  Model is systematically {bias_direction} by ${abs(bias):.2f}")
            print("Consider adjusting model parameters or adding bias correction.")
        else:
            print("✅ No significant systematic bias detected.")

def run_prediction(ticker,
                   train_ratio=0.8, 
                   num_unrollings=60,  
                   num_nodes=[128, 64, 32],  
                   dropout=0.3,  
                   learning_rate=0.001, 
                   epochs=50,  
                   gamma=0.1,
                   prediction_date=None):

    print(f"Starting stock prediction for {ticker}")
    print("=" * 60)

    print("Step 1: Loading stock data...")
    df = load_stock_data(ticker)

    print("\nStep 2: Calculating mid prices...")
    mid_prices = calc_mid_prices(df)

    print("\nStep 3: Splitting and scaling data...")
    train_data, test_data, scaler = scale_data(mid_prices, train_ratio)

    print("\nStep 4: Applying exponential smoothing...")
    train_data = apply_smoothing(train_data, gamma)

    print("\nStep 5: Creating sequences...")
    X_train, y_train = create_sequences(train_data, num_unrollings)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)).astype(np.float32)
    y_train = y_train.reshape((y_train.shape[0])).astype(np.float32)
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    if len(X_train) == 0:
        raise ValueError("No training sequences created. Check if data is sufficient.")

    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("Training data contains NaN or Inf values")

    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        raise ValueError("Training labels contain NaN or Inf values")

    print("\nStep 6: Creating LSTM model...")
    batch_size = min(32, len(X_train) // 10)
    print(f"Using batch size: {batch_size}, sequence length: {num_unrollings}")

    model = create_lstm_model(num_unrollings, num_nodes, dropout, learning_rate)

    print("\nStep 7: Training model...")
    history = train_model(model, X_train, y_train, epochs, batch_size)

    # If prediction_date is specified, use date-based prediction
    if prediction_date is not None:
        print(f"\nStep 8: Making single-step predictions from {prediction_date}...")
        pred_results = predict_from_date_single_step(model, df, scaler, prediction_date, num_unrollings, days_ahead=60)
        
        print("\nStep 9: Creating visualization...")
        plot_date_prediction(pred_results, df, prediction_date, train_data, scaler)
        
        results = {
            'model': model,
            'history': history,
            'prediction_results': pred_results,
            'scaler': scaler,
            'dataframe': df,
            'train_data': train_data
        }
        
    else:
        # Original prediction method
        print("\nStep 8: Making predictions...")
        predictions, mse_norm = make_predictions(model, test_data, num_unrollings)

        print("\nStep 9: Creating visualizations...")
        # Simple full data plot
        plt.figure(figsize=(18,9))
        mid_prices = (df['Low'] + df['High']) / 2.0
        plt.plot(range(df.shape[0]), mid_prices, linewidth=1.5)
        step = max(1, df.shape[0] // 10)
        date_indices = range(0, df.shape[0], step)
        date_labels = [df['Date'].iloc[i].strftime('%Y-%m-%d') for i in date_indices]
        plt.xticks(date_indices, date_labels, rotation=45)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Mid Price ($)', fontsize=18)
        plt.title(f'Stock Price Over Time', fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Calculate MSE and MAE in actual prices
        train_actual = scaler.inverse_transform(train_data.reshape(-1, 1)).flatten()
        test_actual = scaler.inverse_transform(test_data.reshape(-1, 1)).flatten()
        pred_actual = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        test_subset = test_actual[:len(pred_actual)]
        mse_actual = mean_squared_error(test_subset, pred_actual)
        mae_actual = mean_absolute_error(test_subset, pred_actual)

        results = {
            'model': model,
            'history': history,
            'predictions': predictions,
            'mse_normalized': mse_norm,
            'mse_actual': mse_actual,
            'mae_actual': mae_actual,
            'train_data': train_data,
            'test_data': test_data,
            'scaler': scaler,
            'dataframe': df
        }

    print("\nLSTM Stock Market Prediction Complete!")
    print("=" * 60)
    
    return results

# Run the prediction

# Train on all historical data, but show prediction from specific date
results = run_prediction(
    ticker="BWXT",
    prediction_date="2023-06-01"  # Only the plot will focus on this date + 60 days
)

# Alternative: Traditional full train/test view
# results = run_prediction(ticker="BWXT")

if 'prediction_results' in results:
    pred_res = results['prediction_results']
    print(f"\nFinal Results for date prediction:")
    if pred_res['mse'] is not None:
        print(f"MSE: {pred_res['mse']:.2f}")
        print(f"MAE: {pred_res['mae']:.2f}")
    print(f"Predicted {pred_res['days_predicted']} days forward")
else:
    print(f"\nFinal Results:")
    print(f"Test MSE (normalized): {results['mse_normalized']:.5f}")
    print(f"Test MSE (actual $): {results['mse_actual']:.2f}")
    print(f"Test MAE (actual $): {results['mae_actual']:.2f}")