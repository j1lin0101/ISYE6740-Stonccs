import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Load Data

data_source = 'alphavantage'  # Using Alpha Vantage API

if data_source == 'alphavantage':
    # ====================== Loading Data from Alpha Vantage ==================================
    api_key = 'UBZZZOJ2YBZPAITQ'  # Your Alpha Vantage API key
    
    # American Airlines stock market prices
    ticker = "AAL"
    
    # JSON file with all the stock market data for AAL from the last 20 years
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
    
    # Save data to this file
    file_to_save = 'stock_market_data-%s.csv'%ticker
    
    # Check if saved
    # If not make the data frame of time series data
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # extract stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                            float(v['4. close']),float(v['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        print('Data saved to : %s'%file_to_save)        
        df.to_csv(file_to_save)
    
    # Otherwise load CSV
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)

# Sort DataFrame by date
df = df.sort_values('Date')

# Display first few rows
print("First 5 rows of data:")
print(df.head())

# Data visualization
plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.title('Stock Price Over Time')
plt.show()

# Data Preprocessing

# Calculate mid prices from the highest and lowest
high_prices = df.loc[:,'High'].values
low_prices = df.loc[:,'Low'].values
mid_prices = (high_prices + low_prices) / 2.0

print(f"Total data points: {len(mid_prices)}")

# Adjust train size based on available data
total_size = len(mid_prices)
train_ratio = 0.8
train_size = int(total_size * train_ratio)

print(f"Using {train_size} points for training, {total_size - train_size} for testing")

# Split training and test data
train_data = mid_prices[:train_size]
test_data = mid_prices[train_size:]

# Scale the data to be between 0 and 1
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

# Adaptive windowed normalization based on data size
if train_size > 5000:
    # Use windowed normalization for large datasets
    smoothing_window_size = min(2500, train_size // 4)
    num_windows = train_size // smoothing_window_size
    
    for i in range(num_windows):
        start_idx = i * smoothing_window_size
        end_idx = min((i + 1) * smoothing_window_size, train_size)
        
        if end_idx > start_idx:  # Ensure we have data in the window
            scaler_window = MinMaxScaler()
            scaler_window.fit(train_data[start_idx:end_idx])
            train_data[start_idx:end_idx] = scaler_window.transform(train_data[start_idx:end_idx])
    
    # Handle remaining data
    if num_windows * smoothing_window_size < train_size:
        remaining_start = num_windows * smoothing_window_size
        scaler_remaining = MinMaxScaler()
        scaler_remaining.fit(train_data[remaining_start:])
        train_data[remaining_start:] = scaler_remaining.transform(train_data[remaining_start:])
        
    # For test data normalization, use the last scaler
    test_data = scaler_remaining.transform(test_data)
else:
    # For smaller datasets, use simple normalization
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

# Reshape both train and test data
train_data = train_data.reshape(-1)
test_data = test_data.reshape(-1)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(len(train_data)):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)


# Data Generator for LSTM

class DataGeneratorSeq(object):
    """
    Data generator for LSTM training with data augmentation
    """
    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._prices_length:
                self._cursor[b] = np.random.randint(0, (b+1) * self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]

            self._cursor[b] = (self._cursor[b] + 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()    
            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b+1) * self._segments, self._prices_length-1))

# Test data generator
dg = DataGeneratorSeq(train_data,5,5)
u_data, u_labels = dg.unroll_batches()

for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):   
    print('\n\nUnrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ',dat )
    print('\n\tOutput:',lbl)

# LSTM with Tensorflow

# Define hyperparameters
num_unrollings = 50  # Sequence length
batch_size = min(500, train_size // 20)  # Adaptive batch size
num_nodes = [200, 200, 150]  # Number of hidden nodes in each layer
dropout = 0.2  # dropout amount
learning_rate = 0.001
epochs = 30

print(f"Using batch size: {batch_size}, sequence length: {num_unrollings}")

# Prepare data for TensorFlow 2.x
def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Create sequences
X_train, y_train = create_sequences(train_data, num_unrollings)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
y_train = y_train.reshape((y_train.shape[0], 1))

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

# Create the LSTM model using Keras
model = keras.Sequential()

# Add LSTM layers
model.add(keras.layers.LSTM(num_nodes[0], return_sequences=True, dropout=dropout, 
                           input_shape=(num_unrollings, 1)))
model.add(keras.layers.LSTM(num_nodes[1], return_sequences=True, dropout=dropout))
model.add(keras.layers.LSTM(num_nodes[2], dropout=dropout))

# Add output layer
model.add(keras.layers.Dense(1))

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mse',
              metrics=['mae'])

print("Model created successfully!")
model.summary()

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_split=0.1,
                   verbose=1)

print("Training completed!")

# Predicting

# Make predictions on test data
n_predict_once = min(50, len(test_data) // 2)
test_start = len(train_data)

# Create test sequences
predictions_list = []
x_axis_list = []

# Use the last num_unrollings points from training data as the starting sequence
initial_sequence = train_data[-num_unrollings:].reshape(1, num_unrollings, 1)

# Make multiple prediction sequences
num_prediction_points = min(10, len(test_data) // n_predict_once)

for start_idx in range(0, num_prediction_points * n_predict_once, n_predict_once):
    if start_idx + n_predict_once >= len(test_data):
        break
        
    # Reset to a point in the training data for each prediction sequence
    sequence_start = max(0, len(train_data) - num_unrollings + start_idx)
    current_sequence = all_mid_data[sequence_start:sequence_start + num_unrollings].reshape(1, num_unrollings, 1)
    
    predictions = []
    x_axis = []
    
    # Make sequential predictions
    for i in range(n_predict_once):
        # Predict next value
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0, 0])
        x_axis.append(test_start + start_idx + i)
        
        # Update sequence by removing first element and adding prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = pred[0, 0]
    
    predictions_list.append(np.array(predictions))
    x_axis_list.append(x_axis)

print(f"Generated {len(predictions_list)} prediction sequences")

# Calculate test MSE
test_mse_list = []
for i, (preds, x_vals) in enumerate(zip(predictions_list, x_axis_list)):
    mse = 0.0
    for j, (pred, x_val) in enumerate(zip(preds, x_vals)):
        if x_val - test_start < len(test_data):
            true_val = test_data[x_val - test_start]
            mse += (pred - true_val) ** 2
    mse /= len(preds)
    test_mse_list.append(mse)

avg_test_mse = np.mean(test_mse_list)
print(f"Average Test MSE: {avg_test_mse:.5f}")

# Data Visualization

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()

# Plot predictions
plt.figure(figsize=(18, 12))

# Plot 1: Overview with all predictions
plt.subplot(2, 1, 1)
plt.plot(range(len(all_mid_data)), all_mid_data, color='b', label='True', linewidth=1)
for i, (preds, x_vals) in enumerate(zip(predictions_list, x_axis_list)):
    alpha = 0.3 + 0.7 * (i / len(predictions_list))  # Varying transparency
    plt.plot(x_vals, preds, color='r', alpha=alpha, linewidth=1)

plt.title('LSTM Predictions vs True Values (Overview)', fontsize=14)
plt.xlabel('Time Steps')
plt.ylabel('Normalized Price')
plt.legend()
plt.xlim(test_start - 100, len(all_mid_data))

# Plot 2: Detailed view of best predictions
plt.subplot(2, 1, 2)
plt.plot(range(len(all_mid_data)), all_mid_data, color='b', label='True', linewidth=2)

# Plot the prediction with lowest MSE
if test_mse_list:
    best_idx = np.argmin(test_mse_list)
    plt.plot(x_axis_list[best_idx], predictions_list[best_idx], 
             color='r', label=f'Best Prediction (MSE: {test_mse_list[best_idx]:.5f})', linewidth=2)

plt.title('Best LSTM Prediction vs True Values', fontsize=14)
plt.xlabel('Time Steps')
plt.ylabel('Normalized Price')
plt.legend()
plt.xlim(test_start - 50, min(len(all_mid_data), test_start + n_predict_once * 2))
plt.tight_layout()
plt.show()

print("\nLSTM Stock Market Prediction Complete!")