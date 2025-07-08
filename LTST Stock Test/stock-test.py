# Make sure that you have all these libaries available to run the code successfully 
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler


# Alphavantage API Key
api_key = "UBZZZOJ2YBZPAITQ"

ticker = "AAL"

url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

# Save data to this file
file_to_save = 'stock_market_data-%s.csv'%ticker

# Make a JSON request to pull stock data from AlphaVantage
# Save into CSV
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
else:
    print('File already exists. Loading data from CSV')
    df = pd.read_csv(file_to_save)

    df = df.sort_values("Date")
    # print(df.head(10))

    # Plot Data
    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    # plt.show()

    # Split Data in Training and Test Set
    high_prices = df.loc[:,'High'].values
    low_prices = df.loc[:,'Low'].values
    mid_prices = (high_prices+low_prices)/2.0
    train_data = mid_prices[:11000]
    test_data = mid_prices[11000:]  
    print("data split")

    # Scale all Data
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)
    print("data scaled")

    # Smoothing will impact data differently -> use smoothing window
    smoothing_window_size = 2500
    train_data_length = len(train_data)
    num_complete_windows = train_data_length // smoothing_window_size


    for i in range(num_complete_windows):
        start_idx = i * smoothing_window_size
        end_idx = start_idx + smoothing_window_size
        print(f"Smoothing window {start_idx} to {end_idx}")
        
        scaler.fit(train_data[start_idx:end_idx,:])
        train_data[start_idx:end_idx,:] = scaler.transform(train_data[start_idx:end_idx,:])

    # Handle any remaining data
    remaining_start = num_complete_windows * smoothing_window_size
    if remaining_start < train_data_length:
        print(f"Smoothing remaining data from {remaining_start}")
        scaler.fit(train_data[remaining_start:,:])
        train_data[remaining_start:,:] = scaler.transform(train_data[remaining_start:,:])

    # You normalize the last bit of remaining data
    scaler.fit(train_data[i+smoothing_window_size:,:])
    train_data[i+smoothing_window_size:,:] = scaler.transform(train_data[i+smoothing_window_size:,:])

    # Reshape both train and test data
    train_data = train_data.reshape(-1)

    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)

    # Smooth data with exponential moving average
    EMA = 0.0
    gamma = 0.1
    for ti in range(11000):
      EMA = gamma*train_data[ti] + (1-gamma)*EMA
      train_data[ti] = EMA

    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data,test_data],axis=0)


    window_size = 100
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size,N):

        if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx,'Date']

        std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
        std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))


    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
    plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()







