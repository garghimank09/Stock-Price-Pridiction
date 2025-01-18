import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the stock data
data = pd.read_csv('stock_data.csv')  # Replace 'stock_data.csv' with your actual dataset

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare the training data
look_back = 10  # Number of previous days' data to use for prediction
X, Y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:i + look_back, 0])
    Y.append(scaled_data[i + look_back, 0])
X, Y = np.array(X), np.array(Y)
# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train_X, train_Y = X[:train_size], Y[:train_size]
test_X, test_Y = X[train_size:], Y[train_size:]

# Reshape the input data for LSTM
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_X, train_Y, epochs=10, batch_size=32)
# Make predictions
train_predictions = model.predict(train_X)
test_predictions = model.predict(test_X)

# Scale the predictions back to original range
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Evaluate the model
train_rmse = np.sqrt(np.mean(np.square(train_predictions - train_Y)))
test_rmse = np.sqrt(np.mean(np.square(test_predictions - test_Y)))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
