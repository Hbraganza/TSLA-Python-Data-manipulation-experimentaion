import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Define the ticker symbol
ticker_symbol = 'TSLA'

# Download all available data using period='max'
df = yf.download(ticker_symbol, period='max')

# Access the Close price column (if MultiIndex structure exists)
df['Close'] = df[('Close', ticker_symbol)] if isinstance(df.columns, pd.MultiIndex) else df['Close']

# Drop rows with missing values (if any)
df = df.dropna()

# Print shape of data before lagging
print(f"Shape of data before lagging: {df.shape}")

# Set lag_features to a reasonable number based on the size of the data
# Let's limit lag_features to be a maximum of 30 or fewer based on data size
lag_features = min(900, len(df) - 1)  # Use a maximum of 30 lag features

# Create lagged features for Close using pd.concat to avoid fragmentation
lags = [df['Close'].shift(lag) for lag in range(1, lag_features + 1)]

# Combine lag features into a DataFrame
lagged_df = pd.concat(lags, axis=1, keys=[f"Close_t-{lag}" for lag in range(1, lag_features + 1)])

# Add a moving average (7-day window) as a feature
df['Close_MA7'] = df['Close'].rolling(window=7).mean()
df['Close_MA30'] = df['Close'].rolling(window=30).mean()
df['Close_MA100'] = df['Close'].rolling(window=100).mean()

# Concatenate the lagged data to the original df
df = pd.concat([df, lagged_df], axis=1)

# Print shape of data after adding lag features
print(f"Shape of data after adding lag features: {df.shape}")

# Drop rows with NaN values caused by lagging (this will remove rows at the start of the data)
df = df.dropna()

# Print shape of data after dropping NaN values
print(f"Shape of data after dropping NaN values: {df.shape}")

# Prepare the feature set (X) and target variable (y)
X = df[[f"Close_t-{lag}" for lag in range(1, lag_features + 1)]]
y = df['Close']

# Check the shape of X and y before splitting
print(f"Shape of feature set X: {X.shape}")
print(f"Shape of target variable y: {y.shape}")

# Ensure there's enough data for training
if X.shape[0] == 0:
    print("Not enough data for training after creating lag features. Try reducing lag_features.")
    exit()

# Chronological train-test split (80% train, 20% test)
split_idx = int(len(X) * 0.99)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Initialize and train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42, max_depth=10)

# Train the model if enough data is available
if X_train.shape[0] > 0:
    model.fit(X_train, y_train)
else:
    print("Not enough data for training. Please reduce the number of lag features.")
    exit()

# Make predictions for the test set
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# Extend predictions into 2026
last_known_values = df['Close'].iloc[-lag_features:].values.tolist()  # Ensure we have the last `lag_features` values as a list

# Flatten last_known_values
last_known_values = [value[0] for value in last_known_values]  # Extract the number from each inner list

future_predictions = []
current_date = df.index[-1]

# Predict daily Close values up to the end of 2026
while current_date.year < 2026:
    # Ensure last_known_values contains exactly lag_features elements
    input_features = np.array(last_known_values).reshape(1, -1)  # Convert to a 2D NumPy array
    next_close = model.predict(input_features)[0]
    
    # Append the predicted Close value
    future_predictions.append(next_close)
    
    # Update the lagged values list
    last_known_values = last_known_values[1:] + [next_close]  # Shift by one and add the new prediction
    
    # Increment the date
    current_date += pd.Timedelta(days=1)
    
    # Skip weekends if needed (optional)
    while current_date.weekday() > 4:  # Saturday (5) or Sunday (6)
        current_date += pd.Timedelta(days=1)

# Create a DataFrame for future predictions
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=len(future_predictions), freq='B')
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions}).set_index('Date')

# Plot the historical and future predictions
plt.figure(figsize=(12, 8))
plt.plot(df['Close_MA7'], label="Seven Day Average")
plt.plot(df['Close_MA30'], label="Thirty Day Average")
plt.plot(df['Close_MA100'], label="Hundred Day Average")
plt.plot(df.index, df['Close'], label="Historical Close Prices", marker='o')
plt.plot(future_df.index, future_df['Predicted Close'], label="Future Predicted Close Prices", marker='x')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Tesla Close Price Prediction Into 2026")
plt.legend()
plt.grid(True)
plt.show()