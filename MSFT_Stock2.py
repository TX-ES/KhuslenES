from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Download data from Yahoo Finance
symbol = 'MSFT'
start_date = '2024-01-01'
end_date = '2024-02-16'
data = yf.download(symbol, start=start_date, end=end_date)

# Extracting closing prices
df = pd.DataFrame(data['Close'])

# Feature engineering - adding a column with previous day's closing price
df['Previous_Close'] = df['Close'].shift(1)

# Drop rows with NaN values
df = df.dropna()

# Split the data into training and testing sets
X = df[['Previous_Close']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(df.index[-len(y_test):], predictions, label='Predicted Prices')
plt.title('Stock Price Prediction using Linear Regression')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
