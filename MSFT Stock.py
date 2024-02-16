Python 3.10.10 (tags/v3.10.10:aad5f6a, Feb  7 2023, 17:20:36) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import yfinance as yf
import pandas as pd
... import numpy as np
... import matplotlib.pyplot as plt
... from sklearn.ensemble import RandomForestClassifier
... from sklearn.metrics import classification_report
... from sklearn.model_selection import train_test_split
... import datetime as dt
... 
... # Define the ticker symbol and the time period for which data is required
... tickerSymbol = 'MSFT'
... today = dt.date.today()
... startDate = today.replace(year=today.year - 5, month=2, day=9)
... 
... # Download historical data as dataframe
... msft_data = yf.download(tickerSymbol, start=startDate, end=today)['Adj Close']
... 
... # Prepare data for training machine learning model
... data = pd.DataFrame(msft_data)
... data.rename(columns={'Adj Close': 'Actual_Close'}, inplace=True)
... data["Target"] = data["Actual_Close"].shift(-1) > data["Actual_Close"]
... data = data.dropna()
... 
... # Prepare predictors
... predictors = ["Actual_Close"]
... 
... # Define the model
... model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)
... 
... # Train the model
... X_train, X_test, y_train, y_test = train_test_split(data[predictors], data["Target"], test_size=0.2, random_state=1)
... X_train = np.array(X_train)
... X_test = np.array(X_test)
... y_train = np.array(y_train)
... y_test = np.array(y_test)
... model.fit(X_train, y_train)
... 
... # Test the model
... predictions = model.predict(X_test)
