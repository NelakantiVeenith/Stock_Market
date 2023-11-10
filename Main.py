import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to perform linear regression and make predictions
def predict_stock_future(stock_symbol):
    # Convert the stock symbol to uppercase
    stock_symbol = stock_symbol.upper()

    # Add the market suffix if necessary (for Tata Power listed on NSE, '.NS' is required)
    if ' ' in stock_symbol:
        stock_symbol = stock_symbol.replace(' ', '') + '.NS'

    # Fetch the historical data using yfinance
    try:
        stock_data = yf.download(stock_symbol, period="5y")  # Fetch 5 years of data
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Extract the relevant columns: 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return'
    data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    data = data.copy()

    # Calculate daily returns using .loc to avoid SettingWithCopyWarning
    data.loc[:, 'Daily_Return'] = data['Close'].pct_change()

    # Drop rows with missing values using .loc
    data.dropna(inplace=True)

    # Define the features (X) and target (y) with explicit column names
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']]
    X.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']  # Set column names explicitly
    y = data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Visualize the results (optional)
    plt.figure(figsize=(12, 6))
    plt.scatter(X_test['Volume'], y_test, color='blue', label='Actual Prices')
    plt.scatter(X_test['Volume'], y_pred, color='red', label='Predicted Prices')
    plt.xlabel('Volume')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.title(f'Linear Regression Predictions for {stock_symbol}')
    plt.show()

    # Predict the future closing price based on the last available data point
    last_data_point = X.tail(1)
    future_open = last_data_point['Open'].values[0]
    future_high = last_data_point['High'].values[0]
    future_low = last_data_point['Low'].values[0]
    future_close = last_data_point['Close'].values[0]
    future_volume = last_data_point['Volume'].values[0]
    future_return = last_data_point['Daily_Return'].values[0]
    future_price = model.predict([[future_open, future_high, future_low,future_close, future_volume, future_return]])[0]

    # Get the date of the last available data point
    last_date = stock_data.index[-1].strftime('%Y-%m-%d')

    return last_date, future_price

# Example usage:
# Prompt the user for the stock symbol
stock_symbol = input("What stock are you looking for? ").strip()
# Call the function to predict the future stock price
predicted_date, predicted_price = predict_stock_future(stock_symbol)
print(f"Predicted future stock price for {stock_symbol} on {predicted_date}: {predicted_price:.2f}")
