import numpy as np
import pandas as pd
import yfinance as yf
import visualization as vis
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Function to download historical price data for given tickers
def download_price_data(tickers, start_date='2015-01-01', end_date='2024-01-01'):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Fix missing frequency information
    data.index = pd.to_datetime(data.index)
    if data.index.freq is None:
        data = data.asfreq('B')  # Set frequency to business days (B)
        
    return data

# Function to calculate SMA and EWMA
def calculate_moving_averages(data, window=12):
    sma = data.rolling(window=window).mean()
    ewma = data.ewm(span=window).mean()
    return sma, ewma

# ARIMA for mean prediction
def arima_prediction(data, order=(1, 1, 1), steps=1, return_per='ME'):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    
    # Convert forecast to Series for easier indexing
    forecast_index = pd.date_range(start=data.index[-1], periods=steps + 1, freq=return_per)[1:] 
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    return forecast_series

# GARCH for variance prediction
def garch_prediction(data, p=1, q=1, steps=1, return_per='ME'):
    # Rescale data to improve GARCH performance (optional but often useful)
    data_rescaled = data * 10
    
    # Fit the GARCH model
    model = arch_model(data_rescaled, vol='Garch', p=p, q=q)
    model_fit = model.fit(disp="off")
    
    # Forecast variance for the specified horizon (future steps)
    forecast = model_fit.forecast(horizon=steps)
    garch_var = forecast.variance.iloc[-1] / (100)  # Rescale the predicted variance back

    # Return the predicted variance series
    return garch_var, model_fit
    
def extract_single_variance(garch_var):
    return garch_var[-1]

# Main function to analyze and predict future u and sigma
def predict_next_year(ticker, return_per='ME', horizon=12, verbose=False, start_date='2015-01-01', end_date='2024-01-01'):
    # Download historical price data for the ticker
    data = download_price_data(ticker, start_date, end_date)
    data = data.fillna(data.rolling(window=12, min_periods=1).mean())

    data_ret = data.resample(return_per).last()
    returns = data_ret.pct_change(fill_method=None).dropna()

    if verbose:
        print(data.head())
        vis.plt_price(data, ticker)
        # vis.plt_returns(returns, ticker, return_per)
    
    sma, ewma = calculate_moving_averages(returns)
    arima_mean = arima_prediction(returns, steps=horizon, return_per=return_per) 
    hist_variance = returns.var()
    garch_var, model_fit = garch_prediction(returns, steps=horizon, return_per=return_per)

    if verbose:
        vis.plt_moving_averages(returns, sma, ewma, ticker)
        vis.plt_arima_prediction(returns, arima_mean, ticker)
        # vis.plt_garch_variance(returns, hist_variance, garch_var, ticker)
        vis.plt_garch_prediction(returns, model_fit, garch_var, ticker)
        print(f'Historical Variance: {hist_variance:.6f}')
        print(f'GARCH Forecasted Variance: {extract_single_variance(garch_var)}')

    # Returning multiple predictions for both mean (u) and variance (sigma)
    predictions = {
        'SMA_mean': sma.iloc[-1],
        'EWMA_mean': ewma.iloc[-1],
        'ARIMA_mean': arima_mean.iloc[-1],
        'GARCH_variance': extract_single_variance(garch_var)
    }
    
    return predictions

# Example usage with NYSE tickers
if __name__ == "__main__":
    ticker = 'AVGO'  # Example: AAPL, MSFT, GOOGL
    prediction = predict_next_year(ticker, return_per='W', horizon = 52, verbose=True)
    print(prediction)
