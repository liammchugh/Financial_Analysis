
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from utils.download import download_returns

def visualize_portfolio(tickers, series, weight_hist, start, end, clusters, initial_value, per=2, SP500ref=True):
    """
        Visualize the portfolio value over time.

        Args:
            tickers (list): List of stock tickers.
            series (pd.Series): Series containing portfolio values over time.
            weight_hist (pd.DataFrame): DataFrame containing weights for each ticker over time.
            start (str): Start date for the data.
            end (str): End date for the data.
            clusters: Self-learned clusters of assets.
            initial_value (float): Initial value of the portfolio.
            per (int): Rebalance period in days.
        """

    # Optional: Download benchmark (S&P 500) for comparison
    try:
        sp500_raw = yf.download(
        '^GSPC',
        start=start,
        end=end,
        interval="1wk",
        auto_adjust=True,
        actions=False
        )
        if not sp500_raw.empty:
            sp500 = sp500_raw[['Close']].rename(columns={'Close': '^GSPC'})
    except Exception as e:
        print(f"Error downloading S&P 500 data: {e}")  
    sp500_returns = sp500.pct_change().dropna()
    # Align the benchmark returns with the portfolio timestamps
    sp500_returns.index = sp500_returns.index.tz_localize(None)
    sp500_value = initial_value * (1 + sp500_returns).cumprod()
    # Plot the portfolio value and S&P 500 value on the first figure
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(series.index, series, label="Portfolio Value", linestyle='-', alpha=0.7)
    if SP500ref:
        ax1.plot(sp500_value.index, sp500_value, label="S&P 500 Value", linestyle='--', alpha=0.7)
    ax1.set_ylabel('Portfolio & S&P 500 Value')
    ax1.legend(loc='upper left')
    plt.title(f'Portfolio Value Over Time: Rebalancing per {per} Days')
    # Calculate and display average annual return
    total_years = (series.index[-1] - series.index[0]).days / 365.25
    avg_annual_return = ((series.iloc[-1] / initial_value) ** (1 / total_years) - 1) * 100  
    ax1.text(0.5, 0.95, f"Total Return: {avg_annual_return:.2f}%", transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))  
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.show()

    # Plot weight history on the second figure
    # Number of weeks to show in the weight history plot
    hist_N = 30  # Number of days to show in the weight history plot
    weight_context_len = int(hist_N  / per)  # Show the last N weeks of weights
    weight_hist_trimmed = weight_hist.iloc[-weight_context_len:]
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        plt.plot(weight_hist_trimmed.index, weight_hist_trimmed[ticker], label=f"{ticker} Weight", linestyle='--', alpha=0.7)
    plt.title('Equity Weights Over Time (Last Few Weeks)')
    plt.xlabel('Date')
    plt.ylabel('Equity Weights (0-1)')
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


# Function to visualize the price history
def plt_price(data, ticker):
    """
    Visualizes the historical price of a stock.
    
    Parameters:
    data (pd.Series or pd.DataFrame): Adjusted closing prices of the stock.
    ticker (str): The ticker symbol of the stock.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=f'{ticker} Price', color='blue')
    plt.title(f'{ticker} Historical Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Function to visualize the returns
def plt_returns(returns, ticker, period='M'):
    """
    Visualizes the historical returns of a stock.
    
    Parameters:
    returns (pd.Series): Percentage fractional returns of the stock.
    ticker (str): The ticker symbol of the stock.
    period (str): The resampling period for returns calculation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(returns, label=f'{ticker} Returns', color='blue')
    plt.title(f'{ticker} Historical Returns')
    plt.xlabel(f'Date ({period})')
    plt.ylabel(f'Returns [frac]')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Function to visualize the moving averages
def plt_moving_averages(data, sma, ewma, ticker):
    """
    Visualizes the stock's price along with Simple and Exponential Moving Averages.
    
    Parameters:
    data (pd.Series): Adjusted closing prices.
    sma (pd.Series): Simple Moving Average.
    ewma (pd.Series): Exponential Weighted Moving Average.
    ticker (str): The ticker symbol of the stock.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=f'{ticker} Price', color='blue')
    
    # Normalize moving averages to the same scale as the price data for visualization
    if not sma.empty:
        plt.plot(sma, label='SMA', linestyle='--', color='orange')
    if not ewma.empty:
        plt.plot(ewma, label='EWMA', linestyle='--', color='green')
    
    plt.title(f'{ticker} Price with SMA and EWMA')
    plt.xlabel('Date')
    plt.ylabel('Price/Moving Averages')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Function to visualize the GARCH variance prediction
def plt_garch_variance(returns, hist_var, predicted_variance, ticker):
    """
    Visualizes the returns and the predicted variance from GARCH.
    
    Parameters:
    returns (pd.Series): The stock's percentage returns.
    hist_var (float): The historical variance of the returns.
    predicted_variance (pd.Series): The variance predicted by the GARCH model for the forecast horizon.
    ticker (str): The ticker symbol of the stock.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(returns, label=f'{ticker} Historical Returns', color='blue')
    plt.plot(returns.index, [hist_var] * len(returns), label='Historical Variance', color='green', linestyle='--')
    # Generate future dates for the variance prediction
    future_dates = pd.date_range(start=returns.index[-1], periods=len(predicted_variance), freq='W')
    
    # Plot the predicted variance (on the future dates)
    plt.plot(future_dates, predicted_variance, label='Predicted Variance', color='red', linestyle='--')
    
    plt.title(f'{ticker} Returns with GARCH Variance Prediction')
    plt.xlabel('Date')
    plt.ylabel('Returns / Variance')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# Function to visualize the GARCH variance prediction
def plt_garch_prediction(returns, model_fit, forecasted_variance, ticker):
    """
    Visualizes the GARCH model's predicted conditional variance (volatility) and future forecasted variance.
    
    Parameters:
    returns (pd.Series): The historical returns.
    model_fit: The fitted GARCH model object.
    forecast: The forecast object returned by the GARCH model.
    ticker (str): The ticker symbol of the stock.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the historical returns (for context)
    plt.plot(returns.index, returns, label=f'{ticker} Historical Returns', color='blue', alpha=0.5)
    
    # Plot the conditional volatility (the fitted GARCH variance)
    conditional_volatility = model_fit.conditional_volatility / 10  # Rescale back to original data scale
    plt.plot(returns.index, conditional_volatility, label='Conditional Volatility (GARCH)', color='orange')
    
    # Generate future dates for the forecasted variance
    future_dates = pd.date_range(start=returns.index[-1], periods=len(forecasted_variance), freq='W')
    
    # Plot the forecasted variance (for the specified horizon)
    plt.plot(future_dates, forecasted_variance ** 0.5, label='Forecasted Volatility (GARCH)', color='red', linestyle='--')
    
    plt.title(f'{ticker} GARCH Conditional Volatility and Forecasted Volatility')
    plt.xlabel('Date')
    plt.ylabel('Returns / Volatility')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# Function to visualize ARIMA prediction vs actual data
def plt_arima_prediction(data, arima_prediction, ticker):
    """
    Visualizes the ARIMA prediction for the stock's returns.
    
    Parameters:
    data (pd.Series): Historical percentage returns.
    arima_prediction (pd.Series): ARIMA predicted returns for future steps.
    ticker (str): The ticker symbol of the stock.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=f'{ticker} Historical Returns', color='blue')
    
    # Plot ARIMA predictions
    plt.plot(pd.Series(arima_prediction, index=arima_prediction.index),
             label='ARIMA Prediction', color='red', linestyle='--')
    
    plt.title(f'{ticker} Returns with ARIMA Prediction')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# Function to plot the Efficient Frontier and highlight key portfolios
def plt_efficient_frontier(results, weights_record, expected_returns, cov_matrix, tickers):
    """
    Plots the Efficient Frontier using portfolio return and risk results, and highlights 
    the global minimum variance portfolio and the maximum Sharpe ratio portfolio.
    
    Also prints the portfolio weights for the maximum Sharpe ratio portfolio.
    
    Parameters:
    results (ndarray): A 3xN array where each column is a portfolio, containing:
                       - Row 0: Portfolio return
                       - Row 1: Portfolio risk (standard deviation)
                       - Row 2: Sharpe ratio (return/risk)
    weights_record (list): List of portfolio weights used to construct the frontier.
    expected_returns (ndarray): The expected returns of each asset.
    cov_matrix (ndarray): The covariance matrix of the asset returns.
    tickers (list): List of ticker symbols for the assets.
    """
    # Extract the data from the results array
    portfolio_returns = results[0]
    portfolio_risks = results[1]
    portfolio_sharpe = results[2]
    
    # Plot the efficient frontier (risk vs. return)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(portfolio_risks, portfolio_returns, c=portfolio_sharpe, cmap='viridis', marker='o')
    
    # Add color bar for Sharpe ratio
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    # Highlight the maximum Sharpe ratio portfolio
    max_sharpe_idx = np.argmax(portfolio_sharpe)
    plt.scatter(portfolio_risks[max_sharpe_idx], portfolio_returns[max_sharpe_idx], color='red', marker='*', s=100, label='Max Sharpe Ratio')
    
    # Highlight the global minimum variance portfolio
    min_variance_idx = np.argmin(portfolio_risks)
    plt.scatter(portfolio_risks[min_variance_idx], portfolio_returns[min_variance_idx], color='blue', marker='*', s=100, label='Min Variance Portfolio')
    
    # Add labels and title
    plt.title('Efficient Frontier')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Return')
    
    # Add a legend
    plt.legend(loc='best')
    
    # Show the plot
    plt.grid(True)
    plt.show()
    
    # Print the structure of the Max Sharpe Ratio portfolio
    print("\nPortfolio with Maximum Sharpe Ratio:")
    max_sharpe_weights = weights_record[max_sharpe_idx]
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {max_sharpe_weights[i]:.2%}")
    
    # Optional: Print portfolio return and risk
    ret = portfolio_returns[max_sharpe_idx]
    ann = (1 + ret) ** 52 - 1
    print(f"\nExpected Return: {portfolio_returns[max_sharpe_idx]:.4f}")
    print(f"Annualized Return: {ann:.4f}")
    print(f"Risk (Standard Deviation): {portfolio_risks[max_sharpe_idx]:.4f}")
    print(f"Sharpe Ratio: {portfolio_sharpe[max_sharpe_idx]:.4f}")
