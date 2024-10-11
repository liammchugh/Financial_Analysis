import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import visualization as vis
import asset_prediction as asset

# Function to calculate portfolio return and variance
def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return, portfolio_variance

# Function to minimize (negative utility function)
def negative_utility(weights, returns, cov_matrix, risk_aversion):
    portfolio_return, portfolio_variance = portfolio_performance(weights, returns, cov_matrix)
    return -(portfolio_return - risk_aversion * portfolio_variance)

# Efficient Frontier calculation
def efficient_frontier(expected_returns, cov_matrix, num_portfolios=10000):
    num_assets = len(expected_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Expected portfolio return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Expected portfolio variance (covariance matrix)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std_dev = np.sqrt(portfolio_variance)
        
        # Store results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = portfolio_return / portfolio_std_dev  # Sharpe ratio
        
        # Store weights
        weights_record.append(weights)
    
    return results, weights_record

# Example usage with NYSE tickers
if __name__ == "__main__":
    tickers = ['SOXX', 'IYW', 'AAPL', 'MSFT', 'GOOGL', 'FICO', 'GE', 'AVGO']  # Stock tickers
    predictions = {}
    expected_returns = []
    garch_variances = []
    
    for ticker in tickers:
        prediction = asset.predict_next_year(ticker, return_per='W', horizon=52, verbose=False)
        expected_returns.append(prediction['ARIMA_mean'])  # Take ARIMA-predicted mean return
        garch_variances.append(prediction['GARCH_variance'])
    
    # Create a covariance matrix using the variances (for simplicity)
    cov_matrix = np.diag(garch_variances)  # Diagonal covariance matrix using GARCH variances
    
    # Run efficient frontier
    results, weights_record = efficient_frontier(expected_returns, cov_matrix)
    
    # Plot efficient frontier
    vis.plt_efficient_frontier(results, weights_record, expected_returns, cov_matrix, tickers)