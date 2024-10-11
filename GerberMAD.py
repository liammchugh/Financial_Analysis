
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf

# Download price data and calculate weekly returns
def download_weekly_returns(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    weekly_data = data.resample('W').last()
    returns = weekly_data.pct_change().dropna()
    return returns

# Denoising Covariance Matrix (Eigenvalue Clipping)
def denoise_covariance(cov_matrix, clip_threshold=0.5):
    eigenvals, eigenvecs = eigh(cov_matrix)
    
    # Apply the clipping threshold to denoise small eigenvalues
    eigenvals_clipped = np.maximum(eigenvals, clip_threshold)
    
    # Reconstruct the denoised covariance matrix
    cov_matrix_denoised = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
    return cov_matrix_denoised

# Gerber covariance matrix calculation (MAD)
def gerber_covariance(returns, c=0.5):
    std_dev = returns.std()
    thresholds = c * std_dev
    n_assets = returns.shape[1]
    concordance_matrix = np.zeros((n_assets, n_assets))

    for i in range(n_assets):
        for j in range(i, n_assets):
            concordant = ((returns.iloc[:, i] >= thresholds.iloc[i]) & (returns.iloc[:, j] >= thresholds.iloc[j])).sum()
            discordant = ((returns.iloc[:, i] >= thresholds.iloc[i]) & (returns.iloc[:, j] < thresholds.iloc[j])).sum()

            # Ensure no division by zero
            denominator = concordant + discordant
            if denominator == 0:
                concordance_matrix[i, j] = concordance_matrix[j, i] = 0  # Set to 0 to avoid NaN
            else:
                concordance_matrix[i, j] = concordance_matrix[j, i] = (concordant - discordant) / denominator
    
    cov_matrix = np.diag(std_dev) @ concordance_matrix @ np.diag(std_dev)
    
    # Ensure no NaNs or Infs in the covariance matrix
    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return cov_matrix

# Nested Clustering Optimization (NCO)
def nested_clustering(cov_matrix, num_clusters=5):
    # Use hierarchical clustering to group assets
    linkage_matrix = linkage(cov_matrix, method='ward')
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    
    return clusters
# CVaR Optimization with Nested Clustering Constraints
def cvar_optimizer(returns, clusters, target_return, lambda_cvar=0.95, beta=0.05):
    n_assets = returns.shape[1]

    # Calculate cluster-wise covariance matrices
    cov_matrix = gerber_covariance(returns)
    cov_matrix_denoised = denoise_covariance(cov_matrix)

    # Define variables
    weights = cp.Variable(n_assets)
    
    # Use CVXPY's @ operator for matrix multiplication
    portfolio_return = returns.mean().values.flatten() @ weights

    portfolio_risk = cp.quad_form(weights, cov_matrix_denoised)

    # CVaR constraints
    alpha = cp.Variable(1)  # VaR variable
    z = cp.Variable(len(returns))  # Loss variables
    losses = returns.values @ weights  # Use @ operator instead of *

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,  # No short selling
        losses + alpha >= z,  # Loss constraints
        z >= 0
    ]
    
    # Add portfolio return constraint
    constraints.append(portfolio_return >= target_return)
    
    # Enforce nested clustering constraints
    for cluster in np.unique(clusters):
        cluster_mask = (clusters == cluster).astype(float)
        constraints.append(cp.sum(cp.multiply(weights, cluster_mask)) <= 0.5)  # Use multiply for element-wise multiplication
    
    # CVaR constraint
    cvar = alpha + (1 / (beta * len(returns))) * cp.sum(z)
    objective = cp.Minimize(portfolio_risk + lambda_cvar * cvar)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return weights.value

# Download price data and calculate weekly returns
def download_weekly_returns(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    weekly_data = data.resample('W').last()

    # Calculate percentage returns and drop NaN rows
    returns = weekly_data.pct_change().dropna()

    # If NaNs remain, forward fill the NaN values (optional)
    returns = returns.ffill()  # Use the new method instead of 'method=' argument

    return returns, weekly_data


def simulate_portfolio(tickers, start, end, clusters, initial_value=100000, rebalance_interval=2):
    returns, prices = download_weekly_returns(tickers, start, end)
    
    # Initialize portfolio value
    portfolio_value = initial_value
    portfolio_values = []  # Track the portfolio value over time
    timestamps = []
    
    # Initialize portfolio weights (will be updated every 2 weeks)
    n_assets = len(tickers)
    weights = np.ones(n_assets) / n_assets  # Equal allocation to start
    
    # Simulate over each two-week period
    for i in range(0, len(returns), rebalance_interval):
        # Get returns for the next two weeks
        two_week_returns = returns.iloc[i:i + rebalance_interval]
        print(two_week_returns.head())
        print(weights)
        
        # Check if weights is not None before calculating portfolio return
        if weights is None:
            print("Warning: Weights are None. Falling back to equal allocation.")
            weights = np.ones(n_assets) / n_assets  # Reset to equal allocation if weights are None
        
        # Calculate portfolio return over this period
        portfolio_return = np.dot(two_week_returns.mean().values, weights)
        portfolio_value = portfolio_value * (1 + portfolio_return)
        
        # Append the portfolio value and the corresponding timestamp
        portfolio_values.append(portfolio_value)
        timestamps.append(returns.index[i])
        
        # Re-optimize every two weeks
        if i + rebalance_interval < len(returns):
            weights = cvar_optimizer(two_week_returns, clusters, target_return=0.005)
    
    return portfolio_values, timestamps

# Visualization function
def visualize_portfolio(tickers, start, end, initial_value=100000):
    portfolio_values, timestamps = simulate_portfolio(tickers, start, end, clusters)
    
    # Convert to pandas series for easier plotting
    portfolio_series = pd.Series(portfolio_values, index=timestamps)
    
    # Plot the portfolio value over time
    plt.figure(figsize=(10, 6))
    portfolio_series.plot(label="Portfolio Value")
    
    # Optional: Download benchmark (S&P 500) for comparison
    sp500 = yf.download('^GSPC', start=start, end=end)['Adj Close'].resample('W').last()
    sp500_returns = sp500.pct_change().dropna()

    # Ensure matching shapes for sp500_returns
    sp500_value = initial_value * (1 + sp500_returns).cumprod()
    
    sp500_value.plot(label="S&P 500")
    
    # Add labels and legend
    plt.title('Portfolio Value Over Time with Rebalancing Every Two Weeks')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    start = '2023-01-01'
    end = '2024-01-01'
    
    returns = download_weekly_returns(tickers, start, end)[0]
    cov_matrix = gerber_covariance(returns)
    clusters = nested_clustering(cov_matrix)
    
    # Visualize the portfolio
    visualize_portfolio(tickers, start, end, clusters)
