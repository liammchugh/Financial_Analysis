"""
Portfolio optimization framework using modern financial modeling techniques. 
Designed for equity portfolio management, focusing on risk-adjusted returns and diversification.

1. Processing weekly stock price data.
2. Calculating a denoised(eigenvalue clipping) Gerber covariance matrix based on Mean Absolute Deviation (MAD).
3. (optional) Performing Nested Clustering Optimization (NCO) to group assets into clusters.
4. Optimizing portfolio weights using Conditional Value-at-Risk (CVaR) with appropriate constraints.
5. Simulating portfolio performance with periodic rebalancing.
6. Visualizing portfolio value over time & comparing it to a benchmark (e.g., S&P 500).

Top Priorities:
- Deeper investigate denoising and risk hyperparameters.
- Investigate confidence-bound timeseries prediction incorporation.
- Implement semi-supervised equity management framework.

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from utils.download import download_returns

# Denoising Covariance Matrix (Eigenvalue Clipping)
def denoise_covariance(cov_matrix, clip_threshold=0.05):
    eigenvals, eigenvecs = eigh(cov_matrix)
    
    # Apply the clipping threshold to denoise small eigenvalues
    eigenvals_clipped = np.maximum(eigenvals, clip_threshold)
    
    # Reconstruct the denoised covariance matrix
    cov_matrix_denoised = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
    return cov_matrix_denoised

# Gerber covariance matrix calculation (MAD)
def gerber_covariance(returns, c=0.5):
    '''
    Gerber covariance matrix calculation based on Mean Absolute Deviation (MAD)
    Arguments:
    - returns: DataFrame of asset returns
    - c: Threshold multiplier for MAD (default 0.5)
    Returns:
    - cov_matrix: Covariance matrix of the assets
    '''

    # Calculate the mean and standard deviation of returns
    mean_returns = returns.mean()
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
def nested_clustering(cov_matrix, num_clusters=3):
    from scipy.spatial.distance import squareform
    condensed_matrix = squareform(1 - cov_matrix)  # Convert to condensed distance matrix
    linkage_matrix = linkage(condensed_matrix, method='ward')
    # Use hierarchical clustering to group assets
    linkage_matrix = linkage(cov_matrix, method='ward')
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    return clusters


# CVaR Optimization with Nested Clustering Constraints
def cvar_optimize(exp_returns, hist_returns, clusters, target_return, hyper_args=None):
    '''
    Equity portolio optimizer
    Conditional Value-at-Risk constrained optimization
    Using denoised gerber mean-absolute-deviation covariance
    Arguments:
    - exp_returns: DataFrame of expected returns for each asset
    - hist_returns: DataFrame of historical returns for each asset
    - clusters: Array of cluster labels for each asset
    - target_return: Target return for the portfolio
    - hyper_args: Dictionary of hyperparameters for optimization (optional)
        - lambda_cvar: CVaR confidence level (default 0.95)
        - beta: CVaR tail-risk parameter (default 0.05% worst case: higher is more conservative)
    Returns:
    - weights: Optimized portfolio weights
    '''
    n_assets = exp_returns.shape[1]

    # Calculate cluster-wise covariance matrices
    cov_matrix = gerber_covariance(hist_returns)
    cov_matrix_denoised = denoise_covariance(cov_matrix)

    if hyper_args is not None:
        beta = hyper_args.get('beta', 0.05)
        lambda_cvar = hyper_args.get('lambda_cvar', 0.95)
        risk_aversion = hyper_args.get('risk_aversion', 0.5)

    # Define variables
    weights = cp.Variable(n_assets)
    
    # Use CVXPY's @ operator for matrix multiplication
    portfolio_return = exp_returns.mean().values.flatten() @ weights

    # Portfolio risk (variance)
    portfolio_risk = cp.quad_form(weights, cov_matrix_denoised)

    # CVaR constraints
    alpha = cp.Variable(1)  # VaR variable
    z = cp.Variable(len(exp_returns))  # Loss variables
    losses = -exp_returns.values @ weights  # Use @ operator instead of *

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,  # No short selling
        # weights >= -0.1,  # No large short selling
        # cp.sum(weights) >= -0.2,  # enforce mostly-long portfolio
        losses + alpha >= z,  # Loss constraints
        z >= 0
    ]
    
    # Add portfolio return constraint
    constraints.append(portfolio_return >= target_return)
    
    # Cap the sum of weights in each cluster to 0.5
    if clusters is not None:
        for cluster in np.unique(clusters):
            cluster_mask = (clusters == cluster).astype(float)
            constraints.append(cp.sum(cp.multiply(weights, cluster_mask)) <= 0.5)
    
    # CVaR constraint
    cvar = alpha + (1 / (beta * len(exp_returns))) * cp.sum(z)
    return_avg = -cp.sum(losses)/len(exp_returns)  # Average return

    objective = cp.Minimize(
        (risk_aversion*portfolio_risk)
          + (lambda_cvar * cvar)
          - ((1-lambda_cvar)*return_avg)
          )
    # print(f"Objective: {portfolio_risk + lambda_cvar * cvar}: {portfolio_risk}, {lambda_cvar * cvar}")
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)
    # print(f"alpha: {alpha.value}, z: {z.value}, portfolio_return: {portfolio_return.value}, portfolio_risk: {portfolio_risk.value}")
    
    return weights.value

def simulate_expected_returns(returns, i, rel_interval):
    noise = np.random.normal(0, returns.std().mean(), returns.shape)  # Gaussian noise with mean 0 and returns std
    exp_returns = returns.iloc[i:i+rel_interval]
    exp_returns = exp_returns + noise[i:i+rel_interval]  # Add noise to the expected returns
    if exp_returns.isnull().values.any():
        print("NaN values detected in expected returns.")
    exp_returns = exp_returns.replace([np.nan, np.inf, -np.inf], 0)
    return exp_returns

def simulate_portfolio(tickers, returns, prices, tgt_return, initial_value=1e6, rebalance_interval=2, rebalance_hist=10, hyper_args=None):
    '''
    Portfolio Optimization Simulation
    Uses CVaR optimizer every rebalance_interval days
    Arguments:
    - tickers: List of asset tickers
    - returns: DataFrame of asset returns
    - prices: DataFrame of asset prices
    - tgt_return: Target return for the portfolio
    - start: Start date for the simulation
    - end: End date for the simulation
    - initial_value: Initial portfolio value (default 1e6)
    - rebalance_interval: Rebalance interval in days (default 5)
    - hyper_args: Dictionary of hyperparameters for optimization (optional)
        - lambda_cvar: CVaR confidence level (default 0.95)
        - beta: CVaR tail-risk parameter (default 0.05% worst case: higher is more conservative)
    Returns:
    - portfolio_series: Series of portfolio values over time
    - weight_history: List of weights over time
    - timestamps: List of timestamps corresponding to portfolio values
    - final_weights: Final optimized weights
    '''
        
    # Initialize portfolio value
    portfolio_value = initial_value
    portfolio_values = []  # Track the portfolio value over time
    weight_history = []  # Track the weights over time
    timestamps = []
    
    # Initialize portfolio weights (will be updated every rebalance_interval days)
    n_assets = len(tickers)
    weights = np.ones(n_assets) / n_assets  # Equal allocation to start

    # Simulate over each period
    rel_interval = int(rebalance_interval / (returns.index[1] - returns.index[0]).days)  # returns interval datapoints
    rel_hist = int(rebalance_hist / (returns.index[1] - returns.index[0]).days)  # historic interval in returns datapoints

    for i in tqdm(range(rel_hist, len(returns), rel_interval), desc="Simulating Portfolio"):
        hist_returns = returns.iloc[i - rel_hist:i]
        period_returns = returns.iloc[i - rel_interval:i]

        # predicted returns, simulated with noisy next per returns
        exp_returns = simulate_expected_returns(returns, i, rel_interval)

        # print(period_returns.head())
        
        # Check if weights is not None before calculating portfolio return
        if weights is None:
            print("Warning: Weights are None. Falling back to equal allocation.")
            weights = np.ones(n_assets) / n_assets  # Reset to equal allocation
        
        # Calculate portfolio return over prev period
        portfolio_return = np.dot(period_returns.mean().values, weights)
        portfolio_value = portfolio_value * (1 + portfolio_return)
        
        # Append the portfolio value and the corresponding timestamp
        portfolio_values.append(portfolio_value)
        timestamps.append(returns.index[i])
        
        # Re-optimize every interval [weeks]
        if i + rel_interval < len(returns):
            clusters = None  # Placeholder for clusters, if needed
            weights = cvar_optimize(exp_returns, hist_returns, clusters, target_return=tgt_return, hyper_args=hyper_args)
            if weights is None:
                print("Warning: Optimization failed. Adjusting target return.")
                i = 1
                adj_tgt = tgt_return
                while weights is None:
                    adj_tgt = adj_tgt - 0.1*tgt_return # Decrease target return
                    weights = cvar_optimize(exp_returns, hist_returns, clusters, target_return=adj_tgt, hyper_args=hyper_args)
                    i += 1
                    if i > 25:
                        print("Warning: Optimization failed multiple times. Setting equal weights.")
                        weights = np.ones(n_assets) / n_assets  # Reset to equal allocation
                print(f"Adjusted annualized return after {i} iterations: {adj_tgt*(rebalance_interval * 52):.4f}")
            # print(f"weights: {weights_r}")
            weight_history.append(weights)

    
    portfolio_series = pd.Series(portfolio_values, index=timestamps)
    # return portfolio_series, weight_history, final weights
    return portfolio_series, weight_history, timestamps, weights



if __name__ == "__main__":
    tickers = [
        'IYW', 'SOXX', 'AAPL', 'MSFT', 'GOOGL', 
        'AVGO', 'NVDA', 'AMAT', 'INTC', 'TXN', 'QCOM', 
        'TLT', 'SH']  # Treasury Bonds, S&P500 Hedge
    start = '2023-01-01'
    end = '2025-04-09'
    initial_value = 1e4
    rebalance_interval = 1 # days
    rebalance_history = 25 # days
    data_freq = 'D'  # Daily data frequency
    tgt_return = 0.20 # Target return, annual
    hyper_args = {
        'ann_tgt_return': tgt_return,
        'risk_aversion': 1.0,
        'beta': 0.20,
        'lambda_cvar': 0.95
    }

    tgt_return = tgt_return / 356 * rebalance_interval # Convert to rebalance period return
    start_date = pd.to_datetime(start)  # Convert start to datetime
    hist_start = start_date - pd.Timedelta(days=rebalance_history)
    returns, prices = download_returns(tickers, hist_start, end, frequency=data_freq)
    print(len(returns), len(prices))
    if returns is None or prices is None:
        print("Failed to download data. Exiting.")
        exit(1)

    # Cash backstop
    if data_freq == 'H':
        inflation_rate = 0.03 / (24*365)
    elif data_freq == 'D':
        inflation_rate = 0.03 / 365
    elif data_freq == 'W':
        inflation_rate = 0.03 / 52
    elif data_freq == 'M':
        inflation_rate = 0.03 / 12

    cash_returns = pd.Series([-inflation_rate] * len(returns), index=returns.index)
    tickers.append('CASH')
    returns['CASH'] = cash_returns
    prices['CASH'] = 1  # Cash price remains constant

    # cov_matrix = gerber_covariance(returns)
    # # clusters = nested_clustering(cov_matrix)
    clusters = None
    
    portfolio_series, weight_hist, timestamps, final_weight = simulate_portfolio(
        tickers, returns, prices, tgt_return, initial_value, 
        rebalance_interval, rebalance_history,
        hyper_args=hyper_args
    )

    # Clip out the first history interval from the portfolio return series
    portfolio_series = portfolio_series.iloc[rebalance_history:]

    # Calculate the average weight history and print with tickers
    if weight_hist:
        avg_weights = np.mean(weight_hist, axis=0)
        weight_summary = pd.DataFrame({'Ticker': tickers, 'Average Weight': avg_weights})
        print(weight_summary)
    else:
        print("No weight history available.")

    # Create a DataFrame for weight history with associated tickers
    if weight_hist:
        weight_hist_df = pd.DataFrame(weight_hist, columns=tickers, index=timestamps[:len(weight_hist)])
    else:
        print("No weight history available to create DataFrame.")

    rounded_final_weight = np.round(final_weight, 4)
    print(f"Final weights (rounded): {rounded_final_weight}")
    print(f"Tickers: {tickers}")

    from utils.visualization import visualize_portfolio, vis_weight_history

    visualize_portfolio(tickers, portfolio_series, weight_hist_df, start, end, initial_value, per=rebalance_interval)
    hist_days = 14
    vis_weight_history(tickers, weight_hist_df, clusters, hist_days, per=rebalance_interval)

