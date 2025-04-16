import os
import time
import openai
from openai import OpenAI
from utils.download import download_returns
from datetime import datetime, timedelta
from utils.llm_prompt import generate_response
from utils.fetch_news import fetch_news, find_more_news
import pandas as pd
import numpy as np

"""
This script predicts future stock prices using a data & news-integrated LLM. 
It integrates historical stock price data, relevant news articles, and GPT-4's predictive capabilities 
to generate forecasts for specified tickers. 
Calculates returns based on the predictions and visualizes both historical and predicted prices.

Key Features:
- Fetches historical stock price data and interpolates missing values.
- Retrieves relevant news articles for each ticker to enhance prediction accuracy.
- Uses GPT-4 (training cutoff April 2023) to predict future stock prices based on historical data and news.
- Cleans and processes llm output into structured pandas Series. resamples data to specified frequency
- Provides indexed returns.
- Visualizes historical and predicted prices for multiple tickers.

Usage:
- Define the tickers, historical data range, prediction length, and data frequency.
- Run the script to generate predictions and visualize results.

"""

# Set OpenAI API key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def prompt_equity_values(ticker: str, history: str, pred_length: int, rel_news: str) -> str:
    """
    Prompt GPT-4 with an equity value history string and predict the next period of equity values.
    
    Args:
        ticker (str): The stock ticker symbol.
        history (str): The equity value history string.
        pred_length (int): The number of periods to predict.
        rel_news (str): Relevant news articles to consider for prediction.
        
    Returns:
        str: The predicted next period of equity values in the same format as the history.
    """

    prompt = f"""
    You are a financial modeling assistant with the goal of predicting stock prices.

    Provided:
    - Ticker: {ticker}
    - Historical Data in the format "YYYY-MM-DD open" (one row per date)
    - Relevant News: {rel_news}

    Historical Data:
    {history}

    Task:
    Based on the above historical trends and news, predict the next {pred_length} days of {ticker} open prices. 
    **Output** exactly {pred_length} lines, each line in the format: 
    YYYY-MM-DD open

    For example, if you were to predict 3 days for some fictitious data, you would write:
    2023-08-25 175.32
    2023-08-26 176.05
    2023-08-27 177.12

    Now do the actual prediction for {ticker}, for the next {pred_length} days. 
    **Important**: Provide only the series of {pred_length} lines in that format, without any extra text or disclaimers.
    """

    try:
        gpt_version = "gpt-4"
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial modeling tool.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=gpt_version,
            temperature=0.15,
            max_tokens=200,
            stream=True,
        )
        result = ""
        for chunk in response:
            result += chunk.choices[0].delta.content or ""
        return result
    except Exception as e:
        print(f"Error: {e}")
        return ""

def parse_gpt_output(prediction: str) -> pd.Series:
    """
    Parses GPT-generated text into a structured pandas Series.
    """
    import pandas as pd
    from io import StringIO
    from datetime import datetime

    # Create a buffer to read the data into pandas
    data_buffer = StringIO(prediction.strip())

    # Read data into a DataFrame
    df = pd.read_csv(
        data_buffer,
        sep=' ',
        header=None,
        names=['date', 'open'],
        parse_dates=['date'],
        index_col='date'
    )

    # Ensure the 'open' values are numeric
    df['open'] = pd.to_numeric(df['open'], errors='coerce')

    # Return as a Series (dates as index)
    return df['open']

def clean_response(price_series: pd.Series, frequency: str):
    """
    Takes parsed pandas Series of price predictions and calculates returns.
    
    Args:
        price_series (pd.Series): Parsed predicted prices with date index.
        frequency (str): Resampling frequency ('D', 'W', etc.).

    Returns:
        pd.Series: Cleaned price predictions.
        pd.Series: Calculated returns.
    """

    # Ensure data is sorted by date
    price_series = price_series.sort_index()

    # Resample to desired frequency
    period_prices = price_series.resample(frequency).last()

    # Calculate returns (percentage change)
    returns = period_prices.pct_change().dropna()

    return period_prices, returns

# Example usage
def main(tickers, returns, prices, data_freq, pred_len, to_date=datetime.now(), verbose=False):
    returns_dfs = []  # Collect each ticker's returns DataFrame
    prices_dfs = []   # Collect each ticker's prices DataFrame

    for ticker in tickers:
        equity_history = prices[ticker].to_string()
        time_history = prices.index[-1].strftime("%Y-%m-%d")
        equity_history = f"{time_history} {equity_history}"
        if verbose:
            print(f"equity_history: {equity_history}")
            print(f"Ticker: {ticker}")
        
            print(f"Predicting next period of equity values for {ticker}:")
        lookback_days = 14  # Look back two weeks
        limit = 6  # Limit to 6 articles
        service = "newsapi"  # Choose between "newsapi" and "finlight"

        try:
            news = fetch_news(service, ticker, to_date, lookback_days, limit)
            if verbose:
                print("Relevant news articles:")
                print(news)
        except Exception as e:
            print(f"Error fetching news: {e}")
        all_news = find_more_news(news, ticker, service, to_date, lookback_days, limit)

        prediction = prompt_equity_values(ticker, equity_history, pred_len, all_news)
        if verbose:
            print("Predicted output:")
            print(prediction)

        # clean gpt output
        parsed_prediction = parse_gpt_output(prediction)
        price_predictions_series, returns_series = clean_response(parsed_prediction, data_freq)

        # Convert Series to DataFrames with explicit ticker column name
        returns_df = returns_series.to_frame(name=ticker)
        prices_df = price_predictions_series.to_frame(name=ticker)

        # Append DataFrames to lists for later concatenation
        returns_dfs.append(returns_df)
        prices_dfs.append(prices_df)

        if verbose:
            print("Cleaned prediction:")
            print(prices_df)
            print("Returns:")
            print(returns_df)

    # Concatenate all DataFrames after the loop
    all_returns = pd.concat(returns_dfs, axis=1)
    all_prices = pd.concat(prices_dfs, axis=1)

    return all_prices, all_returns

def test_main():
    tickers = [
        'IYW', 'SOXX', 
        'AAPL', 'MSFT', 'AVGO', 'QCOM', 'AMAT', 
        'TLT'
    ]  # Equity tickers
    hist_start = datetime.now() - timedelta(days=365)  # 1 year of historical data
    end = datetime.now()
    data_freq = 'D'  # Daily frequency
    pred_len = 14  # Predicting next period (days)

    returns, prices = download_returns(tickers, hist_start, end, frequency=data_freq)
    # Interpolate missing values (NaNs) in the prices DataFrame
    prices = prices.interpolate(method='linear', limit_direction='forward', axis=0)
    predictions, returns = main(tickers, returns, prices, data_freq, pred_len, to_date=end, verbose=True)
    

    import matplotlib.pyplot as plt
    # Set up tiled layout for plots
    num_tickers = len(tickers)
    cols = 2  # Number of columns in the tiled layout
    rows = (num_tickers + cols - 1) // cols  # Calculate required rows
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    axes = axes.flatten()  # Flatten axes array for easy iteration

    for i, ticker in enumerate(tickers):
        ax = axes[i]

        # Historical data
        ax.plot(prices[ticker].index, prices[ticker], label=f"{ticker} Historical Prices", color='blue')

        # Predicted data
        ax.plot(predictions[ticker].index, predictions[ticker], 
                label=f"{ticker} Predicted Prices", color='orange', linestyle='dotted', marker='o')

        # Vertical line separating historical and predicted data
        ax.axvline(x=prices.index[-1], color='red', linestyle='--', label='Prediction Start')

        # Plot formatting
        ax.set_title(f"{ticker} Historical and Predicted Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)

    # Adjust vertical spacing between rows
    plt.subplots_adjust(top=0.96, bottom=0.129, left=0.044, right=0.987, hspace=0.517, wspace=0.108)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # plt.tight_layout()
    plt.show()

def analyze_hist_performance(tickers):

    cutoff = datetime(2025, 3, 29)  # Cutoff date for historical performance analysis
    hist_len = 365  # Historical length in days
    hist_data_start = cutoff - timedelta(days=hist_len)  # 1 year of historical data
    rebalance_period = 5  # Rebalance every N days
    data_freq = 'D'  # Daily frequency
    pred_len = 14  # Predicting next period (days)

    returns, prices = download_returns(tickers, hist_data_start, datetime.now(), frequency=data_freq)
    predicted_prices_df = pd.DataFrame(index=prices.index, columns=tickers)  # Initialize DataFrame for predicted prices

    # Initialize variables to track errors and periods
    total_error = 0
    num_periods = 0

    # Iterate through rebalance periods, starting from cutoff
    current_date = cutoff
    while current_date < datetime.now():
        # Define the end date for the current period
        end = current_date + timedelta(days=rebalance_period)
        if end > datetime.now():
            end = datetime.now()

        # Interpolate missing values (NaNs) in the prices DataFrame
        per_prices = prices.loc[current_date - timedelta(days=hist_len):current_date].interpolate(method='linear', limit_direction='forward', axis=0)
        per_returns = returns.loc[current_date - timedelta(days=hist_len):current_date]

        # Generate predictions for the current period
        predictions, _ = main(tickers, per_returns, per_prices, data_freq, pred_len, to_date=current_date, verbose=False)

        # Compute the error for the current period
        for ticker in tickers:
            actual_prices = prices[ticker][current_date:end]

            # Handle case where predictions[ticker] might be a DataFrame
            predicted = predictions[ticker][current_date:end]
            if isinstance(predicted, pd.DataFrame):
                predicted_prices = predicted.iloc[:, 0]
            else:
                predicted_prices = predicted

            # Ensure both are Series
            if isinstance(actual_prices, pd.DataFrame):
                actual_prices = actual_prices.squeeze()
            if isinstance(predicted_prices, pd.DataFrame):
                predicted_prices = predicted_prices.squeeze()
            # Align both Series
            actual_prices, predicted_prices = actual_prices.align(predicted_prices, join='inner')

            # Calculate percentage error
            ticker_error = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            total_error += ticker_error

            print(f"Period Error for {ticker}: {ticker_error:.2f}%")
            # Pause operation for 15 seconds
            time.sleep(15)

            # Store error specific to each ticker
            if 'ticker_errors' not in locals():
                ticker_errors = {}
            if ticker not in ticker_errors:
                ticker_errors[ticker] = []
            ticker_errors[ticker].append(ticker_error)
            # Append period price predictions to the predicted_prices DataFrame, only up to the current date
            predicted_prices_df.loc[predictions.index[predictions.index <= datetime.now() - timedelta(days=1)], ticker] = predictions[ticker][predictions.index <= datetime.now() - timedelta(days=1)]

        # Increment the number of periods
        num_periods += 1

        # Move to the next rebalance period
        current_date = end

    # Compute the average percent annual price error for each ticker
    avg_ticker_errors = {ticker: np.mean(errors) for ticker, errors in ticker_errors.items()}
    for ticker, avg_error in avg_ticker_errors.items():
        print(f"Average Percent Annual Return Error for {ticker}: {avg_error:.2f}%")


    # Uncomment the following line to run the historical performance analysis
    return predicted_prices_df, prices

if __name__ == "__main__":
    # Uncomment the following line to run the main function for predictions
    # test_main()

    tickers = [ 
        # 'IYW', 'SOXX', 
        'AAPL', # 'MSFT', 'AVGO', 'QCOM', # 'AMAT',
        # 'TLT'
    ]  # Equity tickers

    # Uncomment the following line to analyze historical performance or load CSVs
    if os.path.exists("predicted_prices.csv") and os.path.exists("actual_prices.csv"):
        predicted_prices_df = pd.read_csv("predicted_prices.csv", index_col=0, parse_dates=True)
        prices = pd.read_csv("actual_prices.csv", index_col=0, parse_dates=True)
    else:
        predicted_prices_df, prices = analyze_hist_performance(tickers)
    # Save predicted prices and actual prices to CSV files
    predicted_prices_df.to_csv("predicted_prices.csv", index=True)
    prices.to_csv("actual_prices.csv", index=True)

    import matplotlib.pyplot as plt

    # Set up tiled layout for scatter plots
    num_tickers = len(tickers)
    cols = 2  # Number of columns in the tiled layout
    rows = (num_tickers + cols - 1) // cols  # Calculate required rows
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    axes = axes.flatten()  # Flatten axes array for easy iteration

    for i, ticker in enumerate(tickers):
        ax = axes[i]

        # Align actual and predicted prices for the ticker
        actual_prices, predicted_prices = prices[ticker].align(predicted_prices_df[ticker], join='inner')

        # Scatter plot of actual vs. predicted prices
        ax.scatter(actual_prices, predicted_prices, alpha=0.6, label=f"{ticker} Prices", color='blue')

        # Plot a diagonal line for reference
        min_price = min(actual_prices.min(), predicted_prices.min())
        max_price = max(actual_prices.max(), predicted_prices.max())
        ax.plot([min_price, max_price], [min_price, max_price], color='red', linestyle='--', label='Ideal Fit')

        # Plot formatting
        ax.set_title(f"{ticker} Predicted vs Actual Prices")
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.legend()
        ax.grid(True)

    # Adjust vertical spacing between rows
    plt.subplots_adjust(top=0.96, bottom=0.129, left=0.044, right=0.987, hspace=0.517, wspace=0.108)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()