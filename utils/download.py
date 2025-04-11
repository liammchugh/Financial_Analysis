import yfinance as yf
import pandas as pd

def download_returns(tickers, start, end, frequency='D'):
    """
    Download adjusted close prices for given tickers and calculate returns.
    Args:
        tickers (list): List of stock tickers.
        start (str): Start date for the data.
        end (str): End date for the data.
        frequency (str): Frequency for resampling 
        (D, W, M, Y). Default is 'D'.
    Returns:
        returns (pd.DataFrame): DataFrame of returns.
        prices (pd.DataFrame): DataFrame of adjusted close prices.
    """
    all_prices = []
    for ticker in tickers:
        # Attempt retry logic per ticker
        for _ in range(3):
            try:
                df_raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval="1d", 
                    auto_adjust=True,  # or just omit, since True is default
                    actions=False
                )
                # Then rename "Close" to your ticker
                if not df_raw.empty:
                    df = df_raw[['Close']].rename(columns={'Close': ticker})
                    all_prices.append(df)
                    break
            except Exception as e:
                print(f"Retrying {ticker} due to: {e}")
        else:
            print(f"Failed to download {ticker}")
            # Optionally append an empty dataframe or skip
            all_prices.append(pd.Series(name=ticker))  

    # Combine into single DataFrame
    data = pd.concat(all_prices, axis=1)

    # Resample
    period_prices = data.resample(frequency).last()
    returns = period_prices.pct_change(fill_method=None).ffill().dropna()

    return returns, period_prices

if __name__ == "__main__":
    tickers = ['IYW', 'SOXX', 'AAPL', 'MSFT', 'GOOGL', 'AVGO']
    start = '2023-01-01'
    end = '2024-06-01'
    returns, prices = download_returns(tickers, start, end, frequency='W')
    
    if returns is not None and prices is not None:
        print("Returns and prices downloaded successfully.")
        print(returns.head())
        print(prices.head())
    else:
        print("Failed to download data.")