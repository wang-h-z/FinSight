import yfinance as yf
import pandas as pd

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical stock data using Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g. 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD'.
        end_date (str): End date in 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame containing historical OHLCV data.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = download_stock_data("AAPL", "2020-01-01", "2023-01-01")
    df.columns.name = None  # remove multi-level column label
    df.to_csv("data/aapl.csv")  # save to file for use later
    print(df.head())