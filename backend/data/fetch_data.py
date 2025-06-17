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
    df.columns.name = None
    df.dropna(inplace=True)
    return df

def save_to_csv(df: pd.DataFrame, path: str):
    """
    Helper function to save a DataFrame to a CSV file readable by preprocessing functions.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): Destination file path.
    """
    df.index.name = "Date"  # Ensure index is named 'Date'
    df.to_csv(path, index=True)

if __name__ == "__main__":
    df = download_stock_data("AAPL", "2020-01-01", "2023-01-01")
    print(df.head())
    save_to_csv(df, "backend/data/aapl.csv")