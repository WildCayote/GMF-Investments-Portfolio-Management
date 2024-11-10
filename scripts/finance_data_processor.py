import pynance as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
import os

class FinanceDataProcessor:
    """
    FinanceDataProcessor handles data fetching, inspection, cleaning, and analysis of financial data from YFinance.
    """

    def __init__(self, storage_dir="../data"):
        """
        Initializes the FinanceDataProcessor instance.

        Parameters:
        - storage_dir (str): Directory to save downloaded data files. Defaults to "../data".
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def fetch_data(self, start_date, end_date, tickers):
        """
        Retrieves historical data for each ticker and saves it as a CSV.

        Parameters:
        - start_date (str): Start date for data fetching.
        - end_date (str): End date for data fetching.
        - tickers (list of str): List of stock symbols.

        Returns:
        - dict: A dictionary with ticker names as keys and paths of saved CSV files as values.
        """
        saved_paths = {}
        
        for ticker in tickers:
            try:
                print(f"Retrieving data for {ticker} from {start_date} to {end_date}...")
                data = pn.data.get(ticker, start=start_date, end=end_date)
                file_path = os.path.join(self.storage_dir, f"{ticker}.csv")
                data.to_csv(file_path)
                saved_paths[ticker] = file_path
                print(f"Data for {ticker} saved to '{file_path}'.")

            except ValueError as ve:
                print(f"Data format issue for {ticker}: {ve}")

            except Exception as e:
                print(f"Failed to retrieve data for {ticker}: {e}")

        return saved_paths
    
    def read_data(self, ticker):
        """
        Loads data from a CSV file for a given ticker.

        Parameters:
        - ticker (str): Stock ticker symbol.

        Returns:
        - pd.DataFrame: Loaded data as a DataFrame.
        """
        file_path = os.path.join(self.storage_dir, f"{ticker}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        else:
            raise FileNotFoundError(f"Data file for ticker '{ticker}' not found. Run `fetch_data()` first.")

    def data_summary(self, data):
        """
        Provides a summary of the data by checking types, missing values, and duplicates.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data.

        Returns:
        - dict: Summary with column types, missing values, and duplicate row count.
        """
        summary = {
            "column_types": data.dtypes,
            "missing_values": data.isnull().sum(),
            "duplicate_rows": data.duplicated().sum()
        }
        print(f"Data summary:\n{summary}")
        return summary

    def identify_outliers(self, data, method="iqr", z_threshold=3):
        """
        Identifies outliers in the data using either the IQR or Z-score method.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data.
        - method (str): Outlier detection method ('iqr' or 'z_score'). Default is 'iqr'.
        - z_threshold (int): Z-score threshold for outliers. Used only with 'z_score' method.

        Returns:
        - pd.DataFrame: Boolean DataFrame marking outliers.
        """
        outliers = pd.DataFrame(index=data.index)

        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in data.columns:
                if method == "z_score":
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outliers[col] = z_scores > z_threshold
                elif method == "iqr":
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers[col] = (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))
        
        print(f"Outliers identified using {method.capitalize()} method.")
        return outliers

    def visualize_outliers(self, data, outliers, ticker):
        """
        Generates box plots to show outliers in the data.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data.
        - outliers (pd.DataFrame): Boolean DataFrame indicating outliers.
        """
        outlier_cols = [col for col in data.columns if col in outliers.columns and outliers[col].any()]

        if not outlier_cols:
            print("No outliers detected.")
            return

        plot_count = len(outlier_cols)
        grid_dim = math.ceil(math.sqrt(plot_count))

        fig, axes = plt.subplots(grid_dim, grid_dim, figsize=(12 * grid_dim, 4 * grid_dim))
        
        if plot_count == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for i, col in enumerate(outlier_cols):
            ax = axes[i]
            ax.plot(data.index, data[col], label=col, color="blue")
            ax.scatter(data.index[outliers[col]], data[col][outliers[col]], color='red', s=20, label="Outliers")
            ax.set_title(f"{col} - Time Series with Outliers of {ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel(col)
            ax.legend()

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def handle_outliers(self, data_dict, outlier_flags):
        """
        Handles outliers by setting them to NaN and applying interpolation.

        Parameters:
        - data_dict (dict): Dictionary containing stock data for each ticker.
        - outlier_flags (dict): Dictionary containing Boolean DataFrames marking outliers.

        Returns:
        - dict: Dictionary with cleaned data for each ticker.
        """
        clean_data = {}

        for ticker, data in data_dict.items():
            cleaned = data.copy()
            
            if ticker in outlier_flags:
                outliers = outlier_flags[ticker]
                cleaned[outliers] = np.nan
                cleaned.interpolate(method="time", inplace=True)
                cleaned.bfill(inplace=True)
                cleaned.ffill(inplace=True)

                print(f"Outliers handled for {ticker} by setting to NaN and filling.")

            clean_data[ticker] = cleaned

        return clean_data

    def scale_data(self, data):
        """
        Normalizes the data columns (except 'Volume') using standard scaling.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data.

        Returns:
        - pd.DataFrame: DataFrame with normalized columns.
        """
        scaler = StandardScaler()
        columns_to_scale = ["Open", "High", "Low", "Close", "Adj Close"]
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
        print("Data scaled with standard scaling.")
        return data

    def summarize_statistics(self, data):
        """
        Provides summary statistics for the data.

        Parameters:
        - data (pd.DataFrame): DataFrame with stock data.

        Returns:
        - dict: Summary statistics including mean, median, and standard deviation.
        """
        stats = {
            "mean": data.mean(),
            "median": data.median(),
            "std_dev": data.std(),
            "missing_values": data.isnull().sum()
        }
        print(f"Calculated statistics:\n{stats}")
        return stats
