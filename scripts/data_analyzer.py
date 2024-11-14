import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

class FinancialDataAnalyzer:
    def __init__(self, data_directory='../data/'):
        """
        Initializes the FinancialDataAnalyzer class.

        Parameters:
        - data_directory (str): Directory containing the data files.
        """
        self.data_directory = data_directory

    def plot_daily_percentage_change(self, stock_data):
        """
        Plots the daily percentage change in the closing price for each stock symbol.

        Parameters:
        - stock_data (dict): Dictionary with stock symbols as keys and their DataFrames as values.
        """
        for symbol, df in stock_data.items():
            if df is None or df.empty:
                print(f"DataFrame for {symbol} is empty.")
                continue
            
            try:
                df['Pct_Change'] = df['Close'].pct_change() * 100
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['Pct_Change'], label=f'{symbol} Daily Percentage Change', color='purple')
                plt.title(f"{symbol} Daily Percentage Change Over Time", fontsize=16)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Percentage Change (%)", fontsize=12)
                plt.legend()
                plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
                plt.show()
            except Exception as e:
                print(f"Error plotting percentage change for {symbol}: {str(e)}")

    def analyze_closing_price_trend(self, stock_data, window_size=30):
        """
        Analyzes and plots the closing price, rolling mean, and volatility (rolling std) over time for multiple symbols.

        Parameters:
        - stock_data (dict): Dictionary with stock symbols as keys and their DataFrames as values.
        - window_size (int): Window size for calculating rolling statistics.
        """
        sns.set(style="darkgrid")

        for symbol, df in stock_data.items():
            if df is None or df.empty:
                print(f"DataFrame for {symbol} is empty.")
                continue

            if 'Close' not in df.columns:
                print(f"'Close' column not found in DataFrame for {symbol}.")
                continue

            try:
                df['Rolling_Mean'] = df['Close'].rolling(window=window_size).mean()
                df['Rolling_Std'] = df['Close'].rolling(window=window_size).std()

                plt.figure(figsize=(12, 6))
                
                sns.lineplot(data=df, x=df.index, y='Close', label=f'{symbol} Closing Price', color="blue", linestyle='solid')
                sns.lineplot(data=df, x=df.index, y='Rolling_Mean', label=f'{symbol} {window_size}-day Rolling Mean', color="magenta", linestyle="--")
                sns.lineplot(data=df, x=df.index, y='Rolling_Std', label=f'{symbol} {window_size}-day Rolling Volatility', color="cyan", linestyle=":")

                plt.title(f"Closing Price Trend, Rolling Mean and Volatility of {symbol} Over Time", fontsize=16)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Value", fontsize=12)
                plt.legend(title="Legend")
                plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)

                y_max = int(plt.ylim()[1])
                plt.yticks(range(0, y_max + 50, 50))

                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Error plotting data for {symbol}: {str(e)}")

    def plot_unusual_returns(self, stock_data, threshold=2.5):
        """
        Calculates and plots daily returns with highlights on unusually high or low return days for each symbol.

        Parameters:
        - stock_data (dict): Dictionary with stock symbols as keys and their DataFrames as values.
        - threshold (float): Threshold (in terms of standard deviations) to define unusual returns.
        """
        sns.set(style="darkgrid")

        for symbol, df in stock_data.items():
            if df is None or df.empty:
                print(f"DataFrame for {symbol} is empty.")
                continue

            try:
                df['Daily_Return'] = df['Close'].pct_change() * 100

                mean_return = df['Daily_Return'].mean()
                std_dev = df['Daily_Return'].std()
                unusual_returns = df[(df['Daily_Return'] > mean_return + threshold * std_dev) |
                                     (df['Daily_Return'] < mean_return - threshold * std_dev)]

                plt.figure(figsize=(12, 6))
                sns.lineplot(x=df.index, y=df['Daily_Return'], label=f'{symbol} Daily Return', color='green')

                plt.scatter(unusual_returns.index, unusual_returns['Daily_Return'], color='red', 
                            label=f"Unusual Returns (±{threshold}σ)", s=50, marker='o')

                plt.title(f"Daily Returns with Unusual Days Highlighted - {symbol}", fontsize=16)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Daily Return (%)", fontsize=12)
                plt.axhline(0, color='black', linestyle='--')
                plt.legend()
                plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Error plotting unusual daily returns for {symbol}: {str(e)}")
