import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesAnalyzer:
    
    def __init__(self):
        """
        Initializes the TimeSeriesAnalyzer class.
        """
        pass
    
    def adf_test(self, series):
        """
        Perform ADF test for stationarity.

        Parameters:
        - series (pd.Series): The time series to test.

        Returns:
        - float: The p-value of the ADF test.
        """
        adf_result = adfuller(series.dropna())
        return adf_result[1]  # Return p-value of ADF test
    
    def difference_series(self, series):
        """
        Apply differencing to the series to make it stationary.

        Parameters:
        - series (pd.Series): The time series to difference.

        Returns:
        - pd.Series: The differenced time series.
        """
        return series.diff().dropna()
    
    def decompose_time_series(self, series, model='additive'):
        """
        Decompose the time series into trend, seasonal, and residual components.

        Parameters:
        - series (pd.Series): The time series to decompose.
        - model (str): The type of decomposition (additive or multiplicative).

        Returns:
        - seasonal_decompose: The decomposed time series components.
        """
        decomposition = seasonal_decompose(series.dropna(), model=model, period=252)  # Assuming daily data, 252 trading days in a year
        return decomposition
    
    def analyze_trends_and_seasonality(self, stock_data, threshold=0.05):
        """
        Analyze seasonality and trends of stock prices by decomposing them.

        Parameters:
        - stock_data (dict): Dictionary with stock symbols as keys and their DataFrames as values.
        - threshold (float): Threshold for the p-value to determine stationarity.
        """
        sns.set(style="darkgrid")

        for symbol, df in stock_data.items():
            if df is None or df.empty:
                print(f"DataFrame for {symbol} is empty.")
                continue

            try:
                # Perform ADF test for stationarity
                p_value = self.adf_test(df['Close'])
                
                print(f"ADF test p-value for {symbol}: {p_value}")

                # If the p-value is greater than the threshold, apply differencing
                if p_value > threshold:
                    print(f"{symbol} series is non-stationary. Differencing the series.")
                    df['Close'] = self.difference_series(df['Close'])

                # After differencing, check again
                p_value = self.adf_test(df['Close'])
                print(f"ADF test p-value after differencing for {symbol}: {p_value}")

                # Decompose the series into trend, seasonal, and residual components
                decomposition = self.decompose_time_series(df['Close'])

                # Plot the decomposition results
                plt.figure(figsize=(12, 8))
                plt.subplot(411)
                plt.plot(df['Close'], label=f'{symbol} Closing Price', color='darkblue')
                plt.title(f'{symbol} Closing Price', color='midnightblue')
                plt.legend(loc='best')

                plt.subplot(412)
                plt.plot(decomposition.trend, label=f'{symbol} Trend', color='coral')
                plt.title(f'{symbol} Trend', color='midnightblue')
                plt.legend(loc='best')

                plt.subplot(413)
                plt.plot(decomposition.seasonal, label=f'{symbol} Seasonal', color='teal')
                plt.title(f'{symbol} Seasonal', color='midnightblue')
                plt.legend(loc='best')

                plt.subplot(414)
                plt.plot(decomposition.resid, label=f'{symbol} Residual', color='orchid')
                plt.title(f'{symbol} Residual', color='midnightblue')
                plt.legend(loc='best')

                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
