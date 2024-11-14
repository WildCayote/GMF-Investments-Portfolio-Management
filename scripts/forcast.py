import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ForecastAnalyzer:
    def __init__(self, historical_csv, forecast_csv, column='Close'):
        """
        Initialize the model ForecastAnalyzer.

        Parameters:
        historical_csv (str): Path to the historical data CSV.
        forecast_csv (str): Path to the forecast data CSV.
        column (str): The column to forecast (default is 'Close').
        """
        self.historical_csv = historical_csv
        self.forecast_csv = forecast_csv
        self.column = column
        self.predictions = {}

        # Load data
        self._load_data()

    def _load_data(self):
        """
        Load historical and forecast data from CSV files.
        """
        try:
            if not self.forecast_csv:
                raise ValueError("Both historical and forecast CSV file paths must be provided.")
            
            # Load historical data
            self.historical_data = pd.read_csv(self.historical_csv, index_col='Date', parse_dates=True)
            
            # Load forecast data
            self.forecast_data = pd.read_csv(self.forecast_csv, index_col='Date', parse_dates=True)
            
            # Extract forecasted values (assuming forecast CSV has a 'forecast' column)
            if 'LSTM' not in self.forecast_data.columns:
                raise ValueError("Forecast CSV must have a 'forecast' column.")
            
            self.predictions['forecast'] = (self.forecast_data['LSTM'].values) * 100
            self.forecast_dates = self.forecast_data.index

        except Exception as e:
            print(f"Error loading data: {e}")
            raise ValueError("Error loading data")

    def plot_forecast(self):
        """
        Plot the historical data alongside the forecast data with confidence intervals.
        """
        try:
            # Combine historical and forecast dates
            forecast_dates = self.forecast_dates
            all_dates = self.historical_data.index.append(forecast_dates)

            # Set up the plot
            plt.figure(figsize=(15, 8))
            plt.plot(self.historical_data.index, self.historical_data[self.column], label='Actual', color='green', linewidth=2)

            # Plot the forecast data
            forecast = self.predictions['forecast']
            plt.plot(forecast_dates, forecast, label='Forecast', linestyle='--', color='orange')

            # Plot the confidence intervals if they exist
            if 'conf_lower' in self.forecast_data.columns and 'conf_upper' in self.forecast_data.columns:
                plt.fill_between(forecast_dates, self.forecast_data['conf_lower'], self.forecast_data['conf_upper'], color='orange', alpha=0.25, label='95% Confidence Interval')

            # Set up labels and title
            plt.xticks(rotation=45)
            plt.title("Historical vs. Forecast Data with Confidence Intervals", fontsize=16)
            plt.xlabel("Date", fontsize=14)
            plt.ylabel(self.column, fontsize=14)
            plt.legend(loc='best')
            sns.set(style="whitegrid")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in plotting forecasts: {e}")
            raise ValueError("Error plotting forecasts")

    def analyze_forecast(self, threshold=0.05):
        """
        Analyze and interpret the forecast results, including trend, volatility, and market opportunities/risk.
        """
        analysis_results = {}

        print("Starting forecast analysis.")
        
        for model_name, forecast in self.predictions.items():
            # Trend Analysis
            trend = "upward" if np.mean(np.diff(forecast)) > 0 else "downward"
            trend_magnitude = np.max(np.diff(forecast))
            print(f"{model_name} forecast shows a {trend} trend.")

            # Volatility and Risk Analysis
            volatility = np.std(forecast)
            volatility_level = "High" if volatility > threshold else "Low"
            max_price = np.max(forecast)
            min_price = np.min(forecast)
            price_range = max_price - min_price
            volatility_analysis = self._volatility_risk_analysis(forecast, threshold)

            # Market Opportunities and Risks
            opportunities_risks = self._market_opportunities_risks(trend, volatility_level)
            
            # Store results in the analysis dictionary
            analysis_results[model_name] = {
                'Trend': trend,
                'Trend_Magnitude': trend_magnitude,
                'Volatility': volatility,
                'Volatility_Level': volatility_level,
                'Max_Price': max_price,
                'Min_Price': min_price,
                'Price_Range': price_range
            }
            print(f"  Volatility and Risk: {volatility_analysis}")
            print(f"  Market Opportunities/Risks: {opportunities_risks}")
        
        # Return the results in a DataFrame for easy viewing
        analysis_df = pd.DataFrame(analysis_results).T
        return analysis_df

    def _volatility_risk_analysis(self, forecast, threshold):
        """
        Analyze the volatility and risk based on forecast data.
        """
        volatility = np.std(forecast)
        volatility_level = "High" if volatility > threshold else "Low"

        # Highlight periods of increasing volatility
        increasing_volatility = any(np.diff(forecast) > np.mean(np.diff(forecast)))
        
        if increasing_volatility:
            return f"Potential increase in volatility, which could lead to market risk."
        else:
            return f"Stable volatility, lower risk."

    def _market_opportunities_risks(self, trend, volatility_level):
        """
        Identify market opportunities or risks based on forecast trends and volatility.
        """
        if trend == "upward":
            if volatility_level == "High":
                return "Opportunity with high risk due to increased volatility."
            else:
                return "Opportunity with moderate risk due to stable volatility."
        elif trend == "downward":
            if volatility_level == "High":
                return "Risk of decline with high uncertainty."
            else:
                return "Moderate risk of decline with low volatility."
        else:
            return "Stable market, with minimal risks."