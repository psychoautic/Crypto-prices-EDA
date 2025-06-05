# Crypto Price Analysis and Prediction Application

This application is a powerful tool for analyzing and predicting cryptocurrency prices using machine learning and natural language processing. It combines Prophet forecasting with LLaMA-based natural language understanding to provide intuitive crypto market analysis.

## Features

- **Cryptocurrency Price Prediction**: Uses Facebook's Prophet model to forecast cryptocurrency prices
- **Multiple Cryptocurrency Support**: Supports various major cryptocurrencies including:
  - Bitcoin (BTC)
  - Ethereum (ETH)
  - Binance Coin (BNB)
  - Cardano (ADA)
  - And many more
- **Interactive Chat Interface**: Custom GUI built with CustomTkinter
- **Natural Language Processing**: Powered by LLaMA model for understanding user queries
- **Multiple Analysis Features**: Supports analysis of:
  - Close prices
  - Volume
  - Market capitalization
  - Other available metrics

## Technical Components

### Main Functions

1. `llamaQuery(prompt: str) -> str`

   - Handles communication with the LLaMA model
   - Processes natural language queries into structured commands

2. `prophetPredict(ticker, feature, periods=365)`

   - Generates price predictions using the Prophet model
   - Supports customizable prediction periods
   - Returns forecast data including upper and lower bounds

3. `compareCryptos(ticker1, ticker2, feature, periods=365)`

   - Compares predictions between two different cryptocurrencies
   - Generates visualization of comparative analysis
   - Supports multiple features for comparison

4. `handleUserQuery(userInput)`
   - Main processing function for user inputs
   - Parses natural language into structured commands
   - Supports various query types including predictions and comparisons

### GUI Components

The application includes a custom GUI built with CustomTkinter, featuring:

- Chat interface for user interaction
- Clear chat functionality
- Message history tracking

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- prophet: Time series forecasting
- matplotlib: Data visualization
- customtkinter: GUI framework
- ollama: LLaMA model integration

## Data Structure

The application expects cryptocurrency data in CSV format stored in the `data/` directory with the following naming convention:
`data/{CryptoName}_hist.csv`

## Usage Examples

1. **Simple Price Prediction**

   ```
   "What will be the price of Bitcoin in 6 months?"
   ```

2. **Comparative Analysis**

   ```
   "Compare ETH and BTC prices for the next year"
   ```

3. **Specific Date Predictions**
   ```
   "Predict BTC price on 2024-12-31"
   ```

## Error Handling

The application includes robust error handling for:

- Missing data files
- Invalid cryptocurrency tickers
- Malformed JSON responses
- Date parsing errors
- Model prediction failures

## Technical Notes

- The Prophet model uses yearly, weekly, and daily seasonality for predictions
- Logarithmic transformation is applied to price data for better forecasting
- Date ranges are handled flexibly with both specific dates and ranges supported
- The application maintains a mapping of ticker symbols to full cryptocurrency names

## Future Improvements

- Add support for more cryptocurrencies
- Implement additional technical indicators
- Enhance visualization options
- Add export functionality for predictions
- Implement real-time data updates
