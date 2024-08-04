# Tradonaut

Tradonaut is a trading system that uses neural networks to analyze stock market data and generate trading signals. It combines real-time and historical data with deep learning techniques to create and execute scalp trading strategies.

## Architecture

- **Convolutional Neural Networks (CNNs):** Extract features from raw data and technical indicators.
- **Long Short-Term Memory Networks (LSTMs):** Find correlation between time series data.
- **Transformer-Based Architectures:** Capture long-term dependencies and contextual relationships.
- **Trend Analysis Model (Optional):** Analyze larger time frame data to provide broader market context.
- **Reinforcement Learning (RL) Model:** Optimize trading strategies and decision-making.

## Workflow

1. Collect and preprocess data from both 1-minute and larger time frames.

2. Extract features using CNNs for both time frames.

3. Periodically perform trend analysis on larger time frame data.

4. Feed CNN output, along with larger timeframe trend, into the LSTM or Transformer models.

5. Optimize trading decisions with the RL model.

6. Generate and execute trading signals, setting appropriate TP and SL levels.

7. Test the integrated approach using historical data.

8. Deploy the system live and monitor performance.

## Resources

- [Investopedia: Neural Networks and Trading](https://www.investopedia.com/articles/trading/06/neuralnetworks.asp)
- [ML4Trading: Machine Learning for Trading](https://ml4trading.io/)
