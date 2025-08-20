

---

# Risk Optimized Pairs Trading Strategy using Bayesian Optimization

## Project Overview

This project implements an optimized algorithmic trading strategy based on **pairs trading**. The strategy involves identifying cointegrated pairs of stocks, forecasting price movements using ARIMA models, and optimizing trading parameters using Bayesian optimization with Advanced Risk Measures. The strategy is backtested and compared to a benchmark to evaluate its performance 

### Key Features:
- **Cointegration Analysis**: Identifying cointegrated pairs of stocks.
- **ARIMA Forecasting**: Using ARIMA models to predict stock prices.
- **Bayesian Optimization**: Optimizing trading parameters for better performance.
- **Backtesting**: Testing the strategy on historical data and comparing it to a benchmark.
- **Risk Measure**: Calculation of Risk Measure to constrol risk in trading strategy.

## Files in This Repository:
- **experiment.ipynb**: Core experiments including data collection, cointegration testing, model training, optimization, and backtesting.
- **src/**: Source code for utilities, models, and optimization.



## Usage

1. **Data Collection**: Download stock price data using Yahoo Finance.
2. **Cointegration Test**: Identify cointegrated pairs using the `cointegration_test()` function.
3. **ARIMA Forecasting**: Train an ARIMA model on selected pairs.
4. **Optimization**: Use **Bayesian Optimization** to fine-tune strategy parameters.



## Conclusion
This project optimizes a pairs trading strategy using statistical methods to achieve superior performance in the financial market by optimizing risk measures to control risk and better performance over naive strategies

