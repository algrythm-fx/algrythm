# algrythm
# AI Scalping Bot for MT5 (XAUUSD)

This project is a high-frequency, AI-powered scalping bot for MetaTrader 5 (MT5), specifically designed to trade XAUUSD using intelligent predictive models and robust risk management techniques.

## 🚀 Features

- 📈 Real-time connection to MT5 using official MetaTrader5 Python API
- 🧠 AI-powered signal prediction using XGBoost and RandomForest
- 🔍 Advanced technical analysis with indicators (RSI, MACD, BB, ATR, Fibonacci, etc.)
- 🧪 Backtesting engine with equity curve visualization
- ⚙️ Genetic Algorithm and Grid Search hyperparameter tuning
- 🔄 Online learning from logged trades
- 🗞️ News-based filtering using NewsAPI
- 🔐 Dynamic risk management with ATR-based SL/TP
- 📊 Multi-timeframe market trend analysis
- 🔁 Pyramiding and trailing stop logic
- 🧠 Optional ensemble (voting) model support
- 📉 Trade log to CSV

## 🛠 Requirements

- Python 3.8+
- MetaTrader5
- pandas, numpy, talib
- xgboost, scikit-learn
- matplotlib, joblib
- deap (for genetic optimization)
- python-dotenv
- requests

Install dependencies:

```bash
pip install -r requirements.txt
