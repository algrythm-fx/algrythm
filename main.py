import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import time
import warnings
from dotenv import load_dotenv
import secrets
import os
import csv
import json # Make sure json is imported at the top

load_dotenv()
if not os.path.exists('.env') or not os.getenv("API_KEY"):
    key = secrets.token_hex(32)
    with open(".env", "w") as f:
        f.write(f"API_KEY={key}\n")
    print(f"Generated new API key and saved to .env: {key}")

import requests
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib # Import joblib for model persistence
import os     # For checking file existence

from deap import base, creator, tools, algorithms
import random

warnings.filterwarnings('ignore')

# --- Configuration ---
SYMBOL = "XAUUSD"  # Example: Gold vs USD
TIMEFRAME = mt5.TIMEFRAME_M5  # 15-minute timeframe
LOT_SIZE = 0.05  # Standard lot size for micro accounts (adjust based on your account and risk)
MAGIC_NUMBER = 12345  # Unique ID for your bot's trades
ACCOUNT_DETAILS = {
    'login': 81435245,  # Your MT5 account login
    'password': 'Dreamweaver247$',  # Your MT5 account password
    'server': 'Exness-MT5Trial10'  # Your MT5 broker server (e.g., 'Exness-MT5server')
}

# New configuration for multiple positions and cooldown
MAX_OPEN_POSITIONS = 20  # Maximum number of concurrent positions allowed by the bot
COOLDOWN_PERIOD_SECONDS = 60 # 1 minute cooldown between NEW TRADE ENTRIES

# File paths for saving models and metadata
MODEL_RF_PATH = 'rf_model.joblib'
MODEL_XGB_PATH = 'xgb_model.joblib'
MODEL_METADATA_PATH = 'model_metadata.joblib' # To save features and label_mapping

# Global variable to track the last time a trade was entered
last_trade_entry_time = None

# --- 1. Data Acquisition and Preprocessing ---

def connect_mt5(account_details):
    if not mt5.initialize(login=account_details['login'],
                          server=account_details['server'],
                          password=account_details['password']):
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return False
    print("MT5 initialized successfully.")
    # print(mt5.terminal_info()) # Uncomment for verbose terminal info
    return True

def get_historical_data(symbol, timeframe, num_bars):
    # Initialize MT5 if not already initialized (might happen if called directly)
    if not mt5.initialize():
        print("MT5 re-initialization failed in get_historical_data")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"Failed to get rates for {symbol}, error code: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]

def calculate_indicators(df):
    # Ensure there's enough data for indicator calculation
    # Max period is 30 for SMA/EMA, 26 for MACD, 20 for BB, 14 for RSI/ATR, 5 for STOCH
    min_bars_needed = max(30, 26, 20, 14, 5)
    if len(df) < min_bars_needed:
        print(f"Not enough data to calculate all indicators. Need at least {min_bars_needed} bars, got {len(df)}.")
        return pd.DataFrame() # Return empty DataFrame to signal failure

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    
    # Moving Averages
    df_copy['SMA_10'] = talib.SMA(df_copy['close'], timeperiod=10)
    df_copy['SMA_30'] = talib.SMA(df_copy['close'], timeperiod=30)
    df_copy['EMA_10'] = talib.EMA(df_copy['close'], timeperiod=10)
    df_copy['EMA_30'] = talib.EMA(df_copy['close'], timeperiod=30)

    # RSI
    df_copy['RSI'] = talib.RSI(df_copy['close'], timeperiod=14)

    # MACD
    macd, macdsignal, macdhist = talib.MACD(df_copy['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df_copy['MACD'] = macd
    df_copy['MACD_Signal'] = macdsignal
    df_copy['MACD_Hist'] = macdhist

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df_copy['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df_copy['BB_Upper'] = upper
    df_copy['BB_Middle'] = middle
    df_copy['BB_Lower'] = lower

    # Stochastic Oscillator
    fastk, fastd = talib.STOCH(df_copy['high'], df_copy['low'], df_copy['close'],
                                 fastk_period=5, slowk_period=3, slowk_matype=0,
                                 slowd_period=3, slowd_matype=0)
    df_copy['STOCH_K'] = fastk
    df_copy['STOCH_D'] = fastd

    # ATR
    df_copy['ATR'] = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)

    # --- Fibonacci Retracement (last 50 bars swing high/low) ---
    lookback = 50
    swing_high = df_copy['high'].rolling(window=lookback).max()
    swing_low = df_copy['low'].rolling(window=lookback).min()

    df_copy['fib_236'] = swing_high - (swing_high - swing_low) * 0.236
    df_copy['fib_382'] = swing_high - (swing_high - swing_low) * 0.382
    df_copy['fib_500'] = swing_high - (swing_high - swing_low) * 0.5
    df_copy['fib_618'] = swing_high - (swing_high - swing_low) * 0.618
    df_copy['fib_786'] = swing_high - (swing_high - swing_low) * 0.786


    # Drop rows with NaN values resulting from indicator calculation
    df_copy.dropna(inplace=True)
    return df_copy

# --- 2. Strategy Development (AI Integration) ---

def label_data(df, future_period_bars=5, atr_multiplier=0.5):
    df_copy = df.copy()

    # Add ATR if not already present
    if 'ATR' not in df_copy.columns:
        df_copy['ATR'] = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)

    # --- Fibonacci Retracement (last 50 bars swing high/low) ---
    lookback = 50
    swing_high = df_copy['high'].rolling(window=lookback).max()
    swing_low = df_copy['low'].rolling(window=lookback).min()

    df_copy['fib_236'] = swing_high - (swing_high - swing_low) * 0.236
    df_copy['fib_382'] = swing_high - (swing_high - swing_low) * 0.382
    df_copy['fib_500'] = swing_high - (swing_high - swing_low) * 0.5
    df_copy['fib_618'] = swing_high - (swing_high - swing_low) * 0.618
    df_copy['fib_786'] = swing_high - (swing_high - swing_low) * 0.786


    df_copy['future_close'] = df_copy['close'].shift(-future_period_bars)
    df_copy['price_change'] = (df_copy['future_close'] - df_copy['close'])

    # Dynamic threshold using ATR
    df_copy['threshold'] = df_copy['ATR'] * atr_multiplier

    df_copy['signal'] = 0
    df_copy.loc[df_copy['price_change'] > df_copy['threshold'], 'signal'] = 1
    df_copy.loc[df_copy['price_change'] < -df_copy['threshold'], 'signal'] = -1

    df_copy.dropna(inplace=True)
    return df_copy

def train_model(df, model_type='random_forest', genetic_optimization=False):
    features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'future_close', 'price_change', 'signal', 'threshold']]
    X = df[features]
    y = df['signal']

    # --- IMPORTANT: Map labels for XGBoost / Scikit-learn ---
    # Convert [-1, 0, 1] to [0, 1, 2] for model compatibility
    # Ensure consistent mapping: -1 -> 0, 0 -> 1, 1 -> 2
    label_mapping = {-1: 0, 0: 1, 1: 2} # Explicitly define mapping
    y_mapped = y.map(label_mapping)
    # Check if all classes are present after mapping
    if len(y_mapped.unique()) < 3:
        print(f"WARNING: After mapping, less than 3 unique classes in y_mapped: {y_mapped.unique()}. This can cause issues.")

    # Only split if you have enough samples for all classes for stratification
    if len(y_mapped.unique()) < 2 or (len(y_mapped[y_mapped == 0]) < 2 or len(y_mapped[y_mapped == 1]) < 2 or len(y_mapped[y_mapped == 2]) < 2): # At least 2 samples per class for stratify
        print("WARNING: Not enough samples per class for stratified split. Falling back to non-stratified split or skipping split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped)


    model = None
    if model_type == 'random_forest':
        if genetic_optimization:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()
            toolbox.register("attr_n_estimators", random.randint, 50, 200)
            toolbox.register("attr_max_depth", random.randint, 5, 20)
            toolbox.register("attr_min_samples_leaf", random.randint, 1, 10)

            toolbox.register("individual", tools.initCycle, creator.Individual,
                             (toolbox.attr_n_estimators, toolbox.attr_max_depth, toolbox.attr_min_samples_leaf), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def evaluate(individual):
                n_estimators, max_depth, min_samples_leaf = individual
                # Ensure parameters are integers for RandomForestClassifier
                n_estimators = int(n_estimators)
                max_depth = int(max_depth)
                min_samples_leaf = int(min_samples_leaf)

                model_ga = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf, random_state=42)
                try:
                    model_ga.fit(X_train, y_train)
                    y_pred = model_ga.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                except Exception as e:
                    print(f"Error during GA RF evaluation: {e}")
                    accuracy = 0.0 # Return poor fitness on error
                return accuracy,

            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutUniformInt, low=[50, 5, 1], up=[200, 20, 10], indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)

            population = toolbox.population(n=20) # Reduced population/generations for faster demo
            algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, verbose=False) # Reduced generations

            best_individual = tools.selBest(population, 1)[0]
            best_n_estimators, best_max_depth, best_min_samples_leaf = best_individual
            print(f"Genetic Algorithm Best Params (RF): n_estimators={best_n_estimators}, max_depth={best_max_depth}, min_samples_leaf={best_min_samples_leaf}")
            model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth,
                                           min_samples_leaf=best_min_samples_leaf, random_state=42)
        else:
            param_grid = {
                'n_estimators': [50, 100], # Reduced search space for quicker runs
                'max_depth': [5, 10],
                'min_samples_leaf': [1, 2]
            }
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Random Forest Best Params (Grid Search): {grid_search.best_params_}")

    elif model_type == 'xgboost':
        if genetic_optimization:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()
            toolbox.register("attr_n_estimators", random.randint, 50, 200)
            toolbox.register("attr_max_depth", random.randint, 3, 10)
            toolbox.register("attr_learning_rate", random.uniform, 0.01, 0.3)

            toolbox.register("individual", tools.initCycle, creator.Individual,
                             (toolbox.attr_n_estimators, toolbox.attr_max_depth, toolbox.attr_learning_rate), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def evaluate_xgb(individual):
                n_estimators, max_depth, learning_rate = individual
                # Ensure parameters are correct types
                n_estimators = int(n_estimators)
                max_depth = int(max_depth)

                model_ga = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=learning_rate, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
                try:
                    model_ga.fit(X_train, y_train)
                    y_pred = model_ga.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                except Exception as e:
                    print(f"Error during GA XGB evaluation: {e}")
                    accuracy = 0.0
                return accuracy,

            toolbox.register("evaluate", evaluate_xgb)
            toolbox.register("mate", tools.cxTwoPoint)
            def mutate_xgb(individual, low, up, indpb):
                for i in range(len(individual)):
                    if random.random() < indpb:
                        if i == 2: # learning_rate is float
                            individual[i] = random.uniform(low[i], up[i])
                        else: # n_estimators, max_depth are int
                            individual[i] = random.randint(low[i], up[i])
                return individual,
            toolbox.register("mutate", mutate_xgb, low=[50, 3, 0.01], up=[200, 10, 0.3], indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)

            population = toolbox.population(n=20) # Reduced population/generations for faster demo
            algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, verbose=False) # Reduced generations

            best_individual = tools.selBest(population, 1)[0]
            best_n_estimators, best_max_depth, best_learning_rate = best_individual
            print(f"Genetic Algorithm Best Params (XGB): n_estimators={int(best_n_estimators)}, max_depth={int(best_max_depth)}, learning_rate={best_learning_rate:.3f}")
            model = XGBClassifier(n_estimators=int(best_n_estimators), max_depth=int(best_max_depth),
                                  learning_rate=best_learning_rate, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        else:
            param_grid = {
                'n_estimators': [50, 100], # Reduced search space
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
            grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"XGBoost Best Params (Grid Search): {grid_search.best_params_}")
            
    # Final fit with the best model found (either by GA or GridSearchCV)
    if model:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("\n--- Model Performance ---")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    else:
        print("Error: No model was trained.")

    return model, features, label_mapping # Only return what's needed for persistence and live trading


# --- Model Persistence Functions ---
def save_model(model, features, label_mapping, model_type='xgboost'):
    """Saves the trained model and associated metadata."""
    if model_type == 'xgboost':
        joblib.dump(model, MODEL_XGB_PATH)
        print(f"XGBoost model saved to {MODEL_XGB_PATH}")
    elif model_type == 'random_forest':
        joblib.dump(model, MODEL_RF_PATH)
        print(f"Random Forest model saved to {MODEL_RF_PATH}")
    
    # Save features and label_mapping
    metadata = {'features': features, 'label_mapping': label_mapping}
    joblib.dump(metadata, MODEL_METADATA_PATH)
    print(f"Model metadata (features, label_mapping) saved to {MODEL_METADATA_PATH}")

def load_model(model_type='xgboost'):
    """Loads the trained model and associated metadata."""
    model = None
    features = None
    label_mapping = None

    if model_type == 'xgboost' and os.path.exists(MODEL_XGB_PATH):
        model = joblib.load(MODEL_XGB_PATH)
        print(f"XGBoost model loaded from {MODEL_XGB_PATH}")
    elif model_type == 'random_forest' and os.path.exists(MODEL_RF_PATH):
        model = joblib.load(MODEL_RF_PATH)
        print(f"Random Forest model loaded from {MODEL_RF_PATH}")
    else:
        print(f"No saved {model_type} model found.")
        return None, None, None
    
    if os.path.exists(MODEL_METADATA_PATH):
        metadata = joblib.load(MODEL_METADATA_PATH)
        features = metadata.get('features')
        label_mapping = metadata.get('label_mapping')
        print(f"Model metadata (features, label_mapping) loaded from {MODEL_METADATA_PATH}")
    else:
        print("No model metadata found.")
        return None, None, None

    return model, features, label_mapping


# --- 3. Backtesting and Evaluation ---

def run_backtest(df, model, features, initial_balance=10000, label_mapping=None):
    """
    Simple backtesting function. For a real bot, you'd want a more sophisticated backtester
    that handles slippage, commissions, real-time data simulation, etc.
    """
    balance = initial_balance
    positions = 0
    trade_history = []
    
    # Predict signals for the entire dataframe (or a subset for backtesting)
    df_backtest = df.copy()
    
    if label_mapping is None:
        print("Error: label_mapping not provided for backtesting.")
        return

    # Ensure to use the correct features for prediction
    # Handle potential DataFrame empty issue or missing features
    if df_backtest[features].empty or len(df_backtest[features].columns) != len(features):
        print("Error: Backtesting DataFrame is empty or features are missing.")
        return

    try:
        df_backtest['predicted_signal_mapped'] = model.predict(df_backtest[features])
    except ValueError as e:
        print(f"Error predicting in backtest: {e}. Check if features match model's expected input.")
        return

    # Reverse the mapping to get the original signal [-1, 0, 1]
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    df_backtest['predicted_signal'] = df_backtest['predicted_signal_mapped'].map(reverse_label_mapping)

    # Simplified lot size calculation for backtest. Assumes constant risk per trade in terms of capital percentage.
    # For backtesting with SL/TP, you would integrate calculate_stop_loss/take_profit into the backtest loop
    # and simulate trades hitting those levels.
    
    # Get symbol info for backtest profit calculation
    symbol_info = mt5.symbol_info(SYMBOL)
    if not symbol_info:
        print("Could not get symbol info for backtesting profit calculation.")
        return

    # Value of one pip for XAUUSD (assuming 0.01 price movement is a pip)
    # And 1 standard lot = 100 units, so 0.01 price change = $1/lot (for XAUUSD)
    pip_value_per_lot = 1.0 # $1 per 0.01 price change for 1 standard lot of XAUUSD
    
    for i in range(len(df_backtest)):
        current_price = df_backtest['close'].iloc[i]
        signal = df_backtest['predicted_signal'].iloc[i]

        # Simplified profit/loss calculation for open positions
        # This backtest assumes immediate closure at current bar's close price if signal changes
        # A more realistic backtest would simulate SL/TP hit logic within the bar.
        
        # Check if there's an open position and a signal to close it
        if positions > 0 and signal == -1: # Close long position if going to sell
            profit_per_unit = (current_price - trade_history[-1]['entry_price'])
            profit = profit_per_unit * (abs(positions) * 100) # Multiply by 100 for XAUUSD units per lot
            balance += profit
            trade_history.append({'type': 'SELL_CLOSE', 'time': df_backtest.index[i], 'entry_price': trade_history[-1]['entry_price'], 'exit_price': current_price, 'profit': profit, 'balance': balance})
            positions = 0
            # print(f"Closed Long at {current_price:.4f}, Profit: {profit:.2f}, Balance: {balance:.2f}")

        elif positions < 0 and signal == 1: # Close short position if going to buy
            profit_per_unit = (trade_history[-1]['entry_price'] - current_price)
            profit = profit_per_unit * (abs(positions) * 100) # Multiply by 100 for XAUUSD units per lot
            balance += profit
            trade_history.append({'type': 'BUY_CLOSE', 'time': df_backtest.index[i], 'entry_price': trade_history[-1]['entry_price'], 'exit_price': current_price, 'profit': profit, 'balance': balance})
            positions = 0
            # print(f"Closed Short at {current_price:.4f}, Profit: {profit:.2f}, Balance: {balance:.2f}")

        # Open new position if no position is open
        if signal == 1 and positions == 0:  # Open long position
            positions = LOT_SIZE # Use constant lot size for simplicity in this backtest
            trade_history.append({'type': 'BUY', 'time': df_backtest.index[i], 'entry_price': current_price, 'balance': balance})
            # print(f"Opened Long at {current_price:.4f}")

        elif signal == -1 and positions == 0:  # Open short position
            positions = -LOT_SIZE # Use constant lot size
            trade_history.append({'type': 'SELL', 'time': df_backtest.index[i], 'entry_price': current_price, 'balance': balance})
            # print(f"Opened Short at {current_price:.4f}")

    print(f"\n--- Backtesting Results ---")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    total_profit = balance - initial_balance
    print(f"Total Profit: ${total_profit:.2f}")
    
    # --- Visualization ---
    equity_curve_values = [initial_balance]
    current_equity = initial_balance
    # Reconstruct equity curve based on trade history
    for trade in trade_history:
        if 'balance' in trade: # Use the balance recorded at each trade point
            equity_curve_values.append(trade['balance'])
        elif 'profit' in trade: # Fallback if balance isn't recorded directly for every point
            current_equity += trade['profit']
            equity_curve_values.append(current_equity)

    if len(equity_curve_values) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve_values)
        plt.title('Equity Curve')
        plt.xlabel('Trade Points')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.show()
    else:
        print("Not enough trades to plot equity curve.")

# --- 4. Risk Management (Placeholder functions) ---
# These now correctly use ATR for dynamic SL/TP
def calculate_stop_loss(entry_price, atr, trade_type):
    # Adjust ATR multiplier as needed. 1.5x ATR is common for SL.
    if trade_type == 'buy':
        return entry_price - (1.5 * atr)
    elif trade_type == 'sell':
        return entry_price + (1.5 * atr)
    return 0.0 # Return 0.0 for invalid types so MT5 doesn't reject SL

def calculate_take_profit(entry_price, atr, trade_type):
    # Adjust ATR multiplier as needed. 2.0x ATR for TP (1:1.33 R:R if SL is 1.5 ATR)
    if trade_type == 'buy':
        return entry_price + (2.0 * atr)
    elif trade_type == 'sell':
        return entry_price - (2.0 * atr)
    return 0.0 # Return 0.0 for invalid types


# --- INTEGRATION: New function to check and log closed trades for online learning ---
def check_and_log_closed_trades(model_type='ensemble'):
    """
    Checks the account history for recently closed trades matching the magic number,
    logs their full details to a CSV, and then checks if the model should be retrained.
    """
    from_date = datetime.utcnow() - timedelta(days=7) # Look at last 7 days of history
    to_date = datetime.utcnow()

    history_deals = mt5.history_deals_get(from_date, to_date)
    if history_deals is None:
        return

    # Path for our detailed trade log
    log_filepath = "logs/trade_history_log.csv"
    os.makedirs("logs", exist_ok=True)
    
    # Read existing tickets to avoid duplicates
    existing_tickets = set()
    if os.path.exists(log_filepath):
        try:
            df_existing = pd.read_csv(log_filepath)
            if 'ticket' in df_existing.columns:
                existing_tickets = set(df_existing['ticket'].astype(int))
        except pd.errors.EmptyDataError:
            pass # File is empty, that's fine

    new_trades_to_log = []
    processed_deal_tickets = set()

    for deal in history_deals:
        # We are looking for "out" deals (closing deals)
        if deal.magic == MAGIC_NUMBER and deal.entry == 1 and deal.ticket not in existing_tickets and deal.ticket not in processed_deal_tickets:
            entry_deal = mt5.history_deals_get(position_id=deal.position_id)
            if entry_deal:
                entry_deal = [d for d in entry_deal if d.entry == 0][0] # Find the "in" deal
                
                trade_record = {
                    'ticket': deal.ticket,
                    'time': datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'BUY' if entry_deal.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': deal.volume,
                    'entry_price': entry_deal.price,
                    'exit_price': deal.price,
                    'profit': deal.profit,
                }
                new_trades_to_log.append(trade_record)
                processed_deal_tickets.add(deal.ticket) # Mark as processed for this run

    # Append new trades to the CSV if any exist
    if new_trades_to_log:
        df_new = pd.DataFrame(new_trades_to_log)
        is_new_file = not os.path.exists(log_filepath)
        df_new.to_csv(log_filepath, mode='a', header=is_new_file, index=False)
        print(f"--- Logged {len(new_trades_to_log)} new closed trades. ---")

    # Now check for retraining based on the log file size
    if should_retrain(log_filepath):
        print("--- INTEGRATION: Trade log threshold reached. Triggering model update. ---")
        send_alert("Trade log threshold reached. Attempting to retrain model.")
        update_model_from_logged_trades(model_type=model_type)



# --- 5. Live Trading Execution ---

def send_order(symbol, trade_type, lot_size, sl_price=0.0, tp_price=0.0):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"Failed to get tick for {symbol}. Cannot send order.")
        return False, None

    if trade_type == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else: # SELL
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "AI Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC, # Good Till Cancel
        "type_filling": mt5.ORDER_FILLING_FOK, # Fill or Kill (must be filled completely or not at all)
    }
    # Only add SL/TP if they are valid (non-zero)
    if sl_price > 0: # Assuming 0.0 is invalid SL
        request["sl"] = sl_price
    if tp_price > 0: # Assuming 0.0 is invalid TP
        request["tp"] = tp_price

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, retcode={result.retcode}, error={mt5.last_error()}")
        print(f"Request details: {request}") # Print full request for debugging
        if result.comment:
            print(f"MT5 Comment: {result.comment}")
        return False, result
    else:
        print(f"{trade_type} order placed successfully. Order: {result.order}, Position: {result.deal}")
        return True, result

def close_position(position_ticket, current_price_bid, current_price_ask):
    position_info = mt5.positions_get(ticket=position_ticket)
    if not position_info:
        print(f"Position {position_ticket} not found.")
        return False
    
    position_info = position_info[0] # Get the single position object

    # Determine closure type and price
    if position_info.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        close_price = current_price_bid # Sell at bid to close buy
    else: # POSITION_TYPE_SELL
        order_type = mt5.ORDER_TYPE_BUY
        close_price = current_price_ask # Buy at ask to close sell

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position_ticket,
        "symbol": SYMBOL,
        "volume": position_info.volume,
        "type": order_type,
        "price": close_price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "AI Bot Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN, # Return partial fills if not complete
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Close order failed, retcode={result.retcode}, error={mt5.last_error()}")
        print(f"Request details: {request}")
        if result.comment:
            print(f"MT5 Comment: {result.comment}")
        return False
    else:
        print(f"Position {position_ticket} closed successfully.")
        return True


# --- Enhancements: Structure + Filters + Risk + News + Online Learning ---

# === Market Structure Filters ===
def detect_nearest_order_block(df, side="buy", window=50, wick_threshold=0.4):
    blocks = []
    for i in range(len(df) - window, len(df)):
        row = df.iloc[i]
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['close'], row['open'])
        lower_wick = min(row['close'], row['open']) - row['low']

        if side == "buy" and lower_wick > wick_threshold * body:
            blocks.append((df.index[i], row['low']))
        elif side == "sell" and upper_wick > wick_threshold * body:
            blocks.append((df.index[i], row['high']))

    if not blocks:
        return None

    if side == "buy":
        nearest = min(blocks, key=lambda x: abs(df['close'].iloc[-1] - x[1]) if x[1] < df['close'].iloc[-1] else float('inf'))
    else:
        nearest = min(blocks, key=lambda x: abs(df['close'].iloc[-1] - x[1]) if x[1] > df['close'].iloc[-1] else float('inf'))
    return nearest[1]

def confirm_break_of_structure(df, direction="bullish", swing_window=5):
    highs = df['high'].rolling(window=swing_window).max()
    lows = df['low'].rolling(window=swing_window).min()

    if direction == "bullish":
        recent_high = highs.iloc[-(swing_window + 1)]
        return df['high'].iloc[-1] > recent_high
    elif direction == "bearish":
        recent_low = lows.iloc[-(swing_window + 1)]
        return df['low'].iloc[-1] < recent_low
    return False

def should_trade(predicted_signal, current_price, df, proba, threshold=0.6):
    if max(proba) < threshold:
        print(f"Low confidence: {max(proba):.2f} < {threshold}, skipping.")
        return False

    if predicted_signal == 1:
        order_block = detect_nearest_order_block(df, side="buy")
        bos = confirm_break_of_structure(df, direction="bullish")
        return bos and order_block and current_price >= order_block
    elif predicted_signal == -1:
        order_block = detect_nearest_order_block(df, side="sell")
        bos = confirm_break_of_structure(df, direction="bearish")
        return bos and order_block and current_price <= order_block
    return False

# === Dynamic Risk & Trend ===
def calculate_dynamic_lot_size(account_balance, risk_percentage, atr, symbol):
    risk_amount = account_balance * risk_percentage
    pip_value_per_lot = 1.0
    stop_loss_pips = atr / 0.01
    if stop_loss_pips == 0:
        return 0.0
    lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
    return round(lot_size, 2)

def determine_market_trend(df):
    highs = df['high'].rolling(window=5).max()
    lows = df['low'].rolling(window=5).min()
    recent_highs = highs.iloc[-5:]
    recent_lows = lows.iloc[-5:]
    if recent_highs.is_monotonic_increasing and recent_lows.is_monotonic_increasing:
        return "uptrend"
    elif recent_highs.is_monotonic_decreasing and recent_lows.is_monotonic_decreasing:
        return "downtrend"
    return "range"

def send_alert(message):
    print(f"[ALERT] {message}")

# === Online Learning ===
def log_trade(features, signal, proba, result):
    os.makedirs("logs", exist_ok=True)
    with open("logs/trade_training_data.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(features + [signal, max(proba), result])

def log_trade_result(trade_type, entry_price, exit_price, proba, features):
    result = 1 if (trade_type == "BUY" and exit_price > entry_price) or (trade_type == "SELL" and exit_price < entry_price) else 0
    signal = 1 if trade_type == "BUY" else -1
    log_trade(features, signal, proba, result)

# === Automatically retrain after every N trades ===
TRADE_LOG_THRESHOLD = 50  # Retrain after every 50 logged trades

def should_retrain(filepath):
    """
    Checks if the model should be retrained based on the number of trades in the log file.
    """
    # THE FIX IS HERE: The function now accepts 'filepath' as an argument.
    
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                # Read all lines. The first line is the header.
                lines = f.readlines()
                num_trades = len(lines) - 1
                
                # Check if the number of trades is a positive multiple of the threshold
                if num_trades > 0 and num_trades % TRADE_LOG_THRESHOLD == 0:
                    return True
        except Exception as e:
            print(f"Could not read log file {filepath}: {e}")
            
    return False

def load_logged_trades():
    filepath = "logs/trade_training_data.csv"
    if not os.path.exists(filepath):
        return None, None
    df = pd.read_csv(filepath, header=None)
    X = df.iloc[:, :-3]
    y = df.iloc[:, -3]
    return X, y

def update_model_from_logged_trades(model_type='xgboost'):
    X, y = load_logged_trades()
    if X is None:
        print("No logged trade data found for retraining.")
        return
    print(f"Retraining {model_type} model on logged trades: {len(X)} samples")
    if model_type == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    else:
        print("Unsupported model type for retrain.")
        return
    model.fit(X, y)
    joblib.dump(model, MODEL_XGB_PATH if model_type=='xgboost' else MODEL_RF_PATH)
    print("Model updated with new training data.")


# --- Multi-Timeframe Analysis ---
def get_trend_from_higher_tf(symbol, higher_tf=mt5.TIMEFRAME_H1, window=30):
    df_higher = get_historical_data(symbol, higher_tf, window)
    if df_higher is None or df_higher.empty:
        print("Failed to fetch higher timeframe data.")
        return "unknown"
    highs = df_higher['high'].rolling(window=5).max()
    lows = df_higher['low'].rolling(window=5).min()
    if highs.iloc[-5:].is_monotonic_increasing and lows.iloc[-5:].is_monotonic_increasing:
        return "uptrend"
    elif highs.iloc[-5:].is_monotonic_decreasing and lows.iloc[-5:].is_monotonic_decreasing:
        return "downtrend"
    return "range"

# --- Volume-Aware Order Block Detection ---
def detect_nearest_order_block(df, side="buy", window=50, wick_threshold=0.4, volume_multiplier=1.2):
    blocks = []
    avg_volume = df['tick_volume'].rolling(window=20).mean().iloc[-1]
    for i in range(len(df) - window, len(df)):
        row = df.iloc[i]
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['close'], row['open'])
        lower_wick = min(row['close'], row['open']) - row['low']
        volume = row['tick_volume']
        if side == "buy" and lower_wick > wick_threshold * body and volume > avg_volume * volume_multiplier:
            blocks.append((df.index[i], row['low']))
        elif side == "sell" and upper_wick > wick_threshold * body and volume > avg_volume * volume_multiplier:
            blocks.append((df.index[i], row['high']))
    if not blocks:
        return None
    current_price = df['close'].iloc[-1]
    if side == "buy":
        nearest = min(blocks, key=lambda x: abs(current_price - x[1]) if x[1] < current_price else float('inf'))
    else:
        nearest = min(blocks, key=lambda x: abs(current_price - x[1]) if x[1] > current_price else float('inf'))
    return nearest[1]

# --- ATR-Based Trailing Stop Loss ---
def calculate_trailing_stop(entry_price, atr, trade_type, trail_multiplier=1.0):
    if trade_type == 'buy':
        return entry_price + (trail_multiplier * atr)
    elif trade_type == 'sell':
        return entry_price - (trail_multiplier * atr)
    return 0.0

# --- SOLUTION: Replace the old update_trailing_stop function with this one ---
def update_trailing_stop(position, atr, trail_multiplier=1.5):
    """
    Updates the trailing stop for a given position, respecting the broker's Stops Level.
    """
    if position.magic != MAGIC_NUMBER:
        return

    symbol = position.symbol
    
    # --- 1. Get symbol info, including the crucial Stops Level and digits ---
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"Could not get symbol info for {symbol}")
        return
        
    stops_level_points = symbol_info.trade_stops_level
    price_digits = symbol_info.digits
    point = symbol_info.point

    # --- 2. Get current market prices ---
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"Could not get tick for {symbol}")
        return
    
    bid_price = tick.bid
    ask_price = tick.ask

    new_sl = 0.0
    current_sl = position.sl

    # --- 3. Calculate potential new SL ---
    # For a BUY position, SL trails below the price
    if position.type == mt5.POSITION_TYPE_BUY:
        # The new SL is based on the current bid price (the exit price)
        potential_sl = bid_price - (trail_multiplier * atr)
        
        # Check if the new SL is profitable (higher than entry) and tighter than the old one
        if potential_sl > current_sl and potential_sl > position.price_open:
            # --- 4. CRITICAL CHECK: Ensure new SL respects the Stops Level ---
            # SL for a BUY must be BELOW the current BID price by at least the stops_level distance
            if potential_sl < bid_price - (stops_level_points * point):
                new_sl = potential_sl
            else:
                # If it's too close, place it exactly at the minimum distance
                new_sl = bid_price - (stops_level_points * point)
                print(f"Position {position.ticket}: Potential SL too close. Adjusting to broker's min distance.")

    # For a SELL position, SL trails above the price
    elif position.type == mt5.POSITION_TYPE_SELL:
        # The new SL is based on the current ask price (the exit price)
        potential_sl = ask_price + (trail_multiplier * atr)
        
        # Check if the new SL is profitable (lower than entry) and tighter than the old one
        if potential_sl < current_sl and potential_sl < position.price_open:
            # --- 4. CRITICAL CHECK: Ensure new SL respects the Stops Level ---
            # SL for a SELL must be ABOVE the current ASK price by at least the stops_level distance
            if potential_sl > ask_price + (stops_level_points * point):
                new_sl = potential_sl
            else:
                new_sl = ask_price + (stops_level_points * point)
                print(f"Position {position.ticket}: Potential SL too close. Adjusting to broker's min distance.")


    # --- 5. Send the modification request ONLY if a valid new SL was calculated ---
    # Also check if the new SL is meaningfully different from the current one
    if new_sl > 0.0 and abs(new_sl - current_sl) > point:
        # Round the SL to the correct number of decimal places for the symbol
        final_sl = round(new_sl, price_digits)
    
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "symbol": symbol,
            "sl": final_sl,
            "tp": position.tp, # Keep original TP
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Trailing SL updated successfully to: {final_sl:.{price_digits}f} for position {position.ticket}")
        else:
            # Provide a much more detailed error report
            print(f"--- FAILED to update trailing SL for position {position.ticket} ---")
            print(f"Retcode: {result.retcode} - {result.comment}")
            print(f"Request Details: SL={final_sl}, Current Bid={bid_price}, Current Ask={ask_price}, StopsLevel={stops_level_points} points")
            mt5.order_send(request) # Send again to get fresh last_error() info if needed
            print(f"MT5 Last Error: {mt5.last_error()}")


# --- Pyramiding ---
MAX_SCALE_INS = 2

def get_bot_positions():
    positions = mt5.positions_get()
    return [p for p in positions if p.magic == MAGIC_NUMBER] if positions else []

def count_active_scale_ins(direction):
    return sum(1 for p in get_bot_positions() if (direction == 1 and p.type == mt5.POSITION_TYPE_BUY) or (direction == -1 and p.type == mt5.POSITION_TYPE_SELL))

def can_add_scale_in(predicted_signal):
    return count_active_scale_ins(predicted_signal) < MAX_SCALE_INS

# --- Backtest Summary + Auto-Recalibration ---
def summarize_backtest(trade_history, initial_balance):
    final_balance = trade_history[-1]['balance'] if trade_history else initial_balance
    total_profit = final_balance - initial_balance
    wins = sum(1 for t in trade_history if t.get("profit", 0) > 0)
    losses = sum(1 for t in trade_history if t.get("profit", 0) < 0)
    total = wins + losses
    win_rate = (wins / total * 100) if total else 0
    print("\n--- Strategy Summary ---")
    print(f"Total Trades: {total}")
    print(f"Wins: {wins}, Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    if win_rate < 70:
        print("⚠️ Win rate below 70%. Triggering model retrain...")
        update_model_from_logged_trades(model_type='xgboost')
    else:
        print("✅ Performance acceptable. No retraining needed.")

# --- Ensemble Voting Model ---
def train_ensemble_model(df):
    features = [col for col in df.columns if col not in ['open','high','low','close','tick_volume','spread','real_volume','future_close','price_change','signal','threshold']]
    X = df[features]
    y = df['signal'].map({-1: 0, 0: 1, 1: 2})
    if len(set(y)) < 3:
        print("Not enough classes for ensemble model.")
        return None, features, {0: -1, 1: 0, 2: 1}
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    rf = RandomForestClassifier(random_state=42)
    ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')
    ensemble.fit(X, y)
    print("✅ Ensemble model trained with XGB + RF")
    return ensemble, features, {0: -1, 1: 0, 2: 1}

# --- RL Logging ---
def log_rl_experience(features, action, reward, done=False):
    os.makedirs("logs", exist_ok=True)
    with open("logs/rl_experience_log.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(features + [action, reward, int(done)])
    print(f"Logged RL experience: action={action}, reward={reward}, done={done}")

# === News API Integration ===
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_BLOCK_WINDOW_MINUTES = 30
HIGH_IMPACT_KEYWORDS = ["interest rate", "CPI", "NFP", "FOMC", "Fed", "jobs report"]

def fetch_news_events(symbol="XAUUSD", lookahead_minutes=60):
    if not NEWS_API_KEY:
        print("No NEWS_API_KEY found in environment.")
        return []
    try:
        url = f"https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": NEWS_API_KEY
        }
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"Failed to fetch news: {resp.status_code}")
            return []
        articles = resp.json().get("articles", [])
        upcoming = []
        now = datetime.utcnow()
        for art in articles:
            published = datetime.strptime(art['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
            if now <= published <= now + timedelta(minutes=lookahead_minutes):
                if any(kw.lower() in art['title'].lower() for kw in HIGH_IMPACT_KEYWORDS):
                    upcoming.append((published, art['title']))
        return upcoming
    except Exception as e:
        print(f"News fetch error: {e}")
        return []

def is_news_time(symbol="XAUUSD"):
    events = fetch_news_events(symbol, lookahead_minutes=NEWS_BLOCK_WINDOW_MINUTES)
    if events:
        print(f"High-impact news within {NEWS_BLOCK_WINDOW_MINUTES} mins:")
        for event in events:
            print(f"- {event[0]} | {event[1]}")
        return True
    return False


def get_open_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []
    # Filter for positions opened by this bot's MAGIC_NUMBER
    return [pos for pos in list(positions) if pos.magic == MAGIC_NUMBER]

def get_current_price(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        return tick.bid, tick.ask
    return None, None

def run_live_trading(model, features, symbol, timeframe, check_interval_seconds=60, label_mapping=None, model_type='ensemble'):
    global last_trade_entry_time

    if not connect_mt5(ACCOUNT_DETAILS):
        print("Could not connect to MT5. Exiting live trading.")
        return

    if label_mapping is None:
        print("Error: label_mapping not provided. Cannot run live trading.")
        return

    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    last_bar_time = None
    print(f"\n--- Starting Enhanced Live Trading for {symbol} ---")
    print(f"Checking every {check_interval_seconds} seconds for new bar...")

    # --- DASHBOARD INTEGRATION: Variables to hold status ---
    from collections import deque
    last_log_messages = deque(maxlen=20) # Store the last 20 log lines
    last_signal, last_confidence, higher_tf_trend = "N/A", 0.0, "N/A"
    
    # Simple logger function
    def log(message):
        print(message)
        last_log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    while True:
        try:
            # --- News Event Filter ---
            if is_news_time(symbol):
                log("--- NEWS ALERT: High-impact news event upcoming. Pausing trading activity. ---")
                time.sleep(check_interval_seconds * 5)
                continue
            
            open_positions = get_open_positions()
            
            # --- Trailing Stop Management ---
            if open_positions:
                current_data_for_atr = get_historical_data(symbol, timeframe, 50)
                if current_data_for_atr is not None and not current_data_for_atr.empty:
                    atr_val = talib.ATR(current_data_for_atr['high'], current_data_for_atr['low'], current_data_for_atr['close'], timeperiod=14).iloc[-1]
                    for pos in open_positions:
                        if (pos.type == mt5.POSITION_TYPE_BUY and pos.price_current > pos.price_open) or \
                           (pos.type == mt5.POSITION_TYPE_SELL and pos.price_current < pos.price_open):
                            update_trailing_stop(pos, atr_val, trail_multiplier=1.5)

            check_and_log_closed_trades(model_type)

            data = get_historical_data(symbol, timeframe, 200)
            if data is None or data.empty:
                log("Failed to get historical data. Retrying...")
                time.sleep(check_interval_seconds)
                continue

            data_with_indicators = calculate_indicators(data)
            if data_with_indicators.empty:
                log("Not enough data to calculate indicators. Waiting for more bars.")
                time.sleep(check_interval_seconds)
                continue

            current_bar_time = data_with_indicators.index[-1]
            can_open_new_trade = True
            if last_trade_entry_time:
                time_since_last_trade = (datetime.utcnow() - last_trade_entry_time).total_seconds()
                if time_since_last_trade < COOLDOWN_PERIOD_SECONDS:
                    can_open_new_trade = False

            if last_bar_time is None or current_bar_time > last_bar_time:
                log(f"\nNew bar detected: {current_bar_time}")
                last_bar_time = current_bar_time

                latest_features = data_with_indicators.iloc[-1][features].values.reshape(1, -1)
                predicted_signal_mapped = model.predict(latest_features)[0]
                predicted_signal = reverse_label_mapping[predicted_signal_mapped]
                proba = model.predict_proba(latest_features)[0]
                
                # --- DASHBOARD INTEGRATION: Update status variables ---
                last_signal = 'BUY' if predicted_signal == 1 else ('SELL' if predicted_signal == -1 else 'HOLD')
                last_confidence = max(proba)
                
                log(f"AI Model Signal: {last_signal} (Confidence: {last_confidence:.2f})")

                higher_tf_trend = get_trend_from_higher_tf(symbol, mt5.TIMEFRAME_H1)
                log(f"Higher Timeframe (H1) Trend: {higher_tf_trend.upper()}")
                
                trade_allowed = False
                if predicted_signal == 1 and higher_tf_trend in ["uptrend", "range"]:
                    trade_allowed = True
                elif predicted_signal == -1 and higher_tf_trend in ["downtrend", "range"]:
                    trade_allowed = True

                if not trade_allowed:
                    log("Signal rejected: Does not align with higher timeframe trend.")
                else:
                    bid, ask = get_current_price(symbol)
                    current_price = ask if predicted_signal == 1 else bid
                    if not should_trade(predicted_signal, current_price, data_with_indicators, proba, threshold=0.65):
                         log("Signal rejected: Market structure or confidence threshold not met.")
                    else:
                        log("Signal confirmed by Market Structure and Confidence.")
                        if not can_add_scale_in(predicted_signal):
                            log("Signal ignored: Max scale-in positions for this direction already open.")
                        elif can_open_new_trade and len(open_positions) < MAX_OPEN_POSITIONS:
                            account_info = mt5.account_info()
                            if account_info:
                                atr_val = data_with_indicators['ATR'].iloc[-1]
                                dynamic_lot_size = calculate_dynamic_lot_size(account_info.balance, 0.01, atr_val, symbol)
                                log(f"Dynamic Lot Size calculated: {dynamic_lot_size} (Risk: 1%)")

                                if dynamic_lot_size > 0.0:
                                    if predicted_signal == 1:
                                        sl = calculate_stop_loss(ask, atr_val, 'buy')
                                        tp = calculate_take_profit(ask, atr_val, 'buy')
                                        success, _ = send_order(symbol, "BUY", dynamic_lot_size, sl_price=sl, tp_price=tp)
                                        if success:
                                            last_trade_entry_time = datetime.utcnow()
                                            send_alert(f"BUY order executed for {symbol} at {ask}")

                                    elif predicted_signal == -1:
                                        sl = calculate_stop_loss(bid, atr_val, 'sell')
                                        tp = calculate_take_profit(bid, atr_val, 'sell')
                                        success, _ = send_order(symbol, "SELL", dynamic_lot_size, sl_price=sl, tp_price=tp)
                                        if success:
                                            last_trade_entry_time = datetime.utcnow()
                                            send_alert(f"SELL order executed for {symbol} at {bid}")
            else:
                log(f"No new bar. Waiting... (Open Positions: {len(open_positions)})")

            # --- DASHBOARD INTEGRATION: Call the update function at the end of every loop ---
            update_dashboard_data(mt5.account_info(), open_positions, last_signal, last_confidence, higher_tf_trend, list(last_log_messages))

        except Exception as e:
            log(f"An error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(check_interval_seconds)


# --- DASHBOARD INTEGRATION: Function to write status files ---
def update_dashboard_data(account_info, open_positions, last_signal, last_confidence, higher_tf_trend, last_log_messages):
    """
    Gathers all relevant data, including trade history, and writes it to a JSON file.
    """
    # Convert MT5 objects to dictionaries
    account_info_dict = {
        'balance': account_info.balance, 'equity': account_info.equity, 'profit': account_info.profit,
        'margin': account_info.margin, 'margin_free': account_info.margin_free, 'margin_level': account_info.margin_level,
    } if account_info else {}

    positions_list = [{
        'ticket': pos.ticket, 'time': datetime.fromtimestamp(pos.time).strftime('%Y-%m-%d %H:%M:%S'),
        'type': 'BUY' if pos.type == 0 else 'SELL', 'volume': pos.volume, 'symbol': pos.symbol,
        'price_open': pos.price_open, 'sl': pos.sl, 'tp': pos.tp,
        'price_current': pos.price_current, 'profit': pos.profit,
    } for pos in open_positions] if open_positions else []

    # --- NEW: Read the trade history log ---
    trade_history = []
    try:
        df_history = pd.read_csv("logs/trade_history_log.csv")
        # Convert to list of dicts for JSON serialization
        trade_history = df_history.to_dict('records')
    except FileNotFoundError:
        pass # No history yet, that's fine.

    dashboard_data = {
        'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'account_info': account_info_dict,
        'open_positions': positions_list,
        'bot_status': {
            'last_signal': last_signal, 'last_confidence': f"{last_confidence:.2f}",
            'higher_tf_trend': higher_tf_trend,
        },
        'log_messages': last_log_messages,
        'trade_history': trade_history, # Add history to the data payload
    }

    try:
        with open('dashboard_data.json', 'w') as f:
            json.dump(dashboard_data, f, indent=4)
    except Exception as e:
        print(f"Error writing to dashboard_data.json: {e}")

        
# --- Main Execution ---

if __name__ == "__main__":
    # --- Step 1: Initialize MT5 Connection ---
    # MT5 connection is now handled inside functions where needed
    # (e.g., connect_mt5, get_historical_data, run_live_trading)

    # --- Step 2: Load or Train the AI Model ---
    model_type = 'xgboost' # Or 'random_forest'
    genetic_optimization = True # Set to True to run GA, False for GridSearchCV

    model, features, label_mapping = load_model(model_type=model_type)

    if model is None or features is None or label_mapping is None:
        print("\n--- No saved model found or loaded successfully. Training new model. ---")
        
        # Connect MT5 for initial data fetch for training
        if not connect_mt5(ACCOUNT_DETAILS):
            exit() # Exit if we can't even connect for initial training data

        print("\n--- Getting Historical Data for Training ---")
        # Get enough bars for comprehensive training. 10000 bars of M15 is about 100 days.
        history_df = get_historical_data(SYMBOL, TIMEFRAME, 10000) 
        if history_df is None or history_df.empty:
            print("No historical data to process for training. Exiting.")
            mt5.shutdown()
            exit()

        print("\n--- Calculating Technical Indicators for Training Data ---")
        history_df = calculate_indicators(history_df)
        if history_df.empty:
            print("Not enough data after indicator calculation for training. Exiting.")
            mt5.shutdown()
            exit()

        print("\n--- Labeling Data for Training ---")
        history_df = label_data(history_df)
        if history_df.empty:
            print("Not enough data after labeling for training. Exiting.")
            mt5.shutdown()
            exit()

        print(f"Dataset size after preprocessing: {len(history_df)} bars")
        print("Signal distribution for training:\n", history_df['signal'].value_counts())

        # Train the model and get the trained model, features, and label_mapping
        model, features, label_mapping = train_model(history_df, model_type=model_type, genetic_optimization=genetic_optimization)
        
        if model is None:
            print("Model training failed. Exiting.")
            mt5.shutdown()
            exit()

        # Save the newly trained model and metadata
        save_model(model, features, label_mapping, model_type=model_type)
    else:
        print(f"\n--- Using loaded {model_type} model. ---")
        # Ensure MT5 is connected before proceeding to live trading/backtesting
        if not connect_mt5(ACCOUNT_DETAILS):
            print("Failed to connect to MT5 despite loading model. Exiting.")
            exit()

    # --- Step 3: Backtesting (Optional, but highly recommended) ---
    # You might want to reload fresh data for backtesting if you trained on older data
    # Or, if you loaded a model, use current historical data for a recent backtest.
    print("\n--- Running Backtest ---")
    # Fetch data specifically for backtesting. Can be a different period than training.
    # E.g., last 2000 bars for backtesting
    backtest_df = get_historical_data(SYMBOL, TIMEFRAME, 2000)
    if backtest_df is None or backtest_df.empty:
        print("No historical data available for backtesting. Skipping backtest.")
    else:
        backtest_df = calculate_indicators(backtest_df)
        if backtest_df.empty:
            print("Not enough data after indicator calculation for backtesting. Skipping backtest.")
        else:
            # We don't need to label this data with future_close for backtesting
            # since we're using the model's prediction directly.
            run_backtest(backtest_df, model, features, initial_balance=10000, label_mapping=label_mapping)


    # --- Step 4 & 5: Live Trading ---
    # WARNING: Live trading can result in real financial losses.
    # Ensure you understand the risks and have thoroughly tested your bot.
    # It is highly recommended to run this on a demo account first.
    run_live_trading(model, features, SYMBOL, TIMEFRAME, check_interval_seconds=60, label_mapping=label_mapping)

    # --- Shut down MT5 connection ---
    print("\nShutting down MT5 connection.")
    mt5.shutdown()



