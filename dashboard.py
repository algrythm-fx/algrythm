# dashboard.py (Upgraded Version)

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, db

# --- Page Configuration (Best to do this once at the top) ---
st.set_page_config(
    page_title="Algrythm AI Trading Bot",
    page_icon="🤖",
    layout="wide"
)

# --- Firebase Initialization (Robust and cached) ---
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            cred_dict = dict(st.secrets.firebase_credentials)
            # Fix private key formatting
            cred_dict["private_key"] = cred_dict["private_key"].replace('\\n', '\n')
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': f"https://{cred_dict['project_id']}-default-rtdb.firebaseio.com/"
            })
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")
        st.stop()

# --- Data Loading (More reasonable cache time) ---
@st.cache_data(ttl=5)  # Cache data for 5 seconds - a much better interval
def load_data_from_firebase():
    """Loads and structures data from Firebase, providing defaults for missing keys."""
    try:
        ref = db.reference('/')
        data = ref.get()
        # If the database is empty or not yet created, return a default structure
        if data is None:
            return {}
        return data
    except Exception as e:
        st.warning(f"Could not fetch data from Firebase: {e}")
        return {} # Return empty dict on error

# --- UI Rendering Functions (for cleaner code) ---

def render_live_status_tab(data):
    """Renders all components for the Live Status tab."""
    account_info = data.get('account_info', {})
    bot_status = data.get('bot_status', {})
    open_positions = data.get('open_positions', [])
    log_messages = data.get('log_messages', [])

    st.subheader("Account Status")
    cols = st.columns(4)
    cols[0].metric("Balance", f"${account_info.get('balance', 0):,.2f}")
    cols[1].metric("Equity", f"${account_info.get('equity', 0):,.2f}")
    cols[2].metric("Live P/L", f"${account_info.get('profit', 0):,.2f}")
    cols[3].metric("Margin Level", f"{account_info.get('margin_level', 0):,.2f}%")

    st.subheader("Bot Status")
    cols = st.columns(3)
    cols[0].metric("Last Signal", bot_status.get('last_signal', 'N/A'))
    cols[1].metric("Confidence", bot_status.get('last_confidence', 'N/A'))
    cols[2].metric("M15 Trend", bot_status.get('higher_tf_trend', 'N/A').upper())

    st.subheader("Open Positions")
    if open_positions:
        df_pos = pd.DataFrame(open_positions)
        st.dataframe(df_pos[['ticket', 'time', 'type', 'volume', 'price_open', 'price_current', 'sl', 'tp', 'profit']], use_container_width=True)
    else:
        st.info("No open positions.")

    st.subheader("Live Log")
    st.code("\n".join(log_messages))

def render_trade_history_tab(data):
    """Renders the Trade History tab."""
    trade_history = data.get('trade_history', [])
    st.subheader("Completed Trade History")
    if trade_history:
        st.dataframe(pd.DataFrame(trade_history), use_container_width=True)
    else:
        st.info("No trade history has been logged yet.")

def calculate_key_metrics(trade_history, initial_balance=10000):
    """Calculate advanced performance metrics"""
    if not trade_history:
        return {
            'net_profit': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
    
    df = pd.DataFrame(trade_history)
    
    # Ensure we have required columns
    if 'profit' not in df.columns:
        if 'exit_price' in df.columns and 'entry_price' in df.columns:
            df['profit'] = df['exit_price'] - df['entry_price']
        else:
            return {
                'net_profit': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
    
    # Basic metrics
    total_trades = len(df)
    wins = len(df[df['profit'] > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    avg_win = df[df['profit'] > 0]['profit'].mean() if wins > 0 else 0
    avg_loss = abs(df[df['profit'] <= 0]['profit'].mean()) if losses > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
    net_profit = df['profit'].sum()
    
    # Advanced metrics
    df['equity'] = initial_balance + df['profit'].cumsum()
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = df['peak'] - df['equity']
    max_drawdown = df['drawdown'].max()
    
    # Calculate Sharpe Ratio (annualized)
    returns = df['profit'] / initial_balance
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
    
    return {
        'net_profit': net_profit,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def render_performance_analytics_tab(data):
    """Renders all components for the Performance Analytics tab with advanced metrics."""
    trade_history = data.get('trade_history', [])
    account_info = data.get('account_info', {})
    
    st.subheader("Performance Analytics")
    
    if trade_history:
        # Calculate initial balance from current balance and net profit
        current_balance = account_info.get('balance', 10000)
        df_history = pd.DataFrame(trade_history)
        
        # Calculate net profit from history if possible
        if 'profit' in df_history.columns:
            net_profit_from_history = df_history['profit'].sum()
        elif 'exit_price' in df_history.columns and 'entry_price' in df_history.columns:
            net_profit_from_history = (df_history['exit_price'] - df_history['entry_price']).sum()
        else:
            net_profit_from_history = 0
            
        initial_balance = current_balance - net_profit_from_history

        metrics = calculate_key_metrics(trade_history, initial_balance)

        # --- Display Key Performance Indicators (KPIs) ---
        st.subheader("Key Performance Indicators")
        cols = st.columns(4)
        cols[0].metric("Net Profit", f"${metrics['net_profit']:,.2f}", 
                      delta=f"{metrics['net_profit']/initial_balance*100:.1f}%" if initial_balance else None)
        cols[1].metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        cols[2].metric("Max Drawdown", f"${metrics['max_drawdown']:,.2f}", 
                      delta=f"-{metrics['max_drawdown']/initial_balance*100:.1f}%" if initial_balance else None)
        cols[3].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        
        cols2 = st.columns(4)
        cols2[0].metric("Total Trades", metrics['total_trades'])
        cols2[1].metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        cols2[2].metric("Avg Win", f"${metrics['avg_win']:,.2f}")
        cols2[3].metric("Avg Loss", f"${metrics['avg_loss']:,.2f}")

        # --- Display Charts ---
        df = pd.DataFrame(trade_history)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
        else:
            df['time'] = pd.to_datetime(df.index)
        
        st.subheader("Equity Curve")
        df['equity'] = initial_balance + df['profit'].cumsum() if 'profit' in df.columns else initial_balance
        st.line_chart(df.rename(columns={'time':'index'}).set_index('index')['equity'])

        st.subheader("Trade Distribution")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if 'profit' in df.columns:
                st.bar_chart(df.set_index('time')['profit'], 
                            color="#FF4B4B" if metrics['net_profit'] < 0 else "#2ca02c")
                st.caption("Profit/Loss per Trade")
            else:
                st.warning("Profit data not available for chart")
        
        with chart_col2:
            if 'profit' in df.columns:
                fig = go.Figure(go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[len(df[df['profit']>0]), len(df[df['profit']<=0])],
                    marker_colors=['#2ca02c', '#d62728'],
                    hole=.3
                ))
                fig.update_layout(title_text='Win/Loss Distribution', 
                                margin=dict(l=20, r=20, t=40, b=20), 
                                height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Profit data not available for pie chart")
    else:
        st.info("No performance data available. Waiting for the first closed trade to be logged.")
# --- Main App Execution ---

# Initialize Firebase connection
init_firebase()

st.title("🤖 Algrythm FX AI Trading Bot Dashboard")

# Load data
data = load_data_from_firebase()
st.write(f"**Last Updated:** {data.get('last_updated', 'N/A')}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Status", "📖 Trade History", "📈 Performance Analytics"])

with tab1:
    render_live_status_tab(data)
with tab2:
    render_trade_history_tab(data)
with tab3:
    render_performance_analytics_tab(data)

# --- Auto-refresh logic (a cleaner, non-blocking way) ---
# This will rerun the script from top to bottom every 10 seconds.
time.sleep(1)
st.rerun()
