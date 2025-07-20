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

def calculate_key_metrics(trade_history):
    """Calculate advanced performance metrics"""
    if not trade_history:
        return {}
    
    df = pd.DataFrame(trade_history)
    df['profit'] = df['exit_price'] - df['entry_price']  # Simplified for example
    
    # Basic metrics
    win_rate = len(df[df['profit'] > 0]) / len(df)
    avg_win = df[df['profit'] > 0]['profit'].mean()
    avg_loss = abs(df[df['profit'] <= 0]['profit'].mean())
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # Advanced metrics
    df['equity'] = df['profit'].cumsum()
    max_drawdown = (df['equity'].cummax() - df['equity']).max()
    sharpe_ratio = df['profit'].mean() / df['profit'].std() * np.sqrt(252)  # Annualized
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def render_performance_analytics_tab(data):
    """Renders all components for the Performance Analytics tab with advanced metrics."""
    trade_history = data.get('trade_history', [])
    account_info = data.get('account_info', {})
    
    st.subheader("Performance Analytics")
    
    if trade_history:
        # --- 1. Calculate the advanced metrics ---
        # Use the current balance to infer the initial balance for a more accurate equity curve
        current_balance = account_info.get('balance', 10000)
        net_profit_from_history = pd.DataFrame(trade_history)['profit'].sum()
        initial_balance = current_balance - net_profit_from_history
        
        metrics = calculate_key_metrics(trade_history, initial_balance)

        # --- 2. Display Key Performance Indicators (KPIs) ---
        st.subheader("Key Performance Indicators")
        cols = st.columns(4)
        cols[0].metric("Net Profit", f"${metrics.get('net_profit', 0):,.2f}")
        cols[1].metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        cols[2].metric("Max Drawdown", f"${metrics.get('max_drawdown', 0):,.2f}")
        cols[3].metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        cols2 = st.columns(4)
        cols2[0].metric("Total Trades", f"{metrics.get('total_trades', 0)}")
        cols2[1].metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
        cols2[2].metric("Average Win", f"${metrics.get('avg_win', 0):,.2f}")
        cols2[3].metric("Average Loss", f"${metrics.get('avg_loss', 0):,.2f}")

        # --- 3. Display Charts ---
        df = pd.DataFrame(trade_history)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')

        st.subheader("Equity Curve")
        df['equity'] = initial_balance + df['profit'].cumsum()
        st.line_chart(df.rename(columns={'time':'index'}).set_index('index')['equity'])

        st.subheader("Trade Distribution")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.bar_chart(df.set_index('time')['profit'], color="#FF4B4B" if metrics.get('net_profit', 0) < 0 else "#2ca02c")
            st.caption("Profit/Loss per Trade")
        with chart_col2:
            fig = go.Figure(go.Pie(
                labels=['Wins', 'Losses'],
                values=[len(df[df['profit']>0]), len(df[df['profit']<=0])],
                marker_colors=['#2ca02c', '#d62728'],
                hole=.3
            ))
            fig.update_layout(title_text='Win/Loss Distribution', margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
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
