# dashboard.py (Final, Corrected, and Optimized Version)

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, db

# --- Constants ---
REFRESH_INTERVAL = 10  # seconds
COLOR_WIN = "#2ca02c"
COLOR_LOSS = "#d62728"

# --- Page Configuration ---
st.set_page_config(
    page_title="Algrythm AI Trading Bot",
    page_icon="🤖",
    layout="wide"
)

# --- Firebase Initialization (Robust and cached) ---
@st.cache_resource
def init_firebase():
    """Initializes the Firebase connection using Streamlit secrets."""
    try:
        if not firebase_admin._apps:
            cred_dict = st.secrets.firebase_credentials
            cred_dict["private_key"] = cred_dict["private_key"].replace('\\n', '\n')
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': f"https://{cred_dict['project_id']}-default-rtdb.firebaseio.com/"
            })
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}. Check your Streamlit Secrets.")
        st.stop()

# --- Data Loading (More reasonable cache time) ---
@st.cache_data(ttl=5)
def load_data_from_firebase():
    """Loads and structures data from Firebase, providing defaults for missing keys."""
    try:
        ref = db.reference('/')
        data = ref.get()
        return data or {} # Return empty dict if data is None
    except Exception as e:
        st.warning(f"Could not fetch data from Firebase: {e}")
        return {}

# --- CORRECTED & UNIFIED Performance Metrics Calculation ---
def calculate_key_metrics(trade_history, initial_balance):
    """
    Calculates a comprehensive set of accurate performance metrics from trade history.
    """
    if not trade_history:
        return {} # Return empty dict if no history

    df = pd.DataFrame(trade_history)
    
    # --- CRITICAL FIX: Ensure 'profit' column is numeric. DO NOT recalculate it. ---
    df['profit'] = pd.to_numeric(df['profit'])
    
    # Separate winning and losing trades
    wins = df[df['profit'] > 0]
    losses = df[df['profit'] <= 0]
    total_trades = len(df)
    
    if total_trades == 0:
        return {}

    # --- P&L Metrics (Corrected) ---
    net_profit = df['profit'].sum()
    gross_profit = wins['profit'].sum()
    gross_loss = abs(losses['profit'].sum())
    
    # --- CRITICAL FIX: Correct Profit Factor calculation ---
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    avg_win = wins['profit'].mean() if not wins.empty else 0
    avg_loss = abs(losses['profit'].mean()) if not losses.empty else 0
    win_rate = (len(wins) / total_trades * 100)
    
    # --- Risk & Return Metrics (Corrected) ---
    # --- CRITICAL FIX: Correct Equity Curve and Drawdown calculation ---
    df['equity'] = initial_balance + df['profit'].cumsum()
    peak = df['equity'].cummax()
    drawdown = peak - df['equity']
    max_drawdown = drawdown.max()
    
    # Sharpe Ratio (simplified)
    daily_returns = df['profit'] / (df['equity'] - df['profit'])
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    return {
        'net_profit': net_profit, 'win_rate': win_rate, 'profit_factor': profit_factor,
        'max_drawdown': max_drawdown, 'sharpe_ratio': sharpe_ratio,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'total_trades': total_trades,
        'equity_df': df[['time', 'equity']] # Return the df for the chart
    }

# --- UI Rendering Functions ---

def render_live_status_tab(data):
     """Render the Live Status tab"""
    display_account_status(data.get('account_info', {}))
    display_bot_status(data.get('bot_status', {}))
    display_positions(data.get('open_positions', []))
    display_logs(data.get('log_messages', []))
    pass 

def render_trade_history_tab(data):
    """Render the Trade History tab"""
    st.subheader("Completed Trade History")
    trades = data.get('trade_history', [])
    if trades:
        st.dataframe(pd.DataFrame(trades), use_container_width=True)
    else:
        st.info("No trade history has been logged yet.")
    pass

def render_performance_analytics_tab(data):
    """Renders the Performance Analytics tab using the corrected metrics."""
    st.subheader("Performance Analytics")
    trade_history = data.get('trade_history', [])
    account_info = data.get('account_info', {})

    if not trade_history:
        st.info("No performance data available. Waiting for the first closed trade.")
        return

    # --- Simplified and Robust Metric Calculation ---
    current_balance = account_info.get('balance', 10000)
    net_profit_from_history = pd.to_numeric(pd.DataFrame(trade_history)['profit']).sum()
    initial_balance = current_balance - net_profit_from_history
    
    metrics = calculate_key_metrics(trade_history, initial_balance)

    if not metrics:
        st.warning("Could not calculate performance metrics from the available trade history.")
        return

    # --- Display KPIs ---
    st.subheader("Key Performance Indicators")
    cols = st.columns(4)
    pf_value = "∞" if metrics.get('profit_factor') == float('inf') else f"{metrics.get('profit_factor', 0):.2f}"
    cols[0].metric("Net Profit", f"${metrics.get('net_profit', 0):,.2f}", 
									delta=f"{metrics['net_profit']/initial_balance*100:.1f}%" if initial_balance else None)
    cols[1].metric("Profit Factor", pf_value)
    cols[2].metric("Max Drawdown", f"${metrics.get('max_drawdown', 0):,.2f}")
    cols[3].metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")

    cols2 = st.columns(4)
    cols2[0].metric("Total Trades", f"{metrics.get('total_trades', 0)}")
    cols2[1].metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
    cols2[2].metric("Average Win", f"${metrics.get('avg_win', 0):,.2f}")
    cols2[3].metric("Average Loss", f"${metrics.get('avg_loss', 0):,.2f}")

    # --- Display Charts ---
    equity_df = metrics.get('equity_df')
    if equity_df is not None and not equity_df.empty:
        st.subheader("Equity Curve")
        equity_df['time'] = pd.to_datetime(equity_df['time'])
        st.line_chart(equity_df.rename(columns={'time':'index'}).set_index('index')['equity'])

    st.subheader("Trade Distribution")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        # We need the full df for this chart
        df_full = pd.DataFrame(trade_history)
        df_full['time'] = pd.to_datetime(df_full['time'])
        st.bar_chart(df_full.set_index('time')['profit'], color=COLOR_WIN if metrics.get('net_profit',0) >= 0 else COLOR_LOSS)
        st.caption("Profit/Loss per Trade")
    with chart_col2:
        wins_count = int(metrics.get('total_trades', 0) * (metrics.get('win_rate', 0) / 100))
        losses_count = metrics.get('total_trades', 0) - wins_count
        fig = go.Figure(go.Pie(
            labels=['Wins', 'Losses'],
            values=[wins_count, losses_count],
            marker_colors=[COLOR_WIN, COLOR_LOSS],
            hole=.3
        ))
        fig.update_layout(title_text='Win/Loss Distribution', margin=dict(l=20, r=20, t=40, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)


# --- Main App ---
def main():
    # (Your main function can stay mostly the same, but it's cleaner to use the render functions)
    configure_page()
    init_firebase()
    
    st.title("🤖 Algrythm FX AI Trading Bot Dashboard")
    data = load_data_from_firebase()
    st.write(f"**Last Updated:** {data.get('last_updated', 'N/A')}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Live Status", "📖 Trade History", "📈 Performance Analytics"])
    with tab1:
        # Re-add the original render functions for cleanliness
        display_account_status(data.get('account_info', {}))
        display_bot_status(data.get('bot_status', {}))
        display_positions(data.get('open_positions', []))
        display_logs(data.get('log_messages', []))
    with tab2:
        render_trade_history_tab(data)
    with tab3:
        render_performance_analytics_tab(data)
    
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

if __name__ == "__main__":
    main()
