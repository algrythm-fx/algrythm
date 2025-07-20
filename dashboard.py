# dashboard.py (Optimized Version)

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, db

# --- Constants ---
DEFAULT_BALANCE = 10000
REFRESH_INTERVAL = 10  # seconds
COLOR_WIN = "#2ca02c"
COLOR_LOSS = "#d62728"

# --- Page Configuration ---
def configure_page():
    st.set_page_config(
        page_title="Algrythm AI Trading Bot",
        page_icon="🤖",
        layout="wide"
    )

# --- Firebase Initialization ---
@st.cache_resource
def init_firebase():
    """Initialize Firebase connection with proper error handling"""
    try:
        if not firebase_admin._apps:
            cred_dict = dict(st.secrets.firebase_credentials)
            cred_dict["private_key"] = cred_dict["private_key"].replace('\\n', '\n')
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': f"https://{cred_dict['project_id']}-default-rtdb.firebaseio.com/"
            })
        return True
    except Exception as e:
        st.error(f"Firebase initialization failed: {str(e)}")
        st.stop()
        return False

# --- Data Loading ---
@st.cache_data(ttl=5)
def load_firebase_data():
    """Load data from Firebase with proper error handling"""
    try:
        ref = db.reference('/')
        data = ref.get() or {}
        return data
    except Exception as e:
        st.warning(f"Data loading error: {str(e)}")
        return {}

# --- UI Components ---
def display_account_status(account_info):
    """Render account status metrics"""
    st.subheader("Account Status")
    cols = st.columns(4)
    cols[0].metric("Balance", f"${account_info.get('balance', 0):,.2f}")
    cols[1].metric("Equity", f"${account_info.get('equity', 0):,.2f}")
    cols[2].metric("Live P/L", f"${account_info.get('profit', 0):,.2f}")
    cols[3].metric("Margin Level", f"{account_info.get('margin_level', 0):,.2f}%")

def display_bot_status(bot_status):
    """Render bot status metrics"""
    st.subheader("Bot Status")
    cols = st.columns(3)
    cols[0].metric("Last Signal", bot_status.get('last_signal', 'N/A'))
    cols[1].metric("Confidence", bot_status.get('last_confidence', 'N/A'))
    cols[2].metric("M15 Trend", bot_status.get('higher_tf_trend', 'N/A').upper())

def display_positions(positions):
    """Render open positions table"""
    st.subheader("Open Positions")
    if positions:
        df = pd.DataFrame(positions)
        cols = ['ticket', 'time', 'type', 'volume', 'price_open', 'price_current', 'sl', 'tp', 'profit']
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.info("No open positions.")

def display_logs(logs):
    """Render live logs"""
    st.subheader("Live Log")
    st.code("\n".join(logs) if logs else "No logs available")

def render_live_status_tab(data):
    """Render the Live Status tab"""
    display_account_status(data.get('account_info', {}))
    display_bot_status(data.get('bot_status', {}))
    display_positions(data.get('open_positions', []))
    display_logs(data.get('log_messages', []))

def render_trade_history_tab(data):
    """Render the Trade History tab"""
    st.subheader("Completed Trade History")
    trades = data.get('trade_history', [])
    if trades:
        st.dataframe(pd.DataFrame(trades), use_container_width=True)
    else:
        st.info("No trade history has been logged yet.")

# --- Performance Metrics ---
def calculate_metrics(df, initial_balance):
    """Calculate all performance metrics"""
    metrics = {
        'total_trades': len(df),
        'net_profit': 0,
        'profit_factor': 0,
        'max_drawdown': 0,
        'sharpe_ratio': 0,
        'win_rate': 0,
        'avg_win': 0,
        'avg_loss': 0
    }
    
    if metrics['total_trades'] == 0:
        return metrics
    
    # Calculate profit if not present
    if 'profit' not in df.columns:
        if all(col in df.columns for col in ['exit_price', 'entry_price']):
            df['profit'] = df['exit_price'] - df['entry_price']
        else:
            return metrics
    
    # Basic metrics
    wins = df[df['profit'] > 0]
    losses = df[df['profit'] <= 0]
    
    metrics.update({
        'net_profit': df['profit'].sum(),
        'win_rate': len(wins) / metrics['total_trades'] * 100,
        'avg_win': wins['profit'].mean() if len(wins) > 0 else 0,
        'avg_loss': abs(losses['profit'].mean()) if len(losses) > 0 else 0,
        'profit_factor': metrics['avg_win'] / metrics['avg_loss'] if metrics['avg_loss'] > 0 else float('inf')
    })
    
    # Advanced metrics
    df['equity'] = initial_balance + df['profit'].cumsum()
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['peak'] - df['equity'])
    metrics['max_drawdown'] = df['drawdown'].max()
    
    # Sharpe Ratio
    returns = df['profit'] / initial_balance
    if len(returns) > 1 and returns.std() > 0:
        metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
    
    return metrics

def render_performance_analytics_tab(data):
    """Render the Performance Analytics tab"""
    st.subheader("Performance Analytics")
    
    trades = data.get('trade_history', [])
    if not trades:
        st.info("No performance data available")
        return
    
    try:
        df = pd.DataFrame(trades)
        
        # Handle datetime conversion
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time']).sort_values('time')
        
        # Calculate metrics
        account_balance = data.get('account_info', {}).get('balance', DEFAULT_BALANCE)
        net_profit = df['profit'].sum() if 'profit' in df.columns else 0
        initial_balance = account_balance - net_profit
        metrics = calculate_metrics(df, initial_balance)
        
        # Display KPIs
        st.subheader("Key Performance Indicators")
        cols = st.columns(4)
        cols[0].metric("Net Profit", f"${metrics['net_profit']:,.2f}", 
                      delta=f"{metrics['net_profit']/initial_balance*100:.1f}%")
        cols[1].metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        cols[2].metric("Max Drawdown", f"${metrics['max_drawdown']:,.2f}")
        cols[3].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        
        cols = st.columns(4)
        cols[0].metric("Total Trades", metrics['total_trades'])
        cols[1].metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        cols[2].metric("Avg Win", f"${metrics['avg_win']:,.2f}")
        cols[3].metric("Avg Loss", f"${metrics['avg_loss']:,.2f}")
        
        # Charts
        st.subheader("Equity Curve")
        st.line_chart(df.set_index('time')['equity'])
        
        st.subheader("Trade Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df.set_index('time')['profit'], 
                        color=COLOR_LOSS if metrics['net_profit'] < 0 else COLOR_WIN)
        
        with col2:
            fig = go.Figure(go.Pie(
                labels=['Wins', 'Losses'],
                values=[metrics['total_trades'] - len(df[df['profit'] <= 0]), len(df[df['profit'] <= 0])],
                marker_colors=[COLOR_WIN, COLOR_LOSS],
                hole=0.3
            ))
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error rendering analytics: {str(e)}")
        if st.checkbox("Show debug info"):
            st.write("Raw trade data:", trades)

# --- Main App ---
def main():
    configure_page()
    init_firebase()
    
    st.title("🤖 Algrythm FX AI Trading Bot Dashboard")
    data = load_firebase_data()
    st.write(f"**Last Updated:** {data.get('last_updated', 'N/A')}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Live Status", "📖 Trade History", "📈 Performance Analytics"])
    with tab1:
        render_live_status_tab(data)
    with tab2:
        render_trade_history_tab(data)
    with tab3:
        render_performance_analytics_tab(data)
    
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

if __name__ == "__main__":
    main()
