import streamlit as st
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, db

# --- Page Configuration ---
st.set_page_config(page_title="Algrythm AI Trading Bot", page_icon="🤖", layout="wide")

# --- Firebase Initialization ---
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

# Initialize Firebase
init_firebase()

# --- Data Loading with Caching ---
@st.cache_data(ttl=2)  # Refresh every 5 seconds
def load_firebase_data():
    try:
        ref = db.reference('/')
        data = ref.get() or {}
        return {
            'last_updated': data.get('last_updated', 'N/A'),
            'account_info': data.get('account_info', {}),
            'open_positions': data.get('open_positions', []),
            'bot_status': data.get('bot_status', {}),
            'log_messages': data.get('log_messages', []),
            'trade_history': data.get('trade_history', [])
        }
    except Exception as e:
        st.warning(f"Data loading error: {e}")
        return {}

# --- Main Dashboard Layout ---
st.title("🤖 Algrythm FX AI Trading Bot Dashboard")

# Load data
data = load_firebase_data()
last_updated = st.empty()  # Single element for last updated

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Status", "📖 Trade History", "📈 Performance Analytics"])

# --- Tab 1: Live Status ---
with tab1:
    st.subheader("Account Status")
    cols = st.columns(4)
    cols[0].metric("Balance", f"${data['account_info'].get('balance', 0):,.2f}")
    cols[1].metric("Equity", f"${data['account_info'].get('equity', 0):,.2f}")
    cols[2].metric("Live P/L", f"${data['account_info'].get('profit', 0):,.2f}")
    cols[3].metric("Margin Level", f"{data['account_info'].get('margin_level', 0):,.2f}%")

    st.subheader("Bot Status")
    cols = st.columns(3)
    cols[0].metric("Last Signal", data['bot_status'].get('last_signal', 'N/A'))
    cols[1].metric("Confidence", data['bot_status'].get('last_confidence', 'N/A'))
    cols[2].metric("M15 Trend", data['bot_status'].get('higher_tf_trend', 'N/A').upper())

    st.subheader("Open Positions")
    if data['open_positions']:
        st.dataframe(pd.DataFrame(data['open_positions']), use_container_width=True)
    else:
        st.info("No open positions")

    st.subheader("Live Log")
    st.code("\n".join(data['log_messages']))

# --- Tab 2: Trade History ---
with tab2:
    st.subheader("Trade History")
    if data['trade_history']:
        st.dataframe(pd.DataFrame(data['trade_history']), use_container_width=True)
    else:
        st.info("No trade history available")

# --- Tab 3: Performance Analytics ---
with tab3:
    if data['trade_history']:
        df = pd.DataFrame(data['trade_history'])
        df['time'] = pd.to_datetime(df['time'])
        
        st.subheader("Performance Metrics")
        cols = st.columns(4)
        profit = df['profit'].sum()
        win_rate = (len(df[df['profit'] > 0]) / len(df)) * 100 if len(df) > 0 else 0
        cols[0].metric("Net Profit", f"${profit:,.2f}")
        cols[1].metric("Total Trades", len(df))
        cols[2].metric("Win Rate", f"{win_rate:.1f}%")
        cols[3].metric("Avg Win", f"${df[df['profit']>0]['profit'].mean():,.2f}")

        st.subheader("Equity Curve")
        df['cumulative'] = df['profit'].cumsum()
        st.line_chart(df.set_index('time')['cumulative'])

        st.subheader("Trade Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df.set_index('time')['profit'])
        with col2:
            fig = go.Figure(go.Pie(
                labels=['Wins', 'Losses'],
                values=[len(df[df['profit']>0]), len(df[df['profit']<=0])],
                marker_colors=['green', 'red']
            ))
            st.plotly_chart(fig, use_container_width=True, key="performance_pie")
    else:
        st.info("No performance data available")

# Update last updated time
last_updated.write(f"Last updated: {data['last_updated']}")

time.sleep(2)
st.experimental_rerun()
