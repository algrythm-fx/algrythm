import streamlit as st
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, db

# --- Page Configuration ---
st.set_page_config(page_title="Algrythm AI Trading Bot", page_icon="🤖", layout="wide")

# --- Firebase Initialization (using Streamlit Secrets) ---
def init_firebase():
    try:
        # Convert the secrets dict to a credentials object
        cred_dict = dict(st.secrets.firebase_credentials)
        cred = credentials.Certificate(cred_dict)

        #conversion
        private_key_with_escapes = cred_dict.get("private_key")
        correct_private_key = private_key_with_escapes.replace('\\n', '\n')
        cred_dict["private_key"] = correct_private_key
        
        # Check if the app is already initialized to avoid errors on rerun
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': f"https://{cred_dict['project_id']}-default-rtdb.firebaseio.com/"
            })
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}. Please check your Streamlit Secrets.")
        st.stop()

# --- Helper Function to Load Data from Firebase ---
@st.cache_data(ttl=5) # Cache data for 5 seconds to avoid hitting Firebase too often
def load_data_from_firebase():
    """Loads the dashboard data from the Firebase Realtime Database."""
    try:
        ref = db.reference('/')
        data = ref.get()
        if data is None: # If database is empty
            return {
                'last_updated': 'N/A', 'account_info': {}, 'open_positions': [],
                'bot_status': {}, 'log_messages': [], 'trade_history': []
            }
        return data
    except Exception as e:
        st.warning(f"Could not fetch data from Firebase: {e}")
        return {} # Return empty dict on error

# Initialize Firebase connection
init_firebase()

# --- The rest of the UI code is the same as before ---
st.title("🤖 Algrythm FX AI Trading Bot Dashboard")
last_updated_placeholder = st.empty()
tab1, tab2, tab3 = st.tabs(["📊 Live Status", "📖 Trade History", "📈 Performance Analytics"])
# ... (The entire UI layout code from the previous step goes here)
# ... I am omitting it for brevity, but you should paste your full UI code below this line.

# For completeness, here is the UI layout part again.
# --- Tab 1: Live Status Content ---
with tab1:
    st.subheader("Account Status")
    acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
    balance_placeholder = acc_col1.empty()
    equity_placeholder = acc_col2.empty()
    profit_placeholder = acc_col3.empty()
    margin_level_placeholder = acc_col4.empty()

    st.subheader("Bot Status")
    bot_col1, bot_col2, bot_col3 = st.columns(3)
    signal_placeholder = bot_col1.empty()
    confidence_placeholder = bot_col2.empty()
    trend_placeholder = bot_col3.empty()

    st.subheader("Open Positions")
    positions_placeholder = st.empty()

    st.subheader("Live Log")
    log_placeholder = st.empty()

# --- Tab 2: Trade History Content ---
with tab2:
    st.subheader("Completed Trade History")
    history_placeholder = st.empty()

# --- Tab 3: Performance Analytics Content ---
with tab3:
    st.subheader("Overall Performance")
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    net_profit_placeholder = perf_col1.empty()
    total_trades_placeholder = perf_col2.empty()
    win_rate_placeholder = perf_col3.empty()
    profit_factor_placeholder = perf_col4.empty()

    st.subheader("Equity Curve")
    equity_curve_placeholder = st.empty()

    st.subheader("Performance Breakdown")
    col1, col2 = st.columns(2)
    pnl_chart_placeholder = col1.empty()
    win_loss_pie_placeholder = col2.empty()


# --- Main Application Loop ---
while True:
    data = load_data_from_firebase() # Use the new function
    
    # Extract data with .get() to avoid errors if a key is missing
    acc_info = data.get('account_info', {})
    bot_status = data.get('bot_status', {})
    open_positions = data.get('open_positions', [])
    log_messages = data.get('log_messages', [])
    trade_history = data.get('trade_history', [])

    # Update placeholders in Tab 1
    last_updated_placeholder.write(f"**Last Updated:** {data.get('last_updated', 'N/A')}")
    # ... (the rest of the UI update logic is exactly the same)
    # ... I am omitting it for brevity. Paste your full update logic here.
    
    # For completeness, here is the update logic part again
    balance_placeholder.metric("Balance", f"${acc_info.get('balance', 0):,.2f}")
    equity_placeholder.metric("Equity", f"${acc_info.get('equity', 0):,.2f}")
    profit_placeholder.metric("Live P/L", f"${acc_info.get('profit', 0):,.2f}")
    margin_level_placeholder.metric("Margin Level", f"{acc_info.get('margin_level', 0):,.2f}%")

    signal_placeholder.metric("Last Signal", bot_status.get('last_signal', 'N/A'))
    confidence_placeholder.metric("Signal Confidence", bot_status.get('last_confidence', 'N/A'))
    trend_placeholder.metric("H1 Trend", bot_status.get('higher_tf_trend', 'N/A').upper())

    with positions_placeholder.container():
        if open_positions:
            df_positions = pd.DataFrame(open_positions)
            st.dataframe(df_positions[['ticket', 'time', 'type', 'volume', 'price_open', 'price_current', 'sl', 'tp', 'profit']], use_container_width=True)
        else:
            st.info("No open positions.")
    
    with log_placeholder.container():
        st.code("\n".join(log_messages), language=None)

    if trade_history:
        df_history = pd.DataFrame(trade_history)
        df_history['time'] = pd.to_datetime(df_history['time'])

        with history_placeholder.container():
            st.dataframe(df_history, use_container_width=True)

        total_profit = df_history['profit'].sum()
        wins = df_history[df_history['profit'] > 0]
        losses = df_history[df_history['profit'] <= 0]
        total_trades = len(df_history)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        profit_from_wins = wins['profit'].sum()
        loss_from_losses = abs(losses['profit'].sum())
        profit_factor = (profit_from_wins / loss_from_losses) if loss_from_losses > 0 else 999

        net_profit_placeholder.metric("Net Profit/Loss", f"${total_profit:,.2f}")
        total_trades_placeholder.metric("Total Trades", total_trades)
        win_rate_placeholder.metric("Win Rate", f"{win_rate:.2f}%")
        profit_factor_placeholder.metric("Profit Factor", f"{profit_factor:.2f}")

        initial_balance = acc_info.get('balance', 10000) - total_profit
        df_history['equity'] = initial_balance + df_history['profit'].cumsum()
        with equity_curve_placeholder.container():
            st.line_chart(df_history.rename(columns={'time':'index'}).set_index('index')['equity'])

        with pnl_chart_placeholder.container():
            st.bar_chart(df_history['profit'], use_container_width=True)
            st.caption("Profit/Loss per Trade")

        with win_loss_pie_placeholder.container():
            fig = go.Figure(data=[go.Pie(labels=['Wins', 'Losses'], values=[len(wins), len(losses)],
                                         marker_colors=['#2ca02c', '#d62728'])])
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        with history_placeholder.container(): st.info("No completed trades have been logged yet.")
        with equity_curve_placeholder.container(): st.info("Awaiting trade history to plot equity curve.")
        with pnl_chart_placeholder.container(): st.info("Awaiting trade history.")
        with win_loss_pie_placeholder.container(): st.info("Awaiting trade history.")

    # Refresh interval
    time.sleep(5)
