import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import time

from strategies_9022 import strategies  # dict {strategy_name: wallet_address}
from pairs import pairs
from hyperliquid_data import PerpHyperliquid

# ---------------------------
# Helper: Fetch data for a single wallet with retry and proper closing
# ---------------------------
async def fetch_data_for_wallet_with_retry(public_address: str, trade_count, retries=3, delay=2):
    hl = PerpHyperliquid(public_api=public_address)
    try:
        for attempt in range(retries):
            try:
                balance = await hl.get_balance()
                open_orders_df = await hl.get_open_orders(pairs)
                ext_pairs = [hl.pair_to_ext_pair(coin) for coin in pairs]
                open_positions_df = await hl.get_open_positions(ext_pairs)
                trade_history_df = await hl.get_trade_history(pairs, count=trade_count)
                break  # Exit the retry loop if successful.
            except Exception as e:
                if "429" in str(e):
                    if attempt < retries - 1:
                        st.warning(f"Rate limit hit. Retrying in {delay} seconds... (attempt {attempt+1}/{retries})")
                        await asyncio.sleep(delay)
                        delay *= 2  # exponential backoff
                        continue
                    else:
                        st.error("Max retries reached. Could not fetch data due to rate limit.")
                        raise
                else:
                    raise
    finally:
        await hl.close()
    return balance, open_orders_df, open_positions_df, trade_history_df

# ---------------------------
# Utility: Filter trades by date
# ---------------------------
def filter_trades_by_date(df: pd.DataFrame, since_dt: datetime) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%y - %H:%M:%S', errors='coerce')
    return df[df["Date"] >= since_dt]

# ---------------------------
# Utility: Compute metrics (ignoring 'neutral' for win rate)
# ---------------------------
def compute_metrics(df: pd.DataFrame):
    if df.empty:
        return 0, 0, 0, 0
    df["Closed PnL"] = df["Closed PnL"].astype(float)
    df["Fee"] = df["Fee"].astype(float)
    total_pnl = df["Closed PnL"].sum()
    total_fees = df["Fee"].sum()
    net_pnl = total_pnl - total_fees

    win_loss_df = df[df["Win/Loss"].str.lower().isin(["win", "loss"])]
    if len(win_loss_df) > 0:
        wins = (win_loss_df["Win/Loss"].str.lower() == "win").sum()
        win_rate = wins / len(win_loss_df) * 100
    else:
        win_rate = 0

    return total_pnl, total_fees, net_pnl, win_rate

# ---------------------------
# Throttling: Use an asyncio semaphore (limit 3 concurrent requests)
# ---------------------------
semaphore = asyncio.Semaphore(3)

async def fetch_with_semaphore(addr: str, trade_count):
    async with semaphore:
        return await fetch_data_for_wallet_with_retry(addr, trade_count)

# ---------------------------
# Caching Aggregated Data
# ---------------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_aggregated_data(trade_count):
    async def fetch_all_strategies():
        tasks = [fetch_with_semaphore(addr, trade_count) for addr in strategies.values()]
        return await asyncio.gather(*tasks)
    return asyncio.run(fetch_all_strategies())

# ---------------------------
# Main App
# ---------------------------
page = st.sidebar.selectbox("Select Page", ["Single Strategy", "Aggregated Data"])

if page == "Single Strategy":
    strategy_names = list(strategies.keys())
    selected_strategy = st.sidebar.selectbox("Select Strategy", strategy_names)
    wallet_address = strategies[selected_strategy]

    st.title(f"{selected_strategy} Dashboard")
    trade_since_option = st.selectbox("Show trades since:", ["Today", "Week", "Month", "All"])
    if trade_since_option == "Today":
        trade_count = 50
    elif trade_since_option == "Week":
        trade_count = 250
    elif trade_since_option == "Month":
        trade_count = 1000
    else:
        trade_count = "All"

    try:
        balance_data, open_orders_df, open_positions_df, trade_history_df = asyncio.run(
            fetch_data_for_wallet_with_retry(wallet_address, trade_count)
        )
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if trade_since_option != "All":
        now = datetime.now()
        if trade_since_option == "Today":
            since_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif trade_since_option == "Week":
            since_dt = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        elif trade_since_option == "Month":
            since_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        trade_history_df = filter_trades_by_date(trade_history_df, since_dt)

    st.subheader("Wallet Metrics")
    st.metric("Balance", f"${balance_data.total:,.2f}")

    total_pnl, total_fees, net_pnl, win_rate = compute_metrics(trade_history_df)
    st.subheader("Aggregated Trade Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total P&L", f"${total_pnl:,.2f}")
    col2.metric("Total Fees", f"${total_fees:,.2f}")
    col3.metric("Net P&L", f"${net_pnl:,.2f}")
    col4.metric("Win Rate", f"{win_rate:.1f}%")

    st.subheader("Open Orders")
    st.table(open_orders_df)

    st.subheader("Open Positions")
    if open_positions_df.empty:
        st.write("No Open Positions")
    else:
        st.table(open_positions_df)

    st.subheader("Trade History")
    st.table(trade_history_df)

elif page == "Aggregated Data":
    st.title("Aggregated Data Across All Strategies")
    trade_since_option = st.selectbox("Show trades since:", ["Today", "Week", "Month", "All"])
    if trade_since_option == "Today":
        trade_count = 50
    elif trade_since_option == "Week":
        trade_count = 250
    elif trade_since_option == "Month":
        trade_count = 1000
    else:
        trade_count = "All"

    try:
        results = get_aggregated_data(trade_count)
    except Exception as e:
        st.error(f"Error fetching aggregated data: {e}")
        st.stop()

    total_balance = 0
    all_open_orders = []
    all_open_positions = []
    all_trade_history = []

    for (balance, oo, op, th) in results:
        total_balance += balance.total
        all_open_orders.append(oo)
        all_open_positions.append(op)
        all_trade_history.append(th)

    aggregated_open_orders = pd.concat(all_open_orders, ignore_index=True)
    aggregated_open_positions = pd.concat(all_open_positions, ignore_index=True)
    aggregated_trade_history = pd.concat(all_trade_history, ignore_index=True)

    if trade_since_option != "All":
        now = datetime.now()
        if trade_since_option == "Today":
            since_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif trade_since_option == "Week":
            since_dt = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        elif trade_since_option == "Month":
            since_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        aggregated_trade_history = filter_trades_by_date(aggregated_trade_history, since_dt)

    aggregated_trade_history.sort_values(by="Date", ascending=False, inplace=True)
    total_pnl, total_fees, net_pnl, win_rate = compute_metrics(aggregated_trade_history)

    st.subheader("Aggregated Metrics")
    st.metric("Total Balance", f"${total_balance:,.2f}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total P&L", f"${total_pnl:,.2f}")
    col2.metric("Total Fees", f"${total_fees:,.2f}")
    col3.metric("Net P&L", f"${net_pnl:,.2f}")
    col4.metric("Win Rate", f"{win_rate:.1f}%")

    st.subheader("Open Orders")
    st.table(aggregated_open_orders)

    st.subheader("Open Positions")
    if aggregated_open_positions.empty:
        st.write("No Open Positions")
    else:
        st.table(aggregated_open_positions)

    st.subheader("Trade History")
    st.table(aggregated_trade_history)
