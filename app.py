import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import os
import time
import numpy as np
import altair as alt

from strategies_9022 import strategies as strategies_9022  # dict {strategy_name: wallet_address}
from strategies_C358 import strategies as strategies_C358  # dict {strategy_name: wallet_address}
from strategies_EE49 import strategies as strategies_EE49
from pairs import pairs
from hyperliquid_data import PerpHyperliquid

# ---------------------------------------------------------------------
# Streamlit Page Config and Custom CSS
# ---------------------------------------------------------------------
st.set_page_config(page_title="Crypto Dashboard", layout="wide", initial_sidebar_state="expanded")

def style_metrics():
    """
    Injects a bit of CSS into the Streamlit app to style the page background
    and the metric elements (cards).
    """
    st.markdown(
        """
        <style>
        /* Dark background for the entire app */
        .stApp {
            background-color: #262730;
            color: #f0f0f0;
        }

        /* Style the metric container cards */
        [data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(250, 250, 250, 0.1);
            padding: 1.5% 1.5% 1.5% 1.5%;
            border-radius: 0.5rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem;
            color: #ccc;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.3rem;
            color: #fff;
        }

        table {
            color: #f8f8f8;
        }
        th {
            color: #e2e2e2;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

style_metrics()  # Apply the custom styling

# ---------------------------------------------------------------------
# Chart Builders
# ---------------------------------------------------------------------
def build_pie_chart(long_value, short_value):
    """
    Builds a small Altair pie chart comparing two values:
    "Long" (green) and "Short" (red).
    No legend, no chart title, transparent background.
    """
    df = pd.DataFrame({
        "Position": ["Long", "Short"],
        "Value": [long_value, short_value]
    })

    chart = (
        alt.Chart(df)
        .mark_arc(outerRadius=50)
        .encode(
            theta="Value",
            color=alt.Color(
                "Position",
                scale=alt.Scale(range=["#4c78a8", "#ccc"]),
                legend=None
            ),
            tooltip=["Position", "Value"]
        )
        .properties(
            width=120,
            height=120,
            background="transparent"
        )
        .configure_view(stroke=None)
    )
    return chart

def build_bar_chart(long_value, short_value):
    """
    Builds a small Altair bar chart comparing two values:
    "Long" (green) and "Short" (red).
    No legend, no chart title, transparent background.
    """
    df = pd.DataFrame({
        "Position": ["Long", "Short"],
        "Value": [long_value, short_value]
    })

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Position:N", sort=None, axis=None),
            y=alt.Y("Value:Q"),
            color=alt.Color(
                "Position:N",
                scale=alt.Scale(range=["#4c78a8", "#ccc"]),
                legend=None
            ),
            tooltip=["Position", "Value"]
        )
        .properties(
            width=120,
            height=120,
            background="transparent"
        )
        .configure_view(stroke=None)
    )
    return chart

def build_cumulative_chart(daily_pnl_df, title="Cumulative Net P&L"):
    """
    Builds an Altair line chart with a dark background for daily cumulative PnL.
    """
    chart_data = daily_pnl_df.reset_index().rename(columns={"Day": "date"})
    chart = (
        alt.Chart(chart_data)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(labelColor="#ccc", tickColor="#ccc")),
            y=alt.Y("Cumulative PnL:Q", title="Cumulative PnL", axis=alt.Axis(labelColor="#ccc", tickColor="#ccc")),
        )
        .properties(
            width="container",
            height=400,
            background="transparent",
            title=title,
        )
        .configure_axis(
            gridColor="#444",
            domainColor="#777",
        )
        .configure_title(
            color="#ccc",
            fontSize=16
        )
    )
    return chart

CSV_FOLDER = "."  # Change if your CSV files are stored in a subfolder
CSV_COLUMNS = ["Trade ID", "Date", "Coin", "Direction", "Price", "Size", "Notional", "Fee", "Closed PnL", "Win/Loss"]

# ---------------------------------------------------------------------
# CSV Load / Save
# ---------------------------------------------------------------------
def load_trade_csv(strategy_name: str) -> pd.DataFrame:
    csv_filename = os.path.join(CSV_FOLDER, f"trade_history_{strategy_name}.csv")
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%y - %H:%M:%S', errors='coerce')
            df = df.sort_values(by="Date", ascending=False)
        return df
    else:
        return pd.DataFrame(columns=CSV_COLUMNS)

def save_trade_csv(strategy_name: str, df: pd.DataFrame):
    csv_filename = os.path.join(CSV_FOLDER, f"trade_history_{strategy_name}.csv")
    if not df.empty:
        df = df.sort_values(by="Date", ascending=False)
        df["Date"] = df["Date"].dt.strftime('%m/%d/%y - %H:%M:%S')
    df.to_csv(csv_filename, index=False)

# ---------------------------------------------------------------------
# Fetching Data from API
# ---------------------------------------------------------------------
async def fetch_api_trades(wallet_address: str, trade_count="All") -> pd.DataFrame:
    hl = PerpHyperliquid(public_api=wallet_address)
    try:
        trade_history_df = await hl.get_trade_history(pairs, count=trade_count)
    finally:
        await hl.close()
    if not trade_history_df.empty:
        trade_history_df["Date"] = pd.to_datetime(trade_history_df["Date"], format='%m/%d/%y - %H:%M:%S', errors='coerce')
    return trade_history_df

async def fetch_balance(wallet_address: str):
    hl = PerpHyperliquid(public_api=wallet_address)
    try:
        balance = await hl.get_balance()
    finally:
        await hl.close()
    return balance

# ---------------------------------------------------------------------
# Update the CSV by fetching only new trades
# ---------------------------------------------------------------------
def update_trade_csv(strategy_name: str, wallet_address: str, trade_count="All") -> pd.DataFrame:
    local_df = load_trade_csv(strategy_name)
    last_timestamp = None
    last_trade_id = None

    if not local_df.empty:
        last_row = local_df.iloc[0]  # most recent trade
        last_timestamp = last_row["Date"]
        last_trade_id = str(last_row["Trade ID"])

    # Fetch new trades from API
    api_df = asyncio.run(fetch_api_trades(wallet_address, trade_count))

    if last_timestamp:
        new_trades = api_df[api_df["Date"] > last_timestamp].copy()
        new_trades = new_trades[new_trades["Trade ID"].astype(str) != last_trade_id]
    else:
        new_trades = api_df.copy()

    # Avoid concatenation warnings by checking if one or both DataFrames are empty
    if new_trades.empty and local_df.empty:
        updated_df = pd.DataFrame(columns=CSV_COLUMNS)
    elif new_trades.empty:
        print(f"No new trades found for strategy: {strategy_name}")
        updated_df = local_df.copy()
    elif local_df.empty:
        updated_df = new_trades.copy()
    else:
        updated_df = pd.concat([new_trades, local_df], ignore_index=True)
    
    updated_df["Date"] = pd.to_datetime(updated_df["Date"], format='%m/%d/%y - %H:%M:%S', errors='coerce')
    updated_df = updated_df.sort_values(by="Date", ascending=False)
    save_trade_csv(strategy_name, updated_df)

    return updated_df

# ---------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame):
    df = df.copy()
    if df.empty:
        # Return 21 metrics (all zeros/empty) when the dataframe is empty.
        return (0, 0, 0, 0, 0, 0, 0, "0.00% / 0.00%", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    # Ensure numeric types
    df["Closed PnL"] = df["Closed PnL"].astype(float)
    df["Fee"] = df["Fee"].astype(float)
    
    # Basic Metrics
    total_pnl = df["Closed PnL"].sum()
    total_fees = df["Fee"].sum()
    net_pnl = total_pnl - total_fees
    volume = df["Notional"].sum()
    
    # Overall win rate (only consider trades marked as "win" or "loss")
    win_loss_df = df[df["Win/Loss"].str.lower().isin(["win", "loss"])]
    if not win_loss_df.empty:
        wins = (win_loss_df["Win/Loss"].str.lower() == "win").sum()
        overall_win_rate = (wins / len(win_loss_df)) * 100
    else:
        overall_win_rate = 0

    # Max Gain and Max Loss (all trades)
    max_gain = df["Closed PnL"].max()
    max_loss = df["Closed PnL"].min()
    
    # Average Return per Trade (Win/Loss) – using return = Closed PnL / Notional
    win_df = df[df["Win/Loss"].str.lower() == "win"].copy()
    loss_df = df[df["Win/Loss"].str.lower() == "loss"].copy()
    if not win_df.empty:
        win_df["Return"] = win_df["Closed PnL"] / win_df["Notional"] * 100
        avg_return_win = win_df["Return"].mean()
    else:
        avg_return_win = 0
    if not loss_df.empty:
        loss_df["Return"] = loss_df["Closed PnL"] / loss_df["Notional"] * 100
        avg_return_loss = loss_df["Return"].mean()
    else:
        avg_return_loss = 0
    avg_return_win_loss = f"{avg_return_win:.2f}% / {avg_return_loss:.2f}%"
    
    # Longs vs. Shorts – use .str.contains (to catch "Open Long", "Close Long", etc.)
    longs_df = df[df["Direction"].str.lower().str.contains("long")]
    shorts_df = df[df["Direction"].str.lower().str.contains("short")]
    longs_count = len(longs_df)
    shorts_count = len(shorts_df)
    total_trades = len(df)
    long_pct = (longs_count / total_trades) * 100 if total_trades > 0 else 0
    short_pct = (shorts_count / total_trades) * 100 if total_trades > 0 else 0
    
    # Long win rate
    long_wl = longs_df[longs_df["Win/Loss"].str.lower().isin(["win", "loss"])]
    if not long_wl.empty:
        long_wins = (long_wl["Win/Loss"].str.lower() == "win").sum()
        long_win_rate = (long_wins / len(long_wl)) * 100
    else:
        long_win_rate = 0
    # Max Gain / Loss for Longs
    max_gain_longs = longs_df["Closed PnL"].max() if not longs_df.empty else 0
    max_loss_longs = longs_df["Closed PnL"].min() if not longs_df.empty else 0
    # Average gain for winning long trades
    winning_longs = longs_df[longs_df["Win/Loss"].str.lower() == "win"]
    avg_gain_longs = winning_longs["Closed PnL"].mean() if not winning_longs.empty else 0
    
    # Short win rate
    short_wl = shorts_df[shorts_df["Win/Loss"].str.lower().isin(["win", "loss"])]
    if not short_wl.empty:
        short_wins = (short_wl["Win/Loss"].str.lower() == "win").sum()
        short_win_rate = (short_wins / len(short_wl)) * 100
    else:
        short_win_rate = 0
    # Max Gain / Loss for Shorts
    max_gain_shorts = shorts_df["Closed PnL"].max() if not shorts_df.empty else 0
    max_loss_shorts = shorts_df["Closed PnL"].min() if not shorts_df.empty else 0
    # Average gain for winning short trades
    winning_shorts = shorts_df[shorts_df["Win/Loss"].str.lower() == "win"]
    avg_gain_shorts = winning_shorts["Closed PnL"].mean() if not winning_shorts.empty else 0

    # Sharpe Ratio – using per-trade return = Closed PnL / Notional
    df["Return"] = df["Closed PnL"] / df["Notional"]
    mean_return = df["Return"].mean()
    std_return = df["Return"].std()
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0

    return (
        total_pnl,         # 1
        total_fees,        # 2
        net_pnl,           # 3
        overall_win_rate,  # 4
        volume,            # 5
        max_gain,          # 6
        max_loss,          # 7
        avg_return_win_loss,  # 8
        longs_count,       # 9
        long_pct,          # 10
        long_win_rate,     # 11
        max_gain_longs,    # 12
        max_loss_longs,    # 13
        avg_gain_longs,    # 14
        shorts_count,      # 15
        short_pct,         # 16
        short_win_rate,    # 17
        max_gain_shorts,   # 18
        max_loss_shorts,   # 19
        avg_gain_shorts,   # 20
        sharpe_ratio       # 21
    )

def filter_trades_by_date(df: pd.DataFrame, since_dt: datetime) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df[df["Date"] >= since_dt]

# ---------------------------------------------------------------------
# Mode Selection
# ---------------------------------------------------------------------
mode = st.sidebar.selectbox("Select Mode", 
    ["Single Strategy", "Set: strategies_9022", "Set: strategies_C358", "Set: strategies_EE49", "All Strategies"])

# Choose which strategies to use based on the mode.
if mode == "Single Strategy":
    all_strategies = {**strategies_9022, **strategies_C358, **strategies_EE49}
    selected_strategy = st.sidebar.selectbox("Select Strategy", list(all_strategies.keys()))
    selected_strategies = {selected_strategy: all_strategies[selected_strategy]}
elif mode == "Set: strategies_9022":
    selected_strategies = strategies_9022
elif mode == "Set: strategies_C358":
    selected_strategies = strategies_C358
elif mode == "Set: strategies_EE49":
    selected_strategies = strategies_EE49
elif mode == "All Strategies":
    selected_strategies = {**strategies_9022, **strategies_C358, **strategies_EE49}

# ---------------------------------------------------------------------
# Timeframe selection for "Show trades since:"
# ---------------------------------------------------------------------
trade_since_option = st.sidebar.selectbox("Show trades since:", 
    ["Today", "Yesterday", "Week", "Month", "All"])
if trade_since_option == "Today":
    trade_count = 50
elif trade_since_option == "Yesterday":
    trade_count = 50
elif trade_since_option == "Week":
    trade_count = 250
elif trade_since_option == "Month":
    trade_count = 1000
else:
    trade_count = "All"

# ---------------------------------------------------------------------
# Main Display
# ---------------------------------------------------------------------
if mode == "Single Strategy":
    # SINGLE STRATEGY MODE
    strategy_name, wallet_address = list(selected_strategies.items())[0]
    st.title(f"{strategy_name} Dashboard")

    # Update CSV with new trades
    updated_trade_history_df = update_trade_csv(strategy_name, wallet_address, trade_count=trade_count)

    # Fetch Balance
    balance_data = asyncio.run(fetch_balance(wallet_address))

    # CHART DataFrame: *all* trades (unfiltered) for the cumulative chart
    chart_df = updated_trade_history_df.copy()

    # ---- Filter trades for metrics & table ----
    now = datetime.now()
    if trade_since_option == "All":
        filtered_df = updated_trade_history_df.copy()
    elif trade_since_option == "Today":
        since_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        filtered_df = filter_trades_by_date(updated_trade_history_df, since_dt)
    elif trade_since_option == "Yesterday":
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today - timedelta(days=1)
        yesterday_end = today - timedelta(seconds=1)
        filtered_df = updated_trade_history_df[
            (updated_trade_history_df["Date"] >= yesterday_start) &
            (updated_trade_history_df["Date"] <= yesterday_end)
        ]
    elif trade_since_option == "Week":
        since_dt = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        filtered_df = filter_trades_by_date(updated_trade_history_df, since_dt)
    elif trade_since_option == "Month":
        since_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        filtered_df = filter_trades_by_date(updated_trade_history_df, since_dt)

    

    # Compute metrics on the *filtered* trades
    (
        total_pnl, total_fees, net_pnl, overall_win_rate, volume,
        max_gain, max_loss, avg_return_win_loss,
        longs_count, long_pct, long_win_rate, max_gain_longs, max_loss_longs, avg_gain_longs,
        shorts_count, short_pct, short_win_rate, max_gain_shorts, max_loss_shorts, avg_gain_shorts,
        sharpe_ratio
    ) = compute_metrics(filtered_df)
    
    st.subheader(f"Trade Metrics")
    
    # Row 1: Balance and Volume
    with st.container():
        r1c1, r1c2,r1c3, r1c4= st.columns(4)
        r1c1.metric("Balance", f"${balance_data.total:,.2f}")
        r1c2.metric("Volume", f"${volume:,.2f}")
        r1c3.metric("Win Rate", f"{overall_win_rate:.2f}%")
        r1c4.metric("Net P&L", f"${net_pnl:,.2f}")
    
    # -----------------------------------------------------------------
    # Cumulative Net PnL Chart (Altair) for *all* trades
    # -----------------------------------------------------------------
    if not chart_df.empty:
        chart_df["Net PnL"] = chart_df["Closed PnL"].astype(float) - chart_df["Fee"].astype(float)
        chart_df["Day"] = chart_df["Date"].dt.date
        daily_pnl = chart_df.groupby("Day")["Net PnL"].sum().reset_index()
        daily_pnl = daily_pnl.sort_values("Day")

        # Fill missing days from min_day to max_day with 0
        min_day = daily_pnl["Day"].min()
        max_day = daily_pnl["Day"].max()
        all_days = pd.date_range(start=min_day, end=max_day, freq="D")
        daily_pnl = daily_pnl.set_index("Day").reindex(all_days, fill_value=0)
        daily_pnl.index.name = "Day"
        # Compute cumulative
        daily_pnl["Cumulative PnL"] = daily_pnl["Net PnL"].cumsum()

        st.subheader("Cumulative Net P&L (All Trades)")
        chart = build_cumulative_chart(daily_pnl, title="Cumulative Net P&L (All Trades)")
        st.altair_chart(chart, use_container_width=True)
    st.markdown("---")

    # Row 2: Win Rate, Max Gain, Max Loss, Sharpe Ratio
    with st.container():
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Max Gain", f"${max_gain:,.2f}")
        r2c2.metric("Max Loss", f"${max_loss:,.2f}")
        r2c3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    st.markdown("---")

    # Row 3: Net P&L, Total Fees, P&L, Avg. Return/Trade (Win/Loss)
    with st.container():
        r3c1, r3c2 = st.columns(2)
        r3c1.metric("Total Fees", f"${total_fees:,.2f}")
        r3c2.metric("Avg. Return/Trade", avg_return_win_loss)
        ## => Add Fundings
    
    st.markdown("---")

    # Row 4: Longs metrics (6 items)
    with st.container():
        r4c1, r4c2, r4c3, r4c4, r4c5, r4c6 = st.columns(6)
        r4c1.metric("Longs Count", f"{longs_count}")
        r4c2.metric("Longs %", f"{long_pct:.2f}%")
        r4c3.metric("Longs Win Rate", f"{long_win_rate:.2f}%")
        r4c4.metric("Max Gain Longs", f"${max_gain_longs:,.2f}")
        r4c5.metric("Max Loss Longs", f"${max_loss_longs:,.2f}")
        r4c6.metric("Avg. Gains Longs", f"${avg_gain_longs:,.2f}")

    st.markdown("---")

    # Row 5: Shorts metrics (6 items)
    with st.container():
        r5c1, r5c2, r5c3, r5c4, r5c5, r5c6 = st.columns(6)
        r5c1.metric("Shorts Count", f"{shorts_count}")
        r5c2.metric("Shorts %", f"{short_pct:.2f}%")
        r5c3.metric("Shorts Win Rate", f"{short_win_rate:.2f}%")
        r5c4.metric("Max Gain Shorts", f"${max_gain_shorts:,.2f}")
        r5c5.metric("Max Loss Shorts", f"${max_loss_shorts:,.2f}")
        r5c6.metric("Avg. Gains Shorts", f"${avg_gain_shorts:,.2f}")

    st.markdown("---")

    # ---------------------------
    # NEW ROW: Pie/Bar Charts
    # ---------------------------
    # 1) Count (pie)
    pie_count = build_pie_chart(longs_count, shorts_count)
    # 2) Percentage (pie)
    pie_pct = build_pie_chart(long_pct, short_pct)
    # 3) Win Rate (pie)
    pie_winrate = build_pie_chart(long_win_rate, short_win_rate)
    # 4) Max Gain (bar)
    bar_maxgain = build_bar_chart(max_gain_longs, max_gain_shorts)
    # 5) Max Loss (bar) -> we use absolute value so a bigger negative is bigger bar
    bar_maxloss = build_bar_chart(abs(max_loss_longs), abs(max_loss_shorts))

    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)

        # Chart 1: Count
        c1.altair_chart(pie_count, use_container_width=True)
        c1.write("Count")

        # Chart 2: %
        c2.altair_chart(pie_pct, use_container_width=True)
        c2.write("%")

        # Chart 3: Win Rate
        c3.altair_chart(pie_winrate, use_container_width=True)
        c3.write("Win Rate")

        # Chart 4: Max Gain
        c4.altair_chart(bar_maxgain, use_container_width=True)
        c4.write("Max Gain")

        # Chart 5: Max Loss
        c5.altair_chart(bar_maxloss, use_container_width=True)
        c5.write("Max Loss")

    st.subheader(f"Trade History ({trade_since_option})")
    st.table(filtered_df)

else:
    # AGGREGATED MODE
    st.title(f"Aggregated Dashboard ({mode})")
    aggregated_trade_history = []
    total_balance = 0

    async def fetch_all_balances(strat_dict):
        tasks = []
        for strat_name, addr in strat_dict.items():
            tasks.append(fetch_balance(addr))
        return await asyncio.gather(*tasks)

    # Update CSV for each selected strategy.
    for strat_name, wallet_addr in selected_strategies.items():
        df_updated = update_trade_csv(strat_name, wallet_addr, trade_count=trade_count)
        aggregated_trade_history.append(df_updated)

    # Fetch balances concurrently.
    all_balances = asyncio.run(fetch_all_balances(selected_strategies))
    total_balance = sum(b.total for b in all_balances)

    # Concatenate all CSV DataFrames for All Time
    if aggregated_trade_history:
        all_time_agg_df = pd.concat(aggregated_trade_history, ignore_index=True)
        all_time_agg_df["Date"] = pd.to_datetime(all_time_agg_df["Date"], errors="coerce")
        all_time_agg_df = all_time_agg_df.sort_values(by="Date", ascending=False)
    else:
        all_time_agg_df = pd.DataFrame(columns=CSV_COLUMNS)

    # For the chart, use all_time_agg_df (full history)
    chart_df = all_time_agg_df.copy()

    # Filter Aggregated Data Based on User Selection
    now = datetime.now()
    if trade_since_option == "All":
        filtered_agg_df = all_time_agg_df.copy()
    elif trade_since_option == "Today":
        since_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        filtered_agg_df = filter_trades_by_date(all_time_agg_df, since_dt)
    elif trade_since_option == "Yesterday":
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today - timedelta(days=1)
        yesterday_end = today - timedelta(seconds=1)
        filtered_agg_df = all_time_agg_df[
            (all_time_agg_df["Date"] >= yesterday_start) &
            (all_time_agg_df["Date"] <= yesterday_end)
        ]
    elif trade_since_option == "Week":
        since_dt = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        filtered_agg_df = filter_trades_by_date(all_time_agg_df, since_dt)
    elif trade_since_option == "Month":
        since_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        filtered_agg_df = filter_trades_by_date(all_time_agg_df, since_dt)

    # Compute metrics on the *filtered* DataFrame
    (
        f_total_pnl, f_total_fees, f_net_pnl, f_overall_win_rate, f_volume,
        f_max_gain, f_max_loss, f_avg_return_win_loss,
        f_longs_count, f_long_pct, f_long_win_rate, f_max_gain_longs, f_max_loss_longs, f_avg_gain_longs,
        f_shorts_count, f_short_pct, f_short_win_rate, f_max_gain_shorts, f_max_loss_shorts, f_avg_gain_shorts,
        f_sharpe_ratio
    ) = compute_metrics(filtered_agg_df)

    # Row 1: total balance, volume
    with st.container():
        ar1c1, ar1c2, ar1c3, ar1c4 = st.columns(4)
        ar1c1.metric("Balance", f"${total_balance:,.2f}")
        ar1c2.metric("Volume", f"${f_volume:,.2f}")
        ar1c3.metric("Win Rate", f"{f_overall_win_rate:.2f}%")
        ar1c4.metric("Net P&L", f"${f_net_pnl:,.2f}")
    
    # -----------------------------------------------------------------
    # Altair Chart: Cumulative Net PnL (All Trades, unfiltered)
    # -----------------------------------------------------------------
    if not chart_df.empty:
        chart_df["Net PnL"] = chart_df["Closed PnL"].astype(float) - chart_df["Fee"].astype(float)
        chart_df["Day"] = chart_df["Date"].dt.date
        daily_pnl = chart_df.groupby("Day")["Net PnL"].sum().reset_index()
        daily_pnl = daily_pnl.sort_values("Day")

        # Fill missing days from min_day to max_day with 0
        min_day = daily_pnl["Day"].min()
        max_day = daily_pnl["Day"].max()
        all_days = pd.date_range(start=min_day, end=max_day, freq="D")
        daily_pnl = daily_pnl.set_index("Day").reindex(all_days, fill_value=0)
        daily_pnl.index.name = "Day"
        daily_pnl["Cumulative PnL"] = daily_pnl["Net PnL"].cumsum()

        chart = build_cumulative_chart(daily_pnl, title="Cumulative Net P&L (All Trades)")
        st.altair_chart(chart, use_container_width=True)

    # Row 4: Longs metrics (6 items)
    with st.container():
        ar4c1, ar4c2, ar4c3, ar4c4, ar4c5, ar4c6 = st.columns(6)
        ar4c1.metric("Longs Count", f"{f_longs_count}")
        ar4c2.metric("Longs %", f"{f_long_pct:.2f}%")
        ar4c3.metric("Longs Win Rate", f"{f_long_win_rate:.2f}%")
        ar4c4.metric("Max Gain Longs", f"${f_max_gain_longs:,.2f}")
        ar4c5.metric("Max Loss Longs", f"${f_max_loss_longs:,.2f}")
        ar4c6.metric("Avg. Gains Longs", f"${f_avg_gain_longs:,.2f}")
    
    st.markdown("---")

    # Row 5: Shorts metrics (6 items)
    with st.container():
        ar5c1, ar5c2, ar5c3, ar5c4, ar5c5, ar5c6 = st.columns(6)
        ar5c1.metric("Shorts Count", f"{f_shorts_count}")
        ar5c2.metric("Shorts %", f"{f_short_pct:.2f}%")
        ar5c3.metric("Shorts Win Rate", f"{f_short_win_rate:.2f}%")
        ar5c4.metric("Max Gain Shorts", f"${f_max_gain_shorts:,.2f}")
        ar5c5.metric("Max Loss Shorts", f"${f_max_loss_shorts:,.2f}")
        ar5c6.metric("Avg. Gains Shorts", f"${f_avg_gain_shorts:,.2f}")

    st.markdown("---")

    # ---------------------------
    # NEW ROW: Pie/Bar Charts
    # ---------------------------
    # 1) Count (pie)
    pie_count = build_pie_chart(f_longs_count, f_shorts_count)
    # 2) Percentage (pie)
    pie_pct = build_pie_chart(f_long_pct, f_short_pct)
    # 3) Win Rate (pie)
    pie_winrate = build_pie_chart(f_long_win_rate, f_short_win_rate)
    # 4) Max Gain (bar)
    bar_maxgain = build_bar_chart(f_max_gain_longs, f_max_gain_shorts)
    # 5) Max Loss (bar) -> abs values
    bar_maxloss = build_bar_chart(abs(f_max_loss_longs), abs(f_max_loss_shorts))

    with st.container():
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)

        pc1.altair_chart(pie_count, use_container_width=True)
        pc1.write("Count")

        pc2.altair_chart(pie_pct, use_container_width=True)
        pc2.write("%")

        pc3.altair_chart(pie_winrate, use_container_width=True)
        pc3.write("Win Rate")

        pc4.altair_chart(bar_maxgain, use_container_width=True)
        pc4.write("Max Gain")

        pc5.altair_chart(bar_maxloss, use_container_width=True)
        pc5.write("Max Loss")

    st.subheader(f"Trade History ({trade_since_option}, Aggregated)")
    st.table(filtered_agg_df)
